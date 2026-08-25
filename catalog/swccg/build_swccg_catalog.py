#!/usr/bin/env python3
"""Example game-specific image downloader for a new CollectorVision catalog.

Purpose
-------
This file is intentionally a small, readable example of the first half of adding
a new card game to CollectorVision: get the game's reference images onto disk
with stable filenames, and write enough metadata to know where they came from.

For SWCCG, the source data comes from the SWCCG Players Committee JSON files.
The script downloads current, physically printed cards only, preferring
Holotable `hires/*.png` images when present and falling back to the `large` image
URL in the JSON record.

Two-Part Catalog Flow
---------------------
1. A game-specific script like this one downloads images into a plain folder
   tree. This script writes files like:

       catalog/swccg/build/images/Premiere/p_94.gif
       catalog/swccg/build/images/Premiere/p_146.png
       catalog/swccg/build/images/Hoth/h_5300.png

   The subfolder layout is flexible. Use set names, UUID prefixes, rarity
   folders, or anything else that makes sense for the source. The important
   part is that each image filename is the ID you want returned by the catalog.

2. Run the generic builder:

       python catalog/build_catalog_from_images.py \
            --image-dir catalog/swccg/build/images \
            --embedder milo1 \
            --game swccg \
            --primary-key-name swccg_id

   It builds the standard CollectorVision NPZ and derives the output filename:

       milo1-swccg-YYYY-MM-DD.npz

   Conceptually, that NPZ contains:

       {
         "embeddings": "float32 array, one row per image",
         "card_ids": ["p_146.png", "p_190.png", "p_191.gif"],
         "source": "swccg",
         "embedder_spec": {"algo_key": "milo1", "image_size": 448}
       }

   A companion `.metadata.jsonl` file maps each returned ID back to the image
   path, for example: {"swccg_id": "p_146.png", "relative_path": "Premiere/p_146.png"}.

Adapting This To Another Game
-----------------------------
Copy this file, replace the source-specific JSON parsing in `normalize_records`,
and keep the output convention simple: download image files under an `images/`
directory with stable, meaningful filenames. Once the files are there, the
generic builder does not care which game they came from.

Example:
    python catalog/swccg/build_swccg_catalog.py --limit 25
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

SOURCE_REPO = "swccgpc/swccg-card-json"
SOURCE_BRANCH = "main"
HOLOTABLE_API_BASE = "https://api.github.com/repos/swccgpc/holotable/contents/Images-HT/starwars"
HOLOTABLE_RAW_BASE = "https://raw.githubusercontent.com/swccgpc/holotable/master/Images-HT/starwars"

CURRENT_CARD_FILES = ("Light.json", "Dark.json")
VIRTUAL_SET_MIN = 200
USER_AGENT = "CollectorVision SWCCG catalog research (https://github.com/HanClinto/CollectorVision)"
_WARNED_HIRES_UNAVAILABLE = False


@dataclass(frozen=True)
class CardRecord:
    swccg_id: str
    source: str
    source_ref: str
    source_file: str
    source_id: int
    gemp_id: str | None
    side: str
    set_id: str
    set_name: str
    set_abbr: str
    rarity: str | None
    face: str
    title: str
    type: str | None
    image_url: str
    preferred_image_url: str
    preferred_image_kind: str
    local_image: str

    def to_json(self) -> dict[str, Any]:
        return {
            "swccg_id": self.swccg_id,
            "source": self.source,
            "source_ref": self.source_ref,
            "source_file": self.source_file,
            "source_id": self.source_id,
            "gemp_id": self.gemp_id,
            "side": self.side,
            "set_id": self.set_id,
            "set_name": self.set_name,
            "set_abbr": self.set_abbr,
            "rarity": self.rarity,
            "face": self.face,
            "title": self.title,
            "type": self.type,
            "image_url": self.image_url,
            "preferred_image_url": self.preferred_image_url,
            "preferred_image_kind": self.preferred_image_kind,
            "local_image": self.local_image,
        }


class ThrottledFetcher:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self._last_request = 0.0

    def open(self, url: str, timeout: float = 30.0):
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        headers = {"User-Agent": USER_AGENT}
        github_token = os.environ.get("GITHUB_TOKEN")
        if github_token and "api.github.com" in url:
            headers["Authorization"] = f"Bearer {github_token}"
        request = Request(url, headers=headers)
        response = urlopen(request, timeout=timeout)
        self._last_request = time.monotonic()
        return response

    def get_json(self, url: str) -> Any:
        with self.open(url) as response:
            return json.load(response)


def read_or_fetch_json(cache_path: Path, url: str, fetcher: ThrottledFetcher) -> Any:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    data = fetcher.get_json(url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def safe_cache_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def safe_path_part(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unknown"


def source_base_url(source_ref: str) -> str:
    return f"https://raw.githubusercontent.com/{SOURCE_REPO}/{source_ref}"


def set_is_physical_current(set_id: str, sets_by_id: dict[str, dict[str, Any]]) -> bool:
    info = sets_by_id.get(set_id)
    if not info or info.get("legacy"):
        return False
    try:
        return int(set_id) < VIRTUAL_SET_MIN
    except ValueError:
        return False


def image_path_parts(image_url: str) -> tuple[str, str] | None:
    marker = "/cards/"
    if marker not in image_url:
        return None
    path = image_url.split(marker, 1)[1]
    if path.startswith("Images-HT/starwars/"):
        path = path[len("Images-HT/starwars/") :]
    parts = path.split("/")
    if len(parts) < 3 or parts[-2] != "large":
        return None
    set_side = parts[-3]
    filename = parts[-1]
    return set_side, filename


def discover_hires_names(
    set_side: str,
    cache_dir: Path,
    fetcher: ThrottledFetcher,
    no_hires: bool,
) -> set[str]:
    global _WARNED_HIRES_UNAVAILABLE

    if no_hires:
        return set()

    cache_path = cache_dir / "holotable_hires" / f"{set_side}.json"
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        url = f"{HOLOTABLE_API_BASE}/{set_side}/hires?ref=master"
        try:
            data = fetcher.get_json(url)
        except HTTPError as exc:
            if exc.code in {403, 404}:
                if exc.code == 403 and not _WARNED_HIRES_UNAVAILABLE:
                    print(
                        "warning: Holotable hires discovery is unavailable from GitHub API; "
                        "falling back to large images. Set GITHUB_TOKEN to raise API limits."
                    )
                    _WARNED_HIRES_UNAVAILABLE = True
                data = []
            else:
                raise
        except URLError:
            if not _WARNED_HIRES_UNAVAILABLE:
                print("warning: Holotable hires discovery failed; falling back to large images.")
                _WARNED_HIRES_UNAVAILABLE = True
            data = []
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    return {item.get("name", "") for item in data if isinstance(item, dict)}


def preferred_image_url(
    image_url: str,
    cache_dir: Path,
    fetcher: ThrottledFetcher,
    no_hires: bool,
) -> tuple[str, str]:
    parts = image_path_parts(image_url)
    if parts is None:
        return image_url, "large"

    set_side, filename = parts
    png_name = f"{Path(filename).stem}.png"
    hires_names = discover_hires_names(set_side, cache_dir, fetcher, no_hires)
    if png_name in hires_names:
        return f"{HOLOTABLE_RAW_BASE}/{set_side}/hires/{png_name}", "hires"
    return image_url, "large"


def image_suffix(image_url: str) -> str:
    return Path(image_url.split("?", 1)[0]).suffix.lower() or ".img"


def make_swccg_id(set_abbr: str, source_id: int, face: str, image_url: str) -> str:
    face_suffix = "" if face == "front" else f"_{face}"
    return f"{safe_path_part(set_abbr).lower()}_{source_id}{face_suffix}{image_suffix(image_url)}"


def local_image_path(images_dir: Path, set_name: str, swccg_id: str) -> Path:
    return images_dir / safe_path_part(set_name) / swccg_id


def normalize_records(
    cache_dir: Path,
    images_dir: Path,
    fetcher: ThrottledFetcher,
    no_hires: bool,
    source_ref: str,
) -> list[CardRecord]:
    source_cache_dir = cache_dir / "source" / safe_cache_name(source_ref)
    source_base = source_base_url(source_ref)
    sets_data = read_or_fetch_json(
        source_cache_dir / "sets.json", f"{source_base}/sets.json", fetcher
    )
    sets_by_id = {str(item["id"]): item for item in sets_data}
    records: list[CardRecord] = []

    for source_file in CURRENT_CARD_FILES:
        data = read_or_fetch_json(
            source_cache_dir / source_file,
            f"{source_base}/{source_file}",
            fetcher,
        )
        for card in data.get("cards", []):
            if card.get("legacy"):
                continue
            set_id = str(card.get("set", ""))
            if not set_is_physical_current(set_id, sets_by_id):
                continue

            set_info = sets_by_id[set_id]
            for face in ("front", "back"):
                face_data = card.get(face) or {}
                image_url = face_data.get("imageUrl")
                title = face_data.get("title")
                if not image_url or not title:
                    continue

                source_id = int(card["id"])
                preferred_url, preferred_kind = preferred_image_url(
                    image_url, cache_dir, fetcher, no_hires
                )
                set_name = str(set_info.get("name", set_id))
                set_abbr = str(set_info.get("abbr", ""))
                swccg_id = make_swccg_id(set_abbr, source_id, face, preferred_url)
                image_path = local_image_path(images_dir, set_name, swccg_id)
                records.append(
                    CardRecord(
                        swccg_id=swccg_id,
                        source="swccgpc-json",
                        source_ref=source_ref,
                        source_file=source_file,
                        source_id=source_id,
                        gemp_id=card.get("gempId"),
                        side=str(card.get("side", "")),
                        set_id=set_id,
                        set_name=set_name,
                        set_abbr=set_abbr,
                        rarity=card.get("rarity"),
                        face=face,
                        title=str(title),
                        type=face_data.get("type"),
                        image_url=str(image_url),
                        preferred_image_url=preferred_url,
                        preferred_image_kind=preferred_kind,
                        local_image=str(image_path),
                    )
                )

    records.sort(key=lambda r: (int(r.set_id), r.side, r.title, r.face, r.source_id))
    metadata = {
        "source_repo": SOURCE_REPO,
        "source_ref": source_ref,
        "card_files": list(CURRENT_CARD_FILES),
        "include_legacy": False,
        "include_pre_errata": False,
        "include_virtual": False,
        "record_count": len(records),
    }
    (cache_dir / "source_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return records


def write_metadata(records: list[CardRecord], metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json(), ensure_ascii=False, sort_keys=True) + "\n")


def is_valid_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def download_image(record: CardRecord, fetcher: ThrottledFetcher, force: bool) -> bool:
    destination = Path(record.local_image)
    if not force and is_valid_image(destination):
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    with fetcher.open(record.preferred_image_url) as response, temp_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 128)
            if not chunk:
                break
            handle.write(chunk)

    if not is_valid_image(temp_path):
        temp_path.unlink(missing_ok=True)
        raise ValueError(f"Downloaded file is not a valid image: {record.preferred_image_url}")

    temp_path.replace(destination)
    return True


def download_images(
    records: list[CardRecord], fetcher: ThrottledFetcher, force: bool
) -> tuple[int, int]:
    downloaded = 0
    cached = 0
    for record in records:
        path = Path(record.local_image)
        if not force and is_valid_image(path):
            cached += 1
            continue
        if download_image(record, fetcher, force):
            downloaded += 1
    return downloaded, cached


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "build")
    parser.add_argument(
        "--source-ref",
        default=SOURCE_BRANCH,
        help="swccg-card-json ref to fetch, preferably a commit SHA for reproducible builds",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit records for a proof run")
    parser.add_argument(
        "--throttle", type=float, default=0.25, help="Seconds between network requests"
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-hires", action="store_true", help="Skip Holotable hires discovery")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    cache_dir = out_dir / "cache"
    images_dir = out_dir / "images"
    metadata_path = out_dir / "metadata.jsonl"

    fetcher = ThrottledFetcher(args.throttle)
    records = normalize_records(cache_dir, images_dir, fetcher, args.no_hires, args.source_ref)
    if args.limit > 0:
        records = records[: args.limit]

    write_metadata(records, metadata_path)
    downloaded, cached = download_images(records, fetcher, args.force_download)

    hires_count = sum(1 for record in records if record.preferred_image_kind == "hires")
    print(f"records: {len(records)}")
    print(f"hires preferred: {hires_count}")
    print(f"images downloaded: {downloaded}")
    print(f"images cached: {cached}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
