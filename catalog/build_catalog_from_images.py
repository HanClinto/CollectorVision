#!/usr/bin/env python3
"""Build a CollectorVision NPZ from a folder of reference images.

This is the generic second half of adding a new game:

1. Put images anywhere under a root directory. Subfolders are fine.
2. Name each image file with the ID you want CollectorVision to return.
3. Run this script against the image root.

Example:
        python catalog/build_catalog_from_images.py \
            --image-dir catalog/swccg/build/images \
            --embedder milo1 \
            --game swccg \
            --primary-key-name swccg_id

That writes files named like:

    catalog/swccg/build/milo1-swccg-YYYY-MM-DD.npz
    catalog/swccg/build/milo1-swccg-YYYY-MM-DD.metadata.jsonl

By default, the catalog ID is the image filename including extension, e.g.
`p_146.png`.

The NPZ has the standard CollectorVision keys: `embeddings`, `card_ids`,
`source`, and `embedder_spec`. The companion metadata JSONL is intentionally
plain, for example:

    {"swccg_id": "p_146.png", "relative_path": "Premiere/p_146.png"}
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}


def safe_name(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")


def catalog_stem(embedder: str, game: str, catalog_date: str) -> str:
    return "-".join(safe_name(part) for part in (embedder, game, catalog_date))


def iter_images(image_root: Path) -> list[Path]:
    return sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def build_catalog(
    image_root: Path,
    output_path: Path,
    metadata_path: Path,
    primary_key_name: str,
    game: str,
    embedder_name: str,
    batch_size: int,
) -> None:
    import collector_vision as cvg

    if embedder_name != "milo1":
        raise ValueError("This example currently supports --embedder milo1")

    image_paths = iter_images(image_root)
    if not image_paths:
        raise ValueError(f"No supported images found under {image_root}")

    ids = [path.name for path in image_paths]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image filenames found; use unique filenames for catalog IDs")

    embedder = cvg.NeuralEmbedder(batch_size=batch_size)
    images = [Image.open(path).convert("RGB") for path in image_paths]
    embeddings = embedder.embed(images).astype(np.float32)
    for image in images:
        image.close()

    embedder_spec = {
        "kind": "neural",
        "algo_key": embedder_name,
        "image_size": 448,
        "game": game,
        "source": game,
        "primary_key_name": primary_key_name,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        card_ids=np.asarray(ids, dtype=str),
        source=np.array(game),
        embedder_spec=np.array(json.dumps(embedder_spec, sort_keys=True)),
    )

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        for path, item_id in zip(image_paths, ids):
            row = {
                primary_key_name: item_id,
                "relative_path": path.relative_to(image_root).as_posix(),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"images: {len(image_paths)}")
    print(f"catalog: {output_path}")
    print(f"metadata: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--embedder", default="milo1")
    parser.add_argument("--game", required=True)
    parser.add_argument("--catalog-date", default=date.today().isoformat())
    parser.add_argument("--primary-key-name", default="card_id")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = (
        args.image_dir.parent / f"{catalog_stem(args.embedder, args.game, args.catalog_date)}.npz"
    )
    metadata_path = output_path.with_suffix(".metadata.jsonl")
    build_catalog(
        args.image_dir,
        output_path,
        metadata_path,
        args.primary_key_name,
        args.game,
        args.embedder,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
