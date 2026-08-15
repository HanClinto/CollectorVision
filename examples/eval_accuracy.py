#!/usr/bin/env python3
"""Evaluate edition and card accuracy on UUID-labelled card images.

The Scryfall UUID must appear somewhere in each filename:
    {uuid}_CardName_date.{jpg,jpeg,png,webp}

Reports:
  Edition accuracy — top result is the exact printing (Scryfall UUID match)
  Card accuracy    — top result is the same card, any printing (oracle_id match)

Usage
-----
    python examples/eval_accuracy.py image.jpg --catalog mtg
    python examples/eval_accuracy.py images/ --catalog mtg --no-rot-invariant
    python examples/eval_accuracy.py images/ --catalog mtg --verbose
    python examples/eval_accuracy.py images/ --catalog mtg --debug
"""

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import cv2

import collector_vision as cvg

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def image_paths(input_path: Path) -> list[Path]:
    """Return supported image paths from a file or directory."""
    if input_path.is_dir():
        return sorted(
            path for path in input_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return [input_path]


def uuid_from_filename(path: Path) -> str | None:
    """Return the first UUID embedded in a filename, if present."""
    match = UUID_RE.search(path.name)
    return match.group(0).lower() if match else None


def search_hits(catalog: cvg.CatalogLike, crop, top_k: int, rot_invariant: bool) -> list[str]:
    """Return top card IDs, optionally trying upright and upside-down crops."""
    crops = [crop]
    if rot_invariant:
        crops.append(cvg.rotate_card_180(crop))

    embeddings = catalog.embedder.embed(crops)
    searches = [catalog.search(embedding, top_k=top_k) for embedding in embeddings]
    best_hits = max(searches, key=lambda hits: hits[0][0])
    return [cid for _, cid in best_hits]


def fetch_scryfall_card(card_id: str, cache: dict[str, dict]) -> dict:
    """Return Scryfall metadata for a card ID, caching lookups for this run."""
    if card_id not in cache:
        req = urllib.request.Request(
            f"https://api.scryfall.com/cards/{card_id}",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            cache[card_id] = json.loads(response.read())
    return cache[card_id]


def card_label(card: dict | None) -> str:
    """Return a compact card name / set label for verbose eval output."""
    if not card:
        return "unknown"
    name = card.get("name", "unknown")
    set_name = card.get("set_name", "unknown set")
    set_code = str(card.get("set", "")).upper()
    return f"{name} / {set_name}" + (f" ({set_code})" if set_code else "")


def print_verbose_result(
    path: Path,
    expected_id: str,
    matched_id: str | None,
    metadata_cache: dict[str, dict],
) -> None:
    expected = fetch_scryfall_card(expected_id, metadata_cache)
    matched = fetch_scryfall_card(matched_id, metadata_cache) if matched_id else None
    print(f"{path.name}")
    print(f"  expected  {card_label(expected)}")
    print(f"  matched   {card_label(matched)}")


def draw_corner_overlay(bgr, detection):
    """Return an image annotated with detector corners and confidence details."""
    out = bgr.copy()
    if detection.corners is not None:
        h, w = bgr.shape[:2]
        pts = (detection.corners * [w, h]).astype("int32")
        for i in range(4):
            cv2.line(out, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
        for label, pt in zip(["TL", "TR", "BR", "BL"], pts):
            cv2.circle(out, tuple(pt), 6, (0, 0, 255), -1)
            cv2.putText(
                out,
                label,
                tuple(pt + [8, -8]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )

    status = "CARD" if detection.card_present else "NO CARD"
    details = [status, f"confidence={detection.confidence:.4f}"]
    if detection.sharpness is not None:
        details.append(f"sharpness={detection.sharpness:.4f}")
    presence = detection.extra.get("presence")
    if presence is not None:
        details.append(f"presence={presence:.4f}")
    cv2.putText(
        out,
        "  ".join(details),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
    )
    return out


def save_debug_artifacts(path: Path, bgr, detection, debug_dir: Path, crop=None) -> None:
    """Save detector overlay and, when possible, aligned card crop for one image."""
    overlays_dir = debug_dir / "overlays"
    aligned_dir = debug_dir / "aligned"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)

    overlay = draw_corner_overlay(bgr, detection)
    cv2.imwrite(str(overlays_dir / f"{path.stem}_corners.jpg"), overlay)

    if detection.card_present:
        aligned = crop if crop is not None else detection.dewarp(bgr)
        aligned.save(aligned_dir / f"{path.stem}_aligned.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path, help="Image file or directory")
    parser.add_argument(
        "--catalog",
        required=True,
        help="Game name (for v2), hf:// reference, or local .npz path",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="One or more k values to report (default: 1 3 5)",
    )
    parser.add_argument(
        "--no-rot-invariant",
        dest="rot_invariant",
        action="store_false",
        help="Only search the dewarped crop instead of also trying a 180-degree rotated copy.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print expected and matched card name / set for each evaluated image.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save aligned card crops and corner-detector overlays for evaluated images.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("eval_accuracy_debug"),
        help="Directory for --debug artifacts (default: eval_accuracy_debug).",
    )
    parser.set_defaults(rot_invariant=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    paths = image_paths(args.input)
    paths = [p for p in paths if uuid_from_filename(p)]
    if not paths:
        sys.exit("No UUID-labelled images found.")

    catalog = cvg.Catalog.load(args.catalog)
    catalog_card_ids = set(catalog.card_ids)
    detector = cvg.NeuralCornerDetector()
    max_k = max(args.top_k)

    detected = total = 0
    edition_hits = {k: 0 for k in args.top_k}
    oracle_hits = {k: 0 for k in args.top_k}
    metadata_cache: dict[str, dict] = {}

    for path in paths:
        true_id = uuid_from_filename(path)
        if true_id is None or true_id not in catalog_card_ids:
            continue
        total += 1

        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        detection = detector.detect(bgr)
        if not detection.card_present:
            if args.debug:
                save_debug_artifacts(path, bgr, detection, args.debug_dir)
            if args.verbose:
                print_verbose_result(path, true_id, None, metadata_cache)
            continue
        detected += 1

        crop = detection.dewarp(bgr)
        if args.debug:
            save_debug_artifacts(path, bgr, detection, args.debug_dir, crop=crop)

        hits = search_hits(catalog, crop, top_k=max_k, rot_invariant=args.rot_invariant)
        if args.verbose:
            print_verbose_result(path, true_id, hits[0] if hits else None, metadata_cache)

        true_oracle = catalog.card_to_oracle.get(true_id)
        hit_oracles = [catalog.card_to_oracle.get(c) for c in hits]

        for k in args.top_k:
            edition_hits[k] += true_id in hits[:k]
            if true_oracle:
                oracle_hits[k] += true_oracle in hit_oracles[:k]

    def pct(n, d):
        return f"{100 * n / d:.1f}%" if d else "—"

    ks = "   ".join(f"top-{k}" for k in args.top_k)
    print(f"Images:   {total}  ({detected} detected)")
    print(f"          {ks}")
    print("Edition   " + "   ".join(f"{pct(edition_hits[k], detected):>5}" for k in args.top_k))
    print("Card      " + "   ".join(f"{pct(oracle_hits[k], detected):>5}" for k in args.top_k))


if __name__ == "__main__":
    main()
