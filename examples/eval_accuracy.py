#!/usr/bin/env python3
"""Evaluate edition and card accuracy on UUID-labelled card images.

The Scryfall UUID must appear somewhere in each filename:
    {uuid}_CardName_date.jpg

Reports:
  Edition accuracy — top result is the exact printing (Scryfall UUID match)
  Card accuracy    — top result is the same card, any printing (oracle_id match)

Usage
-----
    python examples/eval_accuracy.py image.jpg --catalog hf://HanClinto/milo/scryfall-mtg
    python examples/eval_accuracy.py images/   --catalog hf://HanClinto/milo/scryfall-mtg
"""
import argparse
import re
import sys
from pathlib import Path

import cv2
import collector_vision as cvg

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Image file or directory")
    parser.add_argument("--catalog", required=True, help="hf://user/repo/key or .npz path")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    paths = (sorted(args.input.glob("*.jpg"))
             if args.input.is_dir() else [args.input])
    paths = [p for p in paths if UUID_RE.search(p.name)]
    if not paths:
        sys.exit("No UUID-labelled images found.")

    catalog  = cvg.Catalog.load(args.catalog)
    detector = cvg.NeuralCornerDetector()

    detected = edition1 = editionk = oracle1 = oraclek = total = 0

    for path in paths:
        true_id = UUID_RE.search(path.name).group(0).lower()
        if true_id not in set(catalog.card_ids):
            continue
        total += 1

        detection = detector.detect(cv2.imread(str(path)))
        if not detection.card_present:
            continue
        detected += 1

        emb  = catalog.embedder.embed([detection.dewarp(cv2.imread(str(path)))])[0]
        hits = [cid for _, cid in catalog.search(emb, top_k=args.top_k)]

        edition1  += hits[0] == true_id
        editionk  += true_id in hits

        true_oracle  = catalog.card_to_oracle.get(true_id)
        hit_oracles  = [catalog.card_to_oracle.get(c) for c in hits]
        oracle1  += bool(true_oracle) and hit_oracles[0] == true_oracle
        oraclek  += bool(true_oracle) and true_oracle in hit_oracles

    def pct(n, d): return f"{100 * n / d:.1f}%" if d else "—"

    print(f"Images:          {total}  ({detected} detected)")
    print(f"Edition  top-1:  {pct(edition1,  detected)}   top-{args.top_k}: {pct(editionk,  detected)}")
    print(f"Card     top-1:  {pct(oracle1,   detected)}   top-{args.top_k}: {pct(oraclek,   detected)}")


if __name__ == "__main__":
    main()
