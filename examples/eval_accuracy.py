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
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5],
                        help="One or more k values to report (default: 1 3 5)")
    args = parser.parse_args()

    paths = (sorted(args.input.glob("*.jpg"))
             if args.input.is_dir() else [args.input])
    paths = [p for p in paths if UUID_RE.search(p.name)]
    if not paths:
        sys.exit("No UUID-labelled images found.")

    catalog  = cvg.Catalog.load(args.catalog)
    detector = cvg.NeuralCornerDetector()
    max_k    = max(args.top_k)

    detected = total = 0
    edition_hits = {k: 0 for k in args.top_k}
    oracle_hits  = {k: 0 for k in args.top_k}

    for path in paths:
        true_id = UUID_RE.search(path.name).group(0).lower()
        if true_id not in set(catalog.card_ids):
            continue
        total += 1

        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        detection = detector.detect(bgr)
        if not detection.card_present:
            continue
        detected += 1

        emb  = catalog.embedder.embed([detection.dewarp(bgr)])[0]
        hits = [cid for _, cid in catalog.search(emb, top_k=max_k)]

        true_oracle = catalog.card_to_oracle.get(true_id)
        hit_oracles = [catalog.card_to_oracle.get(c) for c in hits]

        for k in args.top_k:
            edition_hits[k] += true_id in hits[:k]
            if true_oracle:
                oracle_hits[k] += true_oracle in hit_oracles[:k]

    def pct(n, d): return f"{100 * n / d:.1f}%" if d else "—"

    ks = "   ".join(f"top-{k}" for k in args.top_k)
    print(f"Images:   {total}  ({detected} detected)")
    print(f"          {ks}")
    print("Edition   " + "   ".join(f"{pct(edition_hits[k], detected):>5}" for k in args.top_k))
    print("Card      " + "   ".join(f"{pct(oracle_hits[k],  detected):>5}" for k in args.top_k))


if __name__ == "__main__":
    main()
