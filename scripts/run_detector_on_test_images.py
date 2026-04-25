#!/usr/bin/env python3
"""Run NeuralCornerDetector on test images and save dewarped crops.

Usage
-----
    python scripts/run_detector_on_test_images.py <input_dir> <output_dir>
    python scripts/run_detector_on_test_images.py <input_dir> <output_dir> --min-sharpness 0.02

Walks all .jpg/.png in input_dir, runs Cornelius, saves:
  - dewarped crop as <stem>_dewarped.jpg  (card_present=True)
  - original with "NO CARD" annotation    (card_present=False)

Prints a summary at the end.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def annotate_no_card(bgr: np.ndarray, sharpness: float | None) -> np.ndarray:
    out = bgr.copy()
    label = f"NO CARD  sharpness={sharpness:.4f}" if sharpness is not None else "NO CARD"
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out


def draw_corners(bgr: np.ndarray, corners_norm: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    out = bgr.copy()
    pts = (corners_norm * [w, h]).astype(np.int32)
    for i in range(4):
        cv2.line(out, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(out, tuple(pt), 6, (0, 0, 255), -1)
    return out


def run(input_dir: Path, output_dir: Path, min_sharpness: float) -> None:
    import collector_vision as cvg

    images = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    if not images:
        print(f"No images found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    crops_dir = output_dir / "dewarped"
    debug_dir = output_dir / "debug_corners"
    crops_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    detector = cvg.NeuralCornerDetector()

    n_detected = 0
    n_missed = 0

    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  SKIP (unreadable): {img_path.name}")
            continue

        detection = detector.detect(bgr, min_sharpness=min_sharpness)
        sharpness = detection.sharpness

        if detection.card_present:
            crop = detection.dewarp(bgr)
            crop_path = crops_dir / f"{img_path.stem}_dewarped.jpg"
            crop.save(str(crop_path))

            debug = draw_corners(bgr, detection.corners)
            cv2.imwrite(str(debug_dir / f"{img_path.stem}_corners.jpg"), debug)

            n_detected += 1
        else:
            annotated = annotate_no_card(bgr, sharpness)
            cv2.imwrite(str(debug_dir / f"{img_path.stem}_nocard.jpg"), annotated)
            n_missed += 1
            print(
                f"  missed: {img_path.name}  sharpness={sharpness:.4f}"
                if sharpness is not None
                else f"  missed: {img_path.name}"
            )

    total = n_detected + n_missed
    print(f"\n{input_dir.name}: {n_detected}/{total} detected  ({100 * n_detected / total:.1f}%)")
    print(f"  dewarped crops → {crops_dir}")
    print(f"  debug overlays → {debug_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--min-sharpness", type=float, default=0.02)
    args = parser.parse_args()

    run(args.input_dir, args.output_dir, args.min_sharpness)


if __name__ == "__main__":
    main()
