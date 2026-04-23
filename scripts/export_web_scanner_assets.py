#!/usr/bin/env python3
"""Export browser-friendly assets for the static web scanner.

Writes:
  examples/web_scanner/assets/manifest.json
  examples/web_scanner/assets/models/*.onnx
  examples/web_scanner/assets/catalog/*.bin / *.json
  examples/web_scanner/assets/samples/*
  examples/web_scanner/vendor/onnxruntime-web/*
  examples/web_scanner/vendor/opencv/opencv.js

The catalog payload is kept simple:
  - embeddings.f16.bin  raw float16 rows
  - card_ids.json       aligned card-id array
"""

from __future__ import annotations

import io
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

from collector_vision.catalog import Catalog
from collector_vision import weights


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "web_scanner" / "assets"
MODELS = OUT / "models"
CATALOG = OUT / "catalog"
SAMPLES = OUT / "samples"
VENDOR = ROOT / "examples" / "web_scanner" / "vendor"
ORT_VENDOR = VENDOR / "onnxruntime-web"
OPENCV_VENDOR = VENDOR / "opencv"
SAMPLE_IMAGE = ROOT / "examples" / "images" / "7286819f-6c57-4503-898c-528786ad86e9_sample.jpg"

ORT_VERSION = "1.24.3"
OPENCV_VERSION = "4.12.0-release.1"

ORT_TARBALL = f"https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{ORT_VERSION}.tgz"
OPENCV_TARBALL = (
    f"https://registry.npmjs.org/@techstark/opencv-js/-/opencv-js-{OPENCV_VERSION}.tgz"
)

ORT_FILES = [
    "package/dist/ort.all.min.mjs",
    "package/dist/ort-wasm-simd-threaded.jsep.mjs",
    "package/dist/ort-wasm-simd-threaded.jsep.wasm",
]
OPENCV_FILES = [
    "package/dist/opencv.js",
]


def _download_tar_member(url: str, member_name: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        member = archive.getmember(member_name)
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"Could not extract {member_name} from {url}")
        return extracted.read()


def _write_vendor_files(url: str, files: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for member_name in files:
        target = out_dir / Path(member_name).name
        target.write_bytes(_download_tar_member(url, member_name))
        print(f"Wrote {target}")


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    CATALOG.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    ORT_VENDOR.mkdir(parents=True, exist_ok=True)
    OPENCV_VENDOR.mkdir(parents=True, exist_ok=True)

    catalog = Catalog.load("hf://HanClinto/milo/scryfall-mtg")

    shutil.copy2(weights.CORNER_DETECTOR, MODELS / "cornelius.onnx")
    shutil.copy2(weights.EMBEDDER, MODELS / "milo.onnx")
    _write_vendor_files(ORT_TARBALL, ORT_FILES, ORT_VENDOR)
    _write_vendor_files(OPENCV_TARBALL, OPENCV_FILES, OPENCV_VENDOR)

    embeddings_path = CATALOG / "scryfall-mtg-embeddings.f16.bin"
    embeddings_path.write_bytes(catalog.embeddings.astype("<f2", copy=False).tobytes())

    card_ids_path = CATALOG / "scryfall-mtg-card-ids.json"
    card_ids_path.write_text(json.dumps(catalog.card_ids), encoding="utf-8")

    sample_path = SAMPLES / "mtg-sample.jpg"
    shutil.copy2(SAMPLE_IMAGE, sample_path)

    manifest = {
        "version": "0.1.0",
        "models": {
            "cornelius": "models/cornelius.onnx",
            "milo": "models/milo.onnx",
        },
        "catalog": {
            "embeddings": "catalog/scryfall-mtg-embeddings.f16.bin",
            "card_ids": "catalog/scryfall-mtg-card-ids.json",
            "rows": int(catalog.embeddings.shape[0]),
            "dims": int(catalog.embeddings.shape[1]),
            "dtype": "float16",
        },
        "sample_frame": "samples/mtg-sample.jpg",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {embeddings_path}")
    print(f"Wrote {card_ids_path}")
    print(f"Wrote {sample_path}")
    print(f"Wrote {OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
