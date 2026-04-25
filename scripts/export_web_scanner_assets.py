#!/usr/bin/env python3
"""Export browser-friendly assets for the static web scanner.

Writes:
  <out-dir>/assets/manifest.json
  <out-dir>/assets/models/*.onnx
  <out-dir>/assets/catalog/*.bin / *.json
  <out-dir>/assets/samples/*
  <out-dir>/vendor/onnxruntime-web/*
  <out-dir>/bundle-metadata.json

The catalog payload is kept simple:
  - embeddings.f16.bin  raw float16 rows
  - card_ids.json       aligned card-id array
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from collector_vision.catalog import Catalog
from collector_vision.hfd import HFD
from collector_vision import weights


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "examples" / "web_scanner"
SAMPLE_IMAGE = ROOT / "examples" / "images" / "7286819f-6c57-4503-898c-528786ad86e9_sample.jpg"
DEFAULT_REPO = "HanClinto/milo"
DEFAULT_CATALOG_KEY = "scryfall-mtg"

ORT_VERSION = "1.24.3"

ORT_TARBALL = f"https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{ORT_VERSION}.tgz"

# CRITICAL: use ort.webgpu.min.mjs (new WebGPU EP), NOT ort.all.min.mjs (legacy JSEP).
# The legacy ort.all.min.mjs + ort-wasm-simd-threaded.jsep.wasm bundle uses the JSEP
# backend which silently returns all-zeros for Conv ops on Android (all ort-web
# versions 1.20-1.24.3).  The new EP fixes this but requires the asyncify WASM variant.
#
# Additionally, even the new WebGPU EP gives numerically wrong outputs for
# cornelius.onnx (corner detector) on Android ARM GPUs (armv81, Chrome 147,
# ort-web 1.24.3).  The scanner.worker.mjs therefore runs the detector on WASM only.
# See ARCHITECTURE.md "Lessons Learned" section for the full history.
ORT_FILES = [
    "package/dist/ort.webgpu.min.mjs",
    "package/dist/ort-wasm-simd-threaded.asyncify.mjs",
    "package/dist/ort-wasm-simd-threaded.asyncify.wasm",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _latest_catalog_filename(repo: str, catalog_key: str) -> str:
    manifest_url = f"https://huggingface.co/{repo}/resolve/main/catalogs/manifest.json"
    with urllib.request.urlopen(manifest_url, timeout=30) as response:
        manifest = json.loads(response.read().decode("utf-8"))
    entry = manifest.get(catalog_key)
    if not entry:
        raise KeyError(f"Catalog key {catalog_key!r} not found in {manifest_url}")
    return entry["latest"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Root directory that should receive assets/, vendor/, and bundle-metadata.json",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face repo id")
    parser.add_argument(
        "--catalog-key",
        default=DEFAULT_CATALOG_KEY,
        help="Catalog key inside catalogs/manifest.json",
    )
    parser.add_argument(
        "--bundle-version",
        default=None,
        help="Optional explicit bundle version string written into bundle-metadata.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    assets_dir = out_dir / "assets"
    models_dir = assets_dir / "models"
    catalog_dir = assets_dir / "catalog"
    samples_dir = assets_dir / "samples"
    vendor_dir = out_dir / "vendor"
    ort_vendor_dir = vendor_dir / "onnxruntime-web"

    models_dir.mkdir(parents=True, exist_ok=True)
    catalog_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    ort_vendor_dir.mkdir(parents=True, exist_ok=True)

    catalog_filename = _latest_catalog_filename(args.repo, args.catalog_key)
    bundle_version = args.bundle_version or Path(catalog_filename).stem
    hfd = HFD(args.repo, args.catalog_key)

    catalog = Catalog.load(hfd)

    cornelius_path = models_dir / "cornelius.onnx"
    milo_path = models_dir / "milo.onnx"
    shutil.copy2(weights.CORNER_DETECTOR, cornelius_path)
    shutil.copy2(weights.EMBEDDER, milo_path)
    _write_vendor_files(ORT_TARBALL, ORT_FILES, ort_vendor_dir)

    embeddings_path = catalog_dir / f"{args.catalog_key}-embeddings.f16.bin"
    embeddings_path.write_bytes(catalog.embeddings.astype("<f2", copy=False).tobytes())

    card_ids_path = catalog_dir / f"{args.catalog_key}-card-ids.json"
    card_ids_path.write_text(json.dumps(catalog.card_ids), encoding="utf-8")

    sample_path = samples_dir / "mtg-sample.jpg"
    shutil.copy2(SAMPLE_IMAGE, sample_path)

    cornelius_hash = _sha256(cornelius_path)
    milo_hash = _sha256(milo_path)

    manifest = {
        "version": bundle_version,
        "models": {
            "cornelius": "models/cornelius.onnx",
            "milo": "models/milo.onnx",
        },
        "model_hashes": {
            "cornelius": cornelius_hash,
            "milo": milo_hash,
        },
        "catalog": {
            "embeddings": f"catalog/{args.catalog_key}-embeddings.f16.bin",
            "card_ids": f"catalog/{args.catalog_key}-card-ids.json",
            "rows": int(catalog.embeddings.shape[0]),
            "dims": int(catalog.embeddings.shape[1]),
            "dtype": "float16",
        },
        "sample_frame": "samples/mtg-sample.jpg",
    }
    manifest_path = assets_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    bundle_metadata = {
        "bundle_version": bundle_version,
        "catalog_key": f"hf://{args.repo}/{args.catalog_key}",
        "catalog_fingerprint": catalog_filename,
        "catalog_rows": int(catalog.embeddings.shape[0]),
        "catalog_dims": int(catalog.embeddings.shape[1]),
        "models": {
            "cornelius": cornelius_hash,
            "milo": milo_hash,
        },
        "vendor": {
            "onnxruntime_web": ORT_VERSION,
        },
        "assets": {
            "manifest": _sha256(manifest_path),
            "embeddings": _sha256(embeddings_path),
            "card_ids": _sha256(card_ids_path),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = out_dir / "bundle-metadata.json"
    metadata_path.write_text(json.dumps(bundle_metadata, indent=2), encoding="utf-8")

    print(f"Wrote {embeddings_path}")
    print(f"Wrote {card_ids_path}")
    print(f"Wrote {sample_path}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
