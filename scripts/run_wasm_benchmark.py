#!/usr/bin/env python3
"""Run model_benchmark.html headlessly via Playwright and print results.

Downloads the ORT runtime from npm on first run (cached in vendor/).
Benchmarks each named model set and prints a side-by-side comparison.

Usage
-----
    # Compare baseline (bundled) vs optimised models
    python scripts/run_wasm_benchmark.py \\
        --set "baseline:weights/cornelius.onnx:weights/milo.onnx" \\
        --set "optimised:/path/to/cornelius.onnx:/path/to/milo.onnx"

    # Benchmark the bundled models only
    python scripts/run_wasm_benchmark.py

All paths are relative to the repo root unless absolute.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import shutil
import socket
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
WEB_DIR  = ROOT / "examples" / "web_scanner"
WEIGHTS  = ROOT / "collector_vision" / "weights"
ORT_VERSION = "1.24.3"
ORT_TARBALL = f"https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{ORT_VERSION}.tgz"
ORT_MEMBERS = [
    "package/dist/ort.webgpu.min.mjs",
    "package/dist/ort-wasm-simd-threaded.asyncify.mjs",
    "package/dist/ort-wasm-simd-threaded.asyncify.wasm",
]


# ---------------------------------------------------------------------------
# COOP/COEP-aware HTTP server (required for SharedArrayBuffer / multi-thread)
# ---------------------------------------------------------------------------

class _COEPHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()

    def log_message(self, *_args) -> None:
        pass  # suppress per-request logs


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(directory: Path) -> tuple[http.server.HTTPServer, int]:
    port = _free_port()

    # Bind the serve directory at handler creation time; SimpleHTTPRequestHandler
    # otherwise uses os.getcwd() at request time which is wrong after chdir.
    class _BoundHandler(_COEPHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

    server = http.server.HTTPServer(("127.0.0.1", port), _BoundHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, port


# ---------------------------------------------------------------------------
# ORT vendor download
# ---------------------------------------------------------------------------

def _ensure_ort(vendor_dir: Path) -> None:
    """Download ORT WASM runtime files from npm if not already present."""
    target = vendor_dir / "onnxruntime-web"
    sentinel = target / "ort.webgpu.min.mjs"
    if sentinel.exists():
        return

    print(f"   Downloading ORT {ORT_VERSION} from npm…", flush=True)
    target.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(ORT_TARBALL, timeout=120) as resp:
        data = resp.read()

    with tarfile.open(fileobj=__import__("io").BytesIO(data), mode="r:gz") as tf:
        for member_name in ORT_MEMBERS:
            member = tf.getmember(member_name)
            src    = tf.extractfile(member)
            dest   = target / Path(member_name).name
            dest.write_bytes(src.read())
            print(f"   {dest.name}  ({dest.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Minimal assets directory for the benchmark page
# ---------------------------------------------------------------------------

def _build_assets(tmp: Path, vendor_dir: Path, cornelius: Path, milo: Path) -> None:
    """Assemble the minimal directory that model_benchmark.html needs."""
    # Vendor ORT runtime (symlink or copy)
    ort_dst = tmp / "vendor" / "onnxruntime-web"
    ort_dst.mkdir(parents=True)
    ort_src = vendor_dir / "onnxruntime-web"
    for f in ort_src.iterdir():
        shutil.copy2(f, ort_dst / f.name)

    # Model files
    models_dst = tmp / "assets" / "models"
    models_dst.mkdir(parents=True)
    shutil.copy2(cornelius, models_dst / "cornelius.onnx")
    shutil.copy2(milo,      models_dst / "milo.onnx")

    # Benchmark HTML (reference the original; serve from a copy to avoid path issues)
    shutil.copy2(WEB_DIR / "model_benchmark.html", tmp / "model_benchmark.html")

    # build.json so the page can read a label (optional — handled gracefully if absent)


# ---------------------------------------------------------------------------
# Playwright runner
# ---------------------------------------------------------------------------

def _run_benchmark(port: int, label: str, timeout: int = 300) -> dict:
    """Open model_benchmark.html in headless Chromium and return the JSON report."""
    from playwright.sync_api import sync_playwright  # noqa: PLC0415

    url = f"http://127.0.0.1:{port}/model_benchmark.html?backend=wasm"

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
            ],
        )
        ctx  = browser.new_context()
        page = ctx.new_page()

        # Forward browser console messages so JS errors are visible
        page.on("console", lambda msg: print(f"   [{label}] console.{msg.type}: {msg.text[:200]}", flush=True)
                if msg.type in ("error", "warning") else None)
        page.on("pageerror", lambda err: print(f"   [{label}] pageerror: {err}", flush=True))

        print(f"   [{label}] navigating to benchmark…", flush=True)
        page.goto(url, wait_until="domcontentloaded")

        # Poll until the status element says "Done."
        deadline = time.time() + timeout
        last_status = ""
        while time.time() < deadline:
            status = page.evaluate("document.getElementById('status')?.textContent || ''")
            if status != last_status:
                print(f"   [{label}] {status}", flush=True)
                last_status = status
            if "Done." in status:
                break
            time.sleep(1)
        else:
            # Print page content to help diagnose
            final = page.evaluate("document.getElementById('status')?.textContent || 'no status'")
            raise TimeoutError(f"Benchmark did not finish within {timeout}s (last status: {final!r})")

        # Extract the JSON from inside the textarea (markdown report)
        markdown = page.evaluate("document.getElementById('comment')?.value || ''")
        browser.close()

    # Parse the embedded JSON from the markdown report
    start = markdown.find("```json")
    end   = markdown.find("```", start + 7)
    if start == -1 or end == -1:
        raise ValueError("Could not find JSON block in benchmark report")
    return json.loads(markdown[start + 7 : end].strip())


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _print_results(label: str, report: dict) -> None:
    ua = report.get("userAgent", "")
    iso = report.get("crossOriginIsolated", False)
    print(f"\n  {'Model':<12} {'Threads':>7} {'Mean ms':>9} {'P90 ms':>9} {'First ms':>10} {'Size MiB':>10}")
    print(f"  {'-'*12} {'-'*7} {'-'*9} {'-'*9} {'-'*10} {'-'*10}")
    for b in report.get("benchmarks", []):
        s = b["latencyMs"]
        size = f"{b['sizeMiB']:.2f}" if b.get("sizeMiB") else "—"
        print(
            f"  {b['name']:<12} {b['threads']:>7} "
            f"{s['mean']:>9.1f} {s['p90']:>9.1f} "
            f"{b['firstRunMs']:>10.1f} {size:>10}"
        )
    print(f"  cross-origin isolated: {iso}  |  UA: {ua[:80]}")


def _compare(results: list[tuple[str, dict]]) -> None:
    if len(results) < 2:
        return
    print("\n── Δ vs baseline ─────────────────────────────────────────")
    base_by_key = {
        (b["name"], b["threads"]): b
        for b in results[0][1].get("benchmarks", [])
    }
    for label, report in results[1:]:
        print(f"\n  {label} vs {results[0][0]}:")
        for b in report.get("benchmarks", []):
            key  = (b["name"], b["threads"])
            base = base_by_key.get(key)
            if not base:
                continue
            delta_mean = b["latencyMs"]["mean"] - base["latencyMs"]["mean"]
            pct        = 100 * delta_mean / base["latencyMs"]["mean"]
            sign       = "+" if delta_mean >= 0 else ""
            bar        = "▲ SLOWER" if delta_mean > 0.5 else ("▼ faster" if delta_mean < -0.5 else "≈ same")
            print(
                f"  {b['name']:<12} {b['threads']:>2}t  "
                f"{sign}{delta_mean:+.1f} ms  ({sign}{pct:.1f}%)  {bar}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run model_benchmark.html headlessly and compare results")
    parser.add_argument(
        "--set", dest="sets", metavar="LABEL:CORNELIUS:MILO", action="append",
        help="Model set: label, cornelius.onnx path, milo.onnx path (colon-separated). "
             "Repeat for multiple sets. Default: bundled weights.",
    )
    parser.add_argument("--timeout", type=int, default=300, help="Per-run timeout in seconds (default 300)")
    args = parser.parse_args()

    # Default to bundled weights only
    if not args.sets:
        args.sets = [f"bundled:{WEIGHTS / 'cornelius.onnx'}:{WEIGHTS / 'milo.onnx'}"]

    model_sets: list[tuple[str, Path, Path]] = []
    for spec in args.sets:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            sys.exit(f"Bad --set format (expected label:cornelius:milo): {spec!r}")
        label, c_path, m_path = parts
        c_abs = Path(c_path) if Path(c_path).is_absolute() else ROOT / c_path
        m_abs = Path(m_path) if Path(m_path).is_absolute() else ROOT / m_path
        for p in (c_abs, m_abs):
            if not p.exists():
                sys.exit(f"Model not found: {p}")
        model_sets.append((label, c_abs, m_abs))

    # Ensure ORT runtime is downloaded
    vendor_dir = WEB_DIR / "vendor"
    _ensure_ort(vendor_dir)

    results: list[tuple[str, dict]] = []

    for label, cornelius, milo in model_sets:
        print(f"\n── {label} ─────────────────────────────────────────────────")
        print(f"   cornelius : {cornelius}  ({cornelius.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"   milo      : {milo}  ({milo.stat().st_size / 1024 / 1024:.1f} MB)")

        with tempfile.TemporaryDirectory(prefix="cv-bench-") as tmp_str:
            tmp = Path(tmp_str)
            _build_assets(tmp, vendor_dir, cornelius, milo)
            server, port = _start_server(tmp)
            try:
                report = _run_benchmark(port, label, timeout=args.timeout)
            finally:
                server.shutdown()

        _print_results(label, report)
        results.append((label, report))

    if len(results) > 1:
        _compare(results)


if __name__ == "__main__":
    main()
