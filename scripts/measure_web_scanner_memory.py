#!/usr/bin/env python3
"""Measure CollectorVision web scanner memory in a local Chromium browser.

This is a lightweight Chrome DevTools Protocol harness using only the Python
standard library.  It launches Chrome/Chromium with fake camera input, opens the
local web scanner, starts scanning, and samples browser-exposed JS heap stats
(`performance.memory`) plus CollectorVision boot/perf breadcrumbs.

Notes:
- `performance.memory` is Chromium-only and JS-heap-only. It does not include
  all native camera, GPU, WebKit/Safari, or OS process memory.
- Mobile mode is Chromium mobile emulation, not a real iPhone memory limit.
- For realistic iOS crash behavior, use a physical iPhone + Safari remote
  inspector / iOS Analytics Jetsam logs.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "examples" / "web_scanner"


def find_chrome() -> str:
    candidates = [
        os.environ.get("CHROME"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("Could not find Chrome/Chromium. Set CHROME=/path/to/chrome.")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_json(url: str, timeout: float = 10.0) -> Any:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - polling browser startup
            last_error = exc
            time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for {url}: {last_error}")


class CDPWebSocket:
    def __init__(self, ws_url: str):
        parsed = urllib.parse.urlparse(ws_url)
        if parsed.scheme != "ws":
            raise ValueError(f"Only ws:// URLs are supported: {ws_url}")
        self.sock = socket.create_connection((parsed.hostname, parsed.port or 80), timeout=5)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        path = parsed.path + (("?" + parsed.query) if parsed.query else "")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {parsed.hostname}:{parsed.port or 80}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(request.encode("ascii"))
        response = self.sock.recv(4096)
        if b" 101 " not in response.split(b"\r\n", 1)[0]:
            raise ConnectionError(response.decode("utf-8", "replace"))
        accept = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
        )
        if accept not in response:
            raise ConnectionError("WebSocket handshake accept mismatch")
        self.next_id = 1

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def send_text(self, text: str) -> None:
        payload = text.encode("utf-8")
        header = bytearray([0x81])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.extend([0x80 | 126, *struct.pack("!H", length)])
        else:
            header.extend([0x80 | 127, *struct.pack("!Q", length)])
        mask = random.randbytes(4) if hasattr(random, "randbytes") else os.urandom(4)
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        self.sock.sendall(bytes(header) + mask + masked)

    def recv_text(self) -> str:
        while True:
            first = self.sock.recv(2)
            if len(first) < 2:
                raise ConnectionError("WebSocket closed")
            opcode = first[0] & 0x0F
            masked = (first[1] & 0x80) != 0
            length = first[1] & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._recv_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._recv_exact(8))[0]
            mask = self._recv_exact(4) if masked else b""
            payload = self._recv_exact(length)
            if masked:
                payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
            if opcode == 1:
                return payload.decode("utf-8")
            if opcode == 8:
                raise ConnectionError("WebSocket close frame")
            if opcode == 9:  # ping -> pong
                self.sock.sendall(b"\x8a\x00")

    def _recv_exact(self, n: int) -> bytes:
        chunks = []
        remaining = n
        while remaining:
            chunk = self.sock.recv(remaining)
            if not chunk:
                raise ConnectionError("WebSocket closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def call(self, method: str, params: dict[str, Any] | None = None, timeout: float = 10.0) -> Any:
        msg_id = self.next_id
        self.next_id += 1
        self.send_text(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        deadline = time.time() + timeout
        while time.time() < deadline:
            message = json.loads(self.recv_text())
            if message.get("id") != msg_id:
                continue
            if "error" in message:
                raise RuntimeError(f"CDP {method} failed: {message['error']}")
            return message.get("result")
        raise TimeoutError(f"Timed out waiting for CDP response to {method}")

    def evaluate(self, expression: str, timeout: float = 10.0) -> Any:
        result = self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
            timeout=timeout,
        )
        remote = result.get("result", {})
        if "value" in remote:
            return remote["value"]
        if "description" in remote:
            return remote["description"]
        return None


def mib(value: int | float | None) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value / (1024 * 1024):.1f} MiB"


def start_server(port: int) -> subprocess.Popen[str]:
    if not (WEB_DIR / "assets" / "manifest.json").exists():
        raise SystemExit(
            "examples/web_scanner/assets/manifest.json is missing. "
            "Run: uv run python scripts/export_web_scanner_assets.py"
        )
    return subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port)],
        cwd=WEB_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def run_case(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    chrome_port = free_port()
    profile_dir = tempfile.TemporaryDirectory(prefix="cv-chrome-profile-")
    chrome = find_chrome()
    page_url = args.url
    separator = "&" if "?" in page_url else "?"
    page_url = f"{page_url}{separator}debug=1&fps=1"

    chrome_args = [
        chrome,
        "about:blank",
        f"--remote-debugging-port={chrome_port}",
        f"--user-data-dir={profile_dir.name}",
        "--no-first-run",
        "--no-default-browser-check",
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        "--enable-precise-memory-info",
    ]
    if args.headless:
        chrome_args.append("--headless=new")
    if mode == "desktop":
        chrome_args.append("--window-size=1300,900")
    else:
        chrome_args.append("--window-size=390,844")

    proc = subprocess.Popen(chrome_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ws: CDPWebSocket | None = None
    try:
        wait_json(f"http://127.0.0.1:{chrome_port}/json/version")
        targets = wait_json(f"http://127.0.0.1:{chrome_port}/json/list")
        page = next(t for t in targets if t.get("type") == "page")
        ws = CDPWebSocket(page["webSocketDebuggerUrl"])
        ws.call("Runtime.enable")
        ws.call("Page.enable")
        if mode == "mobile":
            ws.call(
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": 375,
                    "height": 667,
                    "deviceScaleFactor": 2,
                    "mobile": True,
                },
            )
            ws.call(
                "Emulation.setUserAgentOverride",
                {
                    "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.4 Mobile/15E148 Safari/604.1",
                    "platform": "iPhone",
                },
            )
        ws.call("Page.navigate", {"url": page_url})

        ready_expr = """
        (() => {
          const trace = JSON.parse(localStorage.getItem('cv_boot_trace') || 'null');
          return trace && trace.lastStage === 'boot:ready';
        })()
        """
        deadline = time.time() + args.ready_timeout
        while time.time() < deadline:
            if ws.evaluate(ready_expr):
                break
            time.sleep(0.25)
        else:
            raise TimeoutError("Scanner did not reach boot:ready")

        if args.open_settings:
            ws.evaluate("document.getElementById('settings-toggle')?.click()")
        ws.evaluate("document.getElementById('camera-badge')?.click()")

        samples: list[dict[str, Any]] = []
        sample_expr = """
        (() => {
          const mem = performance.memory ? {
            usedJSHeapSize: performance.memory.usedJSHeapSize,
            totalJSHeapSize: performance.memory.totalJSHeapSize,
            jsHeapSizeLimit: performance.memory.jsHeapSizeLimit,
          } : null;
          const trace = JSON.parse(localStorage.getItem('cv_boot_trace') || 'null');
          const video = document.getElementById('camera-video');
          const processText = document.getElementById('diag-process-canvas')?.textContent || null;
          const timingText = document.getElementById('diag-timing')?.textContent || null;
          const overlay = document.getElementById('perf-overlay')?.textContent || null;
          return {
            now: performance.now(),
            mem,
            lastStage: trace?.lastStage || null,
            lastEntry: trace?.entries?.[trace.entries.length - 1] || null,
            video: video ? { width: video.videoWidth, height: video.videoHeight, readyState: video.readyState } : null,
            processText,
            timingText,
            overlay,
          };
        })()
        """
        end = time.time() + args.duration
        while time.time() < end:
            samples.append(ws.evaluate(sample_expr))
            time.sleep(args.interval)

        used_values = [s.get("mem", {}).get("usedJSHeapSize") for s in samples if s.get("mem")]
        summary = {
            "mode": mode,
            "url": page_url,
            "sampleCount": len(samples),
            "usedJSHeapMin": min(used_values) if used_values else None,
            "usedJSHeapMax": max(used_values) if used_values else None,
            "usedJSHeapLast": used_values[-1] if used_values else None,
            "lastSample": samples[-1] if samples else None,
            "samples": samples if args.keep_samples else [],
        }
        print(
            f"{mode:7} heap used min/max/last: "
            f"{mib(summary['usedJSHeapMin'])} / {mib(summary['usedJSHeapMax'])} / {mib(summary['usedJSHeapLast'])}"
        )
        if samples and samples[-1].get("overlay"):
            print(samples[-1]["overlay"])
        return summary
    finally:
        if ws:
            ws.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        profile_dir.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8040/", help="Scanner URL to measure")
    parser.add_argument(
        "--serve", action="store_true", help="Start a local static server for examples/web_scanner"
    )
    parser.add_argument(
        "--port", type=int, default=8040, help="Local server port when --serve is used"
    )
    parser.add_argument("--mode", choices=["desktop", "mobile", "both"], default="both")
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Seconds to sample after starting camera"
    )
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between samples")
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    parser.add_argument(
        "--open-settings",
        action="store_true",
        help="Open Settings to exercise desktop debug previews",
    )
    parser.add_argument("--headless", action="store_true", help="Run Chrome headless")
    parser.add_argument(
        "--keep-samples", action="store_true", help="Include full sample list in output JSON"
    )
    parser.add_argument("--out", type=Path, default=Path("web_scanner_memory_report.json"))
    args = parser.parse_args()

    server: subprocess.Popen[str] | None = None
    if args.serve:
        server = start_server(args.port)
        args.url = f"http://127.0.0.1:{args.port}/"
        time.sleep(0.5)

    try:
        modes = ["desktop", "mobile"] if args.mode == "both" else [args.mode]
        report = {"createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cases": []}
        for mode in modes:
            report["cases"].append(run_case(args, mode))
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")
    finally:
        if server:
            server.terminate()


if __name__ == "__main__":
    main()
