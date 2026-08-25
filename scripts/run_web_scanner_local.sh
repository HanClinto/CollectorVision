#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-8040}"

cd "$ROOT"
uv run python scripts/export_web_scanner_assets.py
cd examples/web_scanner
exec uv run python -m http.server "$PORT"
