#!/usr/bin/env python3
"""Download and ingest bug-report capture bundles from GitHub issues.

Each issue tagged `bug` may contain a `.json.gz` capture bundle (uploaded via
the **Capture** button in the web scanner).  This script:

1. Queries GitHub for issues labelled ``bug`` (open by default).
2. Finds the ``https://github.com/user-attachments/files/…/*.json.gz`` URL in
   the issue body.
3. Skips files already present in ``tests/fixtures/captures/``.
4. Downloads the bundle and saves it to ``tests/fixtures/captures/``.
5. If the issue body contains an "Expected card" reply, patches
   ``expectedCardId`` into the bundle (gzip round-trip).

Usage::

    # ingest all open bug issues
    python scripts/ingest_bug_reports.py

    # specific issue numbers only
    python scripts/ingest_bug_reports.py 1 3 7

    # include closed issues too
    python scripts/ingest_bug_reports.py --closed

    # dry-run: show what would be downloaded without writing files
    python scripts/ingest_bug_reports.py --dry-run

Environment variables:

    GITHUB_TOKEN   Personal-access-token (or fine-grained token with Issues:read).
                   Optional for public repos within the anonymous rate limit
                   (60 req/hr); strongly recommended otherwise.

Requirements: Python ≥ 3.8 stdlib only (no extra packages needed).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO = "HanClinto/CollectorVision"
API_BASE = "https://api.github.com"
CAPTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "captures"

# Matches raw GitHub attachment URLs ending in .json.gz
ATTACHMENT_RE = re.compile(
    r"https://github\.com/user-attachments/files/\d+/[^\s)\]\"'>]+\.json\.gz"
)

# Matches the "Expected card" section in the issue body.
# Captures the first non-blank, non-comment line after the heading.
EXPECTED_CARD_RE = re.compile(
    r"\*\*Expected card[^*\n]*\*\*:?\s*\n(?:<!--[^\n]*-->\s*\n)*([^\n<>]+)",
    re.IGNORECASE,
)

# UUID / Scryfall ID pattern
SFID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def _headers() -> dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "CollectorVision-ingest-script/1.0",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get(url: str) -> dict | list:
    req = urllib.request.Request(url, headers=_headers())
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise SystemExit(
            f"GitHub API error {exc.code} for {url}:\n{body}"
        ) from exc


def _download_bytes(url: str) -> bytes:
    """Download a URL, following redirects, with optional auth."""
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def list_bug_issues(state: str = "open") -> list[dict]:
    """Return all issues labelled 'bug' with the given state."""
    issues: list[dict] = []
    page = 1
    while True:
        url = (
            f"{API_BASE}/repos/{REPO}/issues"
            f"?labels=bug&state={state}&per_page=100&page={page}"
        )
        batch = _get(url)
        if not batch:
            break
        issues.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    # GitHub returns pull requests in /issues — filter them out.
    return [i for i in issues if "pull_request" not in i]


def get_issue(number: int) -> dict:
    return _get(f"{API_BASE}/repos/{REPO}/issues/{number}")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def find_attachment_url(body: str) -> str | None:
    m = ATTACHMENT_RE.search(body or "")
    return m.group(0) if m else None


def find_expected_card(body: str) -> str | None:
    """Extract the expected card name/SFID from the issue body, if filled in."""
    m = EXPECTED_CARD_RE.search(body or "")
    if not m:
        return None
    value = m.group(1).strip()
    # Treat placeholder-like values as absent.
    if not value or value.startswith("<!--") or value.lower() in {
        "—", "-", "n/a", "unknown", "leave blank if unknown",
    }:
        return None
    return value


def extract_sfid(text: str) -> str | None:
    """Pull the first Scryfall UUID out of a free-text expected-card string."""
    m = SFID_RE.search(text)
    return m.group(0).lower() if m else None


# ---------------------------------------------------------------------------
# Bundle patching
# ---------------------------------------------------------------------------

def patch_expected_card_id(path: Path, expected_card_id: str) -> None:
    """Rewrite the bundle at *path* to set expectedCardId."""
    with gzip.open(path, "rb") as fh:
        bundle = json.load(fh)
    if bundle.get("expectedCardId") == expected_card_id:
        return  # already set — no-op
    bundle["expectedCardId"] = expected_card_id
    with gzip.open(path, "wb") as fh:
        fh.write(json.dumps(bundle).encode())
    print(f"    → patched expectedCardId = {expected_card_id!r}")


def annotate_python_results(path: Path) -> None:
    """Run the Python detector on the capture frame and store results in the bundle.

    Stores ``pythonCorners``, ``pythonSharpness``, and ``pythonCardPresent``
    so the JS Node.js regression tests can use them as the authoritative
    reference (Python CPU results are known-correct).
    """
    try:
        import base64  # noqa: PLC0415

        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import collector_vision as cvg  # noqa: PLC0415
    except ImportError as exc:
        print(f"    → cannot annotate (missing dep: {exc}) — skipping")
        return

    with gzip.open(path, "rb") as fh:
        bundle = json.load(fh)

    png_bytes = base64.b64decode(bundle["framePng"])
    bgr = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        print("    → cannot annotate: failed to decode framePng")
        return

    detector = cvg.NeuralCornerDetector()
    result = detector.detect(bgr)

    bundle["pythonCorners"] = (
        [{"x": float(x), "y": float(y)} for x, y in result.corners]
        if result.corners is not None
        else None
    )
    bundle["pythonSharpness"] = (
        float(result.sharpness) if result.sharpness is not None else None
    )
    bundle["pythonCardPresent"] = bool(result.card_present)

    with gzip.open(path, "wb") as fh:
        fh.write(json.dumps(bundle).encode())

    sharpness_str = f"{result.sharpness:.4f}" if result.sharpness is not None else "n/a"
    print(
        f"    → annotated pythonCorners ({len(bundle['pythonCorners'] or [])} pts), "
        f"sharpness={sharpness_str}, present={result.card_present}"
    )


# ---------------------------------------------------------------------------
# Main ingestion logic
# ---------------------------------------------------------------------------

def ingest_issue(issue: dict, dry_run: bool = False) -> None:
    number = issue["number"]
    title = issue["title"]
    body = issue.get("body") or ""

    attachment_url = find_attachment_url(body)
    if not attachment_url:
        print(f"  #{number} {title!r}: no .json.gz attachment — skipping")
        return

    filename = attachment_url.rsplit("/", 1)[-1]
    dest = CAPTURES_DIR / filename

    expected_raw = find_expected_card(body)
    expected_sfid = extract_sfid(expected_raw) if expected_raw else None

    print(f"  #{number} {title!r}")
    print(f"    attachment : {filename}")
    print(f"    expected   : {expected_raw!r}  (sfid={expected_sfid})")

    if dry_run:
        if dest.exists():
            print(f"    status     : already present (dry-run skip)")
        else:
            print(f"    status     : WOULD download → {dest}")
        return

    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"    status     : already present — skip download")
    else:
        print(f"    status     : downloading…", end=" ", flush=True)
        data = _download_bytes(attachment_url)
        dest.write_bytes(data)
        print(f"{len(data):,} bytes → {dest}")

    if expected_sfid:
        patch_expected_card_id(dest, expected_sfid)
    elif expected_raw:
        # User gave a card name but no UUID — store the raw name string.
        patch_expected_card_id(dest, expected_raw)

    annotate_python_results(dest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CollectorVision bug-report captures from GitHub issues."
    )
    parser.add_argument(
        "issues",
        nargs="*",
        type=int,
        metavar="N",
        help="issue numbers to ingest (default: all open bug issues)",
    )
    parser.add_argument(
        "--closed",
        action="store_true",
        help="include closed issues when scanning all bugs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would happen without writing any files",
    )
    args = parser.parse_args()

    if not os.environ.get("GITHUB_TOKEN"):
        print(
            "Note: GITHUB_TOKEN not set — using anonymous API access "
            "(60 req/hr limit, may fail on private repos)."
        )

    if args.issues:
        issues = []
        for n in args.issues:
            print(f"Fetching issue #{n}…")
            issues.append(get_issue(n))
    else:
        state = "all" if args.closed else "open"
        print(f"Fetching {state} issues labelled 'bug' from {REPO}…")
        issues = list_bug_issues(state=state)
        print(f"Found {len(issues)} issue(s).")

    if not issues:
        print("Nothing to ingest.")
        return

    print()
    for issue in issues:
        ingest_issue(issue, dry_run=args.dry_run)

    print()
    print(
        "Done. Drop any new .json.gz files into tests/fixtures/captures/ "
        "and run:\n  python -m pytest tests/test_capture_regression.py -v"
    )


if __name__ == "__main__":
    main()
