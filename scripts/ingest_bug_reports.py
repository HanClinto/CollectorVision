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
# Import the manifest-generation module (same directory as this script).
# It requires collector_vision + cv2 to be installed; generation is skipped
# gracefully if those are absent.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import generate_manifests as _gm  # type: ignore
except ImportError:
    _gm = None  # type: ignore  # generation will be skipped

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


def _generate_manifests(path: Path, catalog=None) -> None:
    """Generate pipeline manifests for a capture bundle.

    Writes ``<captureId>.python.json``, ``<captureId>.python.dewarp.png``,
    ``<captureId>.js-webgpu.json``, and ``<captureId>.js-webgpu.dewarp.png``
    next to the bundle.  Silently skips if ``generate_manifests`` could not
    be imported (missing collector_vision / cv2 dependencies).
    """
    if _gm is None:
        print("    → generate_manifests unavailable — skipping manifest generation")
        return
    _gm.generate_python_manifest(path, catalog=catalog)
    _gm.extract_webgpu_manifest(path, catalog=catalog)


# ---------------------------------------------------------------------------
# Main ingestion logic
# ---------------------------------------------------------------------------

def ingest_issue(
    issue: dict,
    dry_run: bool = False,
    force: bool = False,
    catalog=None,
) -> bool:
    """Ingest one GitHub issue.

    Returns
    -------
    ``True`` on success.  ``False`` when the capture does not reproduce the
    reported bug (only possible for ``bug``-labelled issues; use ``force`` to
    bypass).
    """
    number = issue["number"]
    title = issue["title"]
    body = issue.get("body") or ""
    state = issue.get("state", "open")

    attachment_url = find_attachment_url(body)
    if not attachment_url:
        print(f"  #{number} {title!r}: no .json.gz attachment — skipping")
        return True

    filename = attachment_url.rsplit("/", 1)[-1]
    dest = CAPTURES_DIR / filename

    expected_raw = find_expected_card(body)
    expected_sfid = extract_sfid(expected_raw) if expected_raw else None

    labels = {lbl["name"] for lbl in issue.get("labels", [])}
    is_bug = "bug" in labels

    print(f"  #{number} {title!r}  [{state}]")
    print(f"    attachment : {filename}")
    print(f"    expected   : {expected_raw!r}  (sfid={expected_sfid})")
    print(f"    is_bug     : {is_bug}")

    if dry_run:
        if dest.exists():
            print(f"    status     : already present (dry-run skip)")
        else:
            print(f"    status     : WOULD download → {dest}")
        return True

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

    # Generate per-pipeline manifests (python + js-webgpu).
    _generate_manifests(dest, catalog=catalog)

    # For bug-labelled issues: verify that this capture actually reproduces
    # the reported discrepancy between the browser and Python pipelines.
    # A capture that doesn't reproduce the bug is not a useful test case.
    if is_bug:
        reproduced = _gm is not None and _gm.verify_reproduction(dest)
        if not reproduced:
            if force:
                print("    → --force: ingesting despite failed reproduction check")
            else:
                return False  # caller will report and exit non-zero

    return True


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
    parser.add_argument(
        "--force",
        action="store_true",
        help="ingest even if the capture does not reproduce the reported bug",
    )
    parser.add_argument(
        "--with-catalog",
        action="store_true",
        help="load the HuggingFace catalog to populate topMatchId in manifests",
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

    catalog = None
    if getattr(args, "with_catalog", False) and not args.dry_run:
        try:
            import collector_vision as cvg  # noqa: PLC0415

            print("Loading catalog from HuggingFace…")
            catalog = cvg.Catalog.load("hf://HanClinto/milo/scryfall-mtg")
            print("Catalog loaded.\n")
        except Exception as exc:
            print(f"Warning: could not load catalog ({exc}) — topMatchId will be null\n")

    print()
    failed_reproduction: list[int] = []
    for issue in issues:
        ok = ingest_issue(
            issue,
            dry_run=args.dry_run,
            force=args.force,
            catalog=catalog,
        )
        if not ok:
            failed_reproduction.append(issue["number"])

    print()
    if failed_reproduction:
        nums = ", ".join(f"#{n}" for n in failed_reproduction)
        print(
            f"ERROR: capture(s) for issue(s) {nums} do not reproduce the reported bug.\n"
            f"  Provide a capture from the affected device/browser, "
            f"or run with --force to ingest anyway."
        )
        sys.exit(1)

    print(
        "Done. Run the consistency tests with:\n"
        "  python -m pytest tests/test_pipeline_consistency.py -v\n"
        "  cd tests/js && npm test  (generates js-cpu manifests)"
    )


if __name__ == "__main__":
    main()
