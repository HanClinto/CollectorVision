"""pytest session-level setup for the CollectorVision test suite.

Auto-fetches capture bundles from GitHub issues before running tests that
depend on them.  This keeps the git repo free of large binary fixtures while
still making the capture-regression and pipeline-consistency test suites work
out-of-the-box in CI and locally.

How it works
------------
Before the test session starts, ``pytest_sessionstart`` checks whether the
captures directory contains any ``.json.gz`` files.  If it is empty (as it
will be on a fresh clone, since the directory contents are gitignored), it
runs ``python scripts/ingest_bug_reports.py`` to download all open bug
captures from GitHub.

Set ``GITHUB_TOKEN`` in the environment for authenticated API access (avoids
the 60 req/hr anonymous rate limit and works on private repos).

Skipping the fetch
------------------
Set ``CV_SKIP_CAPTURE_FETCH=1`` to skip the auto-fetch entirely (useful when
running tests without network access or to avoid re-downloading when bundles
are already present).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPTURES_DIR = ROOT / "tests" / "fixtures" / "captures"
INGEST_SCRIPT = ROOT / "scripts" / "ingest_bug_reports.py"


def pytest_sessionstart(session) -> None:
    """Download capture bundles if the captures directory is empty."""
    if os.environ.get("CV_SKIP_CAPTURE_FETCH"):
        return

    existing = list(CAPTURES_DIR.glob("*.json.gz"))
    if existing:
        return  # already populated — nothing to do

    if not INGEST_SCRIPT.exists():
        return  # script missing — don't block test collection

    print(
        "\n[conftest] captures/ is empty — fetching bundles from GitHub issues…\n"
        "  (set CV_SKIP_CAPTURE_FETCH=1 to suppress)"
    )
    result = subprocess.run(
        [sys.executable, str(INGEST_SCRIPT)],
        cwd=str(ROOT),
        # Don't pass stdin so it can't block; inherit stdout/stderr for visibility.
    )
    if result.returncode != 0:
        print(
            "\n[conftest] WARNING: ingest_bug_reports.py exited non-zero. "
            "Capture-regression tests may be skipped or fail.\n"
            "  Set GITHUB_TOKEN for authenticated access, or run the script manually:\n"
            f"    python {INGEST_SCRIPT.relative_to(ROOT)}"
        )
