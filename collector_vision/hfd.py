"""HFD — HuggingFace Datasets gallery reference with transparent local caching.

Typical usage::

    import collector_vision as cvg

    cvid = cvg.Identifier(cvg.HFD("CollectorVision/galleries", "magic-scryfall-phash16"))

On first use HFD fetches the manifest from HuggingFace, resolves the latest
dated filename (e.g. ``magic-scryfall-phash16-2026-04.npz``), downloads it,
and caches it locally.  On subsequent calls within *cache_refresh* (default
7 days) it returns the local file without any network activity.  After that
window it re-checks the manifest; if the file is unchanged it updates the
manifest timestamp and carries on — no re-download.

The cache lives in ``~/.cache/collectorvision/`` by default (overridable via
the ``COLLECTORVISION_CACHE`` environment variable or the *cache_dir* argument).
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import timedelta
from pathlib import Path


# Default cache refresh window
_DEFAULT_REFRESH = timedelta(days=7)

# HuggingFace Datasets base URL
_HFD_BASE = "https://huggingface.co/datasets/{repo}/resolve/main/"


class HFD:
    """HuggingFace Datasets gallery reference.

    Parameters
    ----------
    repo:
        HuggingFace Datasets repository, e.g. ``"CollectorVision/galleries"``.
    name:
        Gallery base name **without** a date suffix, e.g.
        ``"magic-scryfall-phash16"``.  The latest dated file is resolved
        automatically via the repository manifest.
    cache_refresh:
        How long to trust the local manifest cache before re-checking
        HuggingFace.  Default is 7 days.  Pass ``timedelta(0)`` to always
        check, or ``timedelta(days=365)`` to effectively pin.
    cache_dir:
        Override the default cache directory.  If ``None``, uses
        ``$COLLECTORVISION_CACHE`` or ``~/.cache/collectorvision/``.
    """

    def __init__(
        self,
        repo: str,
        name: str,
        cache_refresh: timedelta = _DEFAULT_REFRESH,
        cache_dir: Path | None = None,
    ) -> None:
        self._repo = repo
        self._name = name
        self._cache_refresh = cache_refresh
        self._cache_dir = cache_dir or _default_cache_dir()
        self._base_url = _HFD_BASE.format(repo=repo)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def resolve(self) -> Path:
        """Return the local path to the gallery file, downloading if needed.

        Called automatically by :class:`~collector_vision.identifier.Identifier`
        — most users never call this directly.
        """
        manifest = self._get_manifest()
        filename = self._resolve_filename(manifest)
        local_path = self._cache_dir / filename

        if not local_path.exists():
            self._evict_old(filename)
            _download(self._base_url + filename, local_path)
        else:
            # File exists — evict any older sibling quietly
            self._evict_old(filename)

        return local_path

    def __repr__(self) -> str:
        return f"HFD({self._repo!r}, {self._name!r})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _manifest_path(self) -> Path:
        return self._cache_dir / "manifest.json"

    def _manifest_age_seconds(self) -> float:
        """Seconds since manifest.json was last written, or inf if absent."""
        p = self._manifest_path()
        if not p.exists():
            return float("inf")
        return time.time() - p.stat().st_mtime

    def _get_manifest(self) -> dict:
        """Return the manifest dict, using local cache when fresh enough.

        Falls back to the stale cache on network failure.  Raises only if
        there is no cached copy at all and the network is unavailable.
        """
        refresh_seconds = self._cache_refresh.total_seconds()

        if self._manifest_age_seconds() < refresh_seconds:
            # Cache is fresh — no network call needed
            return json.loads(self._manifest_path().read_text(encoding="utf-8"))

        # Cache is stale (or absent) — try to fetch a fresh copy
        url = self._base_url + "manifest.json"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._manifest_path().write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
            return data
        except Exception as exc:
            p = self._manifest_path()
            if p.exists():
                # Stale cache is better than nothing
                return json.loads(p.read_text(encoding="utf-8"))
            raise RuntimeError(
                f"Could not fetch manifest from {url!r} and no local cache found.\n"
                f"Check your internet connection, or download the gallery manually.\n"
                f"Original error: {exc}"
            ) from exc

    def _resolve_filename(self, manifest: dict) -> str:
        """Find the filename in the manifest whose base matches self._name.

        Manifest structure::

            {
              "games": {
                "magic": {
                  "phash16": "magic-scryfall-phash16-2026-04.npz",
                  "milo1":   "magic-scryfall-milo1-2026-04.npz"
                }
              }
            }
        """
        prefix = self._name + "-"
        for game_data in manifest.get("games", {}).values():
            for filename in game_data.values():
                if filename.startswith(prefix):
                    return filename

        # Also check a flat "files" list for future manifest formats
        for filename in manifest.get("files", []):
            if filename.startswith(prefix):
                return filename

        available = sorted(
            f
            for game_data in manifest.get("games", {}).values()
            for f in game_data.values()
        )
        raise KeyError(
            f"No gallery matching {self._name!r} found in {self._repo!r} manifest.\n"
            f"Available: {available or '(manifest is empty)'}"
        )

    def _evict_old(self, keep_filename: str) -> None:
        """Delete any cached files matching this name pattern except keep_filename."""
        prefix = self._name + "-"
        for old in self._cache_dir.glob(f"{prefix}*.npz"):
            if old.name != keep_filename:
                try:
                    old.unlink()
                except OSError:
                    pass  # best effort


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_cache_dir() -> Path:
    base = Path(os.environ.get("COLLECTORVISION_CACHE", "~/.cache/collectorvision"))
    return base.expanduser()


def _download(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Download *url* to *dest* with a simple progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".download")
    try:
        print(f"Downloading {dest.name} ...")
        with urllib.request.urlopen(url, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            written = 0
            with tmp.open("wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = written * 100 // total
                        print(f"\r  {pct:3d}%  {written // 1024 // 1024} MB", end="", flush=True)
            if total:
                print()
        tmp.rename(dest)
        print(f"  saved → {dest}")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
