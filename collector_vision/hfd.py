"""HFD — HuggingFace model-repo catalog reference with transparent local caching.

Catalogs live alongside the model weights in the same HuggingFace model repo,
under a ``catalogs/`` subfolder.  HFD wraps one such catalog: on first use it
fetches ``catalogs/manifest.json`` to find the latest snapshot filename,
downloads it, and caches it locally.  Subsequent calls within ``cache_refresh``
(default 7 days) return the cached file with no network activity.

Typical usage::

    catalog = cvg.Catalog.load("hf://HanClinto/milo/scryfall-mtg")

    # Or construct HFD directly for more control:
    from collector_vision.hfd import HFD
    catalog = cvg.Catalog.load(HFD("HanClinto/milo", "scryfall-mtg"))

The cache lives in ``~/.cache/collectorvision/<repo>/<catalog_key>/``.
Override the root with ``$COLLECTORVISION_CACHE`` or the ``cache_dir`` argument.

``catalogs/manifest.json`` format::

    {
      "scryfall-mtg": {
        "latest": "milo1-scryfall-mtg-2026-04.npz",
        "files":  ["milo1-scryfall-mtg-2026-04.npz"]
      },
      "tcgplayer-pokemon": {
        "latest": "milo1-tcgplayer-pokemon-2026-04.npz",
        "files":  ["milo1-tcgplayer-pokemon-2026-04.npz"]
      }
    }
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import timedelta
from pathlib import Path


_DEFAULT_REFRESH = timedelta(days=7)
_HF_MODEL_BASE = "https://huggingface.co/{repo}/resolve/main/"
_CATALOGS_SUBFOLDER = "catalogs"


class HFD:
    """Reference to a catalog stored in a CollectorVision HuggingFace model repo.

    Parameters
    ----------
    repo:
        HuggingFace model repository id, e.g. ``"HanClinto/milo"``.
    catalog_key:
        Which catalog within the repo, e.g. ``"scryfall-mtg"``.
        Matches a key in ``catalogs/manifest.json``.
    cache_refresh:
        How long to trust the local manifest cache before re-checking HF.
        Default 7 days.  Pass ``timedelta(0)`` to always check, or ``None``
        to never recheck once cached.
    cache_dir:
        Override the root cache directory.
    offline:
        If ``True``, never make network calls.  Raises if not cached.
    """

    def __init__(
        self,
        repo: str,
        catalog_key: str,
        cache_refresh: timedelta | None = _DEFAULT_REFRESH,
        cache_dir: Path | None = None,
        offline: bool = False,
    ) -> None:
        self._repo = repo
        self._catalog_key = catalog_key
        self._cache_refresh = cache_refresh
        self._offline = offline
        root = cache_dir or _default_cache_dir()
        self._cache_dir = root / repo.replace("/", "_") / catalog_key
        self._base_url = _HF_MODEL_BASE.format(repo=repo) + _CATALOGS_SUBFOLDER + "/"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def resolve(self) -> Path:
        """Return the local path to the catalog NPZ, downloading if needed."""
        manifest = self._get_manifest()
        entry = manifest.get(self._catalog_key)
        if not entry:
            available = list(manifest.keys())
            raise KeyError(
                f"Catalog key {self._catalog_key!r} not found in {self._repo!r} manifest.\n"
                f"Available: {available or '(manifest is empty)'}"
            )
        filename = entry["latest"]
        local_path = self._cache_dir / filename

        if not local_path.exists():
            if self._offline:
                raise FileNotFoundError(
                    f"Catalog not cached locally: {local_path}\n"
                    "Initialise HFD without offline=True to download it."
                )
            self._evict_old(filename)
            _download(self._base_url + filename, local_path)
        else:
            self._evict_old(filename)

        return local_path

    def __repr__(self) -> str:
        return f"HFD({self._repo!r}, {self._catalog_key!r})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _manifest_path(self) -> Path:
        return self._cache_dir.parent / "manifest.json"

    def _manifest_stale(self) -> bool:
        p = self._manifest_path()
        if not p.exists():
            return True
        if self._cache_refresh is None:
            return False
        return time.time() - p.stat().st_mtime >= self._cache_refresh.total_seconds()

    def _get_manifest(self) -> dict:
        if not self._manifest_stale():
            return json.loads(self._manifest_path().read_text(encoding="utf-8"))

        if self._offline:
            p = self._manifest_path()
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
            raise FileNotFoundError(
                f"No cached manifest for {self._repo!r}.\n"
                "Initialise HFD without offline=True to download it."
            )

        url = self._base_url + "manifest.json"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            self._manifest_path().parent.mkdir(parents=True, exist_ok=True)
            self._manifest_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
            return data
        except Exception as exc:
            p = self._manifest_path()
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
            raise RuntimeError(
                f"Could not fetch manifest from {url!r} and no local cache found.\n"
                "Check your internet connection, or download the catalog manually.\n"
                f"Original error: {exc}"
            ) from exc

    def _evict_old(self, keep_filename: str) -> None:
        """Delete stale NPZ files in this catalog's cache dir."""
        for old in self._cache_dir.glob("*.npz"):
            if old.name != keep_filename:
                try:
                    old.unlink()
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_cache_dir() -> Path:
    base = Path(os.environ.get("COLLECTORVISION_CACHE", "~/.cache/collectorvision"))
    return base.expanduser()


def _download(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
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
