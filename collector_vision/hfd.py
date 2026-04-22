"""HFD — HuggingFace model-repo gallery reference with transparent local caching.

Galleries live alongside the model weights in the same HuggingFace model repo,
under a ``galleries/`` subfolder.  HFD wraps one such gallery: on first use it
fetches ``galleries/manifest.json`` to find the latest snapshot filename,
downloads it, and caches it locally.  Subsequent calls within ``cache_refresh``
(default 7 days) return the cached file with no network activity.

Typical usage::

    import collector_vision as cvg

    # MTG gallery, auto-download
    cvid = cvg.Identifier(cvg.HFD("HanClinto/milo", "scryfall-mtg"))

    # Pokémon gallery
    cvid = cvg.Identifier(cvg.HFD("HanClinto/milo", "tcgplayer-pokemon"))

    # Multi-game — merge into one search index
    cvid = cvg.Identifier(
        cvg.HFD("HanClinto/milo", "scryfall-mtg"),
        cvg.HFD("HanClinto/milo", "tcgplayer-pokemon"),
    )

    # Prefer Gallery.for_game for the common case:
    from collector_vision.games import Game
    gallery = cvg.Gallery.for_game(Game.MTG)

The cache lives in ``~/.cache/collectorvision/<repo>/<gallery_key>/``.
Override the root with ``$COLLECTORVISION_CACHE`` or the ``cache_dir`` argument.

``galleries/manifest.json`` format::

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
# Model repos (not dataset repos) — no "datasets/" in the path
_HF_MODEL_BASE = "https://huggingface.co/{repo}/resolve/main/"
_GALLERIES_SUBFOLDER = "galleries"


class HFD:
    """Reference to a gallery stored in a CollectorVision HuggingFace model repo.

    Parameters
    ----------
    repo:
        HuggingFace model repository id, e.g. ``"HanClinto/milo"``.
    gallery_key:
        Which gallery within the repo, e.g. ``"scryfall-mtg"``.
        Matches a key in ``galleries/manifest.json``.
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
        gallery_key: str,
        cache_refresh: timedelta | None = _DEFAULT_REFRESH,
        cache_dir: Path | None = None,
        offline: bool = False,
    ) -> None:
        self._repo = repo
        self._gallery_key = gallery_key
        self._cache_refresh = cache_refresh
        self._offline = offline
        root = cache_dir or _default_cache_dir()
        # Per-repo, per-gallery subdirectory avoids any filename collisions
        self._cache_dir = root / repo.replace("/", "_") / gallery_key
        self._base_url = _HF_MODEL_BASE.format(repo=repo) + _GALLERIES_SUBFOLDER + "/"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def resolve(self) -> Path:
        """Return the local path to the gallery NPZ, downloading if needed.

        Called automatically by :class:`~collector_vision.identifier.Identifier`.
        """
        manifest = self._get_manifest()
        entry = manifest.get(self._gallery_key)
        if not entry:
            available = list(manifest.keys())
            raise KeyError(
                f"Gallery key {self._gallery_key!r} not found in {self._repo!r} manifest.\n"
                f"Available: {available or '(manifest is empty)'}"
            )
        filename = entry["latest"]
        local_path = self._cache_dir / filename

        if not local_path.exists():
            if self._offline:
                raise FileNotFoundError(
                    f"Gallery not cached locally: {local_path}\n"
                    f"Initialise HFD without offline=True to download it."
                )
            self._evict_old(filename)
            _download(self._base_url + filename, local_path)
        else:
            self._evict_old(filename)

        return local_path

    def __repr__(self) -> str:
        return f"HFD({self._repo!r}, {self._gallery_key!r})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _manifest_path(self) -> Path:
        # Manifest is shared across all gallery keys in a repo
        return (self._cache_dir.parent) / "manifest.json"

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
                f"Initialise HFD without offline=True to download it."
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
                f"Check your internet connection, or download the gallery manually.\n"
                f"Original error: {exc}"
            ) from exc

    def _evict_old(self, keep_filename: str) -> None:
        """Delete stale NPZ files in this gallery's cache dir."""
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
