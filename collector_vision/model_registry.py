"""Registry of supported model identities and their channel aliases.

An alias such as ``"cornelius"`` selects the current supported release for a
model family. Exact IDs such as ``"cornelius-2.12"`` remain stable for callers
that need reproducible model selection.

The packaged JSON document is a bootstrap snapshot of the remote registry. This
module intentionally separates model identity from artifact resolution; a later
resolver will attach immutable Hugging Face revisions and SHA-256 verification.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from datetime import timedelta
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    """A supported model release and its compatibility metadata."""

    id: str
    family: str
    version: str
    task: str
    architecture: str
    input_size: int
    repository: str
    revision: str
    filename: str
    sha256: str
    size_bytes: int


_DEFAULT_REFRESH = timedelta(days=7)
_REGISTRY_URL = "https://huggingface.co/HanClinto/CollectorVision/resolve/main/registry.json"


class ModelRegistry:
    """A validated model registry with exact releases and channel pointers."""

    def __init__(self, data: dict) -> None:
        if data.get("schema_version") != 1:
            raise RuntimeError("Unsupported model registry schema")
        self._models = {
            model_id: ModelSpec(id=model_id, **metadata)
            for model_id, metadata in data["models"].items()
        }
        self._channels: dict[str, dict[str, str]] = data["channels"]

    def get_model(
        self,
        model: str | None = None,
        *,
        family: str | None = None,
        version: str | None = None,
        channel: str = "stable",
    ) -> ModelSpec:
        """Resolve a model ID or family selection to its specification."""
        if model is not None and (family is not None or version is not None):
            raise ValueError("Pass either model or family/version, not both")
        if model is None and family is None:
            raise ValueError("Pass a model ID or family")

        normalized = (model or family or "").lower().strip()
        if model is None and version is not None:
            model_id = f"{normalized}-{version.strip()}"
        elif normalized in self._models:
            model_id = normalized
        else:
            try:
                model_id = self._channels[channel.lower().strip()][normalized]
            except KeyError:
                available = ", ".join(self.available_models(channel=channel))
                raise ValueError(
                    f"Unknown model {normalized!r}. Available models: {available}"
                ) from None

        try:
            return self._models[model_id]
        except KeyError:
            raise RuntimeError(f"Registry channel points to unknown model {model_id!r}") from None

    def available_models(self, channel: str = "stable") -> tuple[str, ...]:
        """Return aliases for a channel and every exact model ID in the registry."""
        try:
            aliases = self._channels[channel.lower().strip()]
        except KeyError:
            known = ", ".join(self.available_channels())
            raise ValueError(
                f"Unknown model channel {channel!r}. Available channels: {known}"
            ) from None
        return tuple(sorted({*aliases, *self._models}))

    def available_channels(self) -> tuple[str, ...]:
        """Return supported model channels."""
        return tuple(sorted(self._channels))


def _load_packaged_registry() -> ModelRegistry:
    registry_path = files("collector_vision").joinpath("data/model_registry.json")
    return ModelRegistry(json.loads(registry_path.read_text(encoding="utf-8")))


_BOOTSTRAP_REGISTRY = _load_packaged_registry()


def load_model_registry(
    *,
    cache_dir: Path | None = None,
    cache_refresh: timedelta | None = _DEFAULT_REFRESH,
    offline: bool = False,
) -> ModelRegistry:
    """Load a cached remote registry, falling back to the packaged snapshot.

    The registry document is small and may refresh independently from the
    installed Python package. Failed refreshes preserve a valid cached or
    packaged registry so existing model selections remain usable.
    """
    root = cache_dir or _default_cache_dir()
    cache_path = root / "registry.json"
    if _is_fresh(cache_path, cache_refresh) or offline:
        return _load_cached_or_bootstrap(cache_path)

    try:
        with urllib.request.urlopen(_REGISTRY_URL, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        registry = ModelRegistry(data)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(".download")
        temp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(cache_path)
        return registry
    except Exception:
        return _load_cached_or_bootstrap(cache_path)


def _load_cached_or_bootstrap(cache_path: Path) -> ModelRegistry:
    if cache_path.exists():
        try:
            return ModelRegistry(json.loads(cache_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    return _BOOTSTRAP_REGISTRY


def _is_fresh(path: Path, refresh: timedelta | None) -> bool:
    if not path.exists():
        return False
    if refresh is None:
        return True
    return time.time() - path.stat().st_mtime < refresh.total_seconds()


def _default_cache_dir() -> Path:
    base = Path(os.environ.get("COLLECTORVISION_CACHE", "~/.cache/collectorvision"))
    return base.expanduser()


def get_model(
    model: str | None = None,
    *,
    family: str | None = None,
    version: str | None = None,
    channel: str = "stable",
) -> ModelSpec:
    """Resolve a model from the package's reproducible bootstrap registry."""
    return _BOOTSTRAP_REGISTRY.get_model(model, family=family, version=version, channel=channel)


def available_models(channel: str = "stable") -> tuple[str, ...]:
    """Return aliases for a channel and every exact model ID in the registry."""
    return _BOOTSTRAP_REGISTRY.available_models(channel)


def available_channels() -> tuple[str, ...]:
    """Return supported model channels."""
    return _BOOTSTRAP_REGISTRY.available_channels()
