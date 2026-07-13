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
from dataclasses import dataclass
from importlib.resources import files


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


def _load_registry() -> tuple[dict[str, ModelSpec], dict[str, dict[str, str]]]:
    registry_path = files("collector_vision").joinpath("data/model_registry.json")
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise RuntimeError("Unsupported model registry schema")

    models = {
        model_id: ModelSpec(id=model_id, **metadata)
        for model_id, metadata in data["models"].items()
    }
    return models, data["channels"]


_MODELS, _CHANNELS = _load_registry()


def get_model(
    model: str | None = None,
    *,
    family: str | None = None,
    version: str | None = None,
    channel: str = "stable",
) -> ModelSpec:
    """Resolve a model ID or family selection to its specification.

    ``model`` accepts an exact ID or a family alias on the requested channel.
    Alternatively, pass ``family`` and an optional exact ``version``. A version
    bypasses channel selection and is stable across future channel updates.
    """
    if model is not None and (family is not None or version is not None):
        raise ValueError("Pass either model or family/version, not both")
    if model is None and family is None:
        raise ValueError("Pass a model ID or family")

    normalized = (model or family or "").lower().strip()
    if model is None and version is not None:
        model_id = f"{normalized}-{version.strip()}"
    elif normalized in _MODELS:
        model_id = normalized
    else:
        try:
            model_id = _CHANNELS[channel.lower().strip()][normalized]
        except KeyError:
            available = ", ".join(available_models(channel=channel))
            raise ValueError(
                f"Unknown model {normalized!r}. Available models: {available}"
            ) from None

    try:
        return _MODELS[model_id]
    except KeyError:
        raise RuntimeError(f"Registry channel points to unknown model {model_id!r}") from None


def available_models(channel: str = "stable") -> tuple[str, ...]:
    """Return aliases for a channel and every exact model ID in the registry."""
    try:
        aliases = _CHANNELS[channel.lower().strip()]
    except KeyError:
        known = ", ".join(available_channels())
        raise ValueError(
            f"Unknown model channel {channel!r}. Available channels: {known}"
        ) from None
    return tuple(sorted({*aliases, *_MODELS}))


def available_channels() -> tuple[str, ...]:
    """Return supported model channels."""
    return tuple(sorted(_CHANNELS))
