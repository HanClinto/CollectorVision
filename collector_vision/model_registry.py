"""Registry of supported model identities and their stable aliases.

An alias such as ``"cornelius"`` selects the current supported release for a
model family. Exact IDs such as ``"cornelius-2.12"`` remain stable for callers
that need reproducible model selection.

This module intentionally separates model identity from artifact resolution.
The current package still supplies bundled ONNX files; a later resolver will
attach immutable Hugging Face revisions and SHA-256 verification to each entry.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """A supported model release and its compatibility metadata."""

    id: str
    family: str
    version: str
    task: str
    architecture: str
    input_size: int


_MODELS: dict[str, ModelSpec] = {
    "cornelius-2.12": ModelSpec(
        id="cornelius-2.12",
        family="cornelius",
        version="2.12",
        task="corner-detection",
        architecture="mobilevit-xxs-simcc",
        input_size=384,
    ),
    "milo-1.0.0": ModelSpec(
        id="milo-1.0.0",
        family="milo",
        version="1.0.0",
        task="card-embedding",
        architecture="mobilevit-xxs-arcface",
        input_size=448,
    ),
}

_DEFAULTS: dict[str, str] = {
    "cornelius": "cornelius-2.12",
    "milo": "milo-1.0.0",
}


def get_model(model: str) -> ModelSpec:
    """Resolve a stable family alias or exact model ID to its specification.

    ``"cornelius"`` resolves to the current default Cornelius release, while
    ``"cornelius-2.12"`` always resolves to that specific version.
    """
    normalized = model.lower().strip()
    model_id = _DEFAULTS.get(normalized, normalized)
    try:
        return _MODELS[model_id]
    except KeyError:
        available = ", ".join(available_models())
        raise ValueError(f"Unknown model {model!r}. Available models: {available}") from None


def available_models() -> tuple[str, ...]:
    """Return stable aliases and exact IDs supported by this registry."""
    return tuple(sorted({*_DEFAULTS, *_MODELS}))
