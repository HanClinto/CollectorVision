"""Download and verify model artifacts declared by the CollectorVision registry."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from collector_vision.model_registry import ModelSpec, load_model_registry


def resolve_registered_model(
    family: str,
    *,
    task: str,
    version: str | None = None,
    channel: str = "stable",
    cache_dir: Path | None = None,
    offline: bool = False,
) -> Path:
    """Resolve a family/channel selection to a verified compatible local model."""
    registry = load_model_registry(cache_dir=cache_dir, offline=offline)
    model = registry.get_model(family=family, version=version, channel=channel)
    if model.task != task:
        raise ValueError(f"Model {model.id!r} has task {model.task!r}; expected {task!r}")
    return resolve_model_artifact(model, cache_dir=cache_dir, offline=offline)


def resolve_model_artifact(
    model: ModelSpec,
    *,
    cache_dir: Path | None = None,
    offline: bool = False,
) -> Path:
    """Return a verified local ONNX path for a registry model.

    The artifact is cached by its SHA-256 digest, so exact model releases never
    overwrite one another. Hugging Face support is imported only when a missing
    model needs downloading; local cached and offline use have no extra runtime
    dependency.
    """
    root = cache_dir or _default_cache_dir()
    destination = root / "models" / model.sha256 / model.filename

    if destination.exists():
        _verify_sha256(destination, model.sha256)
        return destination

    if offline:
        raise FileNotFoundError(
            f"Model {model.id!r} is not cached at {destination}. "
            "Disable offline mode to download it."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    _download_from_hub(model, destination.parent)
    _verify_sha256(destination, model.sha256)
    return destination


def _default_cache_dir() -> Path:
    base = Path(os.environ.get("COLLECTORVISION_CACHE", "~/.cache/collectorvision"))
    return base.expanduser()


def _download_from_hub(model: ModelSpec, destination_dir: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Hugging Face model resolution requires the optional dependency. "
            'Install it with: pip install "collectorvision[hf]"'
        ) from exc

    hf_hub_download(
        repo_id=model.repository,
        filename=model.filename,
        revision=model.revision,
        local_dir=destination_dir,
    )


def _verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"Model artifact checksum mismatch for {path}: expected {expected}, got {actual}"
        )
