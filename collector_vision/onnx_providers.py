"""Small public-to-ONNX provider mapping for neural inference."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from warnings import warn

Provider = Literal["auto", "cpu", "gpu"]

_CPU_PROVIDER = "CPUExecutionProvider"
_CUDA_PROVIDER = "CUDAExecutionProvider"
_AUTO_ACCELERATORS = (
    _CUDA_PROVIDER,
    "CoreMLExecutionProvider",
    "DmlExecutionProvider",
    "ROCMExecutionProvider",
)


def _resolve_provider_names(provider: Provider, available: list[str]) -> list[str]:
    if provider == "cpu":
        return [_CPU_PROVIDER]

    if provider == "gpu":
        selected = [name for name in _AUTO_ACCELERATORS if name in available]
        if not selected:
            raise RuntimeError(
                "GPU provider was requested, but ONNX Runtime does not report any "
                f"accelerator providers as available. Available providers: {available}. "
                "Install an accelerator-enabled ONNX Runtime package for your platform, "
                "such as `collectorvision[gpu]` or `onnxruntime-gpu` for NVIDIA CUDA."
            )
        selected.append(_CPU_PROVIDER)
        return selected

    if provider == "auto":
        selected = [name for name in _AUTO_ACCELERATORS if name in available]
        selected.append(_CPU_PROVIDER)
        return selected

    valid = "', '".join(("auto", "cpu", "gpu"))
    raise ValueError(f"Unknown provider {provider!r}. Expected one of: '{valid}'.")


def create_inference_session(
    onnx_path: Path,
    sess_options: object,
    provider: Provider,
):
    """Create an ONNX Runtime session with simple provider selection.

    ``provider='auto'`` prefers installed accelerator providers, then falls back
    to CPU. Explicit ``'cpu'`` and ``'gpu'`` requests are respected.
    """
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers = _resolve_provider_names(provider, available)

    try:
        return ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=providers,
        )
    except Exception as exc:
        if provider != "auto" or providers == [_CPU_PROVIDER]:
            raise

        warn(
            "ONNX Runtime accelerator session initialization failed; falling back "
            f"to CPUExecutionProvider. Original error: {type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=[_CPU_PROVIDER],
        )
