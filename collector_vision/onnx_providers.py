"""Small public-to-ONNX provider mapping for neural inference."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
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
_NVIDIA_PROVIDERS = (_CUDA_PROVIDER, "TensorrtExecutionProvider")
_WARNED_RUNTIME_CONFLICT = False


def _has_accelerator(providers: list[str]) -> bool:
    return any(name in _AUTO_ACCELERATORS for name in providers)


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
                "such as `onnxruntime-gpu` for NVIDIA CUDA."
            )
        selected.append(_CPU_PROVIDER)
        return selected

    if provider == "auto":
        selected = [name for name in _AUTO_ACCELERATORS if name in available]
        selected.append(_CPU_PROVIDER)
        return selected

    valid = "', '".join(("auto", "cpu", "gpu"))
    raise ValueError(f"Unknown provider {provider!r}. Expected one of: '{valid}'.")


def _installed_onnx_runtime_distributions() -> list[tuple[str, str]]:
    installed: list[tuple[str, str]] = []
    for dist_name in ("onnxruntime", "onnxruntime-gpu"):
        try:
            installed.append((dist_name, version(dist_name)))
        except PackageNotFoundError:
            pass
    return installed


def _warn_if_conflicting_runtimes_installed() -> None:
    global _WARNED_RUNTIME_CONFLICT

    if _WARNED_RUNTIME_CONFLICT:
        return

    installed = _installed_onnx_runtime_distributions()
    if len(installed) <= 1:
        return

    _WARNED_RUNTIME_CONFLICT = True
    formatted = ", ".join(f"{name}=={dist_version}" for name, dist_version in installed)
    warn(
        "Both ONNX Runtime CPU and GPU distributions are installed "
        f"({formatted}). These packages provide the same `onnxruntime` Python "
        "module and can conflict; install exactly one backend, such as "
        "`onnxruntime` for CPU or `onnxruntime-gpu` for NVIDIA GPU.",
        RuntimeWarning,
        stacklevel=3,
    )


def create_inference_session(
    onnx_path: Path,
    num_threads: int,
    provider: Provider,
):
    """Create an ONNX Runtime session with simple provider selection.

    ``provider='auto'`` prefers installed accelerator providers, then falls back
    to CPU. Explicit ``'cpu'`` and ``'gpu'`` requests are respected.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "CollectorVision neural inference requires exactly one ONNX Runtime backend.\n\n"
            "For CPU inference, install:\n"
            "  pip install onnxruntime\n"
            "  uv add onnxruntime\n\n"
            "For NVIDIA GPU inference, install an ONNX Runtime GPU build that matches "
            "your CUDA runtime, for example:\n"
            "  pip install onnxruntime-gpu\n"
            "  uv add onnxruntime-gpu\n\n"
            "If you manage dependencies in requirements.txt or pyproject.toml, add "
            "one of those packages there. Avoid installing both onnxruntime and "
            "onnxruntime-gpu in the same environment."
        ) from exc

    _warn_if_conflicting_runtimes_installed()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1

    available = ort.get_available_providers()
    providers = _resolve_provider_names(provider, available)

    try:
        if any(name in _NVIDIA_PROVIDERS for name in providers) and hasattr(ort, "preload_dlls"):
            ort.preload_dlls(cuda=True, cudnn=True)

        sess = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=providers,
        )
        active_providers = list(sess.get_providers())
        if provider == "gpu" and not _has_accelerator(active_providers):
            raise RuntimeError(
                "GPU provider was requested, but ONNX Runtime initialized without "
                f"an accelerator provider. Requested providers: {providers}. "
                f"Active providers: {active_providers}."
            )
        if (
            provider == "auto"
            and _has_accelerator(providers)
            and not _has_accelerator(active_providers)
        ):
            warn(
                "ONNX Runtime accelerator providers were available but session "
                f"initialized with CPU only. Requested providers: {providers}. "
                f"Active providers: {active_providers}.",
                RuntimeWarning,
                stacklevel=2,
            )
        return sess
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
            sess_options=opts,
            providers=[_CPU_PROVIDER],
        )
