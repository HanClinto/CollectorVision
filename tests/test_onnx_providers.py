import builtins
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from collector_vision.onnx_providers import _resolve_provider_names, create_inference_session


class ProviderResolutionTests(unittest.TestCase):
    def test_auto_prefers_available_accelerators_then_cpu(self) -> None:
        providers = _resolve_provider_names(
            "auto",
            ["CPUExecutionProvider", "CoreMLExecutionProvider", "CUDAExecutionProvider"],
        )

        self.assertEqual(
            providers,
            ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"],
        )

    def test_cpu_forces_cpu_provider(self) -> None:
        providers = _resolve_provider_names(
            "cpu",
            ["CPUExecutionProvider", "CUDAExecutionProvider"],
        )

        self.assertEqual(providers, ["CPUExecutionProvider"])

    def test_gpu_requires_an_accelerator_provider(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "GPU provider was requested"):
            _resolve_provider_names("gpu", ["CPUExecutionProvider"])

    def test_gpu_uses_available_accelerators_then_cpu_graph_fallback(self) -> None:
        providers = _resolve_provider_names(
            "gpu",
            ["CPUExecutionProvider", "CoreMLExecutionProvider"],
        )

        self.assertEqual(providers, ["CoreMLExecutionProvider", "CPUExecutionProvider"])

    def test_unknown_provider_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown provider"):
            _resolve_provider_names("magic", ["CPUExecutionProvider"])  # type: ignore[arg-type]


class CreateInferenceSessionTests(unittest.TestCase):
    def test_missing_onnxruntime_has_actionable_error(self) -> None:
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):  # noqa: ANN001
            if name == "onnxruntime":
                raise ImportError("missing onnxruntime")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(ImportError, "pip install onnxruntime"):
                create_inference_session(Path("model.onnx"), 4, "auto")

    def test_auto_falls_back_to_cpu_when_accelerator_session_fails(self) -> None:
        calls: list[list[str]] = []

        class FakeSession:
            def __init__(self, path, sess_options, providers):  # noqa: ANN001
                calls.append(list(providers))
                if providers != ["CPUExecutionProvider"]:
                    raise RuntimeError("accelerator failed")

        fake_ort = types.SimpleNamespace(
            get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
            InferenceSession=FakeSession,
            SessionOptions=types.SimpleNamespace,
        )

        with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
            with self.assertWarnsRegex(RuntimeWarning, "falling back"):
                create_inference_session(Path("model.onnx"), 4, "auto")

        self.assertEqual(
            calls,
            [
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                ["CPUExecutionProvider"],
            ],
        )

    def test_gpu_rejects_session_that_silently_falls_back_to_cpu(self) -> None:
        class FakeSession:
            def __init__(self, path, sess_options, providers):  # noqa: ANN001
                self.providers = list(providers)

            def get_providers(self) -> list[str]:
                return ["CPUExecutionProvider"]

        fake_ort = types.SimpleNamespace(
            get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
            InferenceSession=FakeSession,
            SessionOptions=types.SimpleNamespace,
        )

        with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
            with self.assertRaisesRegex(RuntimeError, "initialized without an accelerator"):
                create_inference_session(Path("model.onnx"), 4, "gpu")

    def test_auto_warns_when_session_silently_falls_back_to_cpu(self) -> None:
        class FakeSession:
            def __init__(self, path, sess_options, providers):  # noqa: ANN001
                self.providers = list(providers)

            def get_providers(self) -> list[str]:
                return ["CPUExecutionProvider"]

        fake_ort = types.SimpleNamespace(
            get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
            InferenceSession=FakeSession,
            SessionOptions=types.SimpleNamespace,
        )

        with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
            with self.assertWarnsRegex(RuntimeWarning, "CPU only"):
                create_inference_session(Path("model.onnx"), 4, "auto")


if __name__ == "__main__":
    unittest.main()
