"""
create_minimal_model.py
-----------------------
Builds a series of minimal ONNX test models with deterministic weights and
prints the expected output for the standard test input so it can be compared
against onnxruntime-web WebGPU EP output on an Android device.

The goal is to isolate WHICH operator produces wrong output on Android ARM.
Start with the simplest model (single Conv2d) and work up.

Usage
-----
    pip install onnx numpy onnxruntime
    python create_minimal_model.py

Output files
------------
    test_single_conv.onnx      One Conv2d (3→8 channels, 3×3 kernel, no bias)
    test_conv_bn_relu.onnx     Conv2d → BatchNorm → ReLU (common ResNet block)
    test_two_conv.onnx         Two Conv2d in sequence (checks accumulation)
    test_global_avgpool.onnx   Conv2d → GlobalAveragePool → Flatten → Gemm

The expected outputs for a 1×3×64×64 all-0.5 input are printed to stdout.
Copy them into browser_test.html's EXPECTED_OUTPUTS section.

How to use the outputs
----------------------
Load each .onnx in browser_test.html (see that file).  Paste both the
expected outputs (from this script) and the .onnx file path into the HTML,
then open it on the Android device via USB remote debugging.

If a model passes WASM but fails WebGPU, that model's operator set contains
the broken op.  Bisect down to a single operator.
"""

import sys
import textwrap

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

RNG = np.random.default_rng(42)      # deterministic weights
INPUT_SHAPE = (1, 3, 64, 64)         # small enough for fast browser testing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_conv_weights(out_ch, in_ch, kh, kw, name):
    """Deterministic small float32 weights, values in [-0.5, 0.5]."""
    data = RNG.uniform(-0.5, 0.5, (out_ch, in_ch, kh, kw)).astype(np.float32)
    return numpy_helper.from_array(data, name=name)


def make_bn_params(ch, prefix):
    scale  = numpy_helper.from_array(np.ones(ch, dtype=np.float32),               f"{prefix}_scale")
    bias   = numpy_helper.from_array(np.zeros(ch, dtype=np.float32),              f"{prefix}_bias")
    mean   = numpy_helper.from_array(np.zeros(ch, dtype=np.float32),              f"{prefix}_mean")
    var    = numpy_helper.from_array(np.ones(ch, dtype=np.float32) * 0.9,         f"{prefix}_var")
    return scale, bias, mean, var


def run_onnx(path: str, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out  = sess.run(None, {name: x})
    return out[0]


def save_and_check(model, path: str, x: np.ndarray):
    onnx.checker.check_model(model)
    onnx.save(model, path)
    y = run_onnx(path, x)
    print(f"\n{'='*60}")
    print(f"Model: {path}")
    print(f"Input:  shape={x.shape}  mean={x.mean():.4f}  std={x.std():.4f}")
    print(f"Output: shape={y.shape}  dtype={y.dtype}")
    print(f"  min={y.min():.6f}  max={y.max():.6f}  mean={y.mean():.6f}")
    print(f"  first 8 values (flat): {y.ravel()[:8].tolist()}")
    return y


# ---------------------------------------------------------------------------
# Model 1: single Conv2d (the most fundamental op)
# ---------------------------------------------------------------------------

def build_single_conv():
    """1 × Conv2d(3→8, 3×3, stride=1, padding=1, no bias)."""
    name   = "test_single_conv"
    w      = make_conv_weights(8, 3, 3, 3, "w0")

    X      = helper.make_tensor_value_info("X",  TensorProto.FLOAT, INPUT_SHAPE)
    Y      = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, None)

    node   = helper.make_node(
        "Conv", inputs=["X", "w0"], outputs=["Y"],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1],
    )
    graph  = helper.make_graph([node], name, [X], [Y], initializer=[w])
    model  = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model, f"{name}.onnx"


# ---------------------------------------------------------------------------
# Model 2: Conv → BatchNorm → ReLU
# ---------------------------------------------------------------------------

def build_conv_bn_relu():
    """Conv2d(3→8) → BatchNorm → ReLU.  Exercises precision after normalisation."""
    name   = "test_conv_bn_relu"
    w      = make_conv_weights(8, 3, 3, 3, "cbn_w")
    scale, bias, mean, var = make_bn_params(8, "cbn_bn")

    X      = helper.make_tensor_value_info("X",  TensorProto.FLOAT, INPUT_SHAPE)
    Y      = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, None)

    conv   = helper.make_node("Conv", ["X", "cbn_w"], ["conv_out"],
                              kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    bn     = helper.make_node("BatchNormalization",
                              ["conv_out", "cbn_bn_scale", "cbn_bn_bias",
                               "cbn_bn_mean", "cbn_bn_var"],
                              ["bn_out"], epsilon=1e-5, momentum=0.9)
    relu   = helper.make_node("Relu", ["bn_out"], ["Y"])

    graph  = helper.make_graph(
        [conv, bn, relu], name, [X], [Y],
        initializer=[w, scale, bias, mean, var],
    )
    model  = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model, f"{name}.onnx"


# ---------------------------------------------------------------------------
# Model 3: two Conv layers in sequence
# ---------------------------------------------------------------------------

def build_two_conv():
    """Conv2d(3→16) → ReLU → Conv2d(16→8).  Checks multi-layer accumulation."""
    name   = "test_two_conv"
    w0     = make_conv_weights(16, 3,  3, 3, "tc_w0")
    w1     = make_conv_weights(8,  16, 1, 1, "tc_w1")   # 1×1 conv (pointwise)

    X      = helper.make_tensor_value_info("X",  TensorProto.FLOAT, INPUT_SHAPE)
    Y      = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, None)

    conv0  = helper.make_node("Conv", ["X", "tc_w0"], ["c0_out"],
                              kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    relu   = helper.make_node("Relu", ["c0_out"], ["relu_out"])
    conv1  = helper.make_node("Conv", ["relu_out", "tc_w1"], ["Y"],
                              kernel_shape=[1, 1])

    graph  = helper.make_graph([conv0, relu, conv1], name, [X], [Y],
                               initializer=[w0, w1])
    model  = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model, f"{name}.onnx"


# ---------------------------------------------------------------------------
# Model 4: Conv → GlobalAveragePool → Flatten → Gemm
# ---------------------------------------------------------------------------

def build_global_avgpool_gemm():
    """Exercises the full embedder-style pipeline end.

    Conv2d(3→16) → ReLU → GlobalAveragePool → Flatten → Gemm(16→8)

    GlobalAveragePool and Gemm are present in milo.onnx's final layers.
    If the earlier Conv models pass but this one fails, the Gemm or
    GlobalAveragePool is the broken operator.
    """
    name   = "test_global_avgpool_gemm"
    w_conv = make_conv_weights(16, 3, 3, 3, "gap_w_conv")
    # Gemm: (1,16) × (16,8).T → (1,8)
    w_gemm = RNG.uniform(-0.5, 0.5, (8, 16)).astype(np.float32)
    b_gemm = RNG.uniform(-0.1, 0.1, (8,)).astype(np.float32)
    w_gemm_t = numpy_helper.from_array(w_gemm, "gap_w_gemm")
    b_gemm_t = numpy_helper.from_array(b_gemm, "gap_b_gemm")

    X      = helper.make_tensor_value_info("X",  TensorProto.FLOAT, INPUT_SHAPE)
    Y      = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, None)

    conv   = helper.make_node("Conv", ["X", "gap_w_conv"], ["conv_out"],
                              kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    relu   = helper.make_node("Relu", ["conv_out"], ["relu_out"])
    pool   = helper.make_node("GlobalAveragePool", ["relu_out"], ["pool_out"])
    flat   = helper.make_node("Flatten", ["pool_out"], ["flat_out"], axis=1)
    gemm   = helper.make_node("Gemm", ["flat_out", "gap_w_gemm", "gap_b_gemm"], ["Y"],
                              transB=1)

    graph  = helper.make_graph([conv, relu, pool, flat, gemm], name, [X], [Y],
                               initializer=[w_conv, w_gemm_t, b_gemm_t])
    model  = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model, f"{name}.onnx"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    x = np.full(INPUT_SHAPE, 0.5, dtype=np.float32)

    print(textwrap.dedent("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  CollectorVision — WebGPU Android ARM minimal model builder  ║
    ╚══════════════════════════════════════════════════════════════╝

    Builds 4 minimal ONNX models and prints expected outputs.
    Copy the "first 8 values" lines into browser_test.html EXPECTED_OUTPUTS.

    Input for all models: shape=(1,3,64,64), all values = 0.5
    """).strip())

    models = [
        build_single_conv,
        build_conv_bn_relu,
        build_two_conv,
        build_global_avgpool_gemm,
    ]

    results = {}
    for builder in models:
        model, path = builder()
        y = save_and_check(model, path, x)
        results[path] = y

    print(f"\n{'='*60}")
    print("Copy this block into browser_test.html → EXPECTED_OUTPUTS:")
    print()
    for path, y in results.items():
        flat = y.ravel()[:8].tolist()
        print(f'  "{path}": {flat},')

    return 0


if __name__ == "__main__":
    sys.exit(main())
