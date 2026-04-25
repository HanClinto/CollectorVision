# WebGPU EP: Wrong Outputs on Android ARM

**Status:** Workaround in place (WASM fallback). Root cause unconfirmed. Investigation ongoing.

This document records everything we know about why `onnxruntime-web`'s WebGPU execution
provider produces numerically wrong outputs on Android ARM GPUs, and lays out a concrete
plan to isolate the cause and contribute a fix upstream.

---

## 1. Executive Summary

`onnxruntime-web` with the new WebGPU EP (`ort.webgpu.min.mjs`, v1.24.3) produces outputs
that are finite, correctly shaped, and pass all sanity checks — but are numerically wrong
on Android ARM devices. Both models used by CollectorVision (`cornelius.onnx` and
`milo.onnx`) exhibit this. The same inputs through the WASM EP on the same device produce
correct results. There is no error, no NaN, no exception. It fails silently.

The workaround is WASM-only inference, which works correctly on all devices tested.

---

## 2. Device Under Test

| Field           | Value                                                              |
|-----------------|--------------------------------------------------------------------|
| Architecture    | armv81 (ARM v8.1-A)                                               |
| CPU cores       | 8                                                                  |
| RAM             | 8 GB                                                               |
| Browser         | Chrome 147 (Chromium, Blink renderer)                             |
| OS              | Android                                                            |
| GPU family      | Unknown — not yet captured from `adapter.info` (see Action Items) |
| ORT version     | 1.24.3 (`ort.webgpu.min.mjs`, new WebGPU EP)                      |
| WASM variant    | `ort-wasm-simd-threaded.asyncify.mjs` (required pairing)          |

---

## 3. Bug Timeline

Three distinct WebGPU bugs have been found, in order of discovery.

### Bug 1 — Legacy JSEP: Conv outputs are all-zeros (RESOLVED)

| Field        | Value |
|--------------|-------|
| EP bundle    | `ort.all.min.mjs` (legacy JSEP backend) |
| Versions     | All tested: 1.20 – 1.24.3 |
| Symptom      | Every Conv operator output is all-zeros on Android Chrome |
| Visibility   | Silent — no error, valid-looking tensor shape, just zeros |
| GPU families | Replicated on Adreno and Mali |
| Fix          | Switch to `ort.webgpu.min.mjs` (new WebGPU EP) |
| Commit       | `aa0f88f` |

### Bug 2 — New EP, `cornelius.onnx`: wrong corners (RESOLVED)

| Field        | Value |
|--------------|-------|
| EP bundle    | `ort.webgpu.min.mjs` 1.24.3 |
| Model        | `cornelius.onnx` (384×384 CNN corner detector) |
| Symptom      | Corner detector produces a convex, plausible-looking quad that passes all validity checks, but points to an entirely wrong region of the frame with no relationship to the actual card position |
| Confirmed by | Issue #9, build `7ed8f8f` capture bundle |
| Data         | JS WebGPU corners: `0.46,0.31 / 0.81,0.32 / 0.82,0.64 / 0.47,0.66` — coherent quad, wrong area; Python WASM corners: different region entirely |
| Fix          | `executionProviders: ["wasm"]` for cornelius |
| Commit       | `c8defb1` |

### Bug 3 — New EP, `milo.onnx`: wrong embeddings (RESOLVED)

| Field        | Value |
|--------------|-------|
| EP bundle    | `ort.webgpu.min.mjs` 1.24.3 |
| Model        | `milo.onnx` (448×448 image embedder, 128-d float32 output) |
| Symptom      | Embedder produces a wrong embedding vector — cosine similarity with the correct card is low, leading to identification of a different card |
| Confirmed by | Issue #12, build `f6f1c76` capture bundle |
| Control      | Sharp frame (Laplacian = 161.8), corners correct (WASM cornelius, JS vs Python max diff = 0.0003) |
| **JS WebGPU milo** | cardId `fe2bd063`, score **0.387** — wrong card |
| **Python WASM milo** | cardId `995529d1`, score **0.810** — correct card (Drey Keeper) |
| Fix          | `executionProviders: ["wasm"]` for milo |
| Commit       | `c859f9d` |

---

## 4. What We Know It Is NOT

These have been systematically ruled out by the capture data:

- **Not a capture-sync bug.** The atomic capture system (commit `8649512`) ensures the
  frame bitmap, corners, and embedding all come from the same pipeline run in one worker
  tick. Issue #12 was captured on this build.

- **Not a blurry-frame false positive.** Issue #12 Laplacian sharpness = 161.8. Previous
  captures with known-bad frames scored ~40. This is a sharp frame.

- **Not a corner detection bug in issue #12.** Corners were produced by WASM cornelius.
  JS vs Python max diff = 0.0003 — effectively identical.

- **Not a preprocessing bug.** `fillInputTensor` (NCHW layout, ImageNet mean/std
  normalization) is tested in `tests/js/test_pipeline.mjs` against onnxruntime-node CPU
  and matches Python within float32 rounding.

- **Not a dewarp bug (unconfirmed but unlikely).** Since both corner sets are the same to
  4 decimal places and the JS dewarp is a straightforward inverse homography, the crop
  going into milo must be effectively the same as Python's.

- **Not a model weight loading bug.** Models are fetched as raw `ArrayBuffer` and passed
  directly to `ort.InferenceSession.create`. ort-web handles the binary parsing.

- **Not a shape or layout bug.** Outputs are the correct shape and dtype. This rules out
  NCHW/NHWC mixups that would throw errors or produce NaN.

- **Not a fluke.** Two separate models, two separate capture bundles, both fail
  independently. The failure mode is consistent: coherent wrong output, no signal.

---

## 5. The Current ORT Configuration

Relevant settings applied before session creation in `scanner.worker.mjs`:

```js
ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 1, 4);
ort.env.webgpu.adapter  = adapter;    // pre-configured high-performance adapter
ort.env.webgpu.forceFp16 = false;     // do NOT request fp16 session compute
```

The adapter is patched to request `maxStorageBuffersPerShaderStage: 10` (capped to
the device's reported maximum). This is required for ORT's WebGPU shaders to have enough
binding slots.

The `ort.webgpu.min.mjs` bundle uses the *new* WebGPU EP (not the legacy JSEP). It
pairs with `ort-wasm-simd-threaded.asyncify.*` for WASM fallback paths.

---

## 6. Root Cause Hypotheses

### H1 — FP16 mediump precision in mobile GPU WGSL shaders ⭐ Most likely

WebGPU's WGSL spec requires `f32` to be 32-bit, but Android GPU drivers (especially
Adreno and Mali) are known to apply aggressive precision lowering in their shader
compilers. A shader declared as `f32` may execute in 16-bit precision if the driver
decides the precision requirements can be relaxed.

For a Conv2d with 128 channels and a 3×3 kernel, each output activation is the sum of
`128 × 3 × 3 = 1152` multiply-accumulate operations. FP16 accumulation error on 1152
terms can be substantial (~0.01–0.1 per activation depending on weight magnitudes),
and this would compound layer over layer.

`ort.env.webgpu.forceFp16 = false` prevents ORT from *requesting* FP16 session
compute, but does not prevent the GPU driver from using mediump internally.

**Test:** Set `ort.env.webgpu.preferredLayout = "NHWC"` (avoids NCHW→NHWC transpose
shaders) and also try `ort.env.webgpu.forceFp16 = true` to see if the output *changes*
systematically. If it does, the precision path is confirmed as the variable.

### H2 — ORT WGSL shader uses subnormal or denormal floats incorrectly

Some mobile GPUs flush subnormal floats to zero (FTZ mode). If any intermediate
activation passes through a subnormal range (very small absolute value after a ReLU,
for example), FTZ mode would silently zero those values, causing downstream layers to
drift.

**Test:** Inspect whether the models have activations that frequently hit the subnormal
range during WASM inference. If a BatchNorm output has very small values (near 1e-38),
those would survive on CPU but be zeroed on an FTZ GPU.

### H3 — Shader dispatch dimension overflow or workgroup size mismatch

If a layer has a spatial dimension or channel count that causes a workgroup dispatch to
overflow (e.g., `ceil(128/8) = 16` threads when the shader assumes exactly 16), the
last workgroup may read/write out-of-bounds in some ORT shader implementations. This
would produce a wrong but coherent output for the affected layer.

**Test:** Check the model's largest tensor dimension against the device's
`maxComputeWorkgroupsPerDimension` and `maxComputeWorkgroupSizeX`.

### H4 — ORT new EP shader regression for a specific operator version

The new WebGPU EP was introduced post-1.20 and the bugs are present in 1.24.3. ORT's
WebGPU shaders are auto-generated and have changed significantly between versions. A
regression in Conv2d, BatchNorm, or GlobalAveragePooling for mobile feature levels
(specifically the subset of WGSL that mobile GPUs support vs desktop) would explain both
bugs.

**Test:** Version bisect using `tests/js/bisect_webgpu_versions.mjs` with a browser
on device (via remote debugging + port forwarding) to find the first bad version.

---

## 7. Investigation Plan

Ordered by expected effort and information value.

### Step 1 — Capture GPU adapter info (30 min, immediate)

Modify `configureWebGpu()` in `scanner.worker.mjs` to read `adapter.info` (Chrome 113+
supports `GPUAdapterInfo`) and include it in the capture bundle under `adapterInfo`.

```js
const info = await adapter.info?.();
// info.vendor, info.architecture, info.device, info.description
```

This tells us whether the bug is Adreno-specific (Qualcomm), Mali-specific (ARM),
or both. This matters for the upstream bug report and may explain the failure at a
hardware level.

### Step 2 — Minimal Conv2d isolation test (2–4 hours)

Use `experiments/webgpu-android-arm/create_minimal_model.py` to build a tiny 1-layer
ONNX model with known weights and a known input. Load it in
`experiments/webgpu-android-arm/browser_test.html` on the Android device via USB
remote debugging. Compare WASM vs WebGPU outputs to the numpy ground truth.

This answers: **"Is Conv2d itself wrong, or is the bug in a different operator?"**

If Conv2d is wrong → H1 or H2. If Conv2d is correct → go deeper (BatchNorm? Gemm?
GlobalAveragePool?).

### Step 3 — Layer-by-layer output dump (4–8 hours)

Use `onnx` Python tools to add intermediate output nodes to `cornelius.onnx` and/or
`milo.onnx`, re-run the modified model in the browser, and compare each layer's output
to the Python/WASM baseline.

```python
# Quick approach:
import onnx
from onnx import shape_inference
model = onnx.load("milo.onnx")
model = shape_inference.infer_shapes(model)
for node in model.graph.node:
    for output in node.output:
        if output not in [o.name for o in model.graph.output]:
            model.graph.output.append(onnx.helper.make_tensor_value_info(
                output, onnx.TensorProto.FLOAT, None
            ))
onnx.save(model, "milo_with_intermediates.onnx")
```

This pinpoints the **exact layer** where WebGPU and WASM first diverge.

### Step 4 — Version bisect on device (2–3 hours)

`tests/js/bisect_webgpu_versions.mjs` already tests multiple ORT versions using
onnxruntime-node. Extend it (or create a browser-based equivalent) to run on the
Android device with the WebGPU EP. The first version where outputs diverge from WASM
will have a corresponding ORT changelog entry identifying the shader change.

If the bug was present in the very first version of `ort.webgpu.min.mjs`, the
regression is architectural (e.g., missing precision qualifiers from day one) rather
than a point regression.

### Step 5 — Precision flag experiments (1 hour, can run quickly)

Try the following ORT env flags in isolation (via the Settings WebGPU toggle) and
capture bundles for each:

```js
// Experiment A — prefer NHWC (avoids transpose shaders on mobile)
ort.env.webgpu.preferredLayout = "NHWC";

// Experiment B — force fp16 (if output changes, precision is the variable)
ort.env.webgpu.forceFp16 = true;

// Experiment C — disable multi-threading in webgpu dispatch
ort.env.webgpu.numThreads = 1;  // if supported
```

Each experiment produces a capture bundle with `inferenceMode: "WebGPU"`. If any of
these make the score jump toward the WASM score, we've identified the flag that works
around the shader issue.

---

## 8. Upstream Contribution Strategy

Once Step 2 or Step 3 identifies the broken operator, we have a minimal reproducer.

### Primary target: `microsoft/onnxruntime`

**URL:** https://github.com/microsoft/onnxruntime/issues  
**Labels to add:** `ep:webgpu`, `platform:android`, `area:web`  
**Search first:** Look for existing issues mentioning "Android ARM WebGPU wrong output"
or "WASM fallback different results mobile". This is likely not unique to us.

**What to include in the issue:**

1. Minimal ONNX model (from `create_minimal_model.py`) as a file attachment
2. Expected output (numpy ground truth, printed by the script)
3. Actual WASM output (should match expected)
4. Actual WebGPU output (the wrong values)
5. ORT version, browser version, GPU vendor/architecture from `adapter.info`
6. The ORT env configuration used
7. Whether `forceFp16 = true` changes the output

**Template:**

```
Title: WebGPU EP: wrong Conv2d output on Android ARM (Chrome 147, ort-web 1.24.3)

Environment:
- ort-web: 1.24.3 (ort.webgpu.min.mjs)
- Browser: Chrome 147
- Platform: Android ARM (armv81)
- GPU: [from adapter.info]

Steps to reproduce:
1. Load attached minimal_conv_test.onnx
2. Create InferenceSession with executionProviders: ["webgpu"]
3. Run with input: Float32Array of [1, 3, 384, 384] filled with 0.5
4. Compare output to WASM session with same input

Expected (WASM / numpy): [first few values]
Actual (WebGPU): [first few values]

ORT env:
  ort.env.webgpu.forceFp16 = false
  ort.env.webgpu.adapter = <high-performance adapter>
  maxStorageBuffersPerShaderStage = 10
```

### Secondary: Chromium issue tracker

If the minimal model test shows that ORT's generated WGSL is correct (i.e., the shader
source looks right) but Chrome on Android executes it wrong, the bug is in Chrome's
WebGPU implementation. File at https://crbug.com with tag `Internals>GPU>WebGPU`.

Include the WGSL shader (ORT can dump it with `ort.env.webgpu.enableGraphCapture` or
by intercepting the shader module creation).

### Tertiary: Dawn / gpuweb

If the bug is a spec ambiguity (e.g., whether `f32` storage must be full-precision on
mobile), it may warrant a discussion in https://github.com/gpuweb/gpuweb/issues.

---

## 9. Action Items (Ordered)

- [ ] **Capture `adapter.info`** in the WebGPU path of `configureWebGpu()` and add
  `adapterInfo` to the capture bundle JSON. (app.js + scanner.worker.mjs, ~30 min)

- [ ] **Run `create_minimal_model.py`** and note the numpy expected outputs for the
  test inputs. Check in `test_single_conv.onnx` alongside the script.

- [ ] **Test `browser_test.html`** on the Android device via Chrome remote debugging.
  Expected: WASM PASS, WebGPU FAIL for at least one of the test models.

- [ ] **Run Step 2 results → file ORT issue** with the minimal repro if Conv2d is
  the broken op. If not, proceed to Step 3 (layer dump).

- [ ] **Try `preferredLayout = "NHWC"`** as a quick experiment — add it to
  `configureWebGpu()` behind the `enableWebGpu` flag and capture a bundle on device.
  If it fixes scores, it's a cheap fix we can ship immediately while the upstream
  issue is investigated.

---

## 10. Files in This Directory

| File | Purpose |
|------|---------|
| `README.md` | This document |
| `create_minimal_model.py` | Builds small ONNX test models with known weights and prints expected outputs |
| `browser_test.html` | Self-contained test harness — load on device via USB remote debugging to compare WASM vs WebGPU outputs |

---

## 11. Reference

- ORT WebGPU EP source: `js/web/lib/wasm/jsep/webgpu/` in the onnxruntime repo
- ORT WGSL shader generator: `ops/` subdirectory, particularly `conv.ts`, `batch-norm.ts`
- WebGPU precision spec: https://www.w3.org/TR/WGSL/#floating-point-evaluation
- Chromium WebGPU conformance tests: `third_party/dawn/webgpu-cts/`
- Dawn issue tracker (Chromium's WebGPU impl): https://bugs.chromium.org/p/dawn
