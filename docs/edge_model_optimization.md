# Edge model optimization plan

Goal: reduce Cornelius + Milo latency on browser WASM and Raspberry Pi class CPU devices while preserving detector geometry and embedding retrieval quality.

Current bundled models:

| Model | Runtime role | Input | Size | Parameters | Current web latency target |
| --- | --- | ---: | ---: | ---: | ---: |
| Cornelius | card corner detector | 384×384 | 8.23 MiB | ~1.82M | ~300 ms now |
| Milo | card embedder | 448×448 | 4.95 MiB | ~0.99M | ~400 ms now |

## What is likely to help

1. **Keep input resolution reduction as a last resort.**
   - Compute scales roughly with input area for the convolutional stem/backbone.
   - Cornelius 384→320 cuts input pixels by ~31%; 384→288 cuts ~44%.
   - Milo 448→384 cuts ~27%; 448→320 cuts ~49%.
   - This is likely to improve latency, but it risks losing fine set/edition details that Milo needs. Do not lead with this unless precision/model-shape work cannot hit the target.

2. **Static INT8 quantization is the best precision-reduction candidate for Raspberry Pi.**
   - Dynamic quantization mostly affects MatMul/Gemm weights and is less useful when Conv dominates runtime.
   - Static Conv + MatMul quantization can reduce bandwidth and may use ARM int8 kernels on Pi builds of ONNX Runtime.
   - Calibration needs real detector inputs and real dewarped card crops. Validate accuracy before shipping.

3. **Float16 is mainly a WebGPU/accelerator strategy, not a safe WASM/Pi CPU strategy.**
   - WASM and many Pi CPU paths do not accelerate fp16 arithmetic enough to justify it.
   - WebGPU has known correctness issues for these models on Android ARM in the current scanner history, so fp16 WebGPU should be treated as experimental only.

4. **Structured model slimming beats unstructured pruning.**
   - Unstructured sparsity rarely speeds up ONNX Runtime WASM/CPU unless the runtime has sparse kernels for the exact ops.
   - Prefer retraining distilled/mobile variants with smaller channel widths, fewer MobileViT blocks, or smaller heads.
   - For Cornelius specifically, the eight SimCC coordinate heads are a significant parameter-size contributor. They may not dominate latency, but shrinking the projection/head width can reduce file size and memory pressure.

5. **Pipeline gating can reduce average latency even when per-model latency is unchanged.**
   - Run Cornelius every frame, but run Milo only when corners are stable and sharpness is above threshold.
   - Reuse the last embedding/result when the card quad has not moved enough.
   - This does not improve worst-case single-scan latency, but it improves interactive throughput on weak devices.

## INT8 approaches

Calibrated INT8 and INT8-aware training are related but not the same:

- **Post-training static quantization (PTQ)** keeps the trained FP32 model fixed, runs calibration images through it, and inserts quantize/dequantize nodes plus INT8 weights/activations. This is fastest to try and is the first experiment to run on Raspberry Pi.
- **Quantization-aware training (QAT)** fine-tunes the model with fake-quantization modules in the training graph. QAT usually preserves accuracy better than PTQ when activations are sensitive, but requires the training pipeline and representative data.

Recommended order:

1. PTQ with calibration data for both networks.
2. If PTQ loses too much corner or retrieval accuracy, run QAT starting from the FP32 checkpoint.
3. Only after INT8/QAT and pipeline gating are exhausted, test reduced input sizes.

QAT architecture for the training machine:

1. Use PyTorch eager or FX graph-mode QAT with an ARM-friendly `qnnpack`/`xnnpack` style qconfig where possible.
2. Keep model inputs and preprocessing identical to current ONNX export: RGB, ImageNet mean/std, Cornelius 384×384, Milo 448×448.
3. Fuse Conv/BatchNorm/activation blocks before QAT when the training model representation supports it.
4. Prepare separate representative loaders:
   - Cornelius: raw camera/card images with target corners and blank/no-card negatives.
   - Milo: dewarped card crops, balanced across editions/sets that are visually close.
5. Fine-tune with a small learning rate for a short schedule, watching the original validation metrics plus quantization-specific drift:
   - Cornelius: corner MAE/p95, `card_present` false positives/negatives, sharpness distribution.
   - Milo: top-1/top-k retrieval, cosine similarity to FP32 embeddings, confusion among near-identical editions.
6. Export quantized ONNX candidates with stable names such as `cornelius.int8.qat.onnx` and `milo.int8.qat.onnx`.
7. Validate each candidate with this repo’s benchmark and integration tests before replacing web assets.

## Benchmark structure

Use three levels of benchmark, always recording model hash, runtime version, device, backend, warmup, and p50/p90 latency.

1. **Raw ONNX inference**
   - Python/Raspberry Pi: `scripts/benchmark_onnx_models.py`
   - Browser: `examples/web_scanner/model_benchmark.html`
   - Purpose: isolate model/runtime cost from camera, dewarp, catalog search, and UI.
   - The Python report includes the source git commit, branch, commit date, dirty working-tree flag, and porcelain status entries so reports can be tied to the exact benchmark code used.
   - Each model result includes ONNX metadata version/codename/task/architecture plus the exact model file SHA-256, so multiple Cornelius/Milo releases can be compared safely.
   - The Python report includes OS, CPU model, physical/logical core counts, total memory, available memory at benchmark start, Python, ONNX Runtime, and provider information.
   - The Python benchmark records `first_run_ms` separately, then performs per-model warmup runs before measuring steady-state mean/median/P90 latency.
   - Provider sweeps are supported and local-only by default: the default is `--provider all-local`, which tests local providers such as CPU/CoreML. Use `--provider CPUExecutionProvider` for CPU only, or `--provider CPUExecutionProvider,CoreMLExecutionProvider` for an explicit list. Remote-capable providers such as `AzureExecutionProvider` are skipped unless explicitly requested with `--provider all-with-remote`, `--provider all-available`, or by naming the provider directly. Non-CPU providers use CPU fallback for unsupported nodes, and the JSON records both the requested provider and the actual session provider chain.
   - Per-layer/node profiling is available with `--profile`. The report summarizes top operator classes and top individual ONNX Runtime nodes. This is intentionally a hotspot summary rather than a full per-node table because full traces can be large and noisy across warmup/measured runs.
   - Optional result contribution: by default, benchmark comments target https://github.com/HanClinto/CollectorVision/issues/16. Override with `--github-thread-url <url>` or `COLLECTORVISION_BENCHMARK_THREAD_URL=<url>`. The script can post with `gh` when available, or open the browser and copy/write a Markdown comment body.

2. **Pipeline component benchmark**
   - Measure detector preprocessing, Cornelius, dewarp, Milo preprocessing, Milo, and cosine search separately on the same sample frame.
   - Purpose: avoid optimizing the model if JS canvas/dewarp/catalog search is the actual bottleneck on a target device.

3. **Accuracy/performance sweep**
   - For each candidate model, run:
     - detector corner error and `card_present`/sharpness behavior on test captures
     - embedding top-1/top-k retrieval accuracy on held-out cards
     - raw latency on browser WASM and Raspberry Pi
   - Only promote candidates that improve latency without unacceptable detector/retrieval regressions.

## Suggested candidate matrix

| Candidate | Cornelius | Milo | Expected speedup | Risk |
| --- | --- | --- | --- | --- |
| A | 384 static-int8 PTQ | 448 static-int8 PTQ | medium on Pi, unknown web | calibration and runtime support risk |
| B | 384 static-int8 QAT | 448 static-int8 QAT | medium/high on Pi | requires training pipeline, best accuracy-preserving INT8 path |
| C | 384 fp32 slim/distilled | 448 fp32 slim/distilled | medium/high | requires retraining but preserves input detail |
| D | 384 int8 QAT slim | 448 int8 QAT slim | high | highest engineering burden, best edge candidate if accuracy holds |
| E | 320 fp32 or int8 | 384 fp32 or int8 | high | last resort; risks fine edition details |

## Immediate next steps

1. Run raw baselines on target devices:
   - Raspberry Pi: `python scripts/benchmark_onnx_models.py --profile --threads 1,2,4`
   - Browser: open `examples/web_scanner/model_benchmark.html` from the local scanner server, or use the Settings → Model Benchmark shortcut.
   - To collect community CPU results in one place: `python scripts/benchmark_onnx_models.py --profile --threads 1,2,4` and accept the contribution prompt for issue #16.
2. Use the scanner worker pipeline timings to compare preprocessing, ONNX runtime, dewarp, and lookup costs on actual devices.
3. Build a calibration dataset from existing sample/capture frames: detector inputs and dewarped Milo crops.
4. Try static INT8 quantization with calibration, then benchmark on Pi and browser WASM.
5. If PTQ is not accurate enough, run QAT on the training machine.
6. If INT8/QAT is not enough, try reduced-width/distilled variants at the same input resolution.
7. Try reduced-resolution variants only as the final fallback.

## First local CPU observation

On the current development machine, ONNX Runtime CPU profiling shows both models dominated by convolution, transpose/layout movement, attention MatMul/Gemm, Softmax, QuickGelu, and LayerNormalization. Milo is slower primarily because its 448×448 input has ~36% more pixels than Cornelius. This explains why resolution reduction would work, but the preferred path is still INT8/QAT, same-resolution distillation/slimming, and pipeline gating before reducing input detail.
