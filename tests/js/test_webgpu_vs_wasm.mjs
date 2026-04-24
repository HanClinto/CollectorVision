/**
 * tests/js/test_webgpu_vs_wasm.mjs
 *
 * Playwright-based test that runs the Cornelius corner-detector ONNX model
 * under both the "webgpu" and "wasm" execution providers in a real desktop
 * Chrome browser and compares their outputs.
 *
 * Background
 * ----------
 * We observed that Android Chrome produces wrong corner outputs under WebGPU
 * while WASM is correct (verified across 4 captures, pixel diff ≤ 1).
 * This test determines:
 *
 *   A. Desktop Chrome (macOS Metal) WebGPU is ALSO wrong
 *      → ort-web JSEP bug reproducible without Android hardware
 *      → file an issue at microsoft/onnxruntime
 *
 *   B. Desktop Chrome WebGPU matches WASM / Python
 *      → Android-specific: likely Vulkan shader compiler bug on Adreno/Mali
 *      → wrong-output workaround (WASM) is correct; consider filing a Chrome
 *        Android bug with a minimal repro
 *
 * Usage
 * -----
 *   cd tests/js
 *   npm install
 *   npx playwright install chromium    # first time only
 *   node test_webgpu_vs_wasm.mjs       # 3 iterations (default)
 *   node test_webgpu_vs_wasm.mjs 5     # 5 iterations
 *   USE_SYSTEM_CHROME=1 node test_webgpu_vs_wasm.mjs   # use system Chrome
 *
 * The test runs the following inputs on each iteration:
 *   - Each .python-detector-input.npy found in tests/fixtures/captures/
 *     (real 384×384 uint8 RGB frames that produced wrong WebGPU output)
 *   - A synthetic deterministic grey input (sanity check)
 *
 * Exit codes
 *   0 → all WebGPU outputs match WASM within TOLERANCE
 *   1 → at least one mismatch (or error)
 */

import { chromium }                        from 'playwright';
import { readFileSync, existsSync, readdirSync } from 'fs';
import { resolve, dirname }                from 'path';
import { fileURLToPath }                   from 'url';
import { createServer }                    from 'http';

const __dirname    = dirname(fileURLToPath(import.meta.url));
const ROOT         = resolve(__dirname, '../..');
const CAPTURES_DIR = resolve(ROOT, 'tests/fixtures/captures');
const CORNELIUS    = resolve(ROOT, 'collector_vision/weights/cornelius.onnx');
// ORT_DIST_OVERRIDE lets bisect_webgpu_versions.mjs point at a specific
// version's extracted dist/ directory without touching node_modules.
const ORT_DIST     = process.env.ORT_DIST_OVERRIDE
  ?? resolve(__dirname, 'node_modules/onnxruntime-web/dist');

const DETECTOR_SIZE = 384;
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD  = [0.229, 0.224, 0.225];
// Max acceptable corner diff between WebGPU and WASM (both float32, same input).
// Genuine numerical noise between two correct implementations is < 0.001.
// The Android bug produces diffs of 0.3-0.5+.
const TOLERANCE     = 0.02;
const ITERATIONS    = parseInt(process.argv[2] ?? '3');

// ─── input preparation ────────────────────────────────────────────────────

/**
 * Parse a .npy file saved by numpy with shape (384, 384, 3) uint8 (HWC).
 * Returns a Float32Array of shape [3, 384, 384] (CHW) with ImageNet
 * normalisation applied — matching what fillInputTensor produces.
 */
function loadNpyToFloat32Chw(npyPath) {
  const buf   = readFileSync(npyPath);
  const magic = buf.slice(0, 6).toString('binary');
  if (!magic.startsWith('\x93NUMPY')) throw new Error(`Not a .npy file: ${npyPath}`);

  // Header location depends on version byte (index 6).
  // v1.x: headerLen is uint16 at offset 8; v2.x: uint32 at offset 8.
  const major    = buf[6];
  const headerLen = major < 2 ? buf.readUInt16LE(8) : buf.readUInt32LE(8);
  const dataOff  = (major < 2 ? 10 : 12) + headerLen;

  const N  = DETECTOR_SIZE * DETECTOR_SIZE;
  const u8 = new Uint8Array(buf.buffer, buf.byteOffset + dataOff, N * 3);
  const f32 = new Float32Array(3 * N);
  for (let c = 0; c < 3; c++) {
    const base = c * N;
    const mean = IMAGENET_MEAN[c];
    const std  = IMAGENET_STD[c];
    for (let i = 0; i < N; i++) {
      f32[base + i] = (u8[i * 3 + c] / 255 - mean) / std;
    }
  }
  return f32;
}

/**
 * A deterministic pseudo-random input that looks like a "natural" image
 * after normalization.  Used as a self-contained sanity check when no
 * capture fixtures are present.
 */
function makeSyntheticInput() {
  const N = DETECTOR_SIZE * DETECTOR_SIZE;
  const f32 = new Float32Array(3 * N);
  let s = 0xdeadbeef >>> 0;
  for (let i = 0; i < f32.length; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const c   = Math.floor(i / N);
    const pix = (s >>> 24) / 255;  // [0, 1]
    f32[i]    = (pix - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
  }
  return f32;
}

// ─── HTTP server ──────────────────────────────────────────────────────────

function mimeOf(url) {
  if (url.endsWith('.mjs') || url.endsWith('.js')) return 'application/javascript';
  if (url.endsWith('.wasm'))                        return 'application/wasm';
  if (url.endsWith('.html'))                        return 'text/html; charset=utf-8';
  if (url.endsWith('.json'))                        return 'application/json';
  return 'application/octet-stream';
}

/**
 * Start a minimal HTTP server.
 *
 * `staticFiles`: { '/path': () => Buffer }
 * `inputs`:      { id: Float32Array }  — served at /input/<id>.f32
 */
function startServer(staticFiles, inputs) {
  return new Promise(resolve_outer => {
    const srv = createServer((req, res) => {
      const url = (req.url ?? '/').split('?')[0];
      const headers = {
        'Access-Control-Allow-Origin':  '*',
        // COOP + COEP are required for SharedArrayBuffer (multi-threaded WASM).
        // Even with numThreads=1 ort-web may initialise its worker with these.
        'Cross-Origin-Opener-Policy':   'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
        'Cross-Origin-Resource-Policy': 'cross-origin',
        'Content-Type':                 mimeOf(url),
      };

      if (staticFiles[url]) {
        const data = staticFiles[url]();
        res.writeHead(200, headers);
        res.end(data);
        return;
      }

      if (url.startsWith('/input/')) {
        const id  = url.slice('/input/'.length).replace(/\.f32$/, '');
        const inp = inputs[id];
        if (inp) {
          res.writeHead(200, headers);
          res.end(Buffer.from(inp.buffer, inp.byteOffset, inp.byteLength));
        } else {
          res.writeHead(404); res.end(`input not found: ${id}`);
        }
        return;
      }

      res.writeHead(404); res.end(`not found: ${url}`);
    });

    srv.listen(0, '127.0.0.1', () => {
      resolve_outer({ srv, port: srv.address().port });
    });
  });
}

// ─── browser-side test page ───────────────────────────────────────────────

/**
 * Build the HTML test page.
 * The page:
 *   1. Imports ort-web from /ort.all.min.mjs
 *   2. Fetches /inputs.json for the list of input IDs
 *   3. For each input runs WASM then WebGPU inference on the same tensor
 *   4. Stores results in window.__results and sets window.__done = true
 *
 * @param {string[]} inputIds  e.g. ['synthetic', 'cv_2026-04-24T18-59-58']
 * @param {number}   iterations  how many times to repeat each input
 */
function buildPage(inputIds, iterations) {
  return /* html */`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ort-web WebGPU vs WASM — Cornelius</title>
</head>
<body>
<pre id="log" style="font-family:monospace;font-size:12px;white-space:pre-wrap"></pre>
<script type="module">
  const DETECTOR_SIZE = ${DETECTOR_SIZE};
  const ITERATIONS    = ${iterations};
  const INPUT_IDS     = ${JSON.stringify(inputIds)};
  const logEl         = document.getElementById('log');

  function log(s) {
    console.log(s);
    logEl.textContent += s + '\\n';
  }

  async function runSession(session, f32Data) {
    const name   = session.inputNames[0];
    const tensor = new ort.Tensor('float32', f32Data,
                                  [1, 3, DETECTOR_SIZE, DETECTOR_SIZE]);
    const t0  = performance.now();
    const out = await session.run({ [name]: tensor });
    const ms  = performance.now() - t0;
    const corners  = Array.from(out[session.outputNames[0]].data).slice(0, 8);
    const sharpness = session.outputNames[2]
      ? out[session.outputNames[2]].data[0]
      : null;
    return { corners, sharpness, ms };
  }

  try {
    const ort = await import('/ort.all.min.mjs');
    window.ort = ort;   // expose for debugging

    ort.env.wasm.numThreads = 1;
    // Point ort-web at the server root so it finds worker + wasm binaries.
    ort.env.wasm.wasmPaths = location.origin + '/';

    // ── GPU info ──────────────────────────────────────────────────────────
    const gpuAvailable = !!navigator.gpu;
    const gpuInfo = { available: gpuAvailable };
    if (navigator.gpu) {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        try {
          const info = await adapter.requestAdapterInfo();
          Object.assign(gpuInfo, {
            vendor:       info.vendor,
            device:       info.device,
            description:  info.description,
            architecture: info.architecture,
            backend:      info.backend,
          });
        } catch { gpuInfo.infoError = 'requestAdapterInfo() not supported'; }
      } else {
        gpuInfo.adapterError = 'requestAdapter() returned null';
      }
    }
    log('GPU info: ' + JSON.stringify(gpuInfo, null, 2));

    // ── load model ────────────────────────────────────────────────────────
    log('\\nLoading model...');
    const modelBuf = await (await fetch('/model.onnx')).arrayBuffer();
    log('Model: ' + (modelBuf.byteLength / 1024 / 1024).toFixed(1) + ' MB');

    // ── run tests ─────────────────────────────────────────────────────────
    const allResults = { gpuInfo, inputs: [], iterations: ITERATIONS };

    for (const id of INPUT_IDS) {
      log('\\n══ Input: ' + id + ' ══');
      const inputBuf = await (await fetch('/input/' + id + '.f32')).arrayBuffer();
      const f32Data  = new Float32Array(inputBuf);  // [3 × 384 × 384]

      const inputResult = { id, runs: [] };

      for (let iter = 0; iter < ITERATIONS; iter++) {
        log('  ─ iter ' + (iter + 1) + '/' + ITERATIONS);
        const run = {};

        // WASM
        try {
          const sess = await ort.InferenceSession.create(
            modelBuf, { executionProviders: ['wasm'] });
          run.wasm = await runSession(sess, f32Data);
          log('    wasm  : ' + run.wasm.corners.map(v => v.toFixed(4)).join(' ')
              + '  sharpness=' + (run.wasm.sharpness?.toFixed(4) ?? 'n/a')
              + '  ' + run.wasm.ms.toFixed(0) + 'ms');
        } catch (e) {
          run.wasm = { error: e.message };
          log('    wasm  error: ' + e.message);
        }

        // WebGPU
        if (!navigator.gpu || gpuInfo.adapterError) {
          run.webgpu = { error: gpuInfo.adapterError ?? 'navigator.gpu unavailable' };
          log('    webgpu: ' + run.webgpu.error);
        } else {
          try {
            const sess = await ort.InferenceSession.create(
              modelBuf, { executionProviders: ['webgpu'] });
            run.webgpu = await runSession(sess, f32Data);
            log('    webgpu: ' + run.webgpu.corners.map(v => v.toFixed(4)).join(' ')
                + '  sharpness=' + (run.webgpu.sharpness?.toFixed(4) ?? 'n/a')
                + '  ' + run.webgpu.ms.toFixed(0) + 'ms');
          } catch (e) {
            run.webgpu = { error: e.message };
            log('    webgpu error: ' + e.message);
          }
        }

        // Diff
        if (run.wasm?.corners && run.webgpu?.corners) {
          let maxDiff = 0;
          for (let i = 0; i < 8; i++) {
            maxDiff = Math.max(maxDiff,
              Math.abs(run.wasm.corners[i] - run.webgpu.corners[i]));
          }
          run.maxDiff = maxDiff;
          log('    diff  : ' + maxDiff.toFixed(5)
              + (maxDiff > ${TOLERANCE} ? '  ← EXCEEDS TOLERANCE' : '  ✓'));
        }

        inputResult.runs.push(run);
      }

      allResults.inputs.push(inputResult);
    }

    window.__results = allResults;
  } catch (e) {
    console.error(e);
    window.__results = { fatalError: e.message + '\\n' + e.stack };
  }

  window.__done = true;
  document.getElementById('log').textContent += '\\n[DONE]';
</script>
</body>
</html>`;
}

// ─── main ─────────────────────────────────────────────────────────────────

async function main() {
  console.log('='.repeat(60));
  console.log('  ort-web WebGPU vs WASM — Cornelius corner detector');
  console.log(`  ort-web 1.24.3  |  ${ITERATIONS} iterations  |  tolerance ${TOLERANCE}`);
  console.log('='.repeat(60));
  console.log();

  // ── pre-flight checks ──────────────────────────────────────────────────
  if (!existsSync(CORNELIUS)) {
    console.error('ERROR: model not found:', CORNELIUS);
    process.exit(1);
  }

  const ortJs    = resolve(ORT_DIST, 'ort.all.min.mjs');
  const ortWorker = resolve(ORT_DIST, 'ort-wasm-simd-threaded.jsep.mjs');
  const ortWasm  = resolve(ORT_DIST, 'ort-wasm-simd-threaded.jsep.wasm');

  for (const f of [ortJs, ortWorker, ortWasm]) {
    if (!existsSync(f)) {
      console.error('ERROR: ort-web file missing:', f);
      console.error('  Run: cd tests/js && npm install');
      process.exit(1);
    }
  }

  // ── prepare inputs ────────────────────────────────────────────────────
  const inputs = {};

  if (existsSync(CAPTURES_DIR)) {
    for (const fname of readdirSync(CAPTURES_DIR)) {
      if (!fname.endsWith('.python-detector-input.npy')) continue;
      const captureId = fname.replace('.python-detector-input.npy', '');
      const npyPath   = resolve(CAPTURES_DIR, fname);
      try {
        inputs[captureId] = loadNpyToFloat32Chw(npyPath);
        console.log(`Loaded capture input: ${captureId}`);
      } catch (e) {
        console.warn(`  WARNING: could not load ${fname}: ${e.message}`);
      }
    }
  }

  inputs['synthetic'] = makeSyntheticInput();
  console.log('Added synthetic input.');
  console.log();

  const inputIds = Object.keys(inputs);

  // ── HTTP server ───────────────────────────────────────────────────────
  const staticFiles = {
    '/test.html':                      () => Buffer.from(buildPage(inputIds, ITERATIONS)),
    '/model.onnx':                     () => readFileSync(CORNELIUS),
    '/ort.all.min.mjs':                () => readFileSync(ortJs),
    '/ort-wasm-simd-threaded.jsep.mjs':() => readFileSync(ortWorker),
    '/ort-wasm-simd-threaded.jsep.wasm':() => readFileSync(ortWasm),
    '/inputs.json':                    () => Buffer.from(JSON.stringify(inputIds)),
  };

  const { srv, port } = await startServer(staticFiles, inputs);
  const BASE = `http://127.0.0.1:${port}`;
  console.log(`HTTP server: ${BASE}`);

  // ── launch browser ────────────────────────────────────────────────────
  const useSystemChrome = process.env.USE_SYSTEM_CHROME === '1';
  const launchOpts = {
    // headless: false is required for WebGPU access to the real GPU on some
    // platforms.  headless: 'new' (Chrome 112+) also supports WebGPU via
    // software (SwiftShader) on systems without display.
    headless: false,
    args: [
      '--no-sandbox',
      '--disable-gpu-sandbox',
      '--enable-gpu',
      '--enable-webgpu',
      '--enable-webgpu-developer-features',
      '--use-angle=metal',                  // macOS: prefer Metal backend
    ],
  };
  if (useSystemChrome) launchOpts.channel = 'chrome';

  console.log(`Browser: ${useSystemChrome ? 'system Chrome' : 'Playwright Chromium'}`);
  console.log();

  const browser = await chromium.launch(launchOpts);
  const context = await browser.newContext();

  // Capture all browser console output
  const browserLog = [];
  const page = await context.newPage();
  page.on('console', msg =>
    browserLog.push(`[browser ${msg.type()}] ${msg.text()}`));
  page.on('pageerror', err =>
    browserLog.push(`[browser pageerror] ${err.message}`));

  // ── navigate and wait ─────────────────────────────────────────────────
  await page.goto(`${BASE}/test.html`);

  try {
    await page.waitForFunction(() => window.__done === true, { timeout: 120_000 });
  } catch {
    console.error('TIMEOUT waiting for test completion.');
    console.error('Browser console:');
    browserLog.forEach(l => console.error(' ', l));
    await browser.close();
    srv.close();
    process.exit(1);
  }

  const raw = await page.evaluate(() => window.__results);

  await browser.close();
  srv.close();

  // ── print results ─────────────────────────────────────────────────────
  if (raw?.fatalError) {
    console.error('FATAL ERROR from browser:');
    console.error(raw.fatalError);
    process.exit(1);
  }

  console.log('GPU info reported by browser:');
  console.log(JSON.stringify(raw.gpuInfo, null, 2));
  console.log();

  let passed = 0, failed = 0, skipped = 0;

  for (const inp of (raw.inputs ?? [])) {
    console.log(`\n── ${inp.id} ──`);

    for (let i = 0; i < inp.runs.length; i++) {
      const run = inp.runs[i];
      const prefix = `  iter ${i + 1}`;

      if (run.wasm?.error)   console.log(`${prefix}  WASM  error: ${run.wasm.error}`);
      if (run.webgpu?.error) console.log(`${prefix}  WebGPU error: ${run.webgpu.error}`);

      if (run.wasm?.corners) {
        const c = run.wasm.corners.map(v => v.toFixed(3));
        console.log(`${prefix}  wasm  : [${c.join(', ')}]  sharpness=${run.wasm.sharpness?.toFixed(4) ?? 'n/a'}  ${run.wasm.ms?.toFixed(0)}ms`);
      }

      if (run.webgpu?.corners) {
        const c = run.webgpu.corners.map(v => v.toFixed(3));
        console.log(`${prefix}  webgpu: [${c.join(', ')}]  sharpness=${run.webgpu.sharpness?.toFixed(4) ?? 'n/a'}  ${run.webgpu.ms?.toFixed(0)}ms`);
      }

      if (run.maxDiff !== undefined) {
        const ok = run.maxDiff <= TOLERANCE;
        console.log(`${prefix}  diff  : ${run.maxDiff.toFixed(5)} → ${ok ? 'PASS ✓' : 'FAIL ✗'}`);
        if (ok) passed++; else failed++;
      } else if (run.webgpu?.error) {
        console.log(`${prefix}  → SKIP (WebGPU unavailable)`);
        skipped++;
      }
    }
  }

  console.log('\n' + '─'.repeat(60));
  console.log(`Results: ${passed} pass, ${failed} fail, ${skipped} skip`);
  console.log('─'.repeat(60));

  if (failed > 0) {
    console.log(`
INTERPRETATION: desktop Chrome WebGPU output DIFFERS from WASM.
  WebGPU diff exceeds tolerance ${TOLERANCE} (same input, same model).
  This reproduces the Android bug on desktop → likely an ort-web JSEP
  shader bug common to both Metal (macOS) and Vulkan (Android) backends.
  → Consider filing at https://github.com/microsoft/onnxruntime/issues
    with the ort-web version, model op list, and a minimal repro.`);
  } else if (skipped === passed + skipped && passed === 0) {
    console.log(`
INCONCLUSIVE: WebGPU was not available in this browser session.
  Try: USE_SYSTEM_CHROME=1 node test_webgpu_vs_wasm.mjs`);
  } else {
    console.log(`
INTERPRETATION: desktop Chrome WebGPU matches WASM within tolerance.
  The Android bug is specific to that device's GPU driver / WebGPU Vulkan
  backend — not a general ort-web JSEP bug.
  The WASM workaround is correct. Consider filing a Chrome Android bug
  with a minimal WGSL repro if you want a root-cause fix upstream.`);
  }

  process.exit(failed > 0 ? 1 : 0);
}

main().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
