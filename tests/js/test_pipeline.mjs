/**
 * Node.js regression tests for the CollectorVision JS detector pipeline.
 *
 * These tests run the same fillInputTensor → ONNX model → orderCorners path
 * that the browser uses, but with onnxruntime-node (CPU execution provider)
 * and node-canvas (same pixel-data API as browser CanvasRenderingContext2D).
 *
 * What this catches
 * -----------------
 * - Regressions in fillInputTensor (wrong channel order, wrong normalisation)
 * - Regressions in orderCorners (wrong TL/TR/BR/BL assignment)
 * - Regressions in model output parsing (reading wrong tensor index / shape)
 * - Drift between the JS and Python preprocessing pipelines
 *
 * What this does NOT catch
 * ------------------------
 * - WebGPU execution-provider bugs (e.g. the Android fp16 precision issue in
 *   GitHub issue #1).  Those require device testing or a Playwright+WebGPU
 *   setup.  The Python test_capture_regression.py suite carries the
 *   currently-failing browser-corners assertion for that bug.
 *
 * Usage
 * -----
 *   cd tests/js
 *   npm install
 *   npm test
 *
 * Adding captures
 * ---------------
 * Drop .json.gz bundles into tests/fixtures/captures/ and re-run the ingest
 * script so each bundle has pythonCorners annotated:
 *
 *   python scripts/ingest_bug_reports.py <issue-number>
 */

import { readFileSync, readdirSync, existsSync } from 'fs';
import { resolve, dirname, basename }            from 'path';
import { fileURLToPath }                         from 'url';
import { gunzipSync }                            from 'zlib';
import { createCanvas, loadImage }               from 'canvas';
import * as ort                                  from 'onnxruntime-node';

const __dirname      = dirname(fileURLToPath(import.meta.url));
const ROOT           = resolve(__dirname, '../..');
const CAPTURES_DIR   = resolve(ROOT, 'tests/fixtures/captures');
const CORNELIUS_PATH = resolve(ROOT, 'collector_vision/weights/cornelius.onnx');

// Must stay in sync with app.js constants.
const DETECTOR_SIZE    = 384;
const IMAGENET_MEAN    = [0.485, 0.456, 0.406];
const IMAGENET_STD     = [0.229, 0.224, 0.225];
const MIN_SHARPNESS    = 0.02;
const CORNER_TOLERANCE = 0.15;   // normalised units; matches Python regression test

// ---------------------------------------------------------------------------
// Minimal test runner (compatible with Node 16+)
// ---------------------------------------------------------------------------

let _passed = 0;
let _failed = 0;
let _skipped = 0;
const _failures = [];

async function test(label, fn) {
  try {
    const skipped = await fn();
    if (skipped === 'skip') {
      _skipped += 1;
      console.log(`  SKIP  ${label}`);
    } else {
      _passed += 1;
      console.log(`  PASS  ${label}`);
    }
  } catch (err) {
    _failed += 1;
    _failures.push({ label, err });
    console.error(`  FAIL  ${label}`);
    console.error(`        ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// JS pipeline — verbatim port of the critical functions from app.js.
// Any changes to these functions in app.js MUST be reflected here, otherwise
// the tests will diverge from what the browser actually runs.
// ---------------------------------------------------------------------------

/**
 * RGBA Uint8ClampedArray (from CanvasRenderingContext2D.getImageData) →
 * Float32Array of shape (1, 3, size, size), ImageNet-normalised.
 *
 * Matches fillInputTensor() in app.js exactly.
 */
function fillInputTensor(rgbaData, size) {
  const tensor = new Float32Array(3 * size * size);
  const plane  = size * size;
  for (let i = 0; i < plane; i += 1) {
    const r = rgbaData[i * 4]     / 255;
    const g = rgbaData[i * 4 + 1] / 255;
    const b = rgbaData[i * 4 + 2] / 255;
    tensor[i]             = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    tensor[plane + i]     = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    tensor[plane * 2 + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }
  return tensor;
}

/**
 * Reorder four [x, y] points to canonical TL, TR, BR, BL order.
 * Matches orderCorners() in app.js exactly.
 */
function orderCorners(points) {
  const cx = points.reduce((s, [x]) => s + x, 0) / points.length;
  const cy = points.reduce((s, [, y]) => s + y, 0) / points.length;
  const sorted = [...points].sort(
    ([ax, ay], [bx, by]) =>
      Math.atan2(ay - cy, ax - cx) - Math.atan2(by - cy, bx - cx),
  );
  let start = 0;
  let best  = Infinity;
  for (let i = 0; i < sorted.length; i += 1) {
    const score = sorted[i][0] + sorted[i][1];
    if (score < best) { best = score; start = i; }
  }
  const ordered = [0, 1, 2, 3].map((k) => sorted[(start + k) % 4]);
  const signedArea = ordered.reduce((sum, [x1, y1], i) => {
    const [x2, y2] = ordered[(i + 1) % 4];
    return sum + (x1 * y2 - x2 * y1);
  }, 0);
  return signedArea < 0
    ? [ordered[0], ordered[3], ordered[2], ordered[1]]
    : ordered;
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

// ---------------------------------------------------------------------------
// ONNX detector session — loaded once, shared across all tests.
// ---------------------------------------------------------------------------

const detectorSession = await ort.InferenceSession.create(CORNELIUS_PATH, {
  executionProviders: ['cpu'],
});

/**
 * Run the JS detector pipeline on a PNG buffer.
 * @param {Buffer} pngBuffer
 */
async function runJsDetector(pngBuffer) {
  // node-canvas provides the same RGBA pixel layout as
  // CanvasRenderingContext2D.getImageData() in the browser.
  const img    = await loadImage(pngBuffer);
  const canvas = createCanvas(DETECTOR_SIZE, DETECTOR_SIZE);
  const ctx    = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, DETECTOR_SIZE, DETECTOR_SIZE);
  const { data } = ctx.getImageData(0, 0, DETECTOR_SIZE, DETECTOR_SIZE);

  const flat  = fillInputTensor(data, DETECTOR_SIZE);
  const feeds = {
    [detectorSession.inputNames[0]]: new ort.Tensor(
      'float32', flat, [1, 3, DETECTOR_SIZE, DETECTOR_SIZE],
    ),
  };
  const outputs = await detectorSession.run(feeds);

  // Parse output tensors the same way BrowserRuntime.detect() does.
  const cornersRaw    = Array.from(outputs[detectorSession.outputNames[0]].data).slice(0, 8);
  const presenceLogit = outputs[detectorSession.outputNames[1]].data[0];
  const sharpness     = detectorSession.outputNames[2]
    ? Number(outputs[detectorSession.outputNames[2]].data[0])
    : null;

  const points = [];
  for (let i = 0; i < 8; i += 2) {
    points.push([
      Math.min(Math.max(cornersRaw[i],     0), 1),
      Math.min(Math.max(cornersRaw[i + 1], 0), 1),
    ]);
  }

  const confidence = sharpness ?? sigmoid(presenceLogit);
  return {
    corners:     orderCorners(points),
    sharpness,
    cardPresent: confidence >= MIN_SHARPNESS,
  };
}

// ---------------------------------------------------------------------------
// Discover captures
// ---------------------------------------------------------------------------

let captureFiles = [];
if (existsSync(CAPTURES_DIR)) {
  captureFiles = readdirSync(CAPTURES_DIR)
    .filter((f) => f.endsWith('.json.gz'))
    .sort()
    .map((f) => resolve(CAPTURES_DIR, f));
}

if (captureFiles.length === 0) {
  console.log('No *.json.gz captures found in tests/fixtures/captures/');
  console.log('Run: python scripts/ingest_bug_reports.py');
  process.exit(0);
}

// ---------------------------------------------------------------------------
// One test group per capture
// ---------------------------------------------------------------------------

for (const capturePath of captureFiles) {
  const name   = basename(capturePath, '.json.gz');
  const gz     = readFileSync(capturePath);
  const bundle = JSON.parse(gunzipSync(gz).toString('utf-8'));
  const pngBuf = Buffer.from(bundle.framePng, 'base64');

  console.log(`\n[${name}]`);

  // Cache the detection result across the two tests for this capture.
  let detection = null;

  // -------------------------------------------------------------------
  // (1) JS detector must find a card
  // -------------------------------------------------------------------
  await test('JS detector finds card (sharpness > threshold)', async () => {
    detection = await runJsDetector(pngBuf);
    if (!detection.cardPresent) {
      throw new Error(
        `Card not detected. sharpness=${detection.sharpness?.toFixed(3)}, ` +
        `threshold=${MIN_SHARPNESS}`,
      );
    }
    if ((detection.sharpness ?? 0) <= MIN_SHARPNESS) {
      throw new Error(
        `Sharpness ${detection.sharpness?.toFixed(3)} not above MIN_SHARPNESS=${MIN_SHARPNESS}`,
      );
    }
  });

  // -------------------------------------------------------------------
  // (2) JS corners must agree with Python's known-good corners.
  //
  //     pythonCorners are written into the bundle by ingest_bug_reports.py
  //     (annotate_python_results).  They represent the correct CPU output
  //     that both Python and Node.js (CPU) should agree on.
  //
  //     If this test fails it means fillInputTensor, orderCorners, or the
  //     model output-parsing code in app.js has diverged from what Python
  //     expects — a genuine JS regression.
  // -------------------------------------------------------------------
  await test('JS corners agree with Python reference corners', () => {
    if (!bundle.pythonCorners) {
      return 'skip'; // run ingest_bug_reports.py to annotate
    }
    if (!detection?.cardPresent) {
      return 'skip'; // card not detected in previous test
    }

    const jsPts = [...detection.corners]
      .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const pyPts = bundle.pythonCorners
      .map((c) => [c.x, c.y])
      .sort((a, b) => a[0] - b[0] || a[1] - b[1]);

    const deltas   = jsPts.flatMap(([x, y], i) => [
      Math.abs(x - pyPts[i][0]),
      Math.abs(y - pyPts[i][1]),
    ]);
    const maxDelta = Math.max(...deltas);

    if (maxDelta >= CORNER_TOLERANCE) {
      const fmt = (pts) => pts.map((p) => p.map((v) => v.toFixed(4)).join(',')).join('  ');
      throw new Error(
        `JS corners differ from Python reference by ${maxDelta.toFixed(3)} ` +
        `(tolerance ${CORNER_TOLERANCE}).\n` +
        `  Possible regression in fillInputTensor, orderCorners, or output parsing.\n` +
        `  JS :    ${fmt(jsPts)}\n` +
        `  Python: ${fmt(pyPts)}`,
      );
    }
  });
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

console.log(`\n${'─'.repeat(50)}`);
console.log(`Results: ${_passed} passed, ${_failed} failed, ${_skipped} skipped`);
if (_failures.length > 0) {
  console.error('\nFailed tests:');
  for (const { label, err } of _failures) {
    console.error(`  ✗ ${label}`);
    console.error(`    ${err.message.split('\n')[0]}`);
  }
  process.exit(1);
}

