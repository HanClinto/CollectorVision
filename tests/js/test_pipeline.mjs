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
 * - Drift between the JS-CPU and Python preprocessing pipelines
 *
 * What this does NOT catch
 * ------------------------
 * - WebGPU execution-provider bugs (e.g. the Android fp16 precision issue in
 *   GitHub issue #1).  Those require device testing or a Playwright+WebGPU
 *   setup.  The js-webgpu manifest (extracted at ingest time from the live
 *   browser capture) is stored for reference in tests/fixtures/captures/.
 *
 * Usage
 * -----
 *   cd tests/js
 *   npm install
 *   npm test
 *
 * This script writes js-cpu manifests and dewarp PNGs to tests/fixtures/captures/
 * so that test_pipeline_consistency.py can compare across all pipelines.
 *
 * Adding captures
 * ---------------
 * Drop .json.gz bundles into tests/fixtures/captures/ and run:
 *
 *   python scripts/ingest_bug_reports.py <issue-number>
 *
 * That generates the python and js-webgpu manifests, then re-run npm test here
 * to generate the js-cpu manifest.
 */

import { readFileSync, readdirSync, existsSync, writeFileSync } from 'fs';
import { resolve, dirname, basename }            from 'path';
import { fileURLToPath }                         from 'url';
import { gunzipSync }                            from 'zlib';
import { createCanvas, loadImage }               from 'canvas';
import * as ort                                  from 'onnxruntime-node';

const __dirname      = dirname(fileURLToPath(import.meta.url));
const ROOT           = resolve(__dirname, '../..');
const CAPTURES_DIR   = resolve(ROOT, 'tests/fixtures/captures');
const CORNELIUS_PATH = resolve(ROOT, 'collector_vision/weights/cornelius.onnx');
const MILO_PATH      = resolve(ROOT, 'collector_vision/weights/milo.onnx');

// Must stay in sync with app.js constants.
const DETECTOR_SIZE    = 384;
const EMBEDDER_SIZE    = 448;
const DEWARP_W         = 252;
const DEWARP_H         = 352;
const IMAGENET_MEAN    = [0.485, 0.456, 0.406];
const IMAGENET_STD     = [0.229, 0.224, 0.225];
const MIN_SHARPNESS    = 0.02;
const CORNER_TOLERANCE = 0.15;    // normalised units
const EMBEDDING_MIN_DOT = 0.90;   // minimum cosine similarity between pipelines
                                   // (PIL bilinear vs browser canvas drawImage give ~0.93)

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
// Dewarp helpers — verbatim port of the functions from app.js.
// Any changes to these functions in app.js MUST be reflected here.
// ---------------------------------------------------------------------------

function solveLinearSystem(matrix, vector) {
  const size = vector.length;
  const a = matrix.map((row, index) => [...row, vector[index]]);

  for (let col = 0; col < size; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < size; row += 1) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) {
        pivot = row;
      }
    }
    if (Math.abs(a[pivot][col]) < 1e-10) {
      throw new Error('Could not solve dewarp transform.');
    }
    if (pivot !== col) {
      [a[col], a[pivot]] = [a[pivot], a[col]];
    }
    const scale = a[col][col];
    for (let k = col; k <= size; k += 1) {
      a[col][k] /= scale;
    }
    for (let row = 0; row < size; row += 1) {
      if (row === col) continue;
      const factor = a[row][col];
      for (let k = col; k <= size; k += 1) {
        a[row][k] -= factor * a[col][k];
      }
    }
  }
  return a.map((row) => row[size]);
}

function computeHomography(srcPoints, dstPoints) {
  const matrix = [];
  const vector = [];
  for (let i = 0; i < 4; i += 1) {
    const [x, y] = srcPoints[i];
    const [u, v] = dstPoints[i];
    matrix.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    vector.push(u);
    matrix.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    vector.push(v);
  }
  const [h11, h12, h13, h21, h22, h23, h31, h32] = solveLinearSystem(matrix, vector);
  return [h11, h12, h13, h21, h22, h23, h31, h32, 1];
}

function applyHomography(matrix, x, y) {
  const denom = matrix[6] * x + matrix[7] * y + matrix[8];
  return [
    (matrix[0] * x + matrix[1] * y + matrix[2]) / denom,
    (matrix[3] * x + matrix[4] * y + matrix[5]) / denom,
  ];
}

function sampleBilinear(data, width, height, x, y, channel) {
  const clampedX = Math.min(Math.max(x, 0), width - 1);
  const clampedY = Math.min(Math.max(y, 0), height - 1);
  const x0 = Math.floor(clampedX);
  const y0 = Math.floor(clampedY);
  const x1 = Math.min(x0 + 1, width - 1);
  const y1 = Math.min(y0 + 1, height - 1);
  const tx = clampedX - x0;
  const ty = clampedY - y0;
  const i00 = (y0 * width + x0) * 4 + channel;
  const i10 = (y0 * width + x1) * 4 + channel;
  const i01 = (y1 * width + x0) * 4 + channel;
  const i11 = (y1 * width + x1) * 4 + channel;
  const top    = data[i00] * (1 - tx) + data[i10] * tx;
  const bottom = data[i01] * (1 - tx) + data[i11] * tx;
  return top * (1 - ty) + bottom * ty;
}

function normalizeEmbedding(embedding) {
  let norm = 0;
  for (let i = 0; i < embedding.length; i += 1) {
    norm += embedding[i] * embedding[i];
  }
  norm = Math.sqrt(norm);
  if (norm > 1e-8) {
    for (let i = 0; i < embedding.length; i += 1) {
      embedding[i] /= norm;
    }
  }
  return embedding;
}

/**
 * Dewarp a full-resolution source canvas using normalised corners.
 *
 * @param {import('canvas').Canvas} srcCanvas  Full-resolution source image.
 * @param {[number, number][]}       corners    Four [x_norm, y_norm] pairs,
 *                                             TL, TR, BR, BL order.
 * @returns {import('canvas').Canvas}  252×352 px dewarped canvas.
 */
function jsDeWarp(srcCanvas, corners) {
  const width  = srcCanvas.width;
  const height = srcCanvas.height;
  const srcPts = corners.map(([x, y]) => [x * width, y * height]);
  const dstPts = [
    [0,          0         ],
    [DEWARP_W-1, 0         ],
    [DEWARP_W-1, DEWARP_H-1],
    [0,          DEWARP_H-1],
  ];
  // Inverse homography: output pixel → source pixel.
  const inverse = computeHomography(dstPts, srcPts);
  const srcData = srcCanvas
    .getContext('2d', { willReadFrequently: true })
    .getImageData(0, 0, width, height);
  const dstCanvas = createCanvas(DEWARP_W, DEWARP_H);
  const dstCtx    = dstCanvas.getContext('2d');
  const dstData   = dstCtx.createImageData(DEWARP_W, DEWARP_H);

  for (let y = 0; y < DEWARP_H; y += 1) {
    for (let x = 0; x < DEWARP_W; x += 1) {
      const [sx, sy] = applyHomography(inverse, x, y);
      const offset   = (y * DEWARP_W + x) * 4;
      dstData.data[offset]     = sampleBilinear(srcData.data, width, height, sx, sy, 0);
      dstData.data[offset + 1] = sampleBilinear(srcData.data, width, height, sx, sy, 1);
      dstData.data[offset + 2] = sampleBilinear(srcData.data, width, height, sx, sy, 2);
      dstData.data[offset + 3] = 255;
    }
  }
  dstCtx.putImageData(dstData, 0, 0);
  return dstCanvas;
}

// ---------------------------------------------------------------------------
// ONNX sessions — loaded once, shared across all tests.
// ---------------------------------------------------------------------------

const detectorSession = await ort.InferenceSession.create(CORNELIUS_PATH, {
  executionProviders: ['cpu'],
});

const HAS_MILO = existsSync(MILO_PATH);
const embedderSession = HAS_MILO
  ? await ort.InferenceSession.create(MILO_PATH, { executionProviders: ['cpu'] })
  : null;

if (!HAS_MILO) {
  console.warn(`  WARN  milo.onnx not found at ${MILO_PATH} — dewarp/embed tests will be skipped`);
}

/**
 * Run the JS embedder on a 252×352 dewarp canvas.
 * @param {import('canvas').Canvas} dewarpCanvas
 * @returns {Promise<Float32Array>} L2-normalised embedding.
 */
async function jsEmbed(dewarpCanvas) {
  // Scale the 252×352 dewarp up to 448×448 for the embedder, then normalise.
  const scaledCanvas = createCanvas(EMBEDDER_SIZE, EMBEDDER_SIZE);
  scaledCanvas.getContext('2d').drawImage(dewarpCanvas, 0, 0, EMBEDDER_SIZE, EMBEDDER_SIZE);
  const { data } = scaledCanvas.getContext('2d').getImageData(0, 0, EMBEDDER_SIZE, EMBEDDER_SIZE);
  const flat    = fillInputTensor(data, EMBEDDER_SIZE);
  const feeds   = {
    [embedderSession.inputNames[0]]: new ort.Tensor(
      'float32', flat, [1, 3, EMBEDDER_SIZE, EMBEDDER_SIZE],
    ),
  };
  const outputs = await embedderSession.run(feeds);
  const emb     = Float32Array.from(outputs[embedderSession.outputNames[0]].data);
  normalizeEmbedding(emb);
  return emb;
}

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

/**
 * Run the full JS pipeline (detect → dewarp → embed) and write the
 * ``js-cpu`` manifest + dewarp PNG to the captures directory.
 *
 * @param {Buffer} pngBuffer    Raw PNG of the capture frame.
 * @param {string} captureId    Bare capture stem, e.g. ``cv_2026-04-24T16-43-41``.
 * @returns {Promise<object>}   The manifest object.
 */
async function generateJsCpuManifest(pngBuffer, captureId) {
  const img        = await loadImage(pngBuffer);
  const fullCanvas = createCanvas(img.width, img.height);
  fullCanvas.getContext('2d').drawImage(img, 0, 0);

  const detection = await runJsDetector(pngBuffer);

  const manifest = {
    pipeline:      'js-cpu',
    source:        'offline',
    captureId,
    cardPresent:   detection.cardPresent,
    sharpness:     detection.sharpness,
    corners:       detection.corners,
    dewarpPng:     null,
    embedding:     null,
    topMatchId:    null,
    topMatchScore: null,
  };

  if (detection.cardPresent && embedderSession) {
    const dewarpCanvas = jsDeWarp(fullCanvas, detection.corners);

    const dewarpPngPath = resolve(CAPTURES_DIR, `${captureId}.js-cpu.dewarp.png`);
    writeFileSync(dewarpPngPath, dewarpCanvas.toBuffer('image/png'));
    manifest.dewarpPng = `${captureId}.js-cpu.dewarp.png`;

    const emb       = await jsEmbed(dewarpCanvas);
    manifest.embedding = Array.from(emb);
  }

  const manifestPath = resolve(CAPTURES_DIR, `${captureId}.js-cpu.json`);
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  return manifest;
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

  // Run the full JS-CPU pipeline and write the manifest.
  // All per-capture tests below read from this manifest.
  const jsManifest = await generateJsCpuManifest(pngBuf, name);

  // -----------------------------------------------------------------------
  // (1) JS detector must find a card
  // -----------------------------------------------------------------------
  await test('JS detector finds card (sharpness > threshold)', () => {
    if (!jsManifest.cardPresent) {
      throw new Error(
        `Card not detected. sharpness=${jsManifest.sharpness?.toFixed(3)}, ` +
        `threshold=${MIN_SHARPNESS}`,
      );
    }
    if ((jsManifest.sharpness ?? 0) <= MIN_SHARPNESS) {
      throw new Error(
        `Sharpness ${jsManifest.sharpness?.toFixed(3)} not above MIN_SHARPNESS=${MIN_SHARPNESS}`,
      );
    }
  });

  // -----------------------------------------------------------------------
  // (2) JS corners must agree with the Python manifest.
  //
  //     Both the Python and JS-CPU pipelines run on the same frame with the
  //     same model via CPU execution.  Significant disagreement indicates a
  //     regression in fillInputTensor, orderCorners, or output parsing.
  // -----------------------------------------------------------------------
  await test('JS corners agree with python manifest', () => {
    const pyManifestPath = resolve(CAPTURES_DIR, `${name}.python.json`);
    if (!existsSync(pyManifestPath)) {
      return 'skip'; // run: python scripts/generate_manifests.py
    }
    const pyManifest = JSON.parse(readFileSync(pyManifestPath, 'utf-8'));
    if (!pyManifest.corners) return 'skip';
    if (!jsManifest.cardPresent) return 'skip';

    const jsPts = [...jsManifest.corners].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const pyPts = [...pyManifest.corners].sort((a, b) => a[0] - b[0] || a[1] - b[1]);

    const deltas   = jsPts.flatMap(([x, y], i) => [
      Math.abs(x - pyPts[i][0]),
      Math.abs(y - pyPts[i][1]),
    ]);
    const maxDelta = Math.max(...deltas);

    if (maxDelta >= CORNER_TOLERANCE) {
      const fmt = (pts) => pts.map((p) => p.map((v) => v.toFixed(4)).join(',')).join('  ');
      throw new Error(
        `JS corners differ from Python manifest by ${maxDelta.toFixed(3)} ` +
        `(tolerance ${CORNER_TOLERANCE}).\n` +
        `  Possible regression in fillInputTensor, orderCorners, or output parsing.\n` +
        `  JS :    ${fmt(jsPts)}\n` +
        `  Python: ${fmt(pyPts)}`,
      );
    }
  });

  // -----------------------------------------------------------------------
  // (3) JS embedding must agree with the Python manifest.
  //
  //     Embedding dot product > EMBEDDING_MIN_DOT confirms that small
  //     differences in bilinear dewarp don't materially affect the
  //     embedding space.  Failure indicates a preprocessing regression.
  // -----------------------------------------------------------------------
  await test('JS embedding agrees with python manifest', async () => {
    const pyManifestPath = resolve(CAPTURES_DIR, `${name}.python.json`);
    if (!existsSync(pyManifestPath)) return 'skip';
    const pyManifest = JSON.parse(readFileSync(pyManifestPath, 'utf-8'));
    if (!pyManifest.embedding || !jsManifest.embedding) return 'skip';

    const dot = jsManifest.embedding.reduce(
      (s, v, i) => s + v * pyManifest.embedding[i],
      0,
    );
    if (dot < EMBEDDING_MIN_DOT) {
      throw new Error(
        `Embedding dot product ${dot.toFixed(4)} < minimum ${EMBEDDING_MIN_DOT}.\n` +
        `  JS and Python preprocessing pipelines have diverged.`,
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

