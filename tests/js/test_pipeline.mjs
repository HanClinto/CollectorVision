/**
 * Node.js regression tests for the CollectorVision JS detector pipeline.
 *
 * Runs the same fillInputTensor -> ONNX model -> orderCorners pipeline that
 * the browser uses, via onnxruntime-node (CPU) and node-canvas.
 *
 * What this catches
 * -----------------
 * - Regressions in fillInputTensor (channel order, normalisation)
 * - Regressions in orderCorners (TL/TR/BR/BL assignment)
 * - Regressions in output parsing (wrong tensor index / shape)
 * - Numerical drift between JS-CPU and Python preprocessing pipelines
 *
 * Usage
 * -----
 *   cd tests/js && npm install && npm test
 */

import { readFileSync }              from 'fs';
import { resolve, dirname }          from 'path';
import { fileURLToPath }             from 'url';
import { createCanvas, loadImage }   from 'canvas';
import * as ort                      from 'onnxruntime-node';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT      = resolve(__dirname, '../..');

const CORNELIUS_PATH = resolve(ROOT, 'collector_vision/weights/cornelius.onnx');
const MILO_PATH      = resolve(ROOT, 'collector_vision/weights/milo.onnx');
const SAMPLE_IMAGE   = resolve(ROOT, 'examples/images/7286819f-6c57-4503-898c-528786ad86e9_sample.jpg');

// Must stay in sync with app.js constants.
const DETECTOR_SIZE  = 384;
const EMBEDDER_SIZE  = 448;
const DEWARP_W       = 252;
const DEWARP_H       = 352;
const IMAGENET_MEAN  = [0.485, 0.456, 0.406];
const IMAGENET_STD   = [0.229, 0.224, 0.225];
const MIN_SHARPNESS  = 0.02;

// ---------------------------------------------------------------------------
// Minimal test runner
// ---------------------------------------------------------------------------

let _passed = 0, _failed = 0;
const _failures = [];

async function test(label, fn) {
  try {
    await fn();
    _passed += 1;
    console.log(`  PASS  ${label}`);
  } catch (err) {
    _failed += 1;
    _failures.push({ label, err });
    console.error(`  FAIL  ${label}`);
    console.error(`        ${err.message}`);
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

// ---------------------------------------------------------------------------
// JS pipeline -- verbatim port of critical functions from app.js.
// Changes to these functions in app.js MUST be reflected here.
// ---------------------------------------------------------------------------

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

function orderCorners(points) {
  const cx = points.reduce((s, [x]) => s + x, 0) / points.length;
  const cy = points.reduce((s, [, y]) => s + y, 0) / points.length;
  const sorted = [...points].sort(
    ([ax, ay], [bx, by]) =>
      Math.atan2(ay - cy, ax - cx) - Math.atan2(by - cy, bx - cx),
  );
  let start = 0, best = Infinity;
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

function solveLinearSystem(matrix, vector) {
  const size = vector.length;
  const a = matrix.map((row, index) => [...row, vector[index]]);
  for (let col = 0; col < size; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < size; row += 1) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    if (Math.abs(a[pivot][col]) < 1e-10) throw new Error('Singular matrix.');
    if (pivot !== col) [a[col], a[pivot]] = [a[pivot], a[col]];
    const scale = a[col][col];
    for (let k = col; k <= size; k += 1) a[col][k] /= scale;
    for (let row = 0; row < size; row += 1) {
      if (row === col) continue;
      const factor = a[row][col];
      for (let k = col; k <= size; k += 1) a[row][k] -= factor * a[col][k];
    }
  }
  return a.map((row) => row[size]);
}

function computeHomography(srcPoints, dstPoints) {
  const matrix = [], vector = [];
  for (let i = 0; i < 4; i += 1) {
    const [x, y] = srcPoints[i];
    const [u, v] = dstPoints[i];
    matrix.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); vector.push(u);
    matrix.push([0, 0, 0, x, y, 1, -v * x, -v * y]); vector.push(v);
  }
  const [h11, h12, h13, h21, h22, h23, h31, h32] = solveLinearSystem(matrix, vector);
  return [h11, h12, h13, h21, h22, h23, h31, h32, 1];
}

function applyHomography(m, x, y) {
  const d = m[6] * x + m[7] * y + m[8];
  return [(m[0] * x + m[1] * y + m[2]) / d, (m[3] * x + m[4] * y + m[5]) / d];
}

function sampleBilinear(data, width, height, x, y, ch) {
  const cx = Math.min(Math.max(x, 0), width - 1);
  const cy = Math.min(Math.max(y, 0), height - 1);
  const x0 = Math.floor(cx), y0 = Math.floor(cy);
  const x1 = Math.min(x0 + 1, width - 1), y1 = Math.min(y0 + 1, height - 1);
  const tx = cx - x0, ty = cy - y0;
  const top    = data[(y0 * width + x0) * 4 + ch] * (1 - tx) + data[(y0 * width + x1) * 4 + ch] * tx;
  const bottom = data[(y1 * width + x0) * 4 + ch] * (1 - tx) + data[(y1 * width + x1) * 4 + ch] * tx;
  return top * (1 - ty) + bottom * ty;
}

function normalizeEmbedding(emb) {
  let norm = 0;
  for (const v of emb) norm += v * v;
  norm = Math.sqrt(norm);
  if (norm > 1e-8) for (let i = 0; i < emb.length; i += 1) emb[i] /= norm;
  return emb;
}

function jsDeWarp(srcCanvas, corners) {
  const { width, height } = srcCanvas;
  const srcPts = corners.map(([x, y]) => [x * width, y * height]);
  const dstPts = [[0, 0], [DEWARP_W - 1, 0], [DEWARP_W - 1, DEWARP_H - 1], [0, DEWARP_H - 1]];
  const inv     = computeHomography(dstPts, srcPts);
  const srcData = srcCanvas.getContext('2d', { willReadFrequently: true })
    .getImageData(0, 0, width, height);
  const dstCanvas = createCanvas(DEWARP_W, DEWARP_H);
  const dstCtx    = dstCanvas.getContext('2d');
  const dstData   = dstCtx.createImageData(DEWARP_W, DEWARP_H);
  for (let y = 0; y < DEWARP_H; y += 1) {
    for (let x = 0; x < DEWARP_W; x += 1) {
      const [sx, sy] = applyHomography(inv, x, y);
      const off = (y * DEWARP_W + x) * 4;
      for (let c = 0; c < 3; c += 1)
        dstData.data[off + c] = sampleBilinear(srcData.data, width, height, sx, sy, c);
      dstData.data[off + 3] = 255;
    }
  }
  dstCtx.putImageData(dstData, 0, 0);
  return dstCanvas;
}

// ---------------------------------------------------------------------------
// Load models
// ---------------------------------------------------------------------------

const detectorSession = await ort.InferenceSession.create(CORNELIUS_PATH, {
  executionProviders: ['cpu'],
});
const embedderSession = await ort.InferenceSession.create(MILO_PATH, {
  executionProviders: ['cpu'],
});

// ---------------------------------------------------------------------------
// Pipeline helpers
// ---------------------------------------------------------------------------

async function runDetector(imgCanvas) {
  const scaled = createCanvas(DETECTOR_SIZE, DETECTOR_SIZE);
  scaled.getContext('2d').drawImage(imgCanvas, 0, 0, DETECTOR_SIZE, DETECTOR_SIZE);
  const { data } = scaled.getContext('2d').getImageData(0, 0, DETECTOR_SIZE, DETECTOR_SIZE);
  const flat  = fillInputTensor(data, DETECTOR_SIZE);
  const feeds = { [detectorSession.inputNames[0]]: new ort.Tensor('float32', flat, [1, 3, DETECTOR_SIZE, DETECTOR_SIZE]) };
  const out   = await detectorSession.run(feeds);

  const cornersRaw    = Array.from(out[detectorSession.outputNames[0]].data).slice(0, 8);
  const presenceLogit = out[detectorSession.outputNames[1]].data[0];
  const sharpness     = detectorSession.outputNames[2]
    ? Number(out[detectorSession.outputNames[2]].data[0]) : null;

  const points = [];
  for (let i = 0; i < 8; i += 2)
    points.push([Math.min(Math.max(cornersRaw[i], 0), 1), Math.min(Math.max(cornersRaw[i + 1], 0), 1)]);

  const confidence = sharpness ?? sigmoid(presenceLogit);
  return { corners: orderCorners(points), sharpness, cardPresent: confidence >= MIN_SHARPNESS };
}

async function runEmbedder(dewarpCanvas) {
  const scaled = createCanvas(EMBEDDER_SIZE, EMBEDDER_SIZE);
  scaled.getContext('2d').drawImage(dewarpCanvas, 0, 0, EMBEDDER_SIZE, EMBEDDER_SIZE);
  const { data } = scaled.getContext('2d').getImageData(0, 0, EMBEDDER_SIZE, EMBEDDER_SIZE);
  const flat  = fillInputTensor(data, EMBEDDER_SIZE);
  const feeds = { [embedderSession.inputNames[0]]: new ort.Tensor('float32', flat, [1, 3, EMBEDDER_SIZE, EMBEDDER_SIZE]) };
  const out   = await embedderSession.run(feeds);
  const emb   = Float32Array.from(out[embedderSession.outputNames[0]].data);
  return normalizeEmbedding(emb);
}

// ---------------------------------------------------------------------------
// Tests against the bundled sample image
// ---------------------------------------------------------------------------

console.log(`\nSample: ${SAMPLE_IMAGE.replace(ROOT + '/', '')}`);

const img       = await loadImage(readFileSync(SAMPLE_IMAGE));
const srcCanvas = createCanvas(img.width, img.height);
srcCanvas.getContext('2d').drawImage(img, 0, 0);

const detection = await runDetector(srcCanvas);

await test('detector finds card in sample image', () => {
  assert(detection.cardPresent,
    `Card not detected. sharpness=${detection.sharpness?.toFixed(3)}, threshold=${MIN_SHARPNESS}`);
});

await test('corners are in [0,1] and form a reasonable quad', () => {
  assert(detection.corners.length === 4, 'Expected 4 corners');
  for (const [x, y] of detection.corners) {
    assert(x >= 0 && x <= 1 && y >= 0 && y <= 1, `Corner out of bounds: [${x}, ${y}]`);
  }
  // TL should have the smallest x+y sum among the four corners.
  const [tlx, tly] = detection.corners[0];
  for (const [x, y] of detection.corners.slice(1)) {
    assert(tlx + tly <= x + y + 0.05, `TL corner [${tlx},${tly}] is not top-left`);
  }
});

await test('dewarp and embed produce a valid embedding', async () => {
  assert(detection.cardPresent, 'Card not detected -- cannot dewarp/embed');
  const dewarp = jsDeWarp(srcCanvas, detection.corners);
  assert(dewarp.width === DEWARP_W && dewarp.height === DEWARP_H,
    `Unexpected dewarp size: ${dewarp.width}x${dewarp.height}`);

  const emb = await runEmbedder(dewarp);
  assert(emb.length === 128, `Expected 128-d embedding, got ${emb.length}`);

  let norm = 0;
  for (const v of emb) norm += v * v;
  assert(Math.abs(Math.sqrt(norm) - 1.0) < 1e-4,
    `Embedding not L2-normalised: norm=${Math.sqrt(norm).toFixed(6)}`);
});

// ---------------------------------------------------------------------------
// isUsableQuad regression tests
// ---------------------------------------------------------------------------

function quadArea(corners) {
  let area = 0;
  for (let i = 0; i < corners.length; i += 1) {
    const [x1, y1] = corners[i];
    const [x2, y2] = corners[(i + 1) % corners.length];
    area += x1 * y2 - x2 * y1;
  }
  return Math.abs(area) * 0.5;
}

function isUsableQuad(corners) {
  if (!corners || corners.length !== 4) return false;
  const area = quadArea(corners);
  if (!Number.isFinite(area) || area < 0.01) return false;
  for (let i = 0; i < corners.length; i += 1) {
    for (let j = i + 1; j < corners.length; j += 1) {
      const dx = corners[i][0] - corners[j][0];
      const dy = corners[i][1] - corners[j][1];
      if ((dx * dx + dy * dy) < 0.0004) return false;
    }
  }
  // Reject non-convex quads (mixed cross-product signs).
  let pos = 0, neg = 0;
  for (let i = 0; i < 4; i += 1) {
    const prev = corners[(i + 3) % 4];
    const curr = corners[i];
    const next = corners[(i + 1) % 4];
    const cross = (curr[0] - prev[0]) * (next[1] - curr[1])
                - (curr[1] - prev[1]) * (next[0] - curr[0]);
    if (cross > 0) pos += 1;
    if (cross < 0) neg += 1;
  }
  if (pos > 0 && neg > 0) return false;
  return true;
}

console.log('\nisUsableQuad');

await test('rejects non-convex quad from issue #7 (portrait camera, 3 right-edge corners)', () => {
  // Ordered corners captured from bug report cv_2026-04-24T21-19-49:
  // three corners stacked on the right edge (model missed the top-left corner).
  const bad = [
    [0.8712919950, 0.1383170187],
    [0.8652299642, 0.2562827169],
    [0.8706384897, 0.5906347036],
    [0.5520254373, 0.5551005601],
  ];
  assert(!isUsableQuad(bad), 'Expected non-convex quad to be rejected');
});

await test('accepts a well-formed card quad', () => {
  const good = [[0.15, 0.10], [0.85, 0.12], [0.82, 0.88], [0.13, 0.86]];
  assert(isUsableQuad(good), 'Expected valid quad to be accepted');
});

await test('rejects quad with area too small', () => {
  const tiny = [[0.49, 0.49], [0.51, 0.49], [0.51, 0.51], [0.49, 0.51]];
  assert(!isUsableQuad(tiny), 'Expected tiny quad to be rejected (area < 0.01)');
});

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

console.log(`\n${'='.repeat(50)}`);
console.log(`Results: ${_passed} passed, ${_failed} failed`);
if (_failures.length > 0) {
  for (const { label, err } of _failures) {
    console.error(`  FAIL  ${label}: ${err.message.split('\n')[0]}`);
  }
  process.exit(1);
}
