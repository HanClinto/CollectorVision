// scanner.worker.mjs
// Runs the full detect → dewarp → embed → search pipeline on a background
// thread so the camera preview's requestAnimationFrame loop is never blocked.
//
// Message protocol
// ----------------
// main → worker:
//   { type: 'init',   manifest }
//   { type: 'frame',  bitmap: ImageBitmap }   (transferred, zero-copy)
//
// worker → main:
//   { type: 'progress', stage, ratio, loaded?, total?, cached?, inferenceMode? }
//   { type: 'ready',    inferenceMode }
//   { type: 'result',   cardPresent, cornersValid, corners, sharpness,
//                       confidence, cardId, score, rawCorners, detectorInput,
//                       detectorBitmap?, cropBitmap? }
//   { type: 'error',    message }

import * as ort from "./vendor/onnxruntime-web/ort.webgpu.min.mjs";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DETECTOR_SIZE = 384;
const EMBEDDER_SIZE = 448;
const DEWARP_W = EMBEDDER_SIZE;
const DEWARP_H = EMBEDDER_SIZE;
const MIN_SHARPNESS = 0.02;
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

const ASSET_DB_NAME = "collectorvision-web-scanner";
const ASSET_STORE_NAME = "assets";

// ---------------------------------------------------------------------------
// Math helpers (identical to the originals in app.js)
// ---------------------------------------------------------------------------

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function orderCorners(points) {
  const cx = points.reduce((sum, [x]) => sum + x, 0) / points.length;
  const cy = points.reduce((sum, [, y]) => sum + y, 0) / points.length;
  const sorted = [...points].sort(
    ([ax, ay], [bx, by]) => Math.atan2(ay - cy, ax - cx) - Math.atan2(by - cy, bx - cx),
  );
  let start = 0;
  let best = Infinity;
  for (let i = 0; i < sorted.length; i += 1) {
    const score = sorted[i][0] + sorted[i][1];
    if (score < best) {
      best = score;
      start = i;
    }
  }
  const ordered = [
    sorted[start],
    sorted[(start + 1) % 4],
    sorted[(start + 2) % 4],
    sorted[(start + 3) % 4],
  ];
  const signedArea = ordered.reduce((sum, [x1, y1], i) => {
    const [x2, y2] = ordered[(i + 1) % ordered.length];
    return sum + (x1 * y2 - x2 * y1);
  }, 0);
  if (signedArea < 0) {
    return [ordered[0], ordered[3], ordered[2], ordered[1]];
  }
  return ordered;
}

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
  if (!corners || corners.length !== 4) {
    return false;
  }
  const area = quadArea(corners);
  if (!Number.isFinite(area) || area < 0.01) {
    return false;
  }
  for (let i = 0; i < corners.length; i += 1) {
    for (let j = i + 1; j < corners.length; j += 1) {
      const dx = corners[i][0] - corners[j][0];
      const dy = corners[i][1] - corners[j][1];
      if ((dx * dx + dy * dy) < 0.0004) {
        return false;
      }
    }
  }
  // Reject non-convex quads.  When the model misses a corner (e.g. on a
  // portrait camera where one corner is out of frame), it often returns 3
  // nearly-collinear points on one edge and one outlier.  That produces a
  // concave quad where one interior angle is reflex — the cross product at
  // that vertex has the opposite sign to the other three.
  let pos = 0;
  let neg = 0;
  for (let i = 0; i < 4; i += 1) {
    const prev = corners[(i + 3) % 4];
    const curr = corners[i];
    const next = corners[(i + 1) % 4];
    const cross = (curr[0] - prev[0]) * (next[1] - curr[1])
                - (curr[1] - prev[1]) * (next[0] - curr[0]);
    if (cross > 0) pos += 1;
    if (cross < 0) neg += 1;
  }
  if (pos > 0 && neg > 0) {
    return false;
  }
  return true;
}

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
      throw new Error("Could not solve dewarp transform.");
    }
    if (pivot !== col) {
      [a[col], a[pivot]] = [a[pivot], a[col]];
    }
    const scale = a[col][col];
    for (let k = col; k <= size; k += 1) {
      a[col][k] /= scale;
    }
    for (let row = 0; row < size; row += 1) {
      if (row === col) {
        continue;
      }
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

  const top = data[i00] * (1 - tx) + data[i10] * tx;
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

// ---------------------------------------------------------------------------
// Float16 / catalog decode
// ---------------------------------------------------------------------------

function float16ToFloat32(value) {
  const sign = (value & 0x8000) >> 15;
  const exponent = (value & 0x7c00) >> 10;
  const fraction = value & 0x03ff;

  if (exponent === 0) {
    if (fraction === 0) {
      return sign ? -0 : 0;
    }
    return (sign ? -1 : 1) * 2 ** (-14) * (fraction / 1024);
  }

  if (exponent === 0x1f) {
    return fraction ? Number.NaN : (sign ? -Infinity : Infinity);
  }

  return (sign ? -1 : 1) * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function createFloat16LookupTable() {
  const table = new Float32Array(65536);
  for (let value = 0; value < table.length; value += 1) {
    table[value] = float16ToFloat32(value);
  }
  return table;
}

const FLOAT16_LOOKUP = createFloat16LookupTable();

function wrapFloat16Buffer(buffer) {
  return new Uint16Array(buffer);
}

// ---------------------------------------------------------------------------
// IndexedDB asset cache (same DB as the main thread — shared origin storage)
// ---------------------------------------------------------------------------

function openAssetDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(ASSET_DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(ASSET_STORE_NAME)) {
        db.createObjectStore(ASSET_STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function readCachedAsset(key) {
  const db = await openAssetDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(ASSET_STORE_NAME, "readonly");
    const store = tx.objectStore(ASSET_STORE_NAME);
    const request = store.get(key);
    request.onsuccess = () => resolve(request.result ?? null);
    request.onerror = () => reject(request.error);
  });
}

async function writeCachedAsset(key, value) {
  const db = await openAssetDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(ASSET_STORE_NAME, "readwrite");
    const store = tx.objectStore(ASSET_STORE_NAME);
    const request = store.put(value, key);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

// ---------------------------------------------------------------------------
// HTTP helpers with progress reporting + IndexedDB caching
// ---------------------------------------------------------------------------

async function fetchWithProgress(url, responseType, onProgress) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
  }

  const total = Number.parseInt(response.headers.get("content-length") ?? "0", 10) || 0;
  if (!response.body || total === 0) {
    const payload = responseType === "json" ? await response.json() : await response.arrayBuffer();
    onProgress?.(1, total || 1, total || 1);
    return payload;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    chunks.push(value);
    loaded += value.length;
    onProgress?.(loaded / total, loaded, total);
  }

  const blob = new Blob(chunks);
  if (responseType === "json") {
    return JSON.parse(await blob.text());
  }
  return await blob.arrayBuffer();
}

async function fetchJsonCached(url, version, onProgress) {
  const key = `${version}:${url}:json`;
  const cached = await readCachedAsset(key);
  if (cached) {
    onProgress?.(1, 1, 1, true);
    return cached;
  }
  const json = await fetchWithProgress(url, "json", (ratio, loaded, total) => {
    onProgress?.(ratio, loaded, total, false);
  });
  await writeCachedAsset(key, json);
  return json;
}

async function fetchBufferCached(url, version, onProgress) {
  const key = `${version}:${url}:buffer`;
  const cached = await readCachedAsset(key);
  if (cached) {
    onProgress?.(1, cached.byteLength ?? 1, cached.byteLength ?? 1, true);
    return cached;
  }
  const buffer = await fetchWithProgress(url, "buffer", (ratio, loaded, total) => {
    onProgress?.(ratio, loaded, total, false);
  });
  await writeCachedAsset(key, buffer);
  return buffer;
}

// ---------------------------------------------------------------------------
// WebGPU configuration
// ---------------------------------------------------------------------------

/**
 * Attempts to configure the WebGPU adapter for ort-web.
 * Returns true if WebGPU was configured, false if unavailable.
 * Never throws — a false return means ort-web will fall back to WASM.
 */
async function configureWebGpu() {
  if (!("gpu" in navigator)) {
    return false;
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) {
    return false;
  }

  const requestedStorageBuffers = Math.min(
    10,
    adapter.limits.maxStorageBuffersPerShaderStage ?? 8,
  );
  const originalRequestDevice = adapter.requestDevice.bind(adapter);
  Object.defineProperty(adapter, "requestDevice", {
    configurable: true,
    value: async (descriptor = {}) => originalRequestDevice({
      ...descriptor,
      requiredLimits: {
        ...(descriptor.requiredLimits ?? {}),
        maxStorageBuffersPerShaderStage: requestedStorageBuffers,
      },
    }),
  });

  ort.env.webgpu.adapter = adapter;
  // ort.webgpu.min.mjs (new WebGPU EP) — the legacy ort.all.min.mjs JSEP backend
  // silently returned all-zeros for Conv ops on Android (ort-web 1.20–1.24.3).
  // The new EP fixes that, but cornelius.onnx still produces numerically wrong
  // (coherent but incorrect) corner outputs on Android ARM GPUs even with the new EP.
  // See ARCHITECTURE.md “Lessons Learned” for the full history.
  ort.env.webgpu.forceFp16 = false;

  return true;
}

// ---------------------------------------------------------------------------
// Tensor preparation (uses OffscreenCanvas instead of HTMLCanvasElement)
// ---------------------------------------------------------------------------

function fillInputTensorFromContext(ctx, size, tensor) {
  const { data } = ctx.getImageData(0, 0, size, size);
  const plane = size * size;

  for (let i = 0; i < size * size; i += 1) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    tensor[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    tensor[plane + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    tensor[plane * 2 + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }

  return tensor;
}

function createInputTensor(size) {
  return new Float32Array(1 * 3 * size * size);
}

function isIOS() {
  return /iPad|iPhone|iPod/.test(navigator.userAgent)
    || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
}

// ---------------------------------------------------------------------------
// WorkerRuntime — BrowserRuntime ported to run inside a web worker.
// All canvas operations use OffscreenCanvas.
// ---------------------------------------------------------------------------

class WorkerRuntime {
  constructor(manifest, useWebGpu = false) {
    this.manifest = manifest;
    this.useWebGpu = useWebGpu;
    this.detector = null;
    this.embedder = null;
    this.inputNames = {};
    this.embeddings = null;
    this.cardIds = null;
    this.dewarpCanvas = new OffscreenCanvas(DEWARP_W, DEWARP_H);
    this.dewarpCtx = this.dewarpCanvas.getContext("2d", { willReadFrequently: true });
    this.dewarpImageData = this.dewarpCtx.createImageData(DEWARP_W, DEWARP_H);
    // Reusable 384×384 scratch canvas — what was fed to the detector.
    // Transferred to the main thread as a debug bitmap on each result.
    this.detectorScratchCanvas = new OffscreenCanvas(DETECTOR_SIZE, DETECTOR_SIZE);
    this.detectorScratchCtx = this.detectorScratchCanvas.getContext("2d", { willReadFrequently: true });
    this.detectorInputTensor = createInputTensor(DETECTOR_SIZE);
    this.embedderScratchCanvas = new OffscreenCanvas(EMBEDDER_SIZE, EMBEDDER_SIZE);
    this.embedderScratchCtx = this.embedderScratchCanvas.getContext("2d", { willReadFrequently: true });
    this.embedderInputTensor = createInputTensor(EMBEDDER_SIZE);
    this._lastRawCorners = null;
    this._lastDetectorInput = null;
  }

  async load(onStage) {
    const version = this.manifest.version;
    // Use per-model content hashes as cache keys when available so that a new
    // model weight file (same filename, different content) always busts the
    // IndexedDB entry, even if the bundle version string hasn't changed.
    const hashes = this.manifest.model_hashes ?? {};
    const detectorVersion = hashes.cornelius ?? version;
    const embedderVersion = hashes.milo       ?? version;
    const detectorBuffer = await fetchBufferCached(
      `./assets/${this.manifest.models.cornelius}`,
      detectorVersion,
      (ratio, loaded, total, cached) => onStage?.("detector", ratio, loaded, total, cached),
    );
    const embedderBuffer = await fetchBufferCached(
      `./assets/${this.manifest.models.milo}`,
      embedderVersion,
      (ratio, loaded, total, cached) => onStage?.("embedder", ratio, loaded, total, cached),
    );

    // Use as many threads as the device has cores, capped at 4.
    // NOTE: multi-threaded WASM requires SharedArrayBuffer / COOP+COEP headers.
    // Without them ort-web silently falls back to 1 thread.  The COI service
    // worker (coi-serviceworker.js) injects these headers on GitHub Pages so
    // that crossOriginIsolated becomes true and all threads are available.
    // iOS WebKit is prone to killing memory-heavy WASM pages.  Keep the iOS
    // path single-threaded to avoid the extra worker/SAB overhead; other
    // platforms keep the capped multi-threaded fast path.
    const numThreads = isIOS() ? 1 : Math.min(navigator.hardwareConcurrency || 1, 4);
    ort.env.wasm.numThreads = numThreads;
    this.numThreads = numThreads;
    // EP selection: WASM is the safe default.
    // WebGPU is proven broken on Android ARM (issues #9 and #12) but may work
    // on iOS (Metal) and desktop.  The enableWebGpu flag in the init message
    // lets the user opt in from Settings — see ARCHITECTURE.md Lessons Learned.
    const ep = this.useWebGpu ? "webgpu" : "wasm";
    this.detector = await ort.InferenceSession.create(detectorBuffer, {
      executionProviders: [ep],
    });
    this.embedder = await ort.InferenceSession.create(embedderBuffer, {
      executionProviders: [ep],
    });

    this.inputNames.detector = this.detector.inputNames[0];
    this.inputNames.embedder = this.embedder.inputNames[0];

    const embeddingBuffer = await fetchBufferCached(
      `./assets/${this.manifest.catalog.embeddings}`,
      version,
      (ratio, loaded, total, cached) => onStage?.("catalog", ratio * 0.92, loaded, total, cached),
    );
    const ids = await fetchJsonCached(
      `./assets/${this.manifest.catalog.card_ids}`,
      version,
      (ratio, loaded, total, cached) => onStage?.("catalog", 0.92 + ratio * 0.08, loaded, total, cached),
    );
    // Keep the catalog in its packed float16 form.  Expanding the full MTG
    // matrix to Float32Array roughly doubles steady-state catalog memory and
    // can push iOS WebKit into tab reloads.  Search converts individual values
    // through a 256 KB lookup table instead.
    this.embeddings = wrapFloat16Buffer(embeddingBuffer);
    this.cardIds = ids;
  }

  async detect(frameCanvas) {
    const vw = frameCanvas.width;
    const vh = frameCanvas.height;

    // Save a copy of what we fed the model for the debug preview bitmap.
    this.detectorScratchCtx.drawImage(frameCanvas, 0, 0, DETECTOR_SIZE, DETECTOR_SIZE);
    this._lastDetectorInput = `${vw}×${vh} → squash ${DETECTOR_SIZE}×${DETECTOR_SIZE}`;

    const input = fillInputTensorFromContext(
      this.detectorScratchCtx,
      DETECTOR_SIZE,
      this.detectorInputTensor,
    );
    const outputs = await this.detector.run({
      [this.inputNames.detector]: new ort.Tensor("float32", input, [1, 3, DETECTOR_SIZE, DETECTOR_SIZE]),
    });
    const cornersRaw = Array.from(outputs[this.detector.outputNames[0]].data).slice(0, 8);
    const presenceLogit = outputs[this.detector.outputNames[1]].data[0];
    const sharpness = this.detector.outputNames[2]
      ? outputs[this.detector.outputNames[2]].data[0]
      : null;

    const points = [];
    for (let i = 0; i < 8; i += 2) {
      points.push([
        Math.min(Math.max(cornersRaw[i], 0), 1),
        Math.min(Math.max(cornersRaw[i + 1], 0), 1),
      ]);
    }

    this._lastRawCorners = points.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join("  ");

    return {
      corners: orderCorners(points),
      sharpness,
      confidence: sharpness ?? sigmoid(presenceLogit),
      cardPresent: (sharpness ?? sigmoid(presenceLogit)) >= MIN_SHARPNESS,
    };
  }

  dewarp(frameCanvas, corners) {
    const width = frameCanvas.width;
    const height = frameCanvas.height;
    const srcPts = [
      corners[0][0] * width, corners[0][1] * height,
      corners[1][0] * width, corners[1][1] * height,
      corners[2][0] * width, corners[2][1] * height,
      corners[3][0] * width, corners[3][1] * height,
    ];
    const dstPts = [
      0, 0,
      DEWARP_W - 1, 0,
      DEWARP_W - 1, DEWARP_H - 1,
      0, DEWARP_H - 1,
    ];
    const sourcePoints = [
      [srcPts[0], srcPts[1]],
      [srcPts[2], srcPts[3]],
      [srcPts[4], srcPts[5]],
      [srcPts[6], srcPts[7]],
    ];
    const targetPoints = [
      [dstPts[0], dstPts[1]],
      [dstPts[2], dstPts[3]],
      [dstPts[4], dstPts[5]],
      [dstPts[6], dstPts[7]],
    ];
    const inverse = computeHomography(targetPoints, sourcePoints);
    const srcData = frameCanvas.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, width, height);
    const dstData = this.dewarpImageData;

    for (let y = 0; y < DEWARP_H; y += 1) {
      for (let x = 0; x < DEWARP_W; x += 1) {
        const [sx, sy] = applyHomography(inverse, x, y);
        const offset = (y * DEWARP_W + x) * 4;
        dstData.data[offset] = sampleBilinear(srcData.data, width, height, sx, sy, 0);
        dstData.data[offset + 1] = sampleBilinear(srcData.data, width, height, sx, sy, 1);
        dstData.data[offset + 2] = sampleBilinear(srcData.data, width, height, sx, sy, 2);
        dstData.data[offset + 3] = 255;
      }
    }

    this.dewarpCtx.putImageData(dstData, 0, 0);
    return this.dewarpCanvas;
  }

  async embed(cropCanvas) {
    this.embedderScratchCtx.drawImage(cropCanvas, 0, 0, EMBEDDER_SIZE, EMBEDDER_SIZE);
    const input = fillInputTensorFromContext(
      this.embedderScratchCtx,
      EMBEDDER_SIZE,
      this.embedderInputTensor,
    );
    const outputs = await this.embedder.run({
      [this.inputNames.embedder]: new ort.Tensor("float32", input, [1, 3, EMBEDDER_SIZE, EMBEDDER_SIZE]),
    });
    return normalizeEmbedding(Float32Array.from(outputs[this.embedder.outputNames[0]].data));
  }

  search(query) {
    const dims = this.manifest.catalog.dims;
    const rows = this.manifest.catalog.rows;
    let bestScore = -Infinity;
    let bestIndex = -1;

    for (let row = 0; row < rows; row += 1) {
      const offset = row * dims;
      let score = 0;
      for (let col = 0; col < dims; col += 1) {
        score += FLOAT16_LOOKUP[this.embeddings[offset + col]] * query[col];
      }
      if (score > bestScore) {
        bestScore = score;
        bestIndex = row;
      }
    }

    return {
      score: bestScore,
      cardId: this.cardIds[bestIndex],
    };
  }
}

// ---------------------------------------------------------------------------
// Frame pipeline
// ---------------------------------------------------------------------------

let runtime = null;
// Reusable OffscreenCanvas sized to the incoming frame — resized as needed.
let frameCanvas = null;
let frameCtx = null;

async function processFrame(bitmap, captureRequested = false, includeDebugBitmaps = false) {
  if (!frameCanvas || frameCanvas.width !== bitmap.width || frameCanvas.height !== bitmap.height) {
    frameCanvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    frameCtx = frameCanvas.getContext("2d", { willReadFrequently: true });
  }
  frameCtx.drawImage(bitmap, 0, 0);
  bitmap.close();

  const t0 = performance.now();

  // Snapshot the frame for the capture bundle (zero-copy, worker → main).
  let captureFrameBitmap = null;
  if (captureRequested) {
    const snap = new OffscreenCanvas(frameCanvas.width, frameCanvas.height);
    snap.getContext("2d").drawImage(frameCanvas, 0, 0);
    captureFrameBitmap = snap.transferToImageBitmap();
  }

  const t1 = performance.now();
  const detection = await runtime.detect(frameCanvas);
  const t2 = performance.now();

  if (!detection.cardPresent) {
    const transfer0 = captureFrameBitmap ? [captureFrameBitmap] : [];
    self.postMessage({
      type: "result",
      captureRequested,
      captureFrameBitmap,
      cardPresent: false,
      cornersValid: false,
      corners: detection.corners,
      sharpness: detection.sharpness,
      confidence: detection.confidence,
      cardId: null,
      score: null,
      rawCorners: runtime._lastRawCorners,
      detectorInput: runtime._lastDetectorInput,
      detectorBitmap: null,
      cropBitmap: null,
      timing: { detectMs: Math.round(t2 - t1), dewarpMs: 0, embedMs: 0, searchMs: 0, totalMs: Math.round(t2 - t0) },
    }, transfer0);
    return;
  }

  // Transfer debug bitmaps only when the settings preview is open or an
  // explicit capture is requested.  On mobile Safari, transferring ImageBitmap
  // objects every successful frame can put enough pressure on WebKit's GPU
  // process to reload the page.
  const detectorBitmap = includeDebugBitmaps
    ? runtime.detectorScratchCanvas.transferToImageBitmap()
    : null;
  const cornersValid = isUsableQuad(detection.corners);

  if (!cornersValid) {
    const transfer1 = detectorBitmap ? [detectorBitmap] : [];
    if (captureFrameBitmap) transfer1.push(captureFrameBitmap);
    self.postMessage({
      type: "result",
      captureRequested,
      captureFrameBitmap,
      cardPresent: true,
      cornersValid: false,
      corners: detection.corners,
      sharpness: detection.sharpness,
      confidence: detection.confidence,
      cardId: null,
      score: null,
      rawCorners: runtime._lastRawCorners,
      detectorInput: runtime._lastDetectorInput,
      detectorBitmap,
      cropBitmap: null,
      timing: { detectMs: Math.round(t2 - t1), dewarpMs: 0, embedMs: 0, searchMs: 0, totalMs: Math.round(t2 - t0) },
    }, transfer1);
    return;
  }

  const t3 = performance.now();
  const cropCanvas = runtime.dewarp(frameCanvas, detection.corners);
  const t4 = performance.now();
  const embedding = await runtime.embed(cropCanvas);
  const t5 = performance.now();
  const best = runtime.search(embedding);
  const t6 = performance.now();

  // Transfer the dewarp canvas bitmap (zero-copy) then it gets a fresh blank.
  const cropBitmap = includeDebugBitmaps ? cropCanvas.transferToImageBitmap() : null;

  const transfer2 = [];
  if (detectorBitmap) transfer2.push(detectorBitmap);
  if (cropBitmap) transfer2.push(cropBitmap);
  if (captureFrameBitmap) transfer2.push(captureFrameBitmap);
  self.postMessage({
    type: "result",
    captureRequested,
    captureFrameBitmap,
    cardPresent: true,
    cornersValid: true,
    corners: detection.corners,
    sharpness: detection.sharpness,
    confidence: detection.confidence,
    cardId: best.cardId,
    score: best.score,
    rawCorners: runtime._lastRawCorners,
    detectorInput: runtime._lastDetectorInput,
    detectorBitmap,
    cropBitmap,
    timing: {
      detectMs: Math.round(t2 - t1),
      dewarpMs: Math.round(t4 - t3),
      embedMs: Math.round(t5 - t4),
      searchMs: Math.round(t6 - t5),
      totalMs: Math.round(t6 - t0),
    },
  }, transfer2);
}

// ---------------------------------------------------------------------------
// Message handler
// ---------------------------------------------------------------------------

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") {
      const enableWebGpu = data.enableWebGpu === true;
      const webgpuReady = enableWebGpu ? await configureWebGpu() : false;
      const useWebGpu = webgpuReady; // only true if both requested and available
      const inferenceMode = useWebGpu ? "WebGPU" : "WASM";
      self.postMessage({ type: "progress", stage: "webgpu", inferenceMode });
      self.postMessage({ type: "progress", stage: "dewarp", ratio: 1 });

      runtime = new WorkerRuntime(data.manifest, useWebGpu);
      await runtime.load((stage, ratio, loaded, total, cached) => {
        self.postMessage({ type: "progress", stage, ratio, loaded, total, cached });
      });

      self.postMessage({ type: "ready", inferenceMode, numThreads: runtime.numThreads ?? 1 });

    } else if (data.type === "frame") {
      await processFrame(
        data.bitmap,
        data.captureRequested ?? false,
        data.includeDebugBitmaps ?? false,
      );
    }
  } catch (error) {
    self.postMessage({ type: "error", message: error?.message ?? String(error) });
  }
};
