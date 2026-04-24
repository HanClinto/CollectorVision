import * as ort from "./vendor/onnxruntime-web/ort.all.min.mjs";

const BUILD_ID = "2026-04-24-1";

const DETECTOR_SIZE = 384;
const EMBEDDER_SIZE = 448;
const DEWARP_W = 252;
const DEWARP_H = 352;
const MIN_SHARPNESS = 0.02;
const MIN_MATCH_SCORE = 0.75;
const SCAN_INTERVAL_MS = 900;

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

const SOUND_PATHS = {
  scanConfirmed: "./sounds/scan.wav",
  priceHigh: "./sounds/pickup_high.wav",
  priceMid: "./sounds/pickup_mid.wav",
};

const ASSET_DB_NAME = "collectorvision-web-scanner";
const ASSET_STORE_NAME = "assets";

const NOTES = [
  "The scanner now uses the real ONNX weights and the real MTG gallery bundle.",
  "The browser app reads local ./assets files; Hugging Face is a publish-time sync step.",
  "Models and catalog files are cached in IndexedDB after first download.",
  "WebGPU is required. The app does not fall back to WASM-only inference.",
  "Scryfall enrichment runs after confirmation so the recognition loop stays local.",
  "Settings include a bundled sample-frame smoke test for local bring-up.",
  "scan.wav fires on confirm; price-tier sounds fire after Scryfall returns.",
  "Perspective dewarp runs locally in JS so startup stays simple and self-contained.",
];

const LOADING_STEPS = [
  { id: "webgpu", label: "Checking WebGPU" },
  { id: "manifest", label: "Loading manifest" },
  { id: "dewarp", label: "Preparing dewarp" },
  { id: "detector", label: "Loading corner detector" },
  { id: "embedder", label: "Loading embedder" },
  { id: "catalog", label: "Loading catalog" },
];

function describeValue(value) {
  if (value instanceof Error) {
    return value.stack || value.message;
  }
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function createDebugLog() {
  const list = document.getElementById("debug-log");
  const limit = 200;

  function push(level, ...parts) {
    const message = parts.map(describeValue).join(" ");
    console[level === "info" ? "log" : level](`[CollectorVision] ${message}`);

    const item = document.createElement("li");
    item.className = "debug-entry";
    item.dataset.level = level;
    item.innerHTML = `
      <p class="debug-entry__meta">${new Date().toLocaleTimeString()} · ${level.toUpperCase()}</p>
      <p class="debug-entry__message"></p>
    `;
    item.querySelector(".debug-entry__message").textContent = message;
    list.prepend(item);

    while (list.children.length > limit) {
      list.removeChild(list.lastElementChild);
    }
  }

  document.getElementById("clear-debug").addEventListener("click", () => {
    list.innerHTML = "";
  });

  window.addEventListener("error", (event) => {
    push("error", event.message);
  });
  window.addEventListener("unhandledrejection", (event) => {
    push("error", "Unhandled promise rejection", event.reason);
  });

  return {
    info: (...parts) => push("info", ...parts),
    warn: (...parts) => push("warn", ...parts),
    error: (...parts) => push("error", ...parts),
  };
}

function createLoadingScreen() {
  const body = document.body;
  const message = document.getElementById("loading-message");
  const fill = document.getElementById("loading-fill");
  const percent = document.getElementById("loading-percent");
  const steps = document.getElementById("loading-steps");
  const stepEls = new Map();

  for (const step of LOADING_STEPS) {
    const item = document.createElement("li");
    item.className = "loading-screen__step";
    item.dataset.state = "pending";
    item.innerHTML = `
      <span>${step.label}</span>
      <span class="loading-screen__step-note">Pending</span>
    `;
    steps.appendChild(item);
    stepEls.set(step.id, item);
  }

  function updatePercent(value) {
    const clamped = Math.max(0, Math.min(100, value));
    fill.style.width = `${clamped}%`;
    percent.textContent = `${Math.round(clamped)}%`;
  }

  return {
    start(text = "Preparing scanner runtime") {
      body.dataset.loading = "true";
      message.textContent = text;
      updatePercent(0);
    },
    progress(value, text) {
      updatePercent(value);
      if (text) {
        message.textContent = text;
      }
    },
    step(id, state, note) {
      const el = stepEls.get(id);
      if (!el) {
        return;
      }
      el.dataset.state = state;
      el.querySelector(".loading-screen__step-note").textContent = note;
    },
    finish() {
      updatePercent(100);
      message.textContent = "Scanner ready";
      for (const step of LOADING_STEPS) {
        this.step(step.id, "done", "Ready");
      }
      setTimeout(() => {
        delete body.dataset.loading;
      }, 180);
    },
    fail(text) {
      message.textContent = text;
    },
  };
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) {
    el.textContent = value;
  }
}

function formatCurrency(value) {
  if (value === null || value === undefined || value === "") {
    return "Price pending";
  }
  return `$${Number.parseFloat(value).toFixed(2)}`;
}

function buildTextExport(scans) {
  return scans
    .map((scan) => {
      const setCode = (scan.setCode || "mtg").toUpperCase();
      const setName = scan.setName || "Loading...";
      return `${scan.name} — ${setName} (${setCode})`;
    })
    .join("\n");
}

function buildCsvExport(scans) {
  const rows = [["name", "set_code", "set_name", "price_usd", "card_id", "count"]];
  for (const scan of scans) {
    rows.push([
      scan.name,
      scan.setCode,
      scan.setName,
      scan.priceUsd ?? "",
      scan.cardId,
      scan.count,
    ]);
  }
  return rows
    .map((row) => row.map((value) => `"${String(value).replaceAll("\"", "\"\"")}"`).join(","))
    .join("\n");
}

function renderNotes() {
  const list = document.getElementById("notes");
  for (const note of NOTES) {
    const item = document.createElement("li");
    item.textContent = note;
    list.appendChild(item);
  }
}

function renderManifestContract(manifest) {
  const el = document.getElementById("asset-contract");
  el.textContent = JSON.stringify(manifest, null, 2);
}

function renderBuildId() {
  const el = document.getElementById("loading-build");
  if (el) {
    el.textContent = `build ${BUILD_ID}`;
  }
}

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

function decodeFloat16Buffer(buffer) {
  const source = new Uint16Array(buffer);
  const out = new Float32Array(source.length);
  for (let i = 0; i < source.length; i += 1) {
    out[i] = float16ToFloat32(source[i]);
  }
  return out;
}

function createAudioBus() {
  const cache = new Map();

  function getAudio(path) {
    let audio = cache.get(path);
    if (!audio) {
      audio = new Audio(path);
      audio.preload = "auto";
      cache.set(path, audio);
    }
    return audio;
  }

  async function play(path) {
    try {
      const audio = getAudio(path).cloneNode();
      await audio.play();
    } catch (error) {
      console.warn("audio play failed", error);
    }
  }

  return {
    preload() {
      for (const path of Object.values(SOUND_PATHS)) {
        getAudio(path);
      }
    },
    playScanConfirmed() {
      return play(SOUND_PATHS.scanConfirmed);
    },
    playPriceTier(priceUsd) {
      const value = Number.parseFloat(priceUsd ?? "0");
      if (value > 5) {
        return play(SOUND_PATHS.priceHigh);
      }
      if (value > 0.25) {
        return play(SOUND_PATHS.priceMid);
      }
      return Promise.resolve();
    },
  };
}

function assertWebGpu() {
  if (!("gpu" in navigator)) {
    throw new Error("WebGPU is required for this scanner.");
  }
}

async function configureWebGpu(debugLog) {
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) {
    throw new Error("Failed to get a WebGPU adapter.");
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

  debugLog.info(
    "webgpu adapter ready",
    `maxStorageBuffersPerShaderStage=${requestedStorageBuffers}`,
  );
}

function prepareDewarp() {
  return Promise.resolve(true);
}

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

function fillInputTensor(canvas, size) {
  const scratch = document.createElement("canvas");
  scratch.width = size;
  scratch.height = size;
  const ctx = scratch.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(canvas, 0, 0, size, size);
  const { data } = ctx.getImageData(0, 0, size, size);
  const tensor = new Float32Array(1 * 3 * size * size);
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

function updateCropPreview(cropCanvas) {
  const wrapper = document.getElementById("crop-preview");
  const target = document.getElementById("crop-canvas");
  if (!wrapper || !target) {
    return;
  }
  // Draw 1:1 — target canvas is sized to DEWARP_W×DEWARP_H so this shows
  // exactly what the embedder receives, at full resolution.
  const ctx = target.getContext("2d");
  ctx.drawImage(cropCanvas, 0, 0);
  wrapper.hidden = false;
}

class ScanBucket {
  constructor(fillAt = 2, cooldownMs = 3500) {
    this.fillAt = fillAt;
    this.cooldownMs = cooldownMs;
    this.candidate = null;
    this.cooldowns = new Map();
  }

  push(rec) {
    const now = Date.now();
    for (const [id, expiry] of this.cooldowns) {
      if (now >= expiry) {
        this.cooldowns.delete(id);
      }
    }

    if (!rec) {
      if (this.candidate) {
        this.candidate.count = Math.max(0, this.candidate.count - 1);
        if (this.candidate.count === 0) {
          this.candidate = null;
        }
      }
      return null;
    }

    if (this.cooldowns.has(rec.cardId)) {
      return null;
    }

    if (this.candidate?.cardId === rec.cardId) {
      this.candidate.count += 1;
      this.candidate.rec = rec;
      if (this.candidate.count >= this.fillAt) {
        const confirmed = this.candidate.rec;
        this.cooldowns.set(rec.cardId, now + this.cooldownMs);
        this.candidate = null;
        return confirmed;
      }
      return null;
    }

    this.candidate = { cardId: rec.cardId, count: 1, rec };
    return null;
  }
}

class CameraSurface {
  constructor(debugLog) {
    this.page = document.querySelector(".page");
    this.video = document.getElementById("camera-video");
    this.preview = document.getElementById("camera-preview");
    this.previewCtx = this.preview.getContext("2d");
    this.canvas = document.getElementById("camera-overlay");
    this.ctx = this.canvas.getContext("2d");
    this.badge = document.getElementById("camera-badge");
    this.startButton = document.getElementById("camera-start");
    this.debugLog = debugLog;
    this.stream = null;
    this.frameCanvas = document.createElement("canvas");
    this.frameCtx = this.frameCanvas.getContext("2d", { willReadFrequently: true });
    this._resizeHandler = () => this.resize();
    this._previewFrame = null;
  }

  bind(onStart) {
    this.startButton.addEventListener("click", async () => {
      if (this.startButton.disabled) {
        return;
      }
      this.startButton.disabled = true;
      try {
        await this.start();
        await onStart();
      } catch (error) {
        this.debugLog.error("camera start failed", error);
        this.badge.textContent = this.describeCameraError(error);
        this.startButton.disabled = false;
      }
    });
  }

  setLoading(message) {
    this.badge.textContent = message;
    this.startButton.disabled = true;
    this.debugLog.info(message);
  }

  setReady() {
    this.badge.textContent = "Ready to start camera";
    this.startButton.disabled = false;
    this.debugLog.info("scanner runtime loaded; camera can start");
  }

  async start() {
    if (this.stream) {
      return;
    }
    if (!window.isSecureContext) {
      throw new Error("Camera requires HTTPS or localhost.");
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera API unavailable in this browser.");
    }
    this.badge.textContent = "Requesting camera";
    this.debugLog.info("requesting camera stream");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "environment",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    this.stream = stream;
    this.video.playsInline = true;
    this.video.muted = true;
    this.video.srcObject = stream;
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("Camera metadata timed out.")), 5000);
      this.video.onloadedmetadata = () => {
        clearTimeout(timeout);
        resolve();
      };
    });
    await this.video.play();
    this.page.dataset.cameraReady = "true";
    this.badge.textContent = "Camera live";
    this.debugLog.info("camera stream is live", `${this.video.videoWidth}x${this.video.videoHeight}`);
    this.startButton.hidden = true;
    this.resize();
    window.addEventListener("resize", this._resizeHandler);
    this.renderPreview();
  }

  resize() {
    const width = this.preview.clientWidth || this.video.clientWidth || this.video.videoWidth;
    const height = this.preview.clientHeight || this.video.clientHeight || this.video.videoHeight;
    if (!width || !height) {
      return;
    }
    const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
    const nextPreviewWidth = Math.round(width * dpr);
    const nextPreviewHeight = Math.round(height * dpr);
    const nextOverlayWidth = Math.round(width * dpr);
    const nextOverlayHeight = Math.round(height * dpr);
    if (this.preview.width !== nextPreviewWidth || this.preview.height !== nextPreviewHeight) {
      this.preview.width = nextPreviewWidth;
      this.preview.height = nextPreviewHeight;
    }
    if (this.canvas.width !== nextOverlayWidth || this.canvas.height !== nextOverlayHeight) {
      this.canvas.width = nextOverlayWidth;
      this.canvas.height = nextOverlayHeight;
    }
  }

  coverCrop() {
    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    const cssW = this.preview.clientWidth || this.canvas.clientWidth || this.video.clientWidth || vw;
    const cssH = this.preview.clientHeight || this.canvas.clientHeight || this.video.clientHeight || vh;
    const scale = Math.max(cssW / vw, cssH / vh);
    const scaledW = vw * scale;
    const scaledH = vh * scale;
    const cropX = (scaledW - cssW) / 2;
    const cropY = (scaledH - cssH) / 2;
    // sw/sh are the source region in native video pixels.  Use them as the
    // destination size too so captureFrame operates at full camera resolution
    // rather than the (much smaller) CSS layout dimensions.
    const sw = cssW / scale;
    const sh = cssH / scale;
    return {
      sx: cropX / scale,
      sy: cropY / scale,
      sw,
      sh,
      dw: Math.round(sw),
      dh: Math.round(sh),
    };
  }

  captureFrame() {
    this.frameCanvas.width = this.preview.width;
    this.frameCanvas.height = this.preview.height;
    this.frameCtx.drawImage(this.preview, 0, 0);
    return this.frameCanvas;
  }

  renderPreview() {
    if (!this.stream) {
      return;
    }
    this.resize();
    const { sx, sy, sw, sh } = this.coverCrop();
    const width = this.preview.width;
    const height = this.preview.height;
    if (width && height) {
      this.previewCtx.drawImage(this.video, sx, sy, sw, sh, 0, 0, width, height);
    }
    this._previewFrame = requestAnimationFrame(() => this.renderPreview());
  }

  clearOverlay() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  flashConfirmed() {
    this.canvas.classList.add("confirmed-flash");
    setTimeout(() => this.canvas.classList.remove("confirmed-flash"), 400);
  }

  drawCorners(corners, variant = "valid") {
    this.clearOverlay();
    if (!corners || corners.length !== 4) {
      return;
    }
    const width = this.canvas.width;
    const height = this.canvas.height;
    const pts = corners.map(([x, y]) => [x * width, y * height]);

    this.ctx.beginPath();
    this.ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i += 1) {
      this.ctx.lineTo(pts[i][0], pts[i][1]);
    }
    this.ctx.closePath();
    const stroke = variant === "invalid"
      ? "rgba(255, 190, 40, 0.94)"
      : "rgba(0, 230, 120, 0.92)";
    this.ctx.strokeStyle = stroke;
    this.ctx.lineWidth = Math.max(3, Math.round((window.devicePixelRatio || 1) * 2));
    this.ctx.stroke();

    this.ctx.fillStyle = stroke;
    for (const [x, y] of pts) {
      this.ctx.beginPath();
      this.ctx.arc(x, y, Math.max(5, (window.devicePixelRatio || 1) * 4), 0, Math.PI * 2);
      this.ctx.fill();
    }
  }

  describeCameraError(error) {
    if (error?.name === "NotAllowedError") {
      return "Camera permission denied";
    }
    if (error?.name === "NotFoundError") {
      return "No camera found";
    }
    if (error?.name === "NotReadableError") {
      return "Camera is busy in another app";
    }
    return error?.message || "Camera failed";
  }
}

class BrowserRuntime {
  constructor(manifest) {
    this.manifest = manifest;
    this.detector = null;
    this.embedder = null;
    this.inputNames = {};
    this.embeddings = null;
    this.cardIds = null;
    this.dewarpCanvas = document.createElement("canvas");
    this.dewarpCanvas.width = DEWARP_W;
    this.dewarpCanvas.height = DEWARP_H;
    this.dewarpCtx = this.dewarpCanvas.getContext("2d", { willReadFrequently: true });
  }

  async load(onStage) {
    const version = this.manifest.version;
    const detectorBuffer = await fetchBufferCached(
      this.urlFor(this.manifest.models.cornelius),
      version,
      (ratio, loaded, total, cached) => onStage?.("detector", ratio, loaded, total, cached),
    );
    const embedderBuffer = await fetchBufferCached(
      this.urlFor(this.manifest.models.milo),
      version,
      (ratio, loaded, total, cached) => onStage?.("embedder", ratio, loaded, total, cached),
    );

    ort.env.wasm.numThreads = 1;
    this.detector = await ort.InferenceSession.create(detectorBuffer, {
      executionProviders: ["webgpu"],
    });
    this.embedder = await ort.InferenceSession.create(embedderBuffer, {
      executionProviders: ["webgpu"],
    });

    this.inputNames.detector = this.detector.inputNames[0];
    this.inputNames.embedder = this.embedder.inputNames[0];

    const embeddingBuffer = await fetchBufferCached(
      this.urlFor(this.manifest.catalog.embeddings),
      version,
      (ratio, loaded, total, cached) => onStage?.("catalog", ratio * 0.92, loaded, total, cached),
    );
    const ids = await fetchJsonCached(
      this.urlFor(this.manifest.catalog.card_ids),
      version,
      (ratio, loaded, total, cached) => onStage?.("catalog", 0.92 + ratio * 0.08, loaded, total, cached),
    );
    this.embeddings = decodeFloat16Buffer(embeddingBuffer);
    this.cardIds = ids;
  }

  urlFor(path) {
    return `./assets/${path}`;
  }

  async detect(frameCanvas) {
    const input = fillInputTensor(frameCanvas, DETECTOR_SIZE);
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
    const dstData = this.dewarpCtx.createImageData(DEWARP_W, DEWARP_H);

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
    const input = fillInputTensor(cropCanvas, EMBEDDER_SIZE);
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
        score += this.embeddings[offset + col] * query[col];
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

function renderScanList(scans) {
  const list = document.getElementById("scan-list");
  const count = document.getElementById("scan-count");
  const total = document.getElementById("ledger-total");
  list.innerHTML = "";

  if (!scans.length) {
    const empty = document.createElement("li");
    empty.className = "scan-card";
    empty.innerHTML = `
      <p class="scan-card__title">No cards yet</p>
      <p class="scan-card__meta">Confirmed scans will appear here.</p>
    `;
    list.appendChild(empty);
    count.textContent = "0 cards";
    total.textContent = "$0.00";
    return;
  }

  for (const scan of scans) {
    const item = document.createElement("li");
    item.className = "scan-card";
    item.innerHTML = `
      <div class="scan-card__top">
        <div>
          <p class="scan-card__title">${scan.name}</p>
          <p class="scan-card__meta">${scan.setName} (${scan.setCode.toUpperCase()})</p>
        </div>
        <div class="scan-card__right">
          <span class="count-badge">${scan.count}x</span>
          <p class="scan-card__price">${formatCurrency(scan.priceUsd)}</p>
        </div>
      </div>
    `;
    list.appendChild(item);
  }

  const cardCount = scans.reduce((sum, scan) => sum + scan.count, 0);
  const totalValue = scans.reduce(
    (sum, scan) => sum + (Number.parseFloat(scan.priceUsd ?? "0") * scan.count),
    0,
  );
  count.textContent = `${cardCount} cards`;
  total.textContent = `$${totalValue.toFixed(2)}`;
}

async function fetchScryfallCard(cardId, cache) {
  if (cache.has(cardId)) {
    return cache.get(cardId);
  }
  const response = await fetch(`https://api.scryfall.com/cards/${cardId}`, {
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    throw new Error(`Scryfall lookup failed for ${cardId}: HTTP ${response.status}`);
  }
  const data = await response.json();
  cache.set(cardId, data);
  return data;
}

function ensureScanRecord(scans, cardId) {
  let scan = scans.find((entry) => entry.cardId === cardId);
  if (!scan) {
    scan = {
      cardId,
      name: cardId,
      setCode: "mtg",
      setName: "Loading...",
      priceUsd: null,
      count: 0,
    };
    scans.unshift(scan);
  }
  return scan;
}

async function loadImageToCanvas(url) {
  const image = new Image();
  image.decoding = "async";
  image.src = url;
  await image.decode();

  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(image, 0, 0);
  return canvas;
}

function setupSettingsSheet() {
  const page = document.querySelector(".page");
  const sheet = document.getElementById("settings-sheet");
  const open = document.getElementById("settings-toggle");
  const close = document.getElementById("settings-close");

  open.addEventListener("click", () => {
    sheet.hidden = false;
    page.dataset.sheetOpen = "true";
  });

  close.addEventListener("click", () => {
    sheet.hidden = true;
    delete page.dataset.sheetOpen;
  });
}

function setupViewToggle() {
  const page = document.querySelector(".page");
  const button = document.getElementById("view-toggle");
  const glyph = button.querySelector(".camera-chevron__glyph");

  page.dataset.cameraMode = "expanded";
  glyph.textContent = "⌄";
  button.setAttribute("aria-label", "Shrink camera");

  button.addEventListener("click", () => {
    const expanded = page.dataset.cameraMode === "expanded";
    page.dataset.cameraMode = expanded ? "small" : "expanded";
    button.setAttribute("aria-label", expanded ? "Expand camera" : "Shrink camera");
    glyph.textContent = "⌄";
    requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
    setTimeout(() => window.dispatchEvent(new Event("resize")), 200);
  });
}

function setupActions(scans) {
  document.getElementById("copy-list").addEventListener("click", async () => {
    const text = buildTextExport(scans);
    await navigator.clipboard.writeText(text);
  });

  document.getElementById("download-csv").addEventListener("click", () => {
    const csv = buildCsvExport(scans);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "collectorvision-scans.csv";
    link.click();
    URL.revokeObjectURL(url);
  });

  document.getElementById("clear-list").addEventListener("click", () => {
    scans.splice(0, scans.length);
    renderScanList(scans);
  });
}

function createScannerLoop(camera, runtime, scans, audioBus, manifest, debugLog) {
  const bucket = new ScanBucket();
  const scryfallCache = new Map();
  let timer = null;
  let busy = false;

  async function enrich(scan) {
    debugLog.info("fetching scryfall metadata", scan.cardId);
    const data = await fetchScryfallCard(scan.cardId, scryfallCache);
    scan.name = data.name;
    scan.setCode = data.set;
    scan.setName = data.set_name;
    scan.priceUsd = data.prices?.usd ?? null;
    renderScanList(scans);
    await audioBus.playPriceTier(scan.priceUsd);
    debugLog.info("scryfall metadata ready", data.name, data.set, data.prices?.usd ?? "n/a");
  }

  async function processFrame(frame, useBucket = true) {
    const detection = await runtime.detect(frame);
    if (!detection.cardPresent) {
      camera.drawCorners(null);
      if (useBucket) {
        bucket.push(null);
      }
      setText("camera-badge", "No card");
      return;
    }

    if (!isUsableQuad(detection.corners)) {
      camera.drawCorners(detection.corners, "invalid");
      if (useBucket) {
        bucket.push(null);
      }
      setText("camera-badge", "Bad corners");
      debugLog.warn("skipping invalid corner quad", detection.corners);
      return;
    }

    camera.drawCorners(detection.corners);
    const crop = runtime.dewarp(frame, detection.corners);
    updateCropPreview(crop);
    const embedding = await runtime.embed(crop);
    const best = runtime.search(embedding);

    if (!Number.isFinite(best.score) || best.score < MIN_MATCH_SCORE) {
      if (useBucket) {
        bucket.push(null);
      }
      setText("camera-badge", `Low match ${best.score.toFixed(2)}`);
      debugLog.info("rejecting low-confidence match", best.cardId, `score=${best.score.toFixed(4)}`);
      return;
    }

    const confirmed = useBucket
      ? bucket.push({ cardId: best.cardId, score: best.score })
      : { cardId: best.cardId, score: best.score };

    if (!confirmed) {
      return;
    }

    const scan = ensureScanRecord(scans, confirmed.cardId);
    scan.count += 1;
    renderScanList(scans);
    debugLog.info("confirmed scan", confirmed.cardId, `score=${confirmed.score.toFixed(4)}`);
    setText("camera-badge", `Match ${confirmed.score.toFixed(2)}`);
    camera.flashConfirmed();
    await audioBus.playScanConfirmed();
    if (scan.name === scan.cardId) {
      enrich(scan).catch((error) => console.warn("scryfall enrich failed", error));
    }
  }

  return {
    start() {
      if (timer) {
        return;
      }
      setText("camera-badge", "Scanning");
      timer = setInterval(() => {
        if (busy || !camera.stream) {
          return;
        }
        busy = true;
        const frame = camera.captureFrame();
        processFrame(frame, true)
          .catch((error) => {
            debugLog.error("scan tick failed", error);
            setText("camera-badge", error?.message || "Scan error");
          })
          .finally(() => {
            busy = false;
          });
      }, SCAN_INTERVAL_MS);
    },
    async runSample() {
      debugLog.info("running bundled sample frame");
      const sampleFrame = await loadImageToCanvas(`./assets/${manifest.sample_frame}`);
      setText("camera-badge", "Running sample");
      await processFrame(sampleFrame, false);
      setText("camera-badge", "Sample ready");
    },
  };
}

async function loadManifest() {
  const cached = await readCachedAsset("manifest");
  if (cached?.version) {
    setText("manifest-status", `Cached v${cached.version}`);
  }
  const response = await fetch("./assets/manifest.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load manifest: HTTP ${response.status}`);
  }
  const manifest = await response.json();
  await writeCachedAsset("manifest", manifest);
  setText("manifest-status", `Local assets v${manifest.version}`);
  return manifest;
}

function formatBytes(value) {
  if (!value) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(size >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

async function boot() {
  const scans = [];
  const audioBus = createAudioBus();
  const debugLog = createDebugLog();
  const loadingScreen = createLoadingScreen();
  const camera = new CameraSurface(debugLog);

  function setPhase(id, percent, text, note, state = "active") {
    loadingScreen.progress(percent, text);
    loadingScreen.step(id, state, note);
    debugLog.info(text, note);
  }

  renderNotes();
  renderBuildId();
  renderScanList(scans);
  setupSettingsSheet();
  setupViewToggle();
  setupActions(scans);
  audioBus.preload();
  debugLog.info("booting scanner");
  loadingScreen.start("Preparing scanner runtime");
  camera.setLoading("Loading scanner");

  assertWebGpu();
  await configureWebGpu(debugLog);
  setText("webgpu-status", "Available");
  loadingScreen.step("webgpu", "done", "Available");
  loadingScreen.progress(8, "WebGPU available");
  debugLog.info("webgpu available");

  const manifest = await loadManifest();
  renderManifestContract(manifest);
  loadingScreen.step("manifest", "done", `v${manifest.version}`);
  loadingScreen.progress(14, "Manifest loaded");
  debugLog.info("manifest loaded", manifest.version);

  loadingScreen.step("dewarp", "active", "Ready");
  loadingScreen.progress(18, "Preparing dewarp");
  await prepareDewarp();
  setText("models-status", "Loading models");
  loadingScreen.step("dewarp", "done", "Ready");
  loadingScreen.progress(24, "Dewarp ready");
  debugLog.info("dewarp ready");

  const runtime = new BrowserRuntime(manifest);
  loadingScreen.step("detector", "active", "Queued");
  loadingScreen.step("embedder", "active", "Queued");
  loadingScreen.step("catalog", "active", "Queued");
  await runtime.load((stage, ratio, loaded, total, cached) => {
    const ranges = {
      detector: [24, 44],
      embedder: [44, 60],
      catalog: [60, 96],
    };
    const [start, end] = ranges[stage];
    const percent = start + (end - start) * ratio;
    const note = cached ? "Cached" : `${formatBytes(loaded)} / ${formatBytes(total)}`;
    const label = {
      detector: "Loading corner detector",
      embedder: "Loading embedder",
      catalog: "Loading card catalog",
    }[stage];
    setPhase(stage, percent, label, note, ratio >= 1 ? "done" : "active");
  });
  setText("models-status", "Models ready");
  setText("catalog-status", `${manifest.catalog.rows} cards ready`);
  loadingScreen.progress(100, "Scanner ready");
  debugLog.info("models and catalog ready", `${manifest.catalog.rows} rows`);

  const loop = createScannerLoop(camera, runtime, scans, audioBus, manifest, debugLog);
  camera.bind(async () => {
    debugLog.info("starting scan loop");
    loop.start();
  });
  camera.setReady();
  document.getElementById("run-sample").addEventListener("click", () => {
    loop.runSample().catch((error) => {
      debugLog.error("sample run failed", error);
      setText("camera-badge", "Sample failed");
    });
  });
  loadingScreen.finish();
}

boot().catch((error) => {
  console.error(error);
  setText("webgpu-status", error.message);
  document.body.dataset.loading = "true";
  const loadingMessage = document.getElementById("loading-message");
  if (loadingMessage) {
    loadingMessage.textContent = error.message;
  }
});
