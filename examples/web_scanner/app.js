import * as ort from "./vendor/onnxruntime-web/ort.all.min.mjs";

const DETECTOR_SIZE = 384;
const EMBEDDER_SIZE = 448;
const DEWARP_W = 252;
const DEWARP_H = 352;
const MIN_SHARPNESS = 0.02;
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
];

function setText(id, value) {
  document.getElementById(id).textContent = value;
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

async function fetchJsonCached(url, version) {
  const key = `${version}:${url}:json`;
  const cached = await readCachedAsset(key);
  if (cached) {
    return cached;
  }
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
  }
  const json = await response.json();
  await writeCachedAsset(key, json);
  return json;
}

async function fetchBufferCached(url, version) {
  const key = `${version}:${url}:buffer`;
  const cached = await readCachedAsset(key);
  if (cached) {
    return cached;
  }
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
  }
  const buffer = await response.arrayBuffer();
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

function loadOpenCv() {
  if (globalThis.cv?.getPerspectiveTransform) {
    return Promise.resolve(globalThis.cv);
  }
  return new Promise((resolve, reject) => {
    globalThis.Module = {
      onRuntimeInitialized() {
        resolve(globalThis.cv);
      },
    };
    const script = document.createElement("script");
    script.src = "./vendor/opencv/opencv.js";
    script.async = true;
    script.onerror = () => reject(new Error("Failed to load opencv.js"));
    document.head.appendChild(script);
  });
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function orderCorners(points) {
  const sums = points.map(([x, y]) => x + y);
  const diffs = points.map(([x, y]) => x - y);
  return [
    points[sums.indexOf(Math.min(...sums))],
    points[diffs.indexOf(Math.min(...diffs))],
    points[sums.indexOf(Math.max(...sums))],
    points[diffs.indexOf(Math.max(...diffs))],
  ];
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
  constructor() {
    this.page = document.querySelector(".page");
    this.video = document.getElementById("camera-video");
    this.canvas = document.getElementById("camera-overlay");
    this.ctx = this.canvas.getContext("2d");
    this.badge = document.getElementById("camera-badge");
    this.startButton = document.getElementById("camera-start");
    this.stream = null;
    this.frameCanvas = document.createElement("canvas");
    this.frameCtx = this.frameCanvas.getContext("2d", { willReadFrequently: true });
    this._resizeHandler = () => this.resize();
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
        console.error(error);
        this.badge.textContent = this.describeCameraError(error);
        this.startButton.disabled = false;
      }
    });
  }

  setLoading(message) {
    this.badge.textContent = message;
    this.startButton.disabled = true;
  }

  setReady() {
    this.badge.textContent = "Ready to start camera";
    this.startButton.disabled = false;
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
    this.startButton.hidden = true;
    this.resize();
    window.addEventListener("resize", this._resizeHandler);
  }

  resize() {
    const width = this.video.clientWidth || this.video.videoWidth;
    const height = this.video.clientHeight || this.video.videoHeight;
    if (!width || !height) {
      return;
    }
    this.canvas.width = width;
    this.canvas.height = height;
  }

  coverCrop() {
    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    const cssW = this.canvas.clientWidth || this.canvas.width;
    const cssH = this.canvas.clientHeight || this.canvas.height;
    const scale = Math.max(cssW / vw, cssH / vh);
    const scaledW = vw * scale;
    const scaledH = vh * scale;
    const cropX = (scaledW - cssW) / 2;
    const cropY = (scaledH - cssH) / 2;
    return {
      sx: cropX / scale,
      sy: cropY / scale,
      sw: cssW / scale,
      sh: cssH / scale,
      dw: cssW,
      dh: cssH,
    };
  }

  captureFrame() {
    const { sx, sy, sw, sh, dw, dh } = this.coverCrop();
    this.frameCanvas.width = dw;
    this.frameCanvas.height = dh;
    this.frameCtx.drawImage(this.video, sx, sy, sw, sh, 0, 0, dw, dh);
    return this.frameCanvas;
  }

  clearOverlay() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  flashConfirmed() {
    this.canvas.classList.add("confirmed-flash");
    setTimeout(() => this.canvas.classList.remove("confirmed-flash"), 400);
  }

  drawCorners(corners) {
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
    this.ctx.strokeStyle = "rgba(0, 230, 120, 0.92)";
    this.ctx.lineWidth = 3;
    this.ctx.stroke();

    this.ctx.fillStyle = "rgba(0, 230, 120, 0.92)";
    for (const [x, y] of pts) {
      this.ctx.beginPath();
      this.ctx.arc(x, y, 5, 0, Math.PI * 2);
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
  constructor(manifest, cv) {
    this.manifest = manifest;
    this.cv = cv;
    this.detector = null;
    this.embedder = null;
    this.inputNames = {};
    this.embeddings = null;
    this.cardIds = null;
    this.dewarpCanvas = document.createElement("canvas");
    this.dewarpCanvas.width = DEWARP_W;
    this.dewarpCanvas.height = DEWARP_H;
  }

  async load() {
    const version = this.manifest.version;
    const detectorBuffer = await fetchBufferCached(this.urlFor(this.manifest.models.cornelius), version);
    const embedderBuffer = await fetchBufferCached(this.urlFor(this.manifest.models.milo), version);

    ort.env.wasm.numThreads = 1;
    this.detector = await ort.InferenceSession.create(detectorBuffer, {
      executionProviders: ["webgpu"],
    });
    this.embedder = await ort.InferenceSession.create(embedderBuffer, {
      executionProviders: ["webgpu"],
    });

    this.inputNames.detector = this.detector.inputNames[0];
    this.inputNames.embedder = this.embedder.inputNames[0];

    const embeddingBuffer = await fetchBufferCached(this.urlFor(this.manifest.catalog.embeddings), version);
    const ids = await fetchJsonCached(this.urlFor(this.manifest.catalog.card_ids), version);
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
    const cv = this.cv;
    const src = cv.imread(frameCanvas);
    const dst = new cv.Mat();
    const width = frameCanvas.width;
    const height = frameCanvas.height;

    const srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
      corners[0][0] * width, corners[0][1] * height,
      corners[1][0] * width, corners[1][1] * height,
      corners[2][0] * width, corners[2][1] * height,
      corners[3][0] * width, corners[3][1] * height,
    ]);
    const dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
      0, 0,
      DEWARP_W - 1, 0,
      DEWARP_W - 1, DEWARP_H - 1,
      0, DEWARP_H - 1,
    ]);
    const matrix = cv.getPerspectiveTransform(srcPts, dstPts);
    const size = new cv.Size(DEWARP_W, DEWARP_H);
    cv.warpPerspective(src, dst, matrix, size, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
    cv.imshow(this.dewarpCanvas, dst);

    src.delete();
    dst.delete();
    srcPts.delete();
    dstPts.delete();
    matrix.delete();

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

  page.dataset.cameraMode = "small";

  button.addEventListener("click", () => {
    const expanded = page.dataset.cameraMode === "expanded";
    page.dataset.cameraMode = expanded ? "small" : "expanded";
    button.textContent = expanded ? "Expand" : "Shrink";
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

function createScannerLoop(camera, runtime, scans, audioBus, manifest) {
  const bucket = new ScanBucket();
  const scryfallCache = new Map();
  let timer = null;
  let busy = false;

  async function enrich(scan) {
    const data = await fetchScryfallCard(scan.cardId, scryfallCache);
    scan.name = data.name;
    scan.setCode = data.set;
    scan.setName = data.set_name;
    scan.priceUsd = data.prices?.usd ?? null;
    renderScanList(scans);
    await audioBus.playPriceTier(scan.priceUsd);
  }

  async function processFrame(frame, useBucket = true) {
    const detection = await runtime.detect(frame);
    if (!detection.cardPresent) {
      camera.drawCorners(null);
      if (useBucket) {
        bucket.push(null);
      }
      return;
    }

    camera.drawCorners(detection.corners);
    const crop = runtime.dewarp(frame, detection.corners);
    const embedding = await runtime.embed(crop);
    const best = runtime.search(embedding);

    const confirmed = useBucket
      ? bucket.push({ cardId: best.cardId, score: best.score })
      : { cardId: best.cardId, score: best.score };

    if (!confirmed) {
      return;
    }

    const scan = ensureScanRecord(scans, confirmed.cardId);
    scan.count += 1;
    renderScanList(scans);
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
            console.error(error);
            setText("camera-badge", "Scan error");
          })
          .finally(() => {
            busy = false;
          });
      }, SCAN_INTERVAL_MS);
    },
    async runSample() {
      const sampleFrame = await loadImageToCanvas(`./assets/${manifest.sample_frame}`);
      setText("camera-badge", "Running sample");
      await processFrame(sampleFrame, false);
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

async function boot() {
  const scans = [];
  const audioBus = createAudioBus();
  const camera = new CameraSurface();

  renderNotes();
  renderScanList(scans);
  setupSettingsSheet();
  setupViewToggle();
  setupActions(scans);
  audioBus.preload();
  camera.setLoading("Loading scanner");

  assertWebGpu();
  setText("webgpu-status", "Available");

  const manifest = await loadManifest();
  renderManifestContract(manifest);

  const cv = await loadOpenCv();
  setText("models-status", "Loading models");

  const runtime = new BrowserRuntime(manifest, cv);
  await runtime.load();
  setText("models-status", "Models ready");
  setText("catalog-status", `${manifest.catalog.rows} cards ready`);

  const loop = createScannerLoop(camera, runtime, scans, audioBus, manifest);
  camera.bind(async () => {
    loop.start();
  });
  camera.setReady();
  document.getElementById("run-sample").addEventListener("click", () => {
    loop.runSample().catch((error) => {
      console.error(error);
      setText("camera-badge", "Sample failed");
    });
  });
}

boot().catch((error) => {
  console.error(error);
  setText("webgpu-status", error.message);
});
