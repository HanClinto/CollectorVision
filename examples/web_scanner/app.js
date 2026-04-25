// Replaced by the deploy-pages CI workflow with the actual short commit SHA.
const BUILD_ID = "__BUILD_ID__";

const GITHUB_REPO = "HanClinto/CollectorVision";

// DETECTOR_SIZE is kept here for the capture-bundle debug export.
const DETECTOR_SIZE = 384;
const MIN_MATCH_SCORE = 0.70;
const PREVIEW_ASPECT = 16 / 9;
const SCAN_INTERVAL_MS = 900;

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
  "WebGPU is used when available; falls back to WASM-only inference automatically.",
  "Scryfall enrichment runs after confirmation so the recognition loop stays local.",
  "Settings include a bundled sample-frame smoke test for local bring-up.",
  "scan.wav fires on confirm; price-tier sounds fire after Scryfall returns.",
  "Perspective dewarp runs locally in JS so startup stays simple and self-contained.",
];

const LOADING_STEPS = [
  { id: "webgpu", label: "Configuring inference" },
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
  const loading = document.getElementById("loading-build");
  if (loading) {
    loading.textContent = `build ${BUILD_ID}`;
  }
  const settings = document.getElementById("settings-build");
  if (settings) {
    settings.textContent = BUILD_ID;
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

function updateCropPreview(cropBitmap) {
  const wrapper = document.getElementById("crop-preview");
  const target = document.getElementById("crop-canvas");
  if (!wrapper || !target) {
    return;
  }
  const ctx = target.getContext("2d");
  ctx.drawImage(cropBitmap, 0, 0);
  wrapper.hidden = false;
}

function updateDetectorPreview(detectorBitmap) {
  const wrapper = document.getElementById("detector-preview");
  const target = document.getElementById("detector-canvas");
  if (!wrapper || !target) {
    return;
  }
  const ctx = target.getContext("2d");
  ctx.drawImage(detectorBitmap, 0, 0);
  wrapper.hidden = false;
}

// Collect browser / device / camera environment info for the capture bundle
// and for pre-populating GitHub bug reports.
function collectSystemInfo(camera) {
  const info = {
    userAgent: navigator.userAgent,
    platform: navigator.platform || null,
    language: navigator.language,
    hardwareConcurrency: navigator.hardwareConcurrency ?? null,
    deviceMemory: navigator.deviceMemory ?? null,   // Chrome only; undefined elsewhere
    maxTouchPoints: navigator.maxTouchPoints ?? 0,
    screen: {
      width: screen.width,
      height: screen.height,
      colorDepth: screen.colorDepth,
    },
    viewport: {
      width: window.innerWidth,
      height: window.innerHeight,
    },
    webgpuAvailable: typeof navigator.gpu !== "undefined",
    cameraTrackSettings: null,
  };
  if (camera?.stream) {
    const track = camera.stream.getVideoTracks()[0];
    if (track) {
      info.cameraTrackSettings = track.getSettings();
    }
  }
  return info;
}

// Build a GitHub new-issue URL pre-populated with a markdown summary table.
// The title encodes the capture ID so the reporter knows which file to attach.
function buildIssueUrl(captureId, systemInfo) {
  const ua = systemInfo.userAgent.length > 160
    ? systemInfo.userAgent.slice(0, 160) + "\u2026"
    : systemInfo.userAgent;

  const ts = systemInfo.cameraTrackSettings;
  const trackStr = ts
    ? [
        ts.width && ts.height ? `${ts.width}×${ts.height}` : null,
        ts.frameRate ? `@${Math.round(ts.frameRate)}fps` : null,
        ts.facingMode ? `(${ts.facingMode})` : null,
      ].filter(Boolean).join(" ")
    : "—";

  const lines = [
    "## Scanner Bug Report",
    "",
    `**Capture file:** \`${captureId}.json.gz\` *(please attach this file below)*`,
    "",
    "**Problem description:**",
    "<!-- Describe what went wrong and what card you were scanning -->",
    "",
    "**Expected card (name and/or Scryfall ID):**",
    "<!-- e.g. \"Lightning Bolt\" or \"3fabf99f-3a2e-45f6-88e9-cfa3b5b1f24c\" — leave blank if unknown -->",
    "",
    "| Field | Value |",
    "|---|---|",
    `| Build | \`${BUILD_ID}\` |`,
    `| User Agent | ${ua} |`,
    `| Platform | ${systemInfo.platform ?? "—"} |`,
    `| Language | ${systemInfo.language} |`,
    `| Viewport | ${systemInfo.viewport.width}×${systemInfo.viewport.height} |`,
    `| Screen | ${systemInfo.screen.width}×${systemInfo.screen.height} (${systemInfo.screen.colorDepth}bpp) |`,
    `| DPR | ${window.devicePixelRatio ?? 1} |`,
    `| Touch points | ${systemInfo.maxTouchPoints} |`,
    `| CPU cores | ${systemInfo.hardwareConcurrency ?? "—"} |`,
    `| Device memory | ${systemInfo.deviceMemory != null ? `${systemInfo.deviceMemory} GB` : "—"} |`,
    `| Camera track | ${trackStr} |`,
    `| WebGPU | ${systemInfo.webgpuAvailable ? "available" : "unavailable"} |`,
  ];

  const params = new URLSearchParams({
    title: `Bug report: ${captureId}`,
    body: lines.join("\n"),
    labels: "bug",
  });

  return `https://github.com/${GITHUB_REPO}/issues/new?${params}`;
}

// captureState is an object maintained by createScannerLoop:
//   { lastResult, lastDetectorBitmap, lastCropBitmap }
// lastDetectorBitmap / lastCropBitmap are ImageBitmaps transferred from the
// scanner worker and may be null until the first successful detection.
function setupCaptureButton(camera, captureState) {
  const btn = document.getElementById("capture-frame");
  if (!btn) {
    return;
  }

  btn.addEventListener("click", () => {
    if (!camera.stream) {
      btn.textContent = "No stream";
      setTimeout(() => { btn.textContent = "Capture"; }, 1500);
      return;
    }

    const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const captureId = `cv_${ts}`;
    btn.textContent = "\u2026";

    // Register the callback before setting the flag to avoid a race on fast
    // hardware where the worker result arrives before onCapture is set.
    captureState.onCapture = async (data) => {
      try {
        // Draw the captured frame bitmap to a canvas for PNG encoding.
        // framePng, detectorInputRgba, and orderedCorners all come from the
        // same atomic pipeline run (no cross-tick timing race).
        const frameBitmap = data.captureFrameBitmap;
        const snapshotCanvas = document.createElement("canvas");
        snapshotCanvas.width = frameBitmap.width;
        snapshotCanvas.height = frameBitmap.height;
        snapshotCanvas.getContext("2d").drawImage(frameBitmap, 0, 0);
        frameBitmap.close();

        const logEntries = Array.from(
          document.querySelectorAll("#debug-log .debug-entry"),
        ).reverse().map((el) => ({
          level: el.dataset.level,
          meta: el.querySelector(".debug-entry__meta")?.textContent ?? "",
          message: el.querySelector(".debug-entry__message")?.textContent ?? "",
        }));

        // Full-resolution frame — what Python uses to re-run the pipeline.
        const dataUrl = snapshotCanvas.toDataURL("image/png");
        const framePng = dataUrl.slice(dataUrl.indexOf(",") + 1);

        // Raw RGBA pixel bytes from the 384×384 detector input bitmap transferred
        // from the scanner worker.  Stored as base64-encoded Uint8ClampedArray
        // (no PNG encoding, no color-space metadata) so Python can reconstruct
        // exact values with np.frombuffer(...).reshape(384, 384, 4).
        let detectorInputRgba = null;
        if (data.detectorBitmap) {
          const detScratch = document.createElement("canvas");
          detScratch.width = DETECTOR_SIZE;
          detScratch.height = DETECTOR_SIZE;
          detScratch.getContext("2d", { willReadFrequently: true })
            .drawImage(data.detectorBitmap, 0, 0);
          const detInputImageData = detScratch.getContext("2d", { willReadFrequently: true })
            .getImageData(0, 0, DETECTOR_SIZE, DETECTOR_SIZE);
          // Avoid spreading 589 824 bytes as call arguments (stack overflow on mobile).
          const detInputBytes = new Uint8Array(detInputImageData.data.buffer);
          let detInputBinary = "";
          for (let i = 0; i < detInputBytes.length; i++) {
            detInputBinary += String.fromCharCode(detInputBytes[i]);
          }
          detectorInputRgba = btoa(detInputBinary);
        }

        const systemInfo = collectSystemInfo(camera);

        const bundle = {
          captureId,
          buildId: BUILD_ID,
          timestamp: new Date().toISOString(),
          // Expected Scryfall card ID — null until manually identified by a developer.
          // Set this field when filing a regression capture so the test suite can
          // assert the correct identity once the bug is fixed.
          expectedCardId: null,
          systemInfo,
          videoSensor: {
            width: camera.video.videoWidth,
            height: camera.video.videoHeight,
          },
          // processCanvas pixel dimensions — what the worker receives as a bitmap.
          processCanvas: {
            width: snapshotCanvas.width,
            height: snapshotCanvas.height,
          },
          devicePixelRatio: window.devicePixelRatio || 1,
          detectorSize: DETECTOR_SIZE,
          inferenceMode: captureState.inferenceMode ?? null,
          detectorInput: data.detectorInput ?? null,
          rawCorners: data.rawCorners ?? null,
          orderedCorners: data.corners ? data.corners.map(([x, y]) => ({ x, y })) : null,
          sharpness: data.sharpness ?? null,
          cardPresent: data.cardPresent ?? null,
          // JS pipeline result — compare jsScore to Python re-run score to detect
          // embedding divergence between JS (WebGPU/WASM) and Python (CPU ONNX).
          jsCardId: data.cardId ?? null,
          jsScore: data.score ?? null,
          consoleLog: logEntries,
          // Python: cv2.imdecode(np.frombuffer(base64.b64decode(bundle["framePng"]), np.uint8), cv2.IMREAD_COLOR)
          framePng,
          // Decode in Python: np.frombuffer(base64.b64decode(bundle["detectorInputRgba"]), np.uint8).reshape(384, 384, 4)
          // Compare with python-detector-input.npy to find preprocessing divergence.
          detectorInputRgba,
        };

        // Gzip-compress the JSON bundle using the built-in CompressionStream API
        // (Chrome 80+, all modern Android browsers) and download as a single file.
        const jsonBytes = new TextEncoder().encode(JSON.stringify(bundle));
        const cs = new CompressionStream("gzip");
        const writer = cs.writable.getWriter();
        writer.write(jsonBytes);
        writer.close();

        const chunks = [];
        const reader = cs.readable.getReader();
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            break;
          }
          chunks.push(value);
        }

        const compressed = new Blob(chunks, { type: "application/gzip" });
        const url = URL.createObjectURL(compressed);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${captureId}.json.gz`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 5000);

        // Reveal the Report link with a pre-populated GitHub issue URL.
        const reportLink = document.getElementById("report-issue");
        if (reportLink) {
          reportLink.href = buildIssueUrl(captureId, systemInfo);
          reportLink.hidden = false;
        }

        btn.textContent = "Saved!";
        setTimeout(() => { btn.textContent = "Capture"; }, 2000);
      } catch (err) {
        btn.textContent = "Error";
        console.error("capture failed", err);
        setTimeout(() => { btn.textContent = "Capture"; }, 2000);
      }
    };

    captureState.pendingCapture = true;
  });
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

// Writes live geometry values into the Settings sheet so they are readable on
// mobile (open Settings → Sensor & Layout).  Silently ignores unknown IDs so
// it is safe to call before the DOM is fully built.
function createDiagnostics() {
  const IDS = [
    "diag-video", "diag-video-aspect", "diag-source-crop",
    "diag-process-canvas", "diag-display-canvas", "diag-dpr",
    "diag-detector-input", "diag-raw-corners", "diag-corners", "diag-sharpness",
  ];
  const LABELS = {
    "diag-video": "videoSensor",
    "diag-video-aspect": "videoAspect",
    "diag-source-crop": "sourceCrop",
    "diag-process-canvas": "processCanvas",
    "diag-display-canvas": "displayCanvas",
    "diag-dpr": "devicePixelRatio",
    "diag-detector-input": "detectorInput",
    "diag-raw-corners": "rawCorners",
    "diag-corners": "lastCorners",
    "diag-sharpness": "lastSharpness",
  };
  const els = {};
  for (const id of IDS) {
    els[id] = document.getElementById(id);
  }

  const copyBtn = document.getElementById("copy-diag");
  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      const data = { buildId: BUILD_ID };
      for (const id of IDS) {
        data[LABELS[id]] = els[id]?.textContent ?? "—";
      }
      try {
        await navigator.clipboard.writeText(JSON.stringify(data, null, 2));
        const prev = copyBtn.textContent;
        copyBtn.textContent = "Copied!";
        setTimeout(() => { copyBtn.textContent = prev; }, 1500);
      } catch {
        copyBtn.textContent = "Failed";
        setTimeout(() => { copyBtn.textContent = "Copy"; }, 1500);
      }
    });
  }

  return {
    set(id, value) {
      const el = els[id];
      if (el) {
        el.textContent = value;
      }
    },
  };
}

class CameraSurface {
  constructor(debugLog, diag) {
    this.page = document.querySelector(".page");
    this.video = document.getElementById("camera-video");
    this.preview = document.getElementById("camera-preview");
    this.previewCtx = this.preview.getContext("2d");
    this.processCanvas = document.createElement("canvas");
    this.processCtx = this.processCanvas.getContext("2d", { willReadFrequently: true });
    this.canvas = document.getElementById("camera-overlay");
    this.ctx = this.canvas.getContext("2d");
    this.badge = document.getElementById("camera-badge");
    this.startButton = document.getElementById("camera-start");
    this.debugLog = debugLog;
    this.diag = diag;
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
    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    // Use the actual video aspect ratio rather than the hardcoded 16/9
    // PREVIEW_ASPECT so that portrait cameras (e.g. Android 720×1280) are not
    // squeezed into a landscape process canvas.
    const frameAspect = (vw && vh) ? vw / vh : PREVIEW_ASPECT;
    const nextDisplayWidth = Math.round(width * dpr);
    const nextDisplayHeight = Math.round(height * dpr);
    const nextProcessWidth = Math.round(Math.max(width, height * frameAspect) * dpr);
    const nextProcessHeight = Math.round(nextProcessWidth / frameAspect);
    if (this.preview.width !== nextDisplayWidth || this.preview.height !== nextDisplayHeight) {
      this.preview.width = nextDisplayWidth;
      this.preview.height = nextDisplayHeight;
    }
    if (this.canvas.width !== nextDisplayWidth || this.canvas.height !== nextDisplayHeight) {
      this.canvas.width = nextDisplayWidth;
      this.canvas.height = nextDisplayHeight;
    }
    if (
      this.processCanvas.width !== nextProcessWidth
      || this.processCanvas.height !== nextProcessHeight
    ) {
      this.processCanvas.width = nextProcessWidth;
      this.processCanvas.height = nextProcessHeight;
    }

    if (vw && vh) {
      this.diag.set("diag-video", `${vw} × ${vh}`);
      this.diag.set("diag-video-aspect", `${(vw / vh).toFixed(3)} (process ${frameAspect.toFixed(3)})`);
    }
    this.diag.set("diag-dpr", String(dpr));
    this.diag.set("diag-process-canvas", `${nextProcessWidth} × ${nextProcessHeight}`);
    this.diag.set("diag-display-canvas", `${nextDisplayWidth} × ${nextDisplayHeight}`);
    this.debugLog.info(
      "resize",
      `video=${vw}×${vh}`,
      `process=${nextProcessWidth}×${nextProcessHeight}`,
      `display=${nextDisplayWidth}×${nextDisplayHeight}`,
      `dpr=${dpr}`,
    );
  }

  sourceCrop() {
    // Use the full video frame — do not crop to a fixed aspect ratio.
    // A portrait camera (e.g. Android rear camera at 720×1280) would lose
    // ~68% of its height if cropped to 16:9 here.
    return {
      sx: 0,
      sy: 0,
      sw: this.video.videoWidth,
      sh: this.video.videoHeight,
    };
  }

  captureFrame() {
    this.frameCanvas.width = this.processCanvas.width;
    this.frameCanvas.height = this.processCanvas.height;
    this.frameCtx.drawImage(this.processCanvas, 0, 0);
    return this.frameCanvas;
  }

  renderPreview() {
    if (!this.stream) {
      return;
    }
    this.resize();
    const { sx, sy, sw, sh } = this.sourceCrop();
    this.diag.set("diag-source-crop", `${Math.round(sx)},${Math.round(sy)} → ${Math.round(sw)}×${Math.round(sh)}`);
    const processWidth = this.processCanvas.width;
    const processHeight = this.processCanvas.height;
    if (processWidth && processHeight) {
      this.processCtx.drawImage(this.video, sx, sy, sw, sh, 0, 0, processWidth, processHeight);
    }

    const displayWidth = this.preview.width;
    const displayHeight = this.preview.height;
    if (displayWidth && displayHeight && processWidth && processHeight) {
      const scale = Math.max(displayWidth / processWidth, displayHeight / processHeight);
      const drawWidth = processWidth * scale;
      const drawHeight = processHeight * scale;
      const offsetX = (displayWidth - drawWidth) / 2;
      const offsetY = (displayHeight - drawHeight) / 2;
      this.previewCtx.clearRect(0, 0, displayWidth, displayHeight);
      this.previewCtx.drawImage(
        this.processCanvas,
        offsetX,
        offsetY,
        drawWidth,
        drawHeight,
      );
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
    const processWidth = this.processCanvas.width;
    const processHeight = this.processCanvas.height;
    const scale = Math.max(width / processWidth, height / processHeight);
    const offsetX = (processWidth * scale - width) / 2;
    const offsetY = (processHeight * scale - height) / 2;
    const pts = corners.map(([x, y]) => [
      x * processWidth * scale - offsetX,
      y * processHeight * scale - offsetY,
    ]);

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

// createScannerLoop wires together the scanner and enricher workers with the
// camera surface, scan bucket, audio bus, and UI.  All heavy computation
// happens inside the workers; this function only handles message routing and
// DOM updates.
//
// captureState is a shared object updated on every result message:
//   { lastResult, lastDetectorBitmap, lastCropBitmap }
function createScannerLoop(
  camera, scannerWorker, enricherWorker, scans, audioBus, manifest, debugLog, diag, captureState,
) {
  const bucket = new ScanBucket();
  let timer = null;
  let workerBusy = false;

  // Enricher results arrive here on the main thread.
  enricherWorker.addEventListener("message", async ({ data }) => {
    if (data.type === "enriched") {
      const scan = scans.find((s) => s.cardId === data.cardId);
      if (scan) {
        scan.name = data.name;
        scan.setCode = data.set;
        scan.setName = data.setName;
        scan.priceUsd = data.priceUsd;
        renderScanList(scans);
        await audioBus.playPriceTier(data.priceUsd);
        debugLog.info("scryfall metadata ready", data.name, data.set, data.priceUsd ?? "n/a");
      }
    } else if (data.type === "enrichError") {
      debugLog.warn("scryfall enrich failed", data.cardId, data.message);
    }
  });

  // Scanner results arrive here.  The worker has already done detect/dewarp/
  // embed/search; this handler only does UI + bucket + confirm logic.
  scannerWorker.addEventListener("message", async ({ data }) => {
    if (data.type !== "result") {
      return;
    }

    workerBusy = false;

    // Dispatch pending capture callback before any early returns.
    if (data.captureRequested && captureState.onCapture) {
      const cb = captureState.onCapture;
      captureState.onCapture = null;
      cb(data).catch((err) => console.warn("capture callback failed", err));
    }

    // Cache state for the capture button and update debug previews.
    captureState.lastResult = data;
    if (data.detectorBitmap) {
      captureState.lastDetectorBitmap = data.detectorBitmap;
      updateDetectorPreview(data.detectorBitmap);
    }
    if (data.cropBitmap) {
      captureState.lastCropBitmap = data.cropBitmap;
      updateCropPreview(data.cropBitmap);
    }

    diag.set("diag-detector-input", data.detectorInput ?? "—");
    diag.set("diag-raw-corners", data.rawCorners ?? "—");
    diag.set("diag-sharpness", `${data.sharpness?.toFixed(3) ?? "—"} (card ${data.cardPresent ? "yes" : "no"})`);

    if (!data.cardPresent) {
      camera.drawCorners(null);
      bucket.push(null);
      setText("camera-badge", "No card");
      return;
    }

    if (!data.cornersValid) {
      camera.drawCorners(data.corners, "invalid");
      bucket.push(null);
      setText("camera-badge", "Bad corners");
      debugLog.warn("skipping invalid corner quad", data.corners);
      return;
    }

    camera.drawCorners(data.corners);
    diag.set("diag-corners", data.corners.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join("  "));

    if (!Number.isFinite(data.score) || data.score < MIN_MATCH_SCORE) {
      bucket.push(null);
      setText("camera-badge", `Low match ${data.score?.toFixed(2) ?? "—"}`);
      debugLog.info("rejecting low-confidence match", data.cardId, `score=${data.score?.toFixed(4)}`);
      return;
    }

    const confirmed = bucket.push({ cardId: data.cardId, score: data.score });
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
      enricherWorker.postMessage({ type: "enrich", cardId: confirmed.cardId });
    }
  });

  return {
    start() {
      if (timer) {
        return;
      }
      setText("camera-badge", "Scanning");
      timer = setInterval(async () => {
        if (workerBusy || !camera.stream) {
          return;
        }
        workerBusy = true;
        try {
          const captureRequested = captureState.pendingCapture;
          if (captureRequested) captureState.pendingCapture = false;
          const bitmap = await createImageBitmap(camera.processCanvas);
          scannerWorker.postMessage({ type: "frame", bitmap, captureRequested }, [bitmap]);
        } catch (error) {
          workerBusy = false;
          debugLog.error("scan tick failed", error);
          setText("camera-badge", error?.message || "Scan error");
        }
      }, SCAN_INTERVAL_MS);
    },
    runSample() {
      debugLog.info("running bundled sample frame");
      setText("camera-badge", "Running sample");
      scannerWorker.postMessage({ type: "sample", url: `./assets/${manifest.sample_frame}` });
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
  const diag = createDiagnostics();
  const loadingScreen = createLoadingScreen();
  const camera = new CameraSurface(debugLog, diag);

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

  // Canary check: fetch build.json fresh from the server (bypassing all
  // caches) and compare its buildId to ours.  If they differ, this page is
  // stale — redirect to ?v=<live build ID> which the browser has never cached,
  // forcing a fresh index.html fetch and picking up all updated sub-resources.
  try {
    const canary = await fetch(`./build.json?_=${Date.now()}`, { cache: "no-store" })
      .then((r) => r.json());
    if (canary.buildId && canary.buildId !== BUILD_ID) {
      const params = new URLSearchParams(location.search);
      params.set("v", canary.buildId);
      location.replace(location.pathname + "?" + params.toString() + location.hash);
      return; // halt boot — the redirect is in flight
    }
  } catch {
    // Network failure (offline, etc.) — proceed with whatever we have.
  }

  // Load the manifest on the main thread first — it drives both the loading
  // screen text and the worker init message.
  const manifest = await loadManifest();
  renderManifestContract(manifest);
  loadingScreen.step("manifest", "done", `v${manifest.version}`);
  loadingScreen.progress(14, "Manifest loaded");
  debugLog.info("manifest loaded", manifest.version);

  // Create two workers.  The scanner worker does all GPU/CPU inference;
  // the enricher worker handles Scryfall price lookups independently.
  const scannerWorkerUrl = new URL(`./scanner.worker.mjs?v=${BUILD_ID}`, import.meta.url);
  const scannerWorker = new Worker(scannerWorkerUrl, { type: "module" });
  const enricherWorkerUrl = new URL(`./enricher.worker.mjs?v=${BUILD_ID}`, import.meta.url);
  const enricherWorker = new Worker(enricherWorkerUrl, { type: "module" });

  // Wire up init-phase progress messages before posting 'init'.
  const scannerReady = new Promise((resolve, reject) => {
    function onInitMessage({ data }) {
      if (data.type === "progress") {
        if (data.stage === "webgpu") {
          const mode = data.inferenceMode;
          setText("webgpu-status", mode === "WebGPU" ? "Available" : "WASM fallback");
          loadingScreen.step("webgpu", "done", mode);
          loadingScreen.progress(20, `Inference: ${mode}`);
          debugLog.info("inference configured", mode);
        } else if (data.stage === "dewarp") {
          loadingScreen.step("dewarp", "done", "Ready");
          loadingScreen.progress(24, "Dewarp ready");
          debugLog.info("dewarp ready");
        } else {
          const ranges = { detector: [24, 44], embedder: [44, 60], catalog: [60, 96] };
          const [start, end] = ranges[data.stage] ?? [0, 0];
          const percent = start + (end - start) * data.ratio;
          const note = data.cached ? "Cached" : `${formatBytes(data.loaded)} / ${formatBytes(data.total)}`;
          const label = {
            detector: "Loading corner detector",
            embedder: "Loading embedder",
            catalog: "Loading card catalog",
          }[data.stage];
          setPhase(data.stage, percent, label, note, data.ratio >= 1 ? "done" : "active");
        }
      } else if (data.type === "ready") {
        scannerWorker.removeEventListener("message", onInitMessage);
        resolve(data.inferenceMode);
      } else if (data.type === "error") {
        scannerWorker.removeEventListener("message", onInitMessage);
        reject(new Error(data.message));
      }
    }
    scannerWorker.addEventListener("message", onInitMessage);
  });

  loadingScreen.step("webgpu", "active", "Configuring");
  loadingScreen.step("dewarp", "active", "Queued");
  loadingScreen.step("detector", "active", "Queued");
  loadingScreen.step("embedder", "active", "Queued");
  loadingScreen.step("catalog", "active", "Queued");
  setText("models-status", "Loading models");

  scannerWorker.postMessage({ type: "init", manifest });
  const inferenceMode = await scannerReady;

  setText("models-status", "Models ready");
  setText("catalog-status", `${manifest.catalog.rows} cards ready`);
  loadingScreen.progress(100, "Scanner ready");
  debugLog.info("models and catalog ready", `${manifest.catalog.rows} rows`);

  // captureState is updated by the scanner result handler inside createScannerLoop.
  const captureState = { lastResult: null, lastDetectorBitmap: null, lastCropBitmap: null, pendingCapture: false, onCapture: null, inferenceMode };

  const loop = createScannerLoop(
    camera, scannerWorker, enricherWorker, scans, audioBus, manifest, debugLog, diag, captureState,
  );
  camera.bind(async () => {
    debugLog.info("starting scan loop");
    loop.start();
  });
  camera.setReady();
  document.getElementById("run-sample").addEventListener("click", () => {
    loop.runSample();
  });
  setupCaptureButton(camera, captureState);
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
