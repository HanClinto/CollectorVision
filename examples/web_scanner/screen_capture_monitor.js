const BUILD_ID = "__BUILD_ID__";
const CHANNEL_NAME = "collectorvision-monitor";
const ASSET_CHANNELS = {
  stable: "./assets",
  testing: "./testing/assets",
};
const ROI_KEY = "collectorvision_screen_monitor_roi";
const SETTINGS_KEY = "collectorvision_screen_monitor_settings";
const MAX_EVENTS = 250;
const KNOWN_MODEL_VERSIONS = {
  cornelius: {
    "sha256:a90ee87a45781e09e9fc88508162dac87b0492162dff79a2627c52ae773e6a79": "1.205",
    "sha256:8b2edd885d4c813157e15858d3e9a19806e994fcc98f423fc3442cdf704c69ec": "1.210",
    "sha256:56752ee6149ec5e02258c0886350f0ed55bdbab6f112c27b128a61e4e32a7834": "1.221",
    "sha256:650da3cc3e9ac778c6951de631f824ec1e63bdabf3aaa39a35d7435af625612e": "2.12",
  },
  milo: {
    "sha256:bd13d8d60383c69da04dce261f32e93fdaeaa8fd618fbc991e7385f71b3d45df": "1.0.0",
  },
};
const DEFAULT_SETTINGS = {
  matchThreshold: 0.50,
  consecutiveMatches: 2,
  scanIntervalMs: 500,
  minCornerConfidence: 0.02,
  groupBySecondaryId: true,
  showRejectedDetections: true,
};
const DEFAULT_ROI = { x: 0.15, y: 0.15, width: 0.70, height: 0.70 };

const els = {
  captureToggle: document.getElementById("capture-toggle"),
  resetRoi: document.getElementById("reset-roi"),
  sourceMeta: document.getElementById("source-meta"),
  stage: document.getElementById("capture-stage"),
  video: document.getElementById("capture-video"),
  preview: document.getElementById("preview-canvas"),
  overlay: document.getElementById("corner-overlay"),
  versionCornelius: document.getElementById("version-cornelius"),
  versionMilo: document.getElementById("version-milo"),
  versionCatalog: document.getElementById("version-catalog"),
  roiBox: document.getElementById("roi-box"),
  stageStatus: document.getElementById("stage-status"),
  workerStatus: document.getElementById("worker-status"),
  liveResultTitle: document.getElementById("live-result-title"),
  latestCard: document.getElementById("latest-card"),
  latestScore: document.getElementById("latest-score"),
  latestSharpness: document.getElementById("latest-sharpness"),
  latestStatus: document.getElementById("latest-status"),
  confirmedCount: document.getElementById("confirmed-count"),
  matchThreshold: document.getElementById("match-threshold"),
  consecutiveMatches: document.getElementById("consecutive-matches"),
  scanInterval: document.getElementById("scan-interval"),
  scanIntervalValue: document.getElementById("scan-interval-value"),
  cornerThreshold: document.getElementById("corner-threshold"),
  matchSignalFill: document.getElementById("match-signal-fill"),
  matchSignalThreshold: document.getElementById("match-signal-threshold"),
  matchSignalValue: document.getElementById("match-signal-value"),
  cornerSignalFill: document.getElementById("corner-signal-fill"),
  cornerSignalThreshold: document.getElementById("corner-signal-threshold"),
  cornerSignalValue: document.getElementById("corner-signal-value"),
  groupSecondary: document.getElementById("group-secondary"),
  showRejectedDetections: document.getElementById("show-rejected-detections"),
  openOverlay: document.getElementById("open-overlay"),
  copyOverlayLink: document.getElementById("copy-overlay-link"),
  overlaySize: document.getElementById("overlay-size"),
  overlayPosition: document.getElementById("overlay-position"),
  overlayBackground: document.getElementById("overlay-background"),
  overlayDuration: document.getElementById("overlay-duration"),
  overlayShowName: document.getElementById("overlay-show-name"),
  overlayShowSet: document.getElementById("overlay-show-set"),
  overlayShowRarity: document.getElementById("overlay-show-rarity"),
  overlayShowPrice: document.getElementById("overlay-show-price"),
  eventList: document.getElementById("event-list"),
  copyList: document.getElementById("copy-list"),
  downloadCsv: document.getElementById("download-csv"),
  downloadJsonl: document.getElementById("download-jsonl"),
  clearEvents: document.getElementById("clear-events"),
};

const previewCtx = els.preview.getContext("2d");
const overlayCtx = els.overlay.getContext("2d");
const processCanvas = document.createElement("canvas");
const processCtx = processCanvas.getContext("2d");
const channel = new BroadcastChannel(CHANNEL_NAME);
const events = [];

let worker = null;
let workerReady = false;
let stream = null;
let previewFrame = null;
let scanTimer = null;
let workerBusy = false;
let scanStartedAt = null;
let detectionFps = null;
let roi = readRoi();
let settings = readSettings();
let dragState = null;
let lastSource = null;
let bucket = createBucket();

function resolveAssetChannel() {
  const requested = new URLSearchParams(location.search).get("channel") ?? "stable";
  return Object.hasOwn(ASSET_CHANNELS, requested) ? requested : "stable";
}

init();

async function init() {
  applySettingsToInputs();
  applyRoiToBox();
  bindUi();
  await initWorker();
}

function bindUi() {
  updateOverlayLink();
  els.captureToggle.addEventListener("click", () => {
    if (stream) {
      stopCapture();
    } else {
      startCapture().catch((error) => {
        setStageStatus(error?.message || "Could not start capture.");
      });
    }
  });
  els.resetRoi.addEventListener("click", () => {
    roi = { x: 0, y: 0, width: 1, height: 1 };
    saveRoi();
    applyRoiToBox();
    updateSourceMeta();
  });
  els.stage.addEventListener("pointerdown", startRoiDrag);
  window.addEventListener("pointermove", moveRoiDrag);
  window.addEventListener("pointerup", endRoiDrag);
  window.addEventListener("resize", resizeCanvases);
  els.scanInterval.addEventListener("input", updateScanIntervalLabel);
  for (const input of [
    els.matchThreshold,
    els.consecutiveMatches,
    els.scanInterval,
    els.cornerThreshold,
    els.groupSecondary,
    els.showRejectedDetections,
  ]) {
    input.addEventListener("change", updateSettingsFromInputs);
  }
  for (const input of [
    els.overlaySize,
    els.overlayPosition,
    els.overlayBackground,
    els.overlayDuration,
    els.overlayShowName,
    els.overlayShowSet,
    els.overlayShowRarity,
    els.overlayShowPrice,
  ]) {
    input.addEventListener("input", updateOverlayLink);
    input.addEventListener("change", updateOverlayLink);
  }
  els.copyOverlayLink.addEventListener("click", copyOverlayLink);
  els.copyList.addEventListener("click", copyCardList);
  els.downloadCsv.addEventListener("click", () => downloadText("collectorvision-screen-capture.csv", buildCsv(), "text/csv"));
  els.downloadJsonl.addEventListener("click", () => downloadText("collectorvision-screen-capture.jsonl", buildJsonl(), "application/x-ndjson"));
  els.clearEvents.addEventListener("click", () => {
    events.splice(0, events.length);
    renderEvents();
  });
}

function updateOverlayLink() {
  const url = new URL("./screen_capture_overlay.html", location.href);
  url.searchParams.set("size", els.overlaySize.value);
  url.searchParams.set("position", els.overlayPosition.value);
  url.searchParams.set("background", els.overlayBackground.value);
  url.searchParams.set("duration", els.overlayDuration.value);
  url.searchParams.set("name", els.overlayShowName.checked ? "1" : "0");
  url.searchParams.set("set", els.overlayShowSet.checked ? "1" : "0");
  url.searchParams.set("rarity", els.overlayShowRarity.checked ? "1" : "0");
  url.searchParams.set("price", els.overlayShowPrice.checked ? "1" : "0");
  els.openOverlay.href = url.href;
}

async function copyOverlayLink() {
  try {
    await navigator.clipboard.writeText(els.openOverlay.href);
    const original = els.copyOverlayLink.textContent;
    els.copyOverlayLink.textContent = "Copied";
    setTimeout(() => { els.copyOverlayLink.textContent = original; }, 1500);
  } catch (error) {
    console.warn("[CollectorVision monitor] Could not copy overlay link", error);
  }
}

async function initWorker() {
  try {
    const assetChannel = resolveAssetChannel();
    const assetBasePath = ASSET_CHANNELS[assetChannel];
    const manifest = await fetchJson(`${assetBasePath}/manifest.json`);
    const bundleMetadata = await fetchOptionalJson(
      assetChannel === "stable" ? "./bundle-metadata.json" : "./testing/bundle-metadata.json",
    );
    renderVersionDebug(manifest, bundleMetadata);
    worker = new Worker(`./scanner.worker.mjs?v=${BUILD_ID}`, { type: "module" });
    worker.addEventListener("message", handleWorkerMessage);
    worker.addEventListener("error", (event) => {
      workerBusy = false;
      els.workerStatus.textContent = "Worker error";
      els.latestStatus.textContent = event.message || "Scanner worker failed.";
    });
    worker.postMessage({
      type: "init",
      manifest,
      assetBasePath,
      enableWebGpu: false,
      minCornerConfidence: settings.minCornerConfidence,
      rotationInvariant: true,
    });
  } catch (error) {
    els.workerStatus.textContent = "Load failed";
    els.latestStatus.textContent = error?.message || "Could not load scanner assets.";
    renderVersionDebug(null, null);
  }
}

function handleWorkerMessage({ data }) {
  if (data.type === "progress") {
    if (data.stage) {
      const percent = Number.isFinite(data.ratio) ? ` ${Math.round(data.ratio * 100)}%` : "";
      els.workerStatus.textContent = `${data.stage}${percent}`;
    }
    return;
  }
  if (data.type === "ready") {
    workerReady = true;
    els.workerStatus.textContent = `${data.inferenceMode || "WASM"} ready`;
    els.latestStatus.textContent = "Ready. Start capture to scan a shared source.";
    return;
  }
  if (data.type === "error") {
    workerBusy = false;
    els.latestStatus.textContent = data.message || "Scanner error.";
    return;
  }
  if (data.type !== "result") return;

  workerBusy = false;
  updateDetectionFps();
  updateLatestResult(data);
  drawCorners(data);

  const candidate = candidateFromResult(data);
  const confirmed = bucket.push(candidate);
  if (confirmed) {
    emitConfirmed(confirmed, data);
  }
}

async function startCapture() {
  if (!window.isSecureContext) {
    throw new Error("Screen capture requires HTTPS or localhost.");
  }
  if (!navigator.mediaDevices?.getDisplayMedia) {
    throw new Error("Screen capture is not available in this browser.");
  }

  stream = await navigator.mediaDevices.getDisplayMedia({
    video: {
      displaySurface: "browser",
      frameRate: { ideal: 30, max: 60 },
    },
    audio: false,
  });
  const [track] = stream.getVideoTracks();
  track.addEventListener("ended", stopCapture);

  els.video.srcObject = stream;
  await els.video.play();
  lastSource = track.getSettings?.() ?? {};
  els.captureToggle.textContent = "Stop capture";
  updateSourceMeta();
  els.stageStatus.hidden = true;
  resizeCanvases();
  renderPreview();
  startScanLoop();
}

function stopCapture() {
  if (previewFrame !== null) {
    cancelAnimationFrame(previewFrame);
    previewFrame = null;
  }
  if (scanTimer !== null) {
    clearInterval(scanTimer);
    scanTimer = null;
  }
  if (stream) {
    for (const track of stream.getTracks()) {
      track.stop();
    }
  }
  stream = null;
  workerBusy = false;
  scanStartedAt = null;
  detectionFps = null;
  els.liveResultTitle.textContent = "Live result";
  els.video.srcObject = null;
  els.captureToggle.textContent = "Start capture";
  els.sourceMeta.textContent = "No source selected";
  setStageStatus("Choose a screen, window, or tab to begin.");
  overlayCtx.clearRect(0, 0, els.overlay.width, els.overlay.height);
}

function renderPreview() {
  if (!stream) return;
  resizeCanvases();
  const draw = previewDrawRect();
  previewCtx.clearRect(0, 0, els.preview.width, els.preview.height);
  previewCtx.drawImage(els.video, draw.x, draw.y, draw.width, draw.height);
  previewFrame = requestAnimationFrame(renderPreview);
}

function resizeCanvases() {
  const rect = els.stage.getBoundingClientRect();
  const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  for (const canvas of [els.preview, els.overlay]) {
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  }
  applyRoiToBox();
}

function previewDrawRect() {
  const cw = els.preview.width;
  const ch = els.preview.height;
  const vw = els.video.videoWidth || 16;
  const vh = els.video.videoHeight || 9;
  const scale = Math.min(cw / vw, ch / vh);
  const width = vw * scale;
  const height = vh * scale;
  return {
    x: (cw - width) / 2,
    y: (ch - height) / 2,
    width,
    height,
  };
}

function previewDrawCssRect() {
  const rect = els.stage.getBoundingClientRect();
  const vw = els.video.videoWidth || 16;
  const vh = els.video.videoHeight || 9;
  const scale = Math.min(rect.width / vw, rect.height / vh);
  const width = vw * scale;
  const height = vh * scale;
  return {
    x: (rect.width - width) / 2,
    y: (rect.height - height) / 2,
    width,
    height,
  };
}

function startScanLoop() {
  if (scanTimer !== null) clearInterval(scanTimer);
  const interval = Math.max(0, Number(settings.scanIntervalMs) || 0);
  const tick = () => {
    scanOnce().catch((error) => {
      workerBusy = false;
      els.latestStatus.textContent = error?.message || "Scan failed.";
    });
  };
  if (interval <= 0) {
    const loop = async () => {
      if (!stream) return;
      await tick();
      scanTimer = setTimeout(loop, 0);
    };
    loop();
  } else {
    scanTimer = setInterval(tick, interval);
    tick();
  }
}

async function scanOnce() {
  if (!stream || !workerReady || workerBusy || !els.video.videoWidth || !els.video.videoHeight) {
    return;
  }
  workerBusy = true;
  scanStartedAt = performance.now();
  drawRoiToProcessCanvas();
  const bitmap = await createImageBitmap(processCanvas);
  worker.postMessage({ type: "frame", bitmap }, [bitmap]);
}

function updateDetectionFps() {
  if (scanStartedAt === null) return;
  const elapsedMs = performance.now() - scanStartedAt;
  scanStartedAt = null;
  if (elapsedMs <= 0) return;
  const currentFps = 1000 / elapsedMs;
  detectionFps = detectionFps === null ? currentFps : detectionFps * 0.7 + currentFps * 0.3;
  els.liveResultTitle.textContent = `Live result (${detectionFps.toFixed(1)} FPS)`;
}

function drawRoiToProcessCanvas() {
  const vw = els.video.videoWidth;
  const vh = els.video.videoHeight;
  const sx = Math.round(roi.x * vw);
  const sy = Math.round(roi.y * vh);
  const sw = Math.max(1, Math.round(roi.width * vw));
  const sh = Math.max(1, Math.round(roi.height * vh));
  const maxOutput = 1280;
  const scale = Math.min(1, maxOutput / Math.max(sw, sh));
  processCanvas.width = Math.max(1, Math.round(sw * scale));
  processCanvas.height = Math.max(1, Math.round(sh * scale));
  processCtx.drawImage(els.video, sx, sy, sw, sh, 0, 0, processCanvas.width, processCanvas.height);
}

function updateLatestResult(data) {
  const score = Number(data.score);
  const sharpness = Number(data.sharpness);
  const showCard = isAcceptedDetection(data) || settings.showRejectedDetections;
  els.latestCard.textContent = showCard
    ? (data.cardId || (data.cardPresent ? "Card candidate" : "—"))
    : "—";
  els.latestScore.textContent = Number.isFinite(score) ? score.toFixed(3) : "—";
  els.latestSharpness.textContent = Number.isFinite(sharpness) ? sharpness.toFixed(3) : "—";
  updateThresholdMeter("match", settings.matchThreshold, score, 1);
  updateThresholdMeter("corner", settings.minCornerConfidence, sharpness, 0.2);

  if (!data.cardPresent) {
    els.latestStatus.textContent = "No card detected in the ROI.";
  } else if (!data.cornersValid) {
    els.latestStatus.textContent = "Card-like frame, but corner geometry is not usable.";
  } else if (!Number.isFinite(score) || score < settings.matchThreshold) {
    els.latestStatus.textContent = `Candidate below threshold (${Number.isFinite(score) ? score.toFixed(3) : "no score"}).`;
  } else {
    els.latestStatus.textContent = "Candidate accepted; waiting for confirmation bucket.";
  }
}

function isAcceptedDetection(data) {
  const score = Number(data.score);
  return data.cardPresent
    && data.cornersValid
    && Number.isFinite(score)
    && score >= settings.matchThreshold;
}

function updateThresholdMeter(name, threshold, current, maximum) {
  const fill = els[`${name}SignalFill`];
  const marker = els[`${name}SignalThreshold`];
  const value = els[`${name}SignalValue`];
  const thresholdRatio = clamp(threshold / maximum, 0, 1);
  marker.style.left = `${(thresholdRatio * 100).toFixed(1)}%`;
  if (!Number.isFinite(current)) {
    fill.style.width = "0%";
    value.textContent = "Current —";
    return;
  }
  const currentRatio = clamp(current / maximum, 0, 1);
  fill.style.width = `${(currentRatio * 100).toFixed(1)}%`;
  value.textContent = `Current ${current.toFixed(3)}`;
}

function candidateFromResult(data) {
  if (!isAcceptedDetection(data)) return null;
  const score = Number(data.score);
  return {
    cardId: data.cardId,
    secondaryId: data.secondaryId,
    secondaryIdField: data.secondaryIdField,
    score,
    orientation: data.orientation,
  };
}

async function emitConfirmed(confirmed, result) {
  const event = {
    type: "collectorvision.card",
    version: 1,
    timestamp: new Date().toISOString(),
    source: {
      kind: "display-capture",
      label: "user-selected source",
      videoWidth: els.video.videoWidth,
      videoHeight: els.video.videoHeight,
      roi: { ...roi },
    },
    card: {
      cardId: confirmed.cardId,
      secondaryId: confirmed.secondaryId ?? null,
      secondaryIdField: confirmed.secondaryIdField ?? null,
      score: confirmed.score,
      orientation: confirmed.orientation ?? null,
    },
    detection: {
      corners: result.corners ?? null,
      sharpness: result.sharpness ?? null,
      confidence: result.confidence ?? null,
    },
  };
  events.unshift(event);
  while (events.length > MAX_EVENTS) events.pop();
  els.confirmedCount.textContent = String(events.length);
  renderEvents();
  channel.postMessage(event);
  window.dispatchEvent(new CustomEvent("collectorvision:card", { detail: event }));
  enrichEvent(event).catch(() => {});
}

async function enrichEvent(event) {
  if (!event.card.cardId) return;
  const response = await fetch(`https://api.scryfall.com/cards/${encodeURIComponent(event.card.cardId)}`);
  if (!response.ok) return;
  const data = await response.json();
  event.card.name = data.name ?? event.card.name;
  event.card.setCode = data.set ?? event.card.setCode;
  event.card.collectorNumber = data.collector_number ?? event.card.collectorNumber;
  renderEvents();
  channel.postMessage(event);
}

function drawCorners(data) {
  overlayCtx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  if (!data.corners || data.corners.length !== 4) return;
  if (!isAcceptedDetection(data) && !settings.showRejectedDetections) return;
  const draw = previewDrawRect();
  const points = data.corners.map(([x, y]) => ({
    x: draw.x + ((roi.x + x * roi.width) * draw.width),
    y: draw.y + ((roi.y + y * roi.height) * draw.height),
  }));
  overlayCtx.lineWidth = Math.max(3, els.overlay.width / 360);
  overlayCtx.strokeStyle = data.cornersValid ? "#58f29a" : "#ff8a78";
  overlayCtx.fillStyle = data.cornersValid ? "rgba(88, 242, 154, 0.16)" : "rgba(255, 138, 120, 0.16)";
  overlayCtx.beginPath();
  overlayCtx.moveTo(points[0].x, points[0].y);
  for (const point of points.slice(1)) overlayCtx.lineTo(point.x, point.y);
  overlayCtx.closePath();
  overlayCtx.fill();
  overlayCtx.stroke();
}

function createBucket() {
  let candidate = null;
  const cooldowns = new Map();
  return {
    push(next) {
      const now = Date.now();
      for (const [key, expires] of cooldowns) {
        if (now >= expires) cooldowns.delete(key);
      }
      if (!next?.cardId) {
        candidate = null;
        return null;
      }
      const key = bucketKey(next);
      if (cooldowns.has(key)) return null;
      if (candidate?.key === key) {
        candidate.count += 1;
        if (next.score > candidate.best.score) candidate.best = next;
      } else {
        candidate = { key, count: 1, best: next };
      }
      if (candidate.count < settings.consecutiveMatches) return null;
      const confirmed = candidate.best;
      cooldowns.set(key, now + 3000);
      candidate = null;
      return confirmed;
    },
    reset() {
      candidate = null;
      cooldowns.clear();
    },
  };
}

function bucketKey(candidate) {
  if (settings.groupBySecondaryId && candidate.secondaryId) {
    return `secondary:${candidate.secondaryId}`;
  }
  return `card:${candidate.cardId}`;
}

function startRoiDrag(event) {
  const handle = event.target.closest("[data-handle]")?.dataset.handle ?? "move";
  event.preventDefault();
  els.roiBox.setPointerCapture?.(event.pointerId);
  dragState = {
    handle,
    startX: event.clientX,
    startY: event.clientY,
    startRoi: { ...roi },
    moved: false,
  };
}

function moveRoiDrag(event) {
  if (!dragState) return;
  if (Math.abs(event.clientX - dragState.startX) > 3 || Math.abs(event.clientY - dragState.startY) > 3) {
    dragState.moved = true;
  }
  const draw = previewDrawCssRect();
  const dx = (event.clientX - dragState.startX) / draw.width;
  const dy = (event.clientY - dragState.startY) / draw.height;
  const next = { ...dragState.startRoi };
  const minSize = 0.05;

  if (dragState.handle === "move") {
    next.x += dx;
    next.y += dy;
  } else {
    if (dragState.handle.includes("w")) {
      next.x += dx;
      next.width -= dx;
    }
    if (dragState.handle.includes("e")) {
      next.width += dx;
    }
    if (dragState.handle.includes("n")) {
      next.y += dy;
      next.height -= dy;
    }
    if (dragState.handle.includes("s")) {
      next.height += dy;
    }
  }

  if (next.width < minSize) next.width = minSize;
  if (next.height < minSize) next.height = minSize;
  next.x = clamp(next.x, 0, 1 - next.width);
  next.y = clamp(next.y, 0, 1 - next.height);
  next.width = clamp(next.width, minSize, 1 - next.x);
  next.height = clamp(next.height, minSize, 1 - next.y);
  roi = next;
  applyRoiToBox();
  updateSourceMeta();
}

function endRoiDrag(event) {
  if (!dragState) return;
  if (!dragState.moved && dragState.handle === "move") {
    const stage = els.stage.getBoundingClientRect();
    const draw = previewDrawCssRect();
    const clickX = event.clientX - stage.left;
    const clickY = event.clientY - stage.top;
    roi.x = clamp((clickX - draw.x) / draw.width - roi.width / 2, 0, 1 - roi.width);
    roi.y = clamp((clickY - draw.y) / draw.height - roi.height / 2, 0, 1 - roi.height);
    applyRoiToBox();
    updateSourceMeta();
  }
  dragState = null;
  saveRoi();
}

function applyRoiToBox() {
  const stage = els.stage.getBoundingClientRect();
  const draw = previewDrawCssRect();
  const left = draw.x + roi.x * draw.width;
  const top = draw.y + roi.y * draw.height;
  els.roiBox.style.left = `${(left / stage.width) * 100}%`;
  els.roiBox.style.top = `${(top / stage.height) * 100}%`;
  els.roiBox.style.width = `${((roi.width * draw.width) / stage.width) * 100}%`;
  els.roiBox.style.height = `${((roi.height * draw.height) / stage.height) * 100}%`;
}

function readRoi() {
  try {
    return sanitizeRoi(JSON.parse(localStorage.getItem(ROI_KEY) || "null")) ?? { ...DEFAULT_ROI };
  } catch {
    return { ...DEFAULT_ROI };
  }
}

function saveRoi() {
  localStorage.setItem(ROI_KEY, JSON.stringify(roi));
}

function sanitizeRoi(value) {
  if (!value || typeof value !== "object") return null;
  const next = {
    x: Number(value.x),
    y: Number(value.y),
    width: Number(value.width),
    height: Number(value.height),
  };
  if (!Object.values(next).every(Number.isFinite)) return null;
  next.width = clamp(next.width, 0.05, 1);
  next.height = clamp(next.height, 0.05, 1);
  next.x = clamp(next.x, 0, 1 - next.width);
  next.y = clamp(next.y, 0, 1 - next.height);
  return next;
}

function readSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
    return {
      ...DEFAULT_SETTINGS,
      ...saved,
      showRejectedDetections: saved.showRejectedDetections ?? saved.showBelowThresholdMatches ?? true,
    };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

function applySettingsToInputs() {
  els.matchThreshold.value = settings.matchThreshold.toFixed(2);
  els.consecutiveMatches.value = String(settings.consecutiveMatches);
  els.scanInterval.value = String(settings.scanIntervalMs);
  updateScanIntervalLabel();
  els.cornerThreshold.value = settings.minCornerConfidence.toFixed(2);
  els.groupSecondary.checked = settings.groupBySecondaryId === true;
  els.showRejectedDetections.checked = settings.showRejectedDetections !== false;
  updateThresholdMeter("match", settings.matchThreshold, null, 1);
  updateThresholdMeter("corner", settings.minCornerConfidence, null, 0.2);
}

function updateScanIntervalLabel() {
  const interval = Math.max(0, Math.round(Number(els.scanInterval.value) || 0));
  els.scanIntervalValue.textContent = interval === 0 ? "Free-running" : `${interval} ms`;
}

function updateSettingsFromInputs() {
  settings = {
    matchThreshold: clamp(Number(els.matchThreshold.value), 0, 1),
    consecutiveMatches: Math.max(1, Math.round(Number(els.consecutiveMatches.value) || 1)),
    scanIntervalMs: Math.max(0, Math.round(Number(els.scanInterval.value) || 0)),
    minCornerConfidence: clamp(Number(els.cornerThreshold.value), 0, 0.2),
    groupBySecondaryId: els.groupSecondary.checked === true,
    showRejectedDetections: els.showRejectedDetections.checked === true,
  };
  applySettingsToInputs();
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  bucket.reset();
  worker?.postMessage({ type: "config", minCornerConfidence: settings.minCornerConfidence });
  if (stream) startScanLoop();
}

function renderEvents() {
  els.confirmedCount.textContent = String(events.length);
  if (!events.length) {
    els.eventList.innerHTML = '<p class="empty">No confirmed cards yet.</p>';
    return;
  }
  els.eventList.innerHTML = events.map((event) => {
    const card = event.card;
    const name = escapeHtml(card.name || card.cardId || "Unknown card");
    const set = card.setCode ? ` · ${escapeHtml(String(card.setCode).toUpperCase())}` : "";
    const score = Number(card.score);
    return `
      <article class="event-card">
        <div>
          <h3>${name}</h3>
          <p>${escapeHtml(card.cardId || "")}${set}</p>
        </div>
        <span class="event-score">${Number.isFinite(score) ? score.toFixed(3) : "—"}</span>
      </article>
    `;
  }).join("");
}

async function copyCardList() {
  const text = events.map((event) => event.card.name || event.card.cardId).filter(Boolean).join("\n");
  await navigator.clipboard.writeText(text);
}

function buildCsv() {
  const header = [
    "timestamp", "card_id", "secondary_id", "name", "set", "score", "sharpness",
    "video_width", "video_height", "roi_x", "roi_y", "roi_width", "roi_height",
  ];
  const rows = events.map((event) => [
    event.timestamp,
    event.card.cardId,
    event.card.secondaryId,
    event.card.name,
    event.card.setCode,
    event.card.score,
    event.detection.sharpness,
    event.source.videoWidth,
    event.source.videoHeight,
    event.source.roi.x,
    event.source.roi.y,
    event.source.roi.width,
    event.source.roi.height,
  ]);
  return [header, ...rows].map((row) => row.map(csvCell).join(",")).join("\n");
}

function buildJsonl() {
  return events.map((event) => JSON.stringify(event)).join("\n");
}

function downloadText(filename, text, type) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function setStageStatus(message) {
  els.stageStatus.textContent = message;
  els.stageStatus.hidden = false;
}

function updateSourceMeta() {
  if (stream) els.sourceMeta.textContent = describeSource();
}

function describeSource() {
  const width = els.video.videoWidth || lastSource?.width;
  const height = els.video.videoHeight || lastSource?.height;
  const fps = lastSource?.frameRate ? ` · ${Math.round(lastSource.frameRate)} fps` : "";
  if (!width || !height) return "Shared source active";
  const roiWidth = Math.max(1, Math.round(roi.width * width));
  const roiHeight = Math.max(1, Math.round(roi.height * height));
  return `${width} × ${height} · ROI ${roiWidth} × ${roiHeight}${fps}`;
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to load ${url}: ${response.status}`);
  return response.json();
}

async function fetchOptionalJson(url) {
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

function renderVersionDebug(manifest, bundleMetadata) {
  if (!manifest) {
    els.versionCornelius.textContent = "unavailable";
    els.versionMilo.textContent = "unavailable";
    els.versionCatalog.textContent = "unavailable";
    return;
  }
  const modelHashes = manifest.model_hashes ?? {};
  const modelVersions = manifest.model_versions ?? {};
  const metadataModels = bundleMetadata?.models ?? {};
  const metadataModelVersions = bundleMetadata?.model_versions ?? {};
  const detectorKey = manifest.models?.detector ? "detector" : "cornelius";
  const detectorFamily = manifest.detector?.family ?? "cornelius";
  const detectorLabel = document.getElementById("version-detector-label");
  if (detectorLabel) {
    detectorLabel.textContent = detectorFamily;
  }
  const catalog = manifest.catalog ?? {};
  const catalogSource = bundleMetadata?.catalog_fingerprint
    || bundleMetadata?.catalog_key
    || manifest.version
    || catalog.embeddings
    || catalog.card_ids;

  els.versionCornelius.textContent = versionLabel(
    detectorKey,
    modelHashes[detectorKey] || metadataModels[detectorKey],
    modelVersions[detectorKey] || metadataModelVersions[detectorKey],
    manifest.version,
  );
  els.versionMilo.textContent = versionLabel(
    "milo",
    modelHashes.milo || metadataModels.milo,
    modelVersions.milo || metadataModelVersions.milo,
    manifest.version,
  );
  els.versionCatalog.textContent = [
    catalogDateLabel(catalogSource),
    Number.isFinite(catalog.rows) ? `${catalog.rows.toLocaleString()} rows` : null,
  ].filter(Boolean).join(" · ");
}

function versionLabel(modelKey, hash, version, fallback) {
  const knownVersion = version || KNOWN_MODEL_VERSIONS[modelKey]?.[hash];
  const hashLabel = compactHash(hash);
  if (knownVersion && hashLabel) return `${knownVersion} · ${hashLabel}`;
  if (knownVersion) return knownVersion;
  return compactVersion(hash || fallback);
}

function compactHash(value) {
  const text = String(value ?? "").trim();
  if (!text) return "";
  if (text.startsWith("sha256:")) return `sha256:${text.slice(7, 15)}`;
  if (/^[a-f0-9]{40,64}$/i.test(text)) return text.slice(0, 8);
  return "";
}

function compactVersion(value) {
  const text = String(value ?? "").trim();
  if (!text) return "";
  if (text.startsWith("sha256:")) return `sha256:${text.slice(7, 19)}`;
  if (/^[a-f0-9]{40,64}$/i.test(text)) return text.slice(0, 12);
  if (text.length > 28) return `${text.slice(0, 25)}…`;
  return text;
}

function catalogDateLabel(value) {
  const text = String(value ?? "").trim();
  const match = text.match(/\b(20\d{2}-\d{2}-\d{2})\b/);
  if (match) return match[1];
  return compactVersion(text);
}

function csvCell(value) {
  const text = value === null || value === undefined ? "" : String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[char]));
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, Number(value) || 0));
}
