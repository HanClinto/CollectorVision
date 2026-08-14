function versionedUrl(path) {
  const url = new URL(path, import.meta.url);
  url.search = new URL(import.meta.url).search;
  return url.href;
}

const DEFAULT_CONFIG = {
  manifestUrl: new URL("../assets/manifest.json", import.meta.url).href,
  assetBasePath: new URL("../assets", import.meta.url).href,
  workerUrl: versionedUrl("../scanner.worker.mjs"),
  enableWebGpu: false,
  autoStart: true,
  scanIntervalMs: 900,
  minCornerConfidence: 0.02,
  matchThreshold: 0.5,
  consecutiveMatches: 2,
  cooldownMs: 3500,
  groupBySecondaryId: true,
  showFpsOverlay: true,
  overlay: true,
  camera: {
    facingMode: "environment",
    width: { ideal: 1280 },
    height: { ideal: 720 },
  },
  onReady: null,
  onProgress: null,
  onResult: null,
  onCardDetected: null,
  onError: null,
};

function resolveTarget(target) {
  if (typeof target === "string") {
    const element = document.querySelector(target);
    if (!element) {
      throw new Error(`CollectorVision target not found: ${target}`);
    }
    return element;
  }
  if (target instanceof Element) {
    return target;
  }
  throw new Error("CollectorVision requires a target element or selector.");
}

function mergeConfig(config) {
  return {
    ...DEFAULT_CONFIG,
    ...config,
    camera: {
      ...DEFAULT_CONFIG.camera,
      ...(config.camera ?? {}),
    },
  };
}

function clamp01(value) {
  return Math.min(1, Math.max(0, Number(value) || 0));
}

function formatMetric(value) {
  return Number.isFinite(value) ? value.toFixed(2) : "--";
}

function eventDetail(data) {
  return data && typeof data === "object" ? structuredCloneSafe(data) : data;
}

function structuredCloneSafe(value) {
  if (typeof structuredClone === "function") {
    try {
      return structuredClone(value);
    } catch {
      // Fall through to JSON clone.
    }
  }
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    return value;
  }
}

class ConfirmationBucket {
  constructor({ consecutiveMatches, cooldownMs, groupBySecondaryId }) {
    this.consecutiveMatches = Math.max(1, Number(consecutiveMatches) || 1);
    this.cooldownMs = Math.max(0, Number(cooldownMs) || 0);
    this.groupBySecondaryId = groupBySecondaryId === true;
    this.candidate = null;
    this.cooldowns = new Map();
  }

  updateConfig({ consecutiveMatches, cooldownMs, groupBySecondaryId }) {
    if (consecutiveMatches !== undefined) {
      this.consecutiveMatches = Math.max(1, Number(consecutiveMatches) || 1);
    }
    if (cooldownMs !== undefined) {
      this.cooldownMs = Math.max(0, Number(cooldownMs) || 0);
    }
    if (groupBySecondaryId !== undefined) {
      this.groupBySecondaryId = groupBySecondaryId === true;
      this.candidate = null;
      this.cooldowns.clear();
    }
  }

  reset() {
    this.candidate = null;
    this.cooldowns.clear();
  }

  secondaryIdFor(candidate) {
    const explicitSecondaryId = String(candidate?.secondaryId ?? "").trim();
    if (explicitSecondaryId) {
      return explicitSecondaryId;
    }
    const secondaryField = String(candidate?.secondaryIdField ?? "").trim();
    if (!secondaryField) {
      return "";
    }
    return String(candidate?.[secondaryField] ?? "").trim();
  }

  bucketKeyFor(candidate) {
    const primaryCardId = String(candidate?.cardId ?? "").trim();
    if (!primaryCardId) {
      return null;
    }
    if (this.groupBySecondaryId) {
      const secondaryId = this.secondaryIdFor(candidate);
      if (secondaryId) {
        return `secondary:${secondaryId}`;
      }
    }
    return `card:${primaryCardId}`;
  }

  push(candidate) {
    const now = Date.now();
    for (const [bucketKey, expiry] of this.cooldowns) {
      if (now >= expiry) {
        this.cooldowns.delete(bucketKey);
      }
    }

    if (!candidate) {
      if (this.candidate) {
        this.candidate.count = Math.max(0, this.candidate.count - 1);
        if (this.candidate.count === 0) {
          this.candidate = null;
        }
      }
      return null;
    }

    const bucketKey = this.bucketKeyFor(candidate);
    if (!bucketKey) {
      return null;
    }

    if (this.cooldowns.has(bucketKey)) {
      return null;
    }

    if (this.candidate?.bucketKey === bucketKey) {
      this.candidate.count += 1;
      if (!Number.isFinite(this.candidate.bestScore) || candidate.score > this.candidate.bestScore) {
        this.candidate.bestScore = candidate.score;
        this.candidate.bestResult = candidate;
      }
    } else {
      this.candidate = {
        bucketKey,
        count: 1,
        bestScore: candidate.score,
        bestResult: candidate,
      };
    }

    if (this.candidate.count < this.consecutiveMatches) {
      return null;
    }

    const confirmed = this.candidate.bestResult;
    this.cooldowns.set(bucketKey, now + this.cooldownMs);
    this.candidate = null;
    return confirmed;
  }
}

export class CollectorVisionScannerApplet extends EventTarget {
  constructor(config) {
    super();
    this.config = mergeConfig(config ?? {});
    this.target = resolveTarget(this.config.target);
    this.bucket = new ConfirmationBucket(this.config);
    this.worker = null;
    this.manifest = null;
    this.stream = null;
    this.timer = null;
    this.previewFrame = null;
    this.workerBusy = false;
    this.ready = false;
    this.started = false;
    this.lastResult = null;
    this.lastFpsTimestamp = null;
    this.fpsEma = null;
    this.captureCanvas = document.createElement("canvas");
    this.captureCtx = this.captureCanvas.getContext("2d");
    this.elements = this.createElements();
    this.mount();
  }

  async init() {
    try {
      this.setStatus("Loading CollectorVision…");
      this.manifest = await this.loadManifest();
      this.worker = new Worker(this.config.workerUrl, { type: "module" });
      this.worker.addEventListener("message", (event) => this.handleWorkerMessage(event.data));
      this.worker.addEventListener("error", (event) => this.handleError(event.error ?? event.message));
      this.worker.postMessage({
        type: "init",
        manifest: this.manifest,
        assetBasePath: this.config.assetBasePath,
        enableWebGpu: this.config.enableWebGpu === true,
        minCornerConfidence: clamp01(this.config.minCornerConfidence),
      });
      if (this.config.autoStart) {
        await this.start();
      }
      return this;
    } catch (error) {
      this.handleError(error);
      throw error;
    }
  }

  async start() {
    if (this.started) {
      return;
    }
    this.stream = await navigator.mediaDevices.getUserMedia({ video: this.config.camera, audio: false });
    this.elements.video.srcObject = this.stream;
    await this.elements.video.play();
    this.resizeCanvas();
    this.started = true;
    this.setStatus(this.ready ? "Scanning…" : "Camera ready. Loading models…");
    this.restartTickLoop();
    this.startPreviewLoop();
    this.tick();
  }

  stop() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    if (this.previewFrame) {
      cancelAnimationFrame(this.previewFrame);
      this.previewFrame = null;
    }
    this.started = false;
    this.workerBusy = false;
    this.lastFpsTimestamp = null;
    this.fpsEma = null;
    for (const track of this.stream?.getTracks?.() ?? []) {
      track.stop();
    }
    this.stream = null;
    this.elements.video.srcObject = null;
    this.setStatus("Stopped.");
  }

  dispose() {
    this.stop();
    this.worker?.terminate();
    this.worker = null;
    this.target.replaceChildren();
  }

  updateConfig(config) {
    const prevMinCornerConfidence = this.config.minCornerConfidence;
    this.config = mergeConfig({ ...this.config, ...config });
    this.bucket.updateConfig(this.config);
    this.updateFpsVisibility();
    if (config.minCornerConfidence !== undefined && this.worker) {
      const nextMinCornerConfidence = clamp01(this.config.minCornerConfidence);
      if (nextMinCornerConfidence !== clamp01(prevMinCornerConfidence)) {
        this.worker.postMessage({
          type: "config",
          minCornerConfidence: nextMinCornerConfidence,
        });
      }
    }
    if (config.scanIntervalMs !== undefined && this.started) {
      this.restartTickLoop();
    }
  }

  restartTickLoop() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    if (!this.started || Number(this.config.scanIntervalMs) <= 0) {
      return;
    }
    this.timer = window.setInterval(() => this.tick(), this.config.scanIntervalMs);
  }

  createElements() {
    const root = document.createElement("div");
    root.className = "cv-applet";
    root.innerHTML = `
      <style>
        .cv-applet { position: relative; display: grid; gap: 0.65rem; width: 100%; max-width: 28rem; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #f8fafc; }
        .cv-applet__stage { position: relative; overflow: hidden; border-radius: 1rem; background: #020617; aspect-ratio: 16 / 9; box-shadow: 0 1rem 3rem rgba(2, 6, 23, 0.28); }
        .cv-applet__video { display: block; width: 100%; height: 100%; object-fit: cover; }
        .cv-applet__canvas { position: absolute; inset: 0; display: block; width: 100%; height: 100%; pointer-events: none; }
        .cv-applet__fps { position: absolute; right: 0.5rem; top: 0.5rem; margin: 0; padding: 0.2rem 0.45rem; border-radius: 0.45rem; background: rgba(2, 6, 23, 0.75); color: #e2e8f0; font-size: 0.72rem; letter-spacing: 0.02em; font-variant-numeric: tabular-nums; pointer-events: none; }
        .cv-applet__status { margin: 0; padding: 0.65rem 0.75rem; border-radius: 0.75rem; background: rgba(15, 23, 42, 0.86); color: #e2e8f0; font-size: 0.9rem; }
      </style>
      <div class="cv-applet__stage">
        <video class="cv-applet__video" playsinline muted></video>
        <canvas class="cv-applet__canvas"></canvas>
        <p class="cv-applet__fps" aria-live="off">FPS --</p>
      </div>
      <p class="cv-applet__status">Idle.</p>
    `;
    return {
      root,
      video: root.querySelector("video"),
      canvas: root.querySelector("canvas"),
      fps: root.querySelector(".cv-applet__fps"),
      status: root.querySelector(".cv-applet__status"),
    };
  }

  mount() {
    this.target.replaceChildren(this.elements.root);
    this.ctx = this.elements.canvas.getContext("2d");
    this.updateFpsVisibility();
  }

  updateFpsVisibility() {
    if (!this.elements?.fps) {
      return;
    }
    this.elements.fps.hidden = this.config.showFpsOverlay !== true;
  }

  updateFps(now = performance.now()) {
    if (this.config.showFpsOverlay !== true || !this.elements?.fps) {
      return;
    }
    if (this.lastFpsTimestamp === null) {
      this.lastFpsTimestamp = now;
      this.elements.fps.textContent = "FPS --";
      return;
    }
    const deltaMs = now - this.lastFpsTimestamp;
    this.lastFpsTimestamp = now;
    if (!Number.isFinite(deltaMs) || deltaMs <= 0) {
      return;
    }
    const instantaneousFps = 1000 / deltaMs;
    this.fpsEma = this.fpsEma === null
      ? instantaneousFps
      : this.fpsEma * 0.8 + instantaneousFps * 0.2;
    this.elements.fps.textContent = `FPS ${this.fpsEma.toFixed(1)}`;
  }

  async loadManifest() {
    const response = await fetch(this.config.manifestUrl, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load CollectorVision manifest: HTTP ${response.status}`);
    }
    return response.json();
  }

  handleWorkerMessage(data) {
    if (data.type === "progress") {
      this.emit("progress", data);
      this.config.onProgress?.(data, this);
      return;
    }
    if (data.type === "ready") {
      this.ready = true;
      this.setStatus(this.started ? "Scanning…" : "Ready.");
      this.emit("ready", data);
      this.config.onReady?.(data, this);
      if (this.started && Number(this.config.scanIntervalMs) <= 0) {
        this.tick();
      }
      return;
    }
    if (data.type === "result") {
      this.workerBusy = false;
      this.updateFps();
      this.handleResult(data);
      if (this.started && Number(this.config.scanIntervalMs) <= 0) {
        this.tick();
      }
      return;
    }
    if (data.type === "error") {
      this.workerBusy = false;
      this.handleError(data.message);
      if (this.started && Number(this.config.scanIntervalMs) <= 0) {
        this.tick();
      }
    }
  }

  handleResult(result) {
    this.lastResult = result;
    this.emit("result", result);
    this.config.onResult?.(result, this);

    const cornerConfidence = formatMetric(result.confidence);
    const cornerThreshold = clamp01(this.config.minCornerConfidence).toFixed(2);

    if (!result.cardPresent || !result.cornersValid) {
      this.bucket.push(null);
      this.setStatus(
        result.cardPresent
          ? `Card found; waiting for stable corners… corner ${cornerConfidence}/${cornerThreshold}.`
          : `Looking for a card… corner ${cornerConfidence}/${cornerThreshold}.`,
      );
      return;
    }

    if (!Number.isFinite(result.score) || result.score < this.config.matchThreshold) {
      this.bucket.push(null);
      this.setStatus(
        `Candidate below threshold (${result.score?.toFixed(2) ?? "—"}), corner ${cornerConfidence}/${cornerThreshold}.`,
      );
      return;
    }

    const confirmed = this.bucket.push(result);
    this.setStatus(`Candidate ${result.cardId} (${result.score.toFixed(2)}).`);
    if (!confirmed) {
      return;
    }

    const detail = {
      cardId: confirmed.cardId,
      secondaryId: confirmed.secondaryId ?? null,
      secondaryIdField: confirmed.secondaryIdField ?? null,
      score: confirmed.score,
      corners: confirmed.corners,
      sharpness: confirmed.sharpness,
      confidence: confirmed.confidence,
      timing: confirmed.timing,
      raw: confirmed,
      detectedAt: new Date().toISOString(),
    };
    if (detail.secondaryIdField && detail.secondaryId !== null && detail.secondaryId !== undefined) {
      detail[detail.secondaryIdField] = detail.secondaryId;
    }
    this.setStatus(`Detected ${detail.cardId} (${detail.score.toFixed(2)}).`);
    this.emit("cardDetected", detail);
    this.config.onCardDetected?.(detail, this);
  }

  handleError(error) {
    const message = error instanceof Error ? error.message : String(error);
    this.setStatus(message);
    this.emit("error", { message, error });
    this.config.onError?.({ message, error }, this);
  }

  async tick() {
    if (!this.ready || this.workerBusy || !this.started) {
      return;
    }
    if (!this.drawCaptureFrame()) {
      return;
    }
    this.workerBusy = true;
    try {
      const bitmap = await createImageBitmap(this.captureCanvas);
      this.worker.postMessage({ type: "frame", bitmap }, [bitmap]);
    } catch (error) {
      this.workerBusy = false;
      this.handleError(error);
    }
  }

  startPreviewLoop() {
    if (this.previewFrame) {
      return;
    }
    const render = () => {
      this.drawPreview();
      this.previewFrame = this.started ? requestAnimationFrame(render) : null;
    };
    this.previewFrame = requestAnimationFrame(render);
  }

  drawPreview() {
    if (!this.stream || !this.elements.video.videoWidth) {
      return;
    }
    this.resizeCanvas();
    this.ctx.clearRect(0, 0, this.elements.canvas.width, this.elements.canvas.height);
    this.drawOverlay(this.lastResult);
  }

  drawCaptureFrame() {
    if (!this.stream || !this.elements.video.videoWidth) {
      return false;
    }
    this.resizeCanvas();
    this.captureCtx.drawImage(
      this.elements.video,
      0,
      0,
      this.captureCanvas.width,
      this.captureCanvas.height,
    );
    return true;
  }

  resizeCanvas() {
    const width = this.elements.video.videoWidth || 1280;
    const height = this.elements.video.videoHeight || 720;
    if (this.elements.canvas.width !== width || this.elements.canvas.height !== height) {
      this.elements.canvas.width = width;
      this.elements.canvas.height = height;
    }
    if (this.captureCanvas.width !== width || this.captureCanvas.height !== height) {
      this.captureCanvas.width = width;
      this.captureCanvas.height = height;
    }
  }

  drawOverlay(result) {
    if (!this.config.overlay || !result?.cornersValid || !Array.isArray(result.corners)) {
      return;
    }
    const { width, height } = this.elements.canvas;
    this.ctx.save();
    this.ctx.lineWidth = Math.max(3, width * 0.004);
    this.ctx.strokeStyle = "#22c55e";
    this.ctx.beginPath();
    for (let i = 0; i < result.corners.length; i += 1) {
      const [x, y] = result.corners[i];
      const px = clamp01(x) * width;
      const py = clamp01(y) * height;
      if (i === 0) this.ctx.moveTo(px, py);
      else this.ctx.lineTo(px, py);
    }
    this.ctx.closePath();
    this.ctx.stroke();
    this.ctx.restore();
  }

  setStatus(message) {
    this.elements.status.textContent = message;
  }

  emit(type, detail) {
    this.dispatchEvent(new CustomEvent(type, { detail: eventDetail(detail) }));
  }
}

export async function createCollectorVisionScannerApplet(config) {
  const scanner = new CollectorVisionScannerApplet(config);
  await scanner.init();
  return scanner;
}
