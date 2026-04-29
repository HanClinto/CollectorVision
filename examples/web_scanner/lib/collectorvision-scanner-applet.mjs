const DEFAULT_CONFIG = {
  manifestUrl: new URL("../assets/manifest.json", import.meta.url).href,
  workerUrl: new URL("../scanner.worker.mjs", import.meta.url).href,
  enableWebGpu: false,
  autoStart: true,
  scanIntervalMs: 900,
  matchThreshold: 0.6,
  consecutiveMatches: 2,
  cooldownMs: 3500,
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
  constructor({ consecutiveMatches, cooldownMs }) {
    this.consecutiveMatches = Math.max(1, Number(consecutiveMatches) || 1);
    this.cooldownMs = Math.max(0, Number(cooldownMs) || 0);
    this.candidate = null;
    this.cooldowns = new Map();
  }

  updateConfig({ consecutiveMatches, cooldownMs }) {
    if (consecutiveMatches !== undefined) {
      this.consecutiveMatches = Math.max(1, Number(consecutiveMatches) || 1);
    }
    if (cooldownMs !== undefined) {
      this.cooldownMs = Math.max(0, Number(cooldownMs) || 0);
    }
  }

  reset() {
    this.candidate = null;
    this.cooldowns.clear();
  }

  push(candidate) {
    const now = Date.now();
    for (const [cardId, expiry] of this.cooldowns) {
      if (now >= expiry) {
        this.cooldowns.delete(cardId);
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

    if (this.cooldowns.has(candidate.cardId)) {
      return null;
    }

    if (this.candidate?.cardId === candidate.cardId) {
      this.candidate.count += 1;
      this.candidate.result = candidate;
    } else {
      this.candidate = { cardId: candidate.cardId, count: 1, result: candidate };
    }

    if (this.candidate.count < this.consecutiveMatches) {
      return null;
    }

    const confirmed = this.candidate.result;
    this.cooldowns.set(confirmed.cardId, now + this.cooldownMs);
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
    this.workerBusy = false;
    this.ready = false;
    this.started = false;
    this.lastResult = null;
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
        enableWebGpu: this.config.enableWebGpu === true,
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
    this.timer = window.setInterval(() => this.tick(), this.config.scanIntervalMs);
    this.drawPreview();
  }

  stop() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    this.started = false;
    this.workerBusy = false;
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
    this.config = mergeConfig({ ...this.config, ...config });
    this.bucket.updateConfig(this.config);
    if (config.scanIntervalMs !== undefined && this.started) {
      clearInterval(this.timer);
      this.timer = window.setInterval(() => this.tick(), this.config.scanIntervalMs);
    }
  }

  createElements() {
    const root = document.createElement("div");
    root.className = "cv-applet";
    root.innerHTML = `
      <style>
        .cv-applet { position: relative; display: grid; gap: 0.65rem; width: 100%; max-width: 28rem; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #f8fafc; }
        .cv-applet__stage { position: relative; overflow: hidden; border-radius: 1rem; background: #020617; aspect-ratio: 16 / 9; box-shadow: 0 1rem 3rem rgba(2, 6, 23, 0.28); }
        .cv-applet__canvas { display: block; width: 100%; height: 100%; object-fit: cover; }
        .cv-applet__status { margin: 0; padding: 0.65rem 0.75rem; border-radius: 0.75rem; background: rgba(15, 23, 42, 0.86); color: #e2e8f0; font-size: 0.9rem; }
        .cv-applet__video { display: none; }
      </style>
      <div class="cv-applet__stage">
        <canvas class="cv-applet__canvas"></canvas>
      </div>
      <p class="cv-applet__status">Idle.</p>
      <video class="cv-applet__video" playsinline muted></video>
    `;
    return {
      root,
      video: root.querySelector("video"),
      canvas: root.querySelector("canvas"),
      status: root.querySelector(".cv-applet__status"),
    };
  }

  mount() {
    this.target.replaceChildren(this.elements.root);
    this.ctx = this.elements.canvas.getContext("2d");
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
      return;
    }
    if (data.type === "result") {
      this.workerBusy = false;
      this.handleResult(data);
      return;
    }
    if (data.type === "error") {
      this.workerBusy = false;
      this.handleError(data.message);
    }
  }

  handleResult(result) {
    this.lastResult = result;
    this.emit("result", result);
    this.config.onResult?.(result, this);
    this.drawOverlay(result);

    if (!result.cardPresent || !result.cornersValid) {
      this.bucket.push(null);
      this.setStatus(result.cardPresent ? "Card found; waiting for stable corners…" : "Looking for a card…");
      return;
    }

    if (!Number.isFinite(result.score) || result.score < this.config.matchThreshold) {
      this.bucket.push(null);
      this.setStatus(`Candidate below threshold (${result.score?.toFixed(2) ?? "—"}).`);
      return;
    }

    const confirmed = this.bucket.push(result);
    this.setStatus(`Candidate ${result.cardId} (${result.score.toFixed(2)}).`);
    if (!confirmed) {
      return;
    }

    const detail = {
      cardId: confirmed.cardId,
      score: confirmed.score,
      corners: confirmed.corners,
      sharpness: confirmed.sharpness,
      confidence: confirmed.confidence,
      timing: confirmed.timing,
      raw: confirmed,
      detectedAt: new Date().toISOString(),
    };
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
    this.drawPreview();
    if (!this.ready || this.workerBusy || !this.started) {
      return;
    }
    this.workerBusy = true;
    try {
      const bitmap = await createImageBitmap(this.elements.canvas);
      this.worker.postMessage({ type: "frame", bitmap }, [bitmap]);
    } catch (error) {
      this.workerBusy = false;
      this.handleError(error);
    }
  }

  drawPreview() {
    if (!this.stream || !this.elements.video.videoWidth) {
      return;
    }
    this.resizeCanvas();
    this.ctx.drawImage(
      this.elements.video,
      0,
      0,
      this.elements.canvas.width,
      this.elements.canvas.height,
    );
  }

  resizeCanvas() {
    const width = this.elements.video.videoWidth || 1280;
    const height = this.elements.video.videoHeight || 720;
    if (this.elements.canvas.width !== width || this.elements.canvas.height !== height) {
      this.elements.canvas.width = width;
      this.elements.canvas.height = height;
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
