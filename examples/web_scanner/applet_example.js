import { CodeJar } from "https://cdn.jsdelivr.net/npm/codejar@4.3.0/dist/codejar.js";
import {
  createCollectorVisionScannerApplet,
} from "./lib/collectorvision-scanner-applet.mjs?v=__BUILD_ID__";

const CODE_KEY = "collectorvision_applet_example_code";
const PRESET_KEY = "collectorvision_applet_example_preset";
const SETTINGS_KEY = "collectorvision_applet_example_settings";
const ASSET_CHANNELS = {
  stable: "./assets",
  testing: "./testing/assets",
};
const LOG_LIMIT = 12;
const SCRYFALL_CARD_URL = "https://api.scryfall.com/cards/${card.cardId}";
const DEFAULT_SCAN_SETTINGS = {
  minCornerConfidence: 0.02,
  matchThreshold: 0.50,
  consecutiveMatches: 2,
  scanIntervalMs: 900,
  enableWebGpu: false,
  showFpsOverlay: true,
  groupBySecondaryId: true,
};
const MAX_GUI_CORNER_CONFIDENCE = 0.10;

function resolveAssetChannel() {
  const requested = new URLSearchParams(location.search).get("channel") ?? "stable";
  return Object.hasOwn(ASSET_CHANNELS, requested) ? requested : "stable";
}

const assetChannel = resolveAssetChannel();
const assetBasePath = ASSET_CHANNELS[assetChannel];

const PRESETS = [
  {
    id: "table",
    label: "Lookup table",
    code: [
      `  mygui.log("Looking up", card.cardId, "score", card.score.toFixed(3));`,
      "",
      `  const scryfall = await mygui.fetchJson(\`${SCRYFALL_CARD_URL}\`);`,
      "",
      "  mygui.addRow({",
      "    Name: scryfall.name,",
      "    Set: scryfall.set_name,",
      "    Number: scryfall.collector_number,",
      "    Rarity: scryfall.rarity,",
      "    USD: scryfall.prices?.usd ?? \"\",",
      "    Score: card.score.toFixed(3),",
      "  });",
    ].join("\n"),
  },
  {
    id: "color",
    label: "Color mood",
    code: [
      `  const scryfall = await mygui.fetchJson(\`${SCRYFALL_CARD_URL}\`);`,
      "",
      "  const colorMoods = {",
      "    W: \"#dddddd\",",
      "    U: \"#9999ff\",",
      "    B: \"#333333\",",
      "    R: \"#ff9999\",",
      "    G: \"#99ff99\",",
      "    C: \"#999999\",",
      "    M: \"#ffff99\",",
      "  };",
      "",
      "  const colors = scryfall.color_identity ?? [];",
      "  const moodKey = colors.length === 0 ? \"C\" : colors.length > 1 ? \"M\" : colors[0];",
      "  document.body.style.transition = \"background-color 400ms ease\";",
      "  document.body.style.backgroundColor = colorMoods[moodKey] ?? colorMoods.C;",
      "",
      "  mygui.log(\"Page color changed for\", scryfall.name, colors.join(\"\") || \"colorless\");",
    ].join("\n"),
  },
  {
    id: "bounce",
    label: "Bouncing card",
    code: [
      `  const scryfall = await mygui.fetchJson(\`${SCRYFALL_CARD_URL}\`);`,
      "",
      "  mygui.addRow({",
      "    Name: scryfall.name,",
      "    Set: scryfall.set_name,",
      "    USD: scryfall.prices?.usd ?? \"\",",
      "    Score: card.score.toFixed(3),",
      "  });",
      "",
      "  mygui.bounceCard(mygui.cardImageUrl(scryfall), scryfall.name);",
      "  mygui.log(\"Bouncing one copy of\", scryfall.name);",
    ].join("\n"),
  },
  {
    id: "value-party",
    label: "Value party",
    code: [
      `  const scryfall = await mygui.fetchJson(\`${SCRYFALL_CARD_URL}\`);`,
      "  const usd = Number(scryfall.prices?.usd ?? 0);",
      "",
      "  mygui.addRow({",
      "    Name: scryfall.name,",
      "    USD: scryfall.prices?.usd ?? \"\",",
      "    RunningTotal: mygui.addToTotal(usd).toFixed(2),",
      "  });",
      "",
      "  mygui.priceBurst(scryfall.name, usd);",
      "  mygui.log(\"Running total is now $\" + mygui.total.toFixed(2));",
    ].join("\n"),
  },
];

const events = document.getElementById("events");
const editorElement = document.getElementById("handler-code");
const presetSelect = document.getElementById("preset-code");
const cornerThresholdInput = document.getElementById("scan-corner-threshold");
const cornerThresholdLabel = document.getElementById("scan-corner-threshold-label");
const cornerSignalFill = document.getElementById("scan-corner-signal-fill");
const cornerSignalThreshold = document.getElementById("scan-corner-signal-threshold");
const cornerSignalValue = document.getElementById("scan-corner-signal-value");
const matchSignalFill = document.getElementById("scan-match-signal-fill");
const matchSignalThreshold = document.getElementById("scan-match-signal-threshold");
const matchSignalValue = document.getElementById("scan-match-signal-value");
const thresholdInput = document.getElementById("scan-threshold");
const consecutiveInput = document.getElementById("scan-consecutive");
const intervalInput = document.getElementById("scan-interval");
const intervalLabel = document.getElementById("scan-interval-label");
const webGpuInput = document.getElementById("scan-webgpu");
const fpsOverlayInput = document.getElementById("scan-fps-overlay");
const groupSecondaryInput = document.getElementById("scan-group-secondary");
const tableWrap = document.getElementById("table-wrap");
const effectsLayer = document.getElementById("effects-layer");
const totalLabel = document.createElement("div");

const rows = [];
const columns = [];
const logLines = [];
const bouncingCards = [];
let handleCard = null;
let runningTotal = 0;
let scanner = null;

const codeEditor = CodeJar(
  editorElement,
  (editor) => window.Prism.highlightElement(editor),
  { tab: "  " },
);

totalLabel.className = "running-total";
totalLabel.hidden = true;
effectsLayer.append(totalLabel);
populatePresetSelect();
populateScanSettings();
codeEditor.updateCode(localStorage.getItem(CODE_KEY) || activePreset().code, false);
highlightReadonlySignature();
requestAnimationFrame(animateBouncingCards);

function populatePresetSelect() {
  presetSelect.innerHTML = PRESETS.map((preset) => (
    `<option value="${preset.id}">${escapeHtml(preset.label)}</option>`
  )).join("");
  presetSelect.value = localStorage.getItem(PRESET_KEY) || PRESETS[0].id;
}

function activePreset() {
  return PRESETS.find((preset) => preset.id === presetSelect.value) ?? PRESETS[0];
}

function readScanSettings() {
  try {
    return {
      ...DEFAULT_SCAN_SETTINGS,
      ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}"),
    };
  } catch {
    return { ...DEFAULT_SCAN_SETTINGS };
  }
}

function populateScanSettings() {
  const settings = readScanSettings();
  const cornerThreshold = clamp(Number(settings.minCornerConfidence), 0, MAX_GUI_CORNER_CONFIDENCE);
  cornerThresholdInput.value = cornerThreshold.toFixed(2);
  updateCornerThresholdUi(cornerThreshold);
  thresholdInput.value = settings.matchThreshold.toFixed(2);
  updateMatchSignal(null);
  consecutiveInput.value = String(settings.consecutiveMatches);
  intervalInput.value = String(Math.max(0, Math.round(Number(settings.scanIntervalMs) || 0)));
  intervalLabel.textContent = Number(intervalInput.value) <= 0 ? "Max speed" : `${intervalInput.value}ms`;
  webGpuInput.checked = settings.enableWebGpu === true;
  fpsOverlayInput.checked = settings.showFpsOverlay !== false;
  groupSecondaryInput.checked = settings.groupBySecondaryId === true;
}

function updateCornerThresholdUi(value) {
  const threshold = clamp(Number(value), 0, MAX_GUI_CORNER_CONFIDENCE);
  cornerThresholdLabel.textContent = threshold.toFixed(2);
  const ratio = threshold / MAX_GUI_CORNER_CONFIDENCE;
  cornerSignalThreshold.style.left = `${(ratio * 100).toFixed(1)}%`;
}

function updateCornerSignal(confidence) {
  const raw = Math.max(0, Number(confidence) || 0);
  const current = clamp(raw, 0, MAX_GUI_CORNER_CONFIDENCE);
  const ratio = current / MAX_GUI_CORNER_CONFIDENCE;
  cornerSignalFill.style.width = `${(ratio * 100).toFixed(1)}%`;
  cornerSignalValue.textContent = raw > MAX_GUI_CORNER_CONFIDENCE
    ? `Current ${MAX_GUI_CORNER_CONFIDENCE.toFixed(2)}+`
    : `Current ${current.toFixed(2)}`;
}

function updateMatchSignal(score) {
  const threshold = clamp(Number(thresholdInput.value), 0, 1);
  matchSignalThreshold.style.left = `${(threshold * 100).toFixed(1)}%`;
  if (!Number.isFinite(score)) {
    matchSignalFill.style.width = "0%";
    matchSignalValue.textContent = "Current —";
    return;
  }
  const current = clamp(score, 0, 1);
  matchSignalFill.style.width = `${(current * 100).toFixed(1)}%`;
  matchSignalValue.textContent = `Current ${current.toFixed(3)}`;
}

function scanSettingsFromInputs() {
  const minCornerConfidence = clamp(Number(cornerThresholdInput.value), 0, MAX_GUI_CORNER_CONFIDENCE);
  const matchThreshold = clamp(Number(thresholdInput.value), 0, 1);
  const consecutiveMatches = Math.max(1, Math.round(Number(consecutiveInput.value) || 1));
  const scanIntervalMs = Math.max(0, Math.round(Number(intervalInput.value) || 0));
  const enableWebGpu = webGpuInput.checked === true;
  const showFpsOverlay = fpsOverlayInput.checked === true;
  const groupBySecondaryId = groupSecondaryInput.checked === true;
  return {
    minCornerConfidence,
    matchThreshold,
    consecutiveMatches,
    scanIntervalMs,
    enableWebGpu,
    showFpsOverlay,
    groupBySecondaryId,
  };
}

async function applyScanSettings({ announce = true } = {}) {
  const settings = scanSettingsFromInputs();
  cornerThresholdInput.value = settings.minCornerConfidence.toFixed(2);
  updateCornerThresholdUi(settings.minCornerConfidence);
  thresholdInput.value = settings.matchThreshold.toFixed(2);
  updateMatchSignal(null);
  consecutiveInput.value = String(settings.consecutiveMatches);
  intervalInput.value = String(settings.scanIntervalMs);
  intervalLabel.textContent = settings.scanIntervalMs <= 0 ? "Max speed" : `${settings.scanIntervalMs}ms`;
  webGpuInput.checked = settings.enableWebGpu;
  fpsOverlayInput.checked = settings.showFpsOverlay;
  groupSecondaryInput.checked = settings.groupBySecondaryId;
  const shouldRecreateScanner = !!scanner && scanner.config.enableWebGpu !== settings.enableWebGpu;
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  if (shouldRecreateScanner) {
    scanner.dispose();
    scanner = await createScanner(settings);
  } else {
    scanner?.updateConfig(settings);
  }
  if (announce) {
    log(
      "Scan settings:",
      `corner threshold ${settings.minCornerConfidence.toFixed(2)},`,
      `threshold ${settings.matchThreshold.toFixed(2)},`,
      `${settings.consecutiveMatches} consecutive,`,
      settings.scanIntervalMs <= 0 ? "max-speed scanning," : `${settings.scanIntervalMs}ms interval,`,
      settings.enableWebGpu ? "WebGPU on," : "WebGPU off,",
      settings.showFpsOverlay ? "FPS overlay on," : "FPS overlay off,",
      settings.groupBySecondaryId ? "grouping by secondary ID" : "grouping by card ID",
    );
  }
  return settings;
}

intervalInput.addEventListener("input", () => {
  const value = Math.max(0, Math.round(Number(intervalInput.value) || 0));
  intervalLabel.textContent = value <= 0 ? "Max speed" : `${value}ms`;
});

cornerThresholdInput.addEventListener("input", () => {
  updateCornerThresholdUi(cornerThresholdInput.value);
});

function highlightReadonlySignature() {
  document.querySelectorAll(".function-signature code").forEach((element) => {
    window.Prism.highlightElement(element);
  });
}

function log(...parts) {
  const line = parts.map((part) => (
    typeof part === "string" ? part : JSON.stringify(part, null, 2)
  )).join(" ");

  logLines.unshift(`${new Date().toLocaleTimeString()}  ${line}`);
  while (logLines.length > LOG_LIMIT) {
    logLines.pop();
  }
  events.textContent = logLines.join("\n");
}

if (self.crossOriginIsolated) {
  log("COI mode enabled for performance; cross-origin art effects may use local fallback tiles.");
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`Fetch failed ${response.status}: ${url}`);
  }
  return response.json();
}

function addRow(row) {
  rows.unshift(row);
  for (const key of Object.keys(row)) {
    if (!columns.includes(key)) {
      columns.push(key);
    }
  }
  renderTable();
}

function renderTable() {
  if (!rows.length) {
    tableWrap.innerHTML = `<p class="empty">Detected cards will appear here.</p>`;
    return;
  }

  const head = columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("");
  const body = rows.map((row) => (
    `<tr>${columns.map((column) => `<td>${escapeHtml(row[column] ?? "")}</td>`).join("")}</tr>`
  )).join("");

  tableWrap.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "'": "&#39;",
    "\"": "&quot;",
  }[char]));
}

function cardImageUrl(scryfall) {
  return scryfall.image_uris?.small
    ?? scryfall.card_faces?.[0]?.image_uris?.small
    ?? scryfall.image_uris?.normal
    ?? scryfall.card_faces?.[0]?.image_uris?.normal
    ?? "";
}

function cameraRect() {
  return document.getElementById("collectorvision").getBoundingClientRect();
}

function isCrossOriginUrl(url) {
  try {
    return new URL(url, window.location.href).origin !== window.location.origin;
  } catch {
    return false;
  }
}

function bounceCard(imageUrl, label = "Scanned card") {
  const origin = cameraRect();
  const shouldAvoidCrossOriginImage = self.crossOriginIsolated
    && imageUrl
    && isCrossOriginUrl(imageUrl);

  let element;
  if (!imageUrl || shouldAvoidCrossOriginImage) {
    if (!imageUrl) {
      log("No card image available for", label);
    } else {
      log("Cross-origin image blocked under COI; using local fallback tile for", label);
    }
    const fallback = document.createElement("div");
    fallback.className = "bouncing-card";
    fallback.textContent = String(label || "Scanned card").slice(0, 24);
    fallback.style.display = "grid";
    fallback.style.placeItems = "center";
    fallback.style.padding = "0.5rem";
    fallback.style.textAlign = "center";
    fallback.style.fontSize = "0.72rem";
    fallback.style.lineHeight = "1.1";
    fallback.style.fontWeight = "700";
    fallback.style.color = "#e2e8f0";
    fallback.style.background = "linear-gradient(160deg, #0f172a, #334155)";
    element = fallback;
  } else {
    const image = document.createElement("img");
    image.className = "bouncing-card";
    image.referrerPolicy = "no-referrer";
    image.decoding = "async";
    image.alt = label;
    image.addEventListener("error", () => {
      log("Could not load card image; showing name fallback for", label, imageUrl);
    }, { once: true });
    image.src = imageUrl;
    element = image;
  }
  effectsLayer.append(element);

  const width = 92;
  const height = 128;
  const startX = origin.left + origin.width / 2 - width / 2;
  const startY = origin.top + origin.height / 2 - height / 2;
  const speed = 2.2 + Math.random() * 2.6;
  const angle = -Math.PI / 3 + Math.random() * Math.PI * 1.66;

  bouncingCards.push({
    element,
    x: clamp(startX, 0, window.innerWidth - width),
    y: clamp(startY, 0, window.innerHeight - height),
    dx: Math.cos(angle) * speed || speed,
    dy: Math.sin(angle) * speed || speed,
    width,
    height,
  });

  while (bouncingCards.length > 24) {
    bouncingCards.shift().element.remove();
  }
}

function animateBouncingCards() {
  for (const card of bouncingCards) {
    card.x += card.dx;
    card.y += card.dy;

    if (card.x <= 0 || card.x + card.width >= window.innerWidth) {
      card.dx *= -1;
      card.x = clamp(card.x, 0, window.innerWidth - card.width);
    }
    if (card.y <= 0 || card.y + card.height >= window.innerHeight) {
      card.dy *= -1;
      card.y = clamp(card.y, 0, window.innerHeight - card.height);
    }

    card.element.style.transform = `translate(${card.x}px, ${card.y}px)`;
  }
  requestAnimationFrame(animateBouncingCards);
}

function priceBurst(name, usd) {
  const burst = document.createElement("div");
  burst.className = "price-burst";
  burst.textContent = usd > 0 ? `$${usd.toFixed(2)}` : "Priceless ✨";
  effectsLayer.append(burst);

  for (let index = 0; index < 18; index += 1) {
    const sparkle = document.createElement("span");
    sparkle.className = "sparkle";
    sparkle.textContent = ["✦", "✨", "✧", "★"][index % 4];
    sparkle.style.setProperty("--x", `${Math.cos(index) * (60 + Math.random() * 140)}px`);
    sparkle.style.setProperty("--y", `${Math.sin(index) * (60 + Math.random() * 140)}px`);
    burst.append(sparkle);
  }

  window.setTimeout(() => burst.remove(), 1600);
  log(name, usd > 0 ? `is worth $${usd.toFixed(2)}` : "has no USD price today");
}

function updateTotalLabel({ pulse = false } = {}) {
  totalLabel.textContent = `Total $${runningTotal.toFixed(2)}`;
  totalLabel.hidden = runningTotal <= 0;
  if (pulse && runningTotal > 0) {
    totalLabel.classList.remove("running-total--pulse");
    void totalLabel.offsetWidth;
    totalLabel.classList.add("running-total--pulse");
  }
}

function addToTotal(amount) {
  runningTotal += Number.isFinite(amount) ? amount : 0;
  updateTotalLabel({ pulse: amount > 0 });
  return runningTotal;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function compileHandler() {
  const source = codeEditor.toString();
  localStorage.setItem(CODE_KEY, source);
  handleCard = new Function(
    "card",
    "mygui",
    `"use strict"; async function onCardScanned(card, mygui) {\n${source}\n}\nreturn onCardScanned(card, mygui);`,
  );
  log("Handler applied. Scan a card to run it.");
}

const mygui = {
  addRow,
  addToTotal,
  bounceCard,
  cameraRect,
  cardImageUrl,
  clear() {
    rows.length = 0;
    columns.length = 0;
    runningTotal = 0;
    updateTotalLabel();
    renderTable();
  },
  fetchJson,
  log,
  priceBurst,
  get rows() {
    return rows;
  },
  get total() {
    return runningTotal;
  },
};

document.getElementById("apply-code").addEventListener("click", () => {
  try {
    compileHandler();
  } catch (error) {
    log("Handler error:", error.message);
  }
});

document.getElementById("reset-code").addEventListener("click", () => {
  localStorage.setItem(PRESET_KEY, presetSelect.value);
  codeEditor.updateCode(activePreset().code, false);
  compileHandler();
});

document.getElementById("clear-table").addEventListener("click", mygui.clear);
document.getElementById("apply-scan-settings").addEventListener("click", async () => {
  await applyScanSettings();
});

presetSelect.addEventListener("change", () => {
  localStorage.setItem(PRESET_KEY, presetSelect.value);
  log("Preset selected:", activePreset().label, "Press Load preset to use it.");
});

compileHandler();

async function createScanner(settings) {
  return createCollectorVisionScannerApplet({
    target: "#collectorvision",
    manifestUrl: `${assetBasePath}/manifest.json`,
    assetBasePath,
    ...settings,
    overlay: true,
    onResult(result) {
      updateCornerSignal(result?.sharpness ?? 0);
      updateMatchSignal(result?.score);
    },
    async onCardDetected(card) {
      try {
        await handleCard?.(card, { ...mygui, scanner });
      } catch (error) {
        log("Handler error:", error.message);
      }
    },
    onError(error) {
      log("Scanner error:", error.message);
    },
  });
}

scanner = await createScanner(await applyScanSettings({ announce: false }));

log(`Using ${assetChannel} assets.`);

window.collectorVisionScanner = scanner;
