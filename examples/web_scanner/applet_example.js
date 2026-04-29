import { CodeJar } from "https://cdn.jsdelivr.net/npm/codejar@4.3.0/dist/codejar.js";
import { createCollectorVisionScannerApplet } from "./lib/collectorvision-scanner-applet.mjs";

const CODE_KEY = "collectorvision_applet_example_code";
const PRESET_KEY = "collectorvision_applet_example_preset";
const LOG_LIMIT = 12;
const SCRYFALL_CARD_URL = "https://api.scryfall.com/cards/${card.cardId}";

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
const tableWrap = document.getElementById("table-wrap");
const effectsLayer = document.getElementById("effects-layer");

const rows = [];
const columns = [];
const logLines = [];
const bouncingCards = [];
let handleCard = null;
let runningTotal = 0;

const codeEditor = CodeJar(
  editorElement,
  (editor) => window.Prism.highlightElement(editor),
  { tab: "  " },
);

populatePresetSelect();
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

function bounceCard(imageUrl, label = "Scanned card") {
  if (!imageUrl) {
    log("No card image available for", label);
    return;
  }

  const origin = cameraRect();
  const image = document.createElement("img");
  image.className = "bouncing-card";
  image.src = imageUrl;
  image.alt = label;
  effectsLayer.append(image);

  const width = 92;
  const height = 128;
  const startX = origin.left + origin.width / 2 - width / 2;
  const startY = origin.top + origin.height / 2 - height / 2;
  const speed = 2.2 + Math.random() * 2.6;
  const angle = -Math.PI / 3 + Math.random() * Math.PI * 1.66;

  bouncingCards.push({
    element: image,
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

function addToTotal(amount) {
  runningTotal += Number.isFinite(amount) ? amount : 0;
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

presetSelect.addEventListener("change", () => {
  localStorage.setItem(PRESET_KEY, presetSelect.value);
  log("Preset selected:", activePreset().label, "Press Load preset to use it.");
});

compileHandler();

const scanner = await createCollectorVisionScannerApplet({
  target: "#collectorvision",
  matchThreshold: 0.60,
  consecutiveMatches: 2,
  scanIntervalMs: 900,
  overlay: true,
  enableWebGpu: false,
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

window.collectorVisionScanner = scanner;
