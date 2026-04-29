import { CodeJar } from "https://cdn.jsdelivr.net/npm/codejar@4.3.0/dist/codejar.js";
import { createCollectorVisionScannerApplet } from "./lib/collectorvision-scanner-applet.mjs";

const CODE_KEY = "collectorvision_applet_example_code";
const LOG_LIMIT = 12;
const SCRYFALL_CARD_URL = "https://api.scryfall.com/cards/${card.cardId}";

const DEFAULT_HANDLER = [
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
].join("\n");

const events = document.getElementById("events");
const editorElement = document.getElementById("handler-code");
const tableWrap = document.getElementById("table-wrap");

const rows = [];
const columns = [];
const logLines = [];
let handleCard = null;

const codeEditor = CodeJar(
  editorElement,
  (editor) => window.Prism.highlightElement(editor),
  { tab: "  " },
);
codeEditor.updateCode(localStorage.getItem(CODE_KEY) || DEFAULT_HANDLER, false);
highlightReadonlySignature();

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
  clear() {
    rows.length = 0;
    columns.length = 0;
    renderTable();
  },
  fetchJson,
  log,
  get rows() {
    return rows;
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
  codeEditor.updateCode(DEFAULT_HANDLER, false);
  compileHandler();
});

document.getElementById("clear-table").addEventListener("click", mygui.clear);

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
