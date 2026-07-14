const CHANNEL_NAME = "collectorvision-monitor";
const CACHE_KEY = "collectorvision_overlay_scryfall_cache_v1";
const CACHE_MAX_ENTRIES = 80;
const DEFAULT_HIDE_MS = 8000;

const card = document.getElementById("overlay-card");
const artEl = document.getElementById("card-art");
const nameEl = document.getElementById("card-name");
const setEl = document.getElementById("card-set");
const rarityEl = document.getElementById("card-rarity");
const priceEl = document.getElementById("card-price");
let hideTimer = null;
let latestRequest = 0;
let scryfallCache = readCache();
const display = readDisplayOptions();

function readDisplayOptions() {
  const params = new URLSearchParams(location.search);
  const enabled = (name, fallback = true) => {
    const value = params.get(name);
    return value === null ? fallback : value !== "0" && value !== "false";
  };
  const choice = (name, choices, fallback) => {
    const value = params.get(name);
    return choices.includes(value) ? value : fallback;
  };
  const durationSeconds = Number(params.get("duration"));
  return {
    showName: enabled("name"),
    showSet: enabled("set"),
    showRarity: enabled("rarity"),
    showPrice: enabled("price"),
    background: choice("background", ["glass", "solid", "transparent"], "glass"),
    position: choice("position", ["top-left", "top-right", "bottom-left", "bottom-right"], "bottom-right"),
    size: choice("size", ["medium", "large", "huge"], "large"),
    hideMs: Number.isFinite(durationSeconds) && durationSeconds >= 0
      ? durationSeconds * 1000
      : DEFAULT_HIDE_MS,
  };
}

card.classList.add(
  `background-${display.background}`,
  `position-${display.position}`,
  `size-${display.size}`,
);
card.classList.toggle("show-name", display.showName);
card.classList.toggle("show-set", display.showSet);
card.classList.toggle("show-rarity", display.showRarity);
card.classList.toggle("show-price", display.showPrice);

function readCache() {
  try {
    return JSON.parse(localStorage.getItem(CACHE_KEY) || "{}");
  } catch (error) {
    console.warn("[CollectorVision overlay] Could not read Scryfall cache", error);
    return {};
  }
}

function writeCache() {
  const entries = Object.entries(scryfallCache)
    .sort(([, a], [, b]) => (b.cachedAt ?? 0) - (a.cachedAt ?? 0))
    .slice(0, CACHE_MAX_ENTRIES);
  scryfallCache = Object.fromEntries(entries);
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify(scryfallCache));
  } catch (error) {
    console.warn("[CollectorVision overlay] Could not write Scryfall cache", error);
  }
}

function normalizeScryfall(cardData) {
  const imageUrl = cardData.image_uris?.normal
    ?? cardData.image_uris?.large
    ?? cardData.card_faces?.[0]?.image_uris?.normal
    ?? cardData.card_faces?.[0]?.image_uris?.large
    ?? null;
  return {
    name: cardData.name,
    setName: cardData.set_name,
    rarity: cardData.rarity,
    priceUsd: cardData.prices?.usd ?? cardData.prices?.usd_foil ?? null,
    imageUrl,
    cachedAt: Date.now(),
  };
}

async function fetchScryfall(cardId) {
  const cached = scryfallCache[cardId];
  if (cached?.name) return cached;

  const response = await fetch(`https://api.scryfall.com/cards/${encodeURIComponent(cardId)}`);
  if (!response.ok) {
    throw new Error(`Scryfall lookup failed: HTTP ${response.status}`);
  }
  const data = normalizeScryfall(await response.json());
  if (!data.name) {
    throw new Error("Scryfall response did not include a card name.");
  }
  scryfallCache[cardId] = data;
  writeCache();
  return data;
}

async function preloadImage(src) {
  if (!src) return;
  await new Promise((resolve) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = resolve;
    image.onerror = resolve;
    image.src = src;
  });
}

function showScryfallCard(cardData) {
  nameEl.textContent = cardData.name;
  setEl.textContent = cardData.setName ?? "";
  rarityEl.textContent = cardData.rarity ?? "";
  priceEl.textContent = formatPrice(cardData.priceUsd);
  if (cardData.imageUrl) {
    artEl.src = cardData.imageUrl;
    artEl.alt = cardData.name;
  } else {
    artEl.removeAttribute("src");
    artEl.alt = "";
  }
  card.hidden = false;
  requestAnimationFrame(() => card.classList.add("is-visible"));

  clearTimeout(hideTimer);
  if (display.hideMs > 0) {
    hideTimer = setTimeout(() => {
      card.classList.remove("is-visible");
    }, display.hideMs);
  }
}

function formatPrice(price) {
  const value = Number(price);
  if (!Number.isFinite(value)) return "";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(value);
}

async function handleCardEvent(event) {
  const cardId = String(event?.card?.cardId ?? "").trim();
  if (!cardId) return;
  const requestId = ++latestRequest;
  try {
    const cardData = await fetchScryfall(cardId);
    await preloadImage(cardData.imageUrl);
    if (requestId !== latestRequest) return;
    showScryfallCard(cardData);
  } catch (error) {
    console.warn("[CollectorVision overlay] Scryfall lookup failed", error);
  }
}

const channel = new BroadcastChannel(CHANNEL_NAME);
channel.addEventListener("message", (event) => {
  if (event.data?.type === "collectorvision.card") {
    handleCardEvent(event.data);
  }
});
