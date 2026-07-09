const CHANNEL_NAME = "collectorvision-monitor";
const CACHE_KEY = "collectorvision_overlay_scryfall_cache_v1";
const CACHE_MAX_ENTRIES = 80;
const DEFAULT_HIDE_MS = 8000;

const card = document.getElementById("overlay-card");
const artEl = document.getElementById("card-art");
const nameEl = document.getElementById("card-name");
const metaEl = document.getElementById("card-meta");
let hideTimer = null;
let latestRequest = 0;
let scryfallCache = readCache();

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
  metaEl.textContent = [cardData.setName, cardData.rarity].filter(Boolean).join(" · ");
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
  hideTimer = setTimeout(() => {
    card.classList.remove("is-visible");
  }, DEFAULT_HIDE_MS);
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
