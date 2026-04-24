// enricher.worker.mjs
// Handles Scryfall card metadata lookups on a dedicated background thread.
// This keeps price fetches completely decoupled from the scan pipeline.
//
// Message protocol
// ----------------
// main → worker:   { type: 'enrich', cardId: string }
// worker → main:   { type: 'enriched',   cardId, name, set, setName, priceUsd }
//                  { type: 'enrichError', cardId, message }

const cache = new Map();

async function fetchScryfallCard(cardId) {
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

self.onmessage = async ({ data }) => {
  if (data.type !== "enrich") {
    return;
  }
  try {
    const card = await fetchScryfallCard(data.cardId);
    self.postMessage({
      type: "enriched",
      cardId: data.cardId,
      name: card.name,
      set: card.set,
      setName: card.set_name,
      priceUsd: card.prices?.usd ?? null,
    });
  } catch (error) {
    self.postMessage({
      type: "enrichError",
      cardId: data.cardId,
      message: error?.message ?? String(error),
    });
  }
};
