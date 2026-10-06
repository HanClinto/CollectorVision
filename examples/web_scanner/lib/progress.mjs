export function createProgressTracker({ catalogMode, bundledCatalogBytes = 0 } = {}) {
  const stages = new Map();

  return function trackProgress(data) {
    const previous = stages.get(data.stage);
    const rawLoaded = Math.max(0, Number(data.loaded) || 0);
    const rawTotal = Math.max(0, Number(data.total) || 0);
    let loaded = rawLoaded;
    let total = rawTotal;
    let ratio = Math.max(0, Math.min(1, Number(data.ratio) || 0));
    let offset = previous?.offset ?? 0;

    // Older workers report each bundled catalog file independently. Combine
    // those reports into the manifest's authoritative aggregate size.
    if (data.stage === "catalog" && catalogMode === "v1" && bundledCatalogBytes > 0) {
      if (rawTotal === bundledCatalogBytes) {
        offset = 0;
      } else if (previous && rawLoaded < previous.rawLoaded) {
        offset = previous.loaded;
      }
      loaded = Math.min(bundledCatalogBytes, offset + rawLoaded);
      total = bundledCatalogBytes;
      ratio = loaded / total;
    }

    // Keep completion-only events from erasing useful byte details, and keep
    // every stage monotonic if messages arrive late or restart at zero.
    if (previous && loaded < previous.loaded) {
      loaded = previous.loaded;
      total = previous.total;
    }
    if (previous) {
      ratio = Math.max(ratio, previous.ratio);
    }

    const progress = { loaded, total, ratio, rawLoaded, offset };
    stages.set(data.stage, progress);
    return progress;
  };
}
