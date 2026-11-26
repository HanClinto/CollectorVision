import assert from "node:assert/strict";

import { createProgressTracker } from "../../examples/web_scanner/lib/progress.mjs";

const EMBEDDINGS_BYTES = 28_086_016;
const CARD_IDS_BYTES = 4_408_130;
const ORACLE_IDS_BYTES = 4_388_440;
const CATALOG_BYTES = EMBEDDINGS_BYTES + CARD_IDS_BYTES + ORACLE_IDS_BYTES;

function progress(stage, loaded, total, ratio = total > 0 ? loaded / total : 0) {
  return { stage, loaded, total, ratio };
}

{
  const track = createProgressTracker({
    catalogMode: "v1",
    bundledCatalogBytes: CATALOG_BYTES,
  });
  const updates = [
    progress("catalog", 0, EMBEDDINGS_BYTES),
    progress("catalog", EMBEDDINGS_BYTES, EMBEDDINGS_BYTES),
    progress("catalog", 0, CARD_IDS_BYTES),
    progress("catalog", CARD_IDS_BYTES, CARD_IDS_BYTES),
    progress("catalog", 0, ORACLE_IDS_BYTES),
    progress("catalog", ORACLE_IDS_BYTES, ORACLE_IDS_BYTES),
  ].map(track);

  assert.deepEqual(
    updates.map(({ loaded }) => loaded),
    [
      0,
      EMBEDDINGS_BYTES,
      EMBEDDINGS_BYTES,
      EMBEDDINGS_BYTES + CARD_IDS_BYTES,
      EMBEDDINGS_BYTES + CARD_IDS_BYTES,
      CATALOG_BYTES,
    ],
  );
  assert.ok(updates.every(({ total }) => total === CATALOG_BYTES));
  assert.ok(updates.every((update, index) => index === 0 || update.ratio >= updates[index - 1].ratio));
}

{
  const track = createProgressTracker({
    catalogMode: "v1",
    bundledCatalogBytes: CATALOG_BYTES,
  });
  const updates = [
    progress("catalog", EMBEDDINGS_BYTES, CATALOG_BYTES),
    progress("catalog", EMBEDDINGS_BYTES + CARD_IDS_BYTES, CATALOG_BYTES),
    progress("catalog", CATALOG_BYTES, CATALOG_BYTES),
  ].map(track);

  assert.deepEqual(updates.map(({ loaded }) => loaded), [
    EMBEDDINGS_BYTES,
    EMBEDDINGS_BYTES + CARD_IDS_BYTES,
    CATALOG_BYTES,
  ]);
}

{
  const track = createProgressTracker();
  track(progress("catalog", 12_000, 20_000, 0.6));
  const completed = track(progress("catalog", 0, 0, 1));

  assert.equal(completed.loaded, 12_000);
  assert.equal(completed.total, 20_000);
  assert.equal(completed.ratio, 1);
}

console.log("Catalog progress regression tests passed");
