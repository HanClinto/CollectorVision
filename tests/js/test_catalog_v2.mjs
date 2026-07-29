import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";

import {
  BrowserCatalogV2,
  CatalogV2BrowserClient,
  CatalogV2Error,
} from "../../examples/web_scanner/lib/collectorvision-catalog-v2.mjs";

if (!globalThis.crypto) globalThis.crypto = webcrypto;

const encoder = new TextEncoder();
const tag = "catalog-v2-beta.1-2026-07-24";
const key = "milo1/scryfall/mtg";

async function gzip(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function asset(filename, payload) {
  const compressed = await gzip(payload);
  return {
    compressed,
    descriptor: {
      filename,
      size: compressed.byteLength,
      sha256: await sha256(compressed),
    },
  };
}

async function sha256(bytes) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...digest].map((value) => value.toString(16).padStart(2, "0")).join("");
}

function jsonLines(values) {
  return encoder.encode(values.map((value) => JSON.stringify(value)).join("\n") + "\n");
}

function fp16(values) {
  const bytes = new Uint8Array(values.length * 2);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setUint16(index * 2, value, true));
  return bytes;
}

async function fixture() {
  const rows = await asset(
    "demo.recognition.jsonl.gz",
    jsonLines([
      { key: "card:a:face:0", identifiers: { source_card: "a" } },
      { key: "card:b:face:1", identifiers: { source_card: "b" }, face_index: 1 },
    ]),
  );
  const matrix = await asset("demo.recognition.f16.gz", fp16([0x3c00, 0, 0, 0x3c00]));
  const metadata = await asset(
    "demo.metadata.jsonl.gz",
    jsonLines([
      { key: "card:a:face:0", metadata: { name: "Alpha" } },
      { key: "card:b:face:1", metadata: { name: "Beta" } },
    ]),
  );
  const manifest = {
    schema_version: 2,
    catalog_key: key,
    version: tag,
    embedding_model: "collectorvision@test:milo",
    rows: 2,
    dim: 2,
    dtype: "float16",
    descriptor: {
      game: "magic-the-gathering",
      source: "scryfall",
      profile: "printings",
      description: "Demo",
      result_identifier: "source_card",
      recommended: true,
    },
    assets: {
      recognition_rows: rows.descriptor,
      recognition_matrix: matrix.descriptor,
      metadata_rows: metadata.descriptor,
    },
  };
  const manifestBytes = encoder.encode(JSON.stringify(manifest));
  const index = {
    schema_version: 2,
    release_version: tag,
    catalogs: {
      [key]: {
        manifest_filename: "demo.manifest.json",
        sha256: await sha256(manifestBytes),
      },
    },
  };
  const files = new Map([
    [
      "catalog-feed-v2.json",
      encoder.encode(
        JSON.stringify({
          schema_version: 2,
          release_version: tag,
          catalogs: {
            [key]: {
              base: {
                version: tag,
                manifest_filename: "demo.manifest.json",
                sha256: await sha256(manifestBytes),
              },
              deltas: [],
            },
          },
        }),
      ),
    ],
    [`${tag}/catalog-index-v2.json`, encoder.encode(JSON.stringify(index))],
    [`${tag}/demo.manifest.json`, manifestBytes],
    [`${tag}/${rows.descriptor.filename}`, rows.compressed],
    [`${tag}/${matrix.descriptor.filename}`, matrix.compressed],
    [`${tag}/${metadata.descriptor.filename}`, metadata.compressed],
  ]);
  return files;
}

function mockFetch(files) {
  return async (url) => {
    const path = new URL(url).pathname.replace(/^\/+/, "");
    const payload = files.get(path);
    return payload
      ? new Response(payload, { status: 200 })
      : new Response("missing", { status: 404 });
  };
}

class MemorySnapshotCache {
  constructor() {
    this.snapshots = new Map();
  }

  async get(tag, catalogKey, includeMetadata) {
    return this.snapshots.get(`${tag}\0${catalogKey}\0${includeMetadata}`) ?? null;
  }

  async put(catalog) {
    this.snapshots.set(
      `${catalog.version}\0${catalog.catalogKey}\0${catalog.metadataLoaded}`,
      catalog,
    );
  }
}

const files = await fixture();
const snapshotCache = new MemorySnapshotCache();
const client = new CatalogV2BrowserClient({
  releaseBaseUrl: "https://catalog.test/",
  fetchImpl: mockFetch(files),
  cache: snapshotCache,
});
const catalog = await client.load(tag, key, { includeMetadata: true });

assert(catalog instanceof BrowserCatalogV2);
assert.equal(catalog.records.length, 2);
assert.equal(catalog.recordForIndex(1).metadata.name, "Beta");
assert.deepEqual(catalog.search(new Float32Array([0, 1]), 1), [[1, "b"]]);
assert.equal(catalog.recordForIndex(1).face_index, 1);

const simpleCatalog = await BrowserCatalogV2.forGame("mtg", {
  releaseBaseUrl: "https://catalog.test/",
  fetchImpl: mockFetch(files),
});
assert.equal(simpleCatalog.catalogKey, key);
await assert.rejects(
  () =>
    BrowserCatalogV2.forGame("pokemon", {
      source: "scryfall",
      fetchImpl: mockFetch(files),
    }),
  /only available for MTG/,
);

const nextTag = "catalog-v2-beta.2-2026-07-25";
const operations = await asset(
  "demo.delta.jsonl.gz",
  jsonLines([
    { op: "delete", key: "card:a:face:0" },
    {
      op: "upsert",
      record: { key: "card:B:face:0", identifiers: { source_card: "c" } },
      embedding_index: 0,
    },
  ]),
);
const deltaMatrix = await asset("demo.delta.f16.gz", fp16([0x3c00, 0]));
const metadataDelta = await asset(
  "demo.metadata.delta.jsonl.gz",
  jsonLines([
    { op: "delete", key: "card:a:face:0" },
    { op: "upsert", key: "card:B:face:0", metadata: { name: "Gamma" } },
  ]),
);
const nextManifest = {
  ...JSON.parse(new TextDecoder().decode(files.get(`${tag}/demo.manifest.json`))),
  version: nextTag,
  delta: {
    base_version: tag,
    requires_exact_base: true,
    operations: 2,
    metadata_operations: 2,
  },
};
nextManifest.assets = {
  ...nextManifest.assets,
  delta_operations: operations.descriptor,
  delta_matrix: deltaMatrix.descriptor,
  metadata_delta: metadataDelta.descriptor,
};
const nextManifestBytes = encoder.encode(JSON.stringify(nextManifest));
const nextIndex = {
  schema_version: 2,
  release_version: nextTag,
  catalogs: {
    [key]: {
      manifest_filename: "demo.manifest.json",
      sha256: await sha256(nextManifestBytes),
    },
  },
};
files.set(`${nextTag}/catalog-index-v2.json`, encoder.encode(JSON.stringify(nextIndex)));
files.set(`${nextTag}/demo.manifest.json`, nextManifestBytes);
files.set(`${nextTag}/${operations.descriptor.filename}`, operations.compressed);
files.set(`${nextTag}/${deltaMatrix.descriptor.filename}`, deltaMatrix.compressed);
files.set(`${nextTag}/${metadataDelta.descriptor.filename}`, metadataDelta.compressed);
files.set(
  "catalog-feed-v2.json",
  encoder.encode(
    JSON.stringify({
      schema_version: 2,
      release_version: nextTag,
      catalogs: {
        [key]: {
          base: {
            version: tag,
            manifest_filename: "demo.manifest.json",
            sha256: JSON.parse(new TextDecoder().decode(files.get(`${tag}/catalog-index-v2.json`)))
              .catalogs[key].sha256,
          },
          deltas: [
            {
              from: tag,
              to: nextTag,
              manifest_filename: "demo.manifest.json",
              sha256: await sha256(nextManifestBytes),
            },
          ],
        },
      },
    }),
  ),
);

const reloadedClient = new CatalogV2BrowserClient({
  releaseBaseUrl: "https://catalog.test/",
  fetchImpl: mockFetch(files),
  cache: snapshotCache,
});
const updated = await reloadedClient.loadFromFeed(key, {
  includeMetadata: true,
});
assert.deepEqual(updated.records.map((record) => record.key), [
  "card:B:face:0",
  "card:b:face:1",
]);
assert.deepEqual(updated.search(new Float32Array([1, 0]), 1), [[1, "c"]]);
assert.equal(updated.recordForIndex(0).metadata.name, "Gamma");

const updatedWithoutMetadata = await client.load(nextTag, key, {
  previous: catalog,
});
assert.equal("metadata" in updatedWithoutMetadata.recordForIndex(0), false);
assert.equal("metadata" in updatedWithoutMetadata.recordForIndex(1), false);

const deleteTag = "catalog-v2-beta.3-2026-07-26";
const deleteOperations = await asset(
  "demo.delete.delta.jsonl.gz",
  jsonLines([{ op: "delete", key: "card:B:face:0" }]),
);
const deleteManifest = {
  ...nextManifest,
  version: deleteTag,
  rows: 1,
  delta: {
    base_version: nextTag,
    requires_exact_base: true,
    operations: 1,
    metadata_operations: 0,
  },
  assets: {
    ...nextManifest.assets,
    delta_operations: deleteOperations.descriptor,
  },
};
delete deleteManifest.assets.delta_matrix;
delete deleteManifest.assets.metadata_delta;
const deleteManifestBytes = encoder.encode(JSON.stringify(deleteManifest));
files.set(
  `${deleteTag}/catalog-index-v2.json`,
  encoder.encode(
    JSON.stringify({
      schema_version: 2,
      release_version: deleteTag,
      catalogs: {
        [key]: {
          manifest_filename: "demo.manifest.json",
          sha256: await sha256(deleteManifestBytes),
        },
      },
    }),
  ),
);
files.set(`${deleteTag}/demo.manifest.json`, deleteManifestBytes);
files.set(
  `${deleteTag}/${deleteOperations.descriptor.filename}`,
  deleteOperations.compressed,
);
const deleteOnly = await client.load(deleteTag, key, {
  previous: updatedWithoutMetadata,
});
assert.deepEqual(deleteOnly.records.map((record) => record.key), ["card:b:face:1"]);

await assert.rejects(
  () => client.load("latest", key),
  /explicit immutable beta tag/,
);
const tampered = new Map(files);
tampered.set(`${tag}/demo.recognition.f16.gz`, new Uint8Array([1, 2, 3]));
const tamperedClient = new CatalogV2BrowserClient({
  releaseBaseUrl: "https://catalog.test/",
  fetchImpl: mockFetch(tampered),
});
await assert.rejects(
  () => tamperedClient.load(tag, key),
  CatalogV2Error,
);

const warnings = [];
const originalWarn = console.warn;
console.warn = (...values) => warnings.push(values);
try {
  const failingCacheClient = new CatalogV2BrowserClient({
    releaseBaseUrl: "https://catalog.test/",
    fetchImpl: mockFetch(files),
    cache: {
      async get() {
        throw new Error("read unavailable");
      },
      async put() {
        throw new Error("quota exceeded");
      },
    },
  });
  assert.equal((await failingCacheClient.load(tag, key)).records.length, 2);

  const incompatibleCache = new MemorySnapshotCache();
  incompatibleCache.snapshots.set(`${tag}\0${key}\0false`, updated);
  const repairingClient = new CatalogV2BrowserClient({
    releaseBaseUrl: "https://catalog.test/",
    fetchImpl: mockFetch(files),
    cache: incompatibleCache,
  });
  assert.equal((await repairingClient.load(tag, key)).version, tag);
} finally {
  console.warn = originalWarn;
}
assert.equal(warnings.length, 3);

console.log("Catalog v2 browser tests passed");
