/**
 * Node.js regression tests for the CollectorVision Catalog v2 browser client.
 *
 * These tests build fully synthetic, deterministic feeds and gzip assets in
 * memory (no network, no real CollectorVisionCatalog data) that follow the
 * live `catalog-feed-v2.json` contract described in
 * HanClinto/CollectorVisionCatalog `docs/catalog-v2.md` and
 * `docs/versioning.md`: a feed of embedding families, each nesting catalogs
 * keyed by `family/local-key`, each catalog advertising a base snapshot and
 * zero or more exact-predecessor updates through `current_version`.
 *
 * Usage
 * -----
 *   cd tests/js && npm install && npm test
 */

import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";

import {
  BrowserCatalogV2,
  CatalogV2FeedClient,
  CatalogV2Error,
} from "../../examples/web_scanner/lib/collectorvision-catalog-v2.mjs";

if (!globalThis.crypto) globalThis.crypto = webcrypto;

// ---------------------------------------------------------------------------
// Minimal test runner (mirrors the style used by test_pipeline.mjs)
// ---------------------------------------------------------------------------

let _passed = 0;
let _failed = 0;
const _failures = [];

async function test(label, fn) {
  try {
    await fn();
    _passed += 1;
    console.log(`  PASS  ${label}`);
  } catch (err) {
    _failed += 1;
    _failures.push({ label, err });
    console.error(`  FAIL  ${label}`);
    console.error(`        ${err.stack ?? err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Synthetic feed/asset fixture helpers
// ---------------------------------------------------------------------------

const encoder = new TextEncoder();
const BASE_URL = "https://catalog.test/catalog-v2/";
const FEED_URL = `${BASE_URL}catalog-feed-v2.json`;

// Bit patterns for the IEEE-754 half-precision values these tests need.
const F16 = {
  0: 0x0000,
  1: 0x3c00,
  [-1]: 0xbc00,
  0.5: 0x3800,
  [-0.5]: 0xb800,
  0.25: 0x3400,
  2: 0x4000,
};

function f16(values) {
  const bytes = new Uint8Array(values.length * 2);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => {
    const bits = value in F16 ? F16[value] : F16[Object.is(value, -0) ? 0 : value];
    if (bits === undefined) throw new Error(`unsupported fp16 test value ${value}`);
    view.setUint16(index * 2, bits, true);
  });
  return bytes;
}

async function gzip(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function sha256(bytes) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...digest].map((value) => value.toString(16).padStart(2, "0")).join("");
}

function jsonLines(values) {
  return encoder.encode(values.map((value) => JSON.stringify(value)).join("\n") + "\n");
}

/** A tiny in-memory HTTPS asset store that gzips JSON-lines/binary payloads and
 * mints authoritative {url, size, sha256} references, matching the live feed
 * contract where every asset reference is trusted on its own. */
class FeedFixture {
  constructor(baseUrl = BASE_URL) {
    this.baseUrl = baseUrl;
    this.files = new Map();
    this.calls = [];
  }

  fetchImpl() {
    return async (url) => {
      const key = url.toString();
      this.calls.push(key);
      const bytes = this.files.get(key);
      return bytes ? new Response(bytes, { status: 200 }) : new Response("missing", { status: 404 });
    };
  }

  setJson(path, value) {
    this.files.set(`${this.baseUrl}${path}`, encoder.encode(JSON.stringify(value)));
  }

  async putRows(path, values) {
    return this.#putGzip(path, jsonLines(values));
  }

  async putEmbeddings(path, values) {
    return this.#putGzip(path, f16(values));
  }

  async #putGzip(path, bytes) {
    const compressed = await gzip(bytes);
    const url = `${this.baseUrl}${path}`;
    this.files.set(url, compressed);
    return { url, size: compressed.byteLength, sha256: await sha256(compressed) };
  }

  replace(url, bytes) {
    this.files.set(url, bytes);
  }
}

function descriptor(overrides = {}) {
  return {
    game: "magic-the-gathering",
    source: "scryfall",
    profile: "default",
    description: "Demo Scryfall catalog.",
    result_identifier: "scryfall_card",
    recommended: true,
    ...overrides,
  };
}

const EMBEDDING = Object.freeze({
  model: "collectorvision@test:milo-1.0.0",
  dimensions: 2,
  dtype: "float16",
  byte_order: "little",
  layout: "row-major",
});

class MemorySnapshotCache {
  constructor() {
    this.snapshots = new Map();
    this.getCalls = 0;
    this.putCalls = 0;
  }

  async get(version, catalogKey, includeMetadata) {
    this.getCalls += 1;
    return this.snapshots.get(`${version}\0${catalogKey}\0${includeMetadata}`) ?? null;
  }

  async put(catalog) {
    this.putCalls += 1;
    this.snapshots.set(
      `${catalog.version}\0${catalog.catalogKey}\0${catalog.metadataLoaded}`,
      catalog,
    );
  }
}

/** Builds a single-family, single-catalog fixture with a base snapshot of two
 * rows ("card-a" front face, "card-b" back face) and, optionally, a chain of
 * updates through version 2 (add "card-c", update "card-a"; then delete
 * "card-b", update "card-c"). Returns the fixture plus the full catalog key. */
async function buildFixture({ withUpdates = false, withMetadata = false } = {}) {
  const fixture = new FeedFixture();
  const key = "fam1/scryfall/mtg";

  const baseIdentifiers = await fixture.putRows("scryfall-mtg/version/0/base/identifiers.jsonl.gz", [
    { id: "card-a", identifiers: { scryfall_oracle: "oracle-a" } },
    { id: "card-b", identifiers: {}, face_index: 1, finishes: ["foil", "nonfoil"] },
  ]);
  const baseEmbeddings = await fixture.putEmbeddings("scryfall-mtg/version/0/base/embeddings.f16.gz", [
    1, 0, 0, 1,
  ]);
  const baseMetadata = await fixture.putRows("scryfall-mtg/version/0/base/metadata.jsonl.gz", [
    { name: "Alpha" },
    withMetadata ? null : { name: "Beta" },
  ]);

  const catalog = {
    public_name: "scryfall-mtg",
    descriptor: descriptor(),
    current_version: 0,
    rows: 2,
    source_updated_at: "2026-07-24T00:00:00Z",
    base: {
      version: 0,
      rows: 2,
      source_updated_at: "2026-07-24T00:00:00Z",
      recognition: { assets: { embeddings: baseEmbeddings, identifiers: baseIdentifiers } },
      metadata: { assets: { records: baseMetadata } },
    },
    updates: {},
  };

  if (withUpdates) {
    const v1Identifiers = await fixture.putRows(
      "scryfall-mtg/version/1/delta-from-0/identifiers.jsonl.gz",
      [
        { op: "upsert", record: { id: "card-a", identifiers: { scryfall_oracle: "oracle-a-2" } }, embedding_index: 0 },
        { op: "upsert", record: { id: "card-c", identifiers: {} }, embedding_index: 1 },
      ],
    );
    const v1Embeddings = await fixture.putEmbeddings(
      "scryfall-mtg/version/1/delta-from-0/embeddings.f16.gz",
      [-1, 0, 0.5, 0.5],
    );
    const v1Metadata = await fixture.putRows(
      "scryfall-mtg/version/1/delta-from-0/metadata.jsonl.gz",
      [
        { op: "upsert", id: "card-a", metadata: { name: "Alpha II" } },
        { op: "upsert", id: "card-c", metadata: { name: "Gamma" } },
      ],
    );
    catalog.updates["1"] = {
      from_version: 0,
      to_version: 1,
      rows: { added: 1, updated: 1, deleted: 0 },
      source_updated_at: "2026-07-25T00:00:00Z",
      recognition: { rows: 2, assets: { embeddings: v1Embeddings, identifiers: v1Identifiers } },
      metadata: { rows: 2, assets: { records: v1Metadata } },
    };

    const v2Identifiers = await fixture.putRows(
      "scryfall-mtg/version/2/delta-from-1/identifiers.jsonl.gz",
      [
        { op: "delete", id: "card-b", face_index: 1 },
        { op: "upsert", record: { id: "card-c", identifiers: {} }, embedding_index: 0 },
      ],
    );
    const v2Embeddings = await fixture.putEmbeddings(
      "scryfall-mtg/version/2/delta-from-1/embeddings.f16.gz",
      [2, 0],
    );
    const v2Metadata = await fixture.putRows(
      "scryfall-mtg/version/2/delta-from-1/metadata.jsonl.gz",
      [{ op: "upsert", id: "card-c", metadata: { name: "Gamma II" } }],
    );
    catalog.updates["2"] = {
      from_version: 1,
      to_version: 2,
      rows: { added: 0, updated: 1, deleted: 1 },
      source_updated_at: "2026-07-26T00:00:00Z",
      recognition: { rows: 2, assets: { embeddings: v2Embeddings, identifiers: v2Identifiers } },
      metadata: { rows: 1, assets: { records: v2Metadata } },
    };
    catalog.current_version = 2;
    catalog.rows = 2; // card-a, card-c (card-b deleted)
  }

  fixture.setJson("catalog-feed-v2.json", {
    checked_at: "2026-07-26T12:00:00Z",
    families: {
      fam1: {
        embedding: EMBEDDING,
        catalogs: { "scryfall/mtg": catalog },
      },
    },
  });

  return { fixture, key, catalog };
}

function client(fixture, extra = {}) {
  return new CatalogV2FeedClient({ fetchImpl: fixture.fetchImpl(), feedUrl: FEED_URL, ...extra });
}

// ---------------------------------------------------------------------------
// Base load
// ---------------------------------------------------------------------------

console.log("Base snapshot loading");

await test("loads a base-only catalog via BrowserCatalogV2.forGame", async () => {
  const { fixture, key } = await buildFixture();
  const catalog = await BrowserCatalogV2.forGame("mtg", {
    fetchImpl: fixture.fetchImpl(),
    feedUrl: FEED_URL,
  });
  assert(catalog instanceof BrowserCatalogV2);
  assert.equal(catalog.catalogKey, key);
  assert.equal(catalog.familyKey, "fam1");
  assert.equal(catalog.version, 0);
  assert.equal(catalog.rows, 2);
  assert.equal(catalog.dimension, 2);
  assert.equal(catalog.metadataLoaded, false);
});

await test("orders rows deterministically and reports public identifiers/finishes", async () => {
  const { fixture } = await buildFixture();
  const catalog = await client(fixture).loadGame("mtg");
  assert.deepEqual(
    catalog.records.map((r) => [r.id, r.faceIndex]),
    [
      ["card-a", 0],
      ["card-b", 1],
    ],
  );
  const a = catalog.recordForIndex(0);
  assert.equal(a.id, "card-a");
  assert.equal(a.identifiers.scryfall_card, "card-a");
  assert.equal(a.identifiers.scryfall_oracle, "oracle-a");
  assert.equal(a.face_index, 0);
  assert.equal("finishes" in a, false);
  const b = catalog.recordForIndex(1);
  assert.deepEqual(b.finishes, ["foil", "nonfoil"]);
  assert.equal(b.face_index, 1);
});

await test("search() dequantizes packed float16 embeddings and returns [score, id]", async () => {
  const { fixture } = await buildFixture();
  const catalog = await client(fixture).loadGame("mtg");
  assert.deepEqual(catalog.search(new Float32Array([0, 1]), 1), [[1, "card-b"]]);
  assert.deepEqual(catalog.search(new Float32Array([1, 0]), 1), [[1, "card-a"]]);
});

// ---------------------------------------------------------------------------
// Metadata optionality
// ---------------------------------------------------------------------------

console.log("\nMetadata optionality");

await test("recognition-only load never fetches or exposes metadata", async () => {
  const { fixture } = await buildFixture();
  const catalog = await client(fixture).loadCatalog("fam1/scryfall/mtg", { includeMetadata: false });
  assert.equal(catalog.metadataLoaded, false);
  assert.equal("metadata" in catalog.recordForIndex(0), false);
  assert(fixture.calls.every((url) => !url.includes("metadata")));
});

await test("metadata load treats rows generically, including null and promo/layout fields", async () => {
  const { fixture } = await buildFixture({ withMetadata: true });
  const catalog = await client(fixture).loadCatalog("fam1/scryfall/mtg", { includeMetadata: true });
  assert.equal(catalog.metadataLoaded, true);
  assert.deepEqual(catalog.recordForIndex(0).metadata, { name: "Alpha" });
  assert.equal("metadata" in catalog.recordForIndex(1), false, "null metadata rows expose no metadata field");
});

await test("metadata objects pass through opaque fields such as promo and layout untouched", async () => {
  const fixture = new FeedFixture();
  const identifiers = await fixture.putRows("g/version/0/base/identifiers.jsonl.gz", [
    { id: "x" },
  ]);
  const embeddings = await fixture.putEmbeddings("g/version/0/base/embeddings.f16.gz", [1, 0]);
  const metadata = await fixture.putRows("g/version/0/base/metadata.jsonl.gz", [
    { name: "Wretched Gift", promo: true, layout: "normal", cmc: 2 },
  ]);
  fixture.setJson("catalog-feed-v2.json", {
    checked_at: "2026-07-26T12:00:00Z",
    families: {
      fam1: {
        embedding: EMBEDDING,
        catalogs: {
          "scryfall/mtg": {
            public_name: "scryfall-mtg",
            descriptor: descriptor(),
            current_version: 0,
            rows: 1,
            source_updated_at: "2026-07-24T00:00:00Z",
            base: {
              version: 0,
              rows: 1,
              source_updated_at: "2026-07-24T00:00:00Z",
              recognition: { assets: { embeddings, identifiers } },
              metadata: { assets: { records: metadata } },
            },
            updates: {},
          },
        },
      },
    },
  });
  const catalog = await client(fixture).loadCatalog("fam1/scryfall/mtg", { includeMetadata: true });
  assert.deepEqual(catalog.recordForIndex(0).metadata, {
    name: "Wretched Gift",
    promo: true,
    layout: "normal",
    cmc: 2,
  });
});

// ---------------------------------------------------------------------------
// Base + multiple contiguous updates
// ---------------------------------------------------------------------------

console.log("\nBase + multiple updates");

await test("applies a base and two contiguous updates through current_version", async () => {
  const { fixture } = await buildFixture({ withUpdates: true, withMetadata: true });
  const catalog = await client(fixture).loadCatalog("fam1/scryfall/mtg", { includeMetadata: true });
  assert.equal(catalog.version, 2);
  assert.equal(catalog.rows, 2);
  assert.deepEqual(
    catalog.records.map((r) => r.id),
    ["card-a", "card-c"],
  );
  assert.equal(catalog.recordForIndex(0).identifiers.scryfall_oracle, "oracle-a-2");
  assert.equal(catalog.recordForIndex(0).metadata.name, "Alpha II");
  assert.equal(catalog.recordForIndex(1).metadata.name, "Gamma II");
  assert.deepEqual(catalog.search(new Float32Array([1, 0]), 1), [[2, "card-c"]]);
});

await test("reuses a compatible previous snapshot and only fetches remaining updates", async () => {
  const { fixture } = await buildFixture({ withUpdates: true });
  const feedClient = client(fixture);
  const v1 = await feedClient.loadCatalog("fam1/scryfall/mtg", {
    previous: null,
  });
  // Force the intermediate snapshot down to v1 for this assertion by loading
  // a fixture whose current_version is 1 first is unnecessary: instead verify
  // that supplying the final v2 snapshot back in as `previous` short-circuits
  // all further network access.
  assert.equal(v1.version, 2);
  fixture.calls.length = 0;
  const replayed = await feedClient.loadCatalog("fam1/scryfall/mtg", { previous: v1 });
  assert.equal(replayed, v1);
  assert.deepEqual(fixture.calls, [FEED_URL]);
});

await test("continues from a mid-chain previous snapshot instead of restarting at the base", async () => {
  const { fixture, catalog: catalogV2 } = await buildFixture({ withUpdates: true });
  // Build a standalone v0-current feed to obtain a real v1 snapshot to hand
  // back in as `previous`.
  const v0Fixture = new FeedFixture();
  v0Fixture.files = new Map(fixture.files);
  const v1OnlyFeed = {
    checked_at: "2026-07-25T12:00:00Z",
    families: {
      fam1: {
        embedding: EMBEDDING,
        catalogs: {
          "scryfall/mtg": {
            ...catalogV2,
            current_version: 1,
            rows: 3,
            updates: { 1: catalogV2.updates["1"] },
          },
        },
      },
    },
  };
  v0Fixture.setJson("catalog-feed-v2.json", v1OnlyFeed);
  const v1 = await client(v0Fixture).loadCatalog("fam1/scryfall/mtg");
  assert.equal(v1.version, 1);

  fixture.calls.length = 0;
  const v2 = await client(fixture).loadCatalog("fam1/scryfall/mtg", { previous: v1 });
  assert.equal(v2.version, 2);
  assert(
    fixture.calls.some((url) => url.includes("version/2/delta-from-1")),
    "expected the v1->v2 delta assets to be fetched",
  );
  assert(
    fixture.calls.every((url) => !url.includes("version/0/base") && !url.includes("version/1/delta-from-0")),
    "must not re-fetch the base or the v0->v1 delta when continuing from v1",
  );
});

// ---------------------------------------------------------------------------
// Persistent cache: latest hit and stale/incompatible fallback
// ---------------------------------------------------------------------------

console.log("\nPersistent cache");

await test("a cached snapshot already at current_version skips every asset fetch", async () => {
  const { fixture } = await buildFixture({ withUpdates: true });
  const cache = new MemorySnapshotCache();
  const warm = await client(fixture, { cache }).loadCatalog("fam1/scryfall/mtg");
  assert.equal(warm.version, 2);
  assert.equal(cache.snapshots.size, 3); // base, v1, v2 all persisted along the way

  fixture.calls.length = 0;
  const cached = await client(fixture, { cache }).loadCatalog("fam1/scryfall/mtg");
  assert.equal(cached.version, 2);
  assert.deepEqual(fixture.calls, [FEED_URL], "only the feed document should be fetched on a cache hit");
});

await test("an incompatible cached snapshot is discarded and the feed is used instead", async () => {
  const { fixture, key } = await buildFixture({ withUpdates: true });
  const cache = new MemorySnapshotCache();
  const warnings = [];
  const originalWarn = console.warn;
  console.warn = (...values) => warnings.push(values);
  try {
    const good = await client(fixture, { cache }).loadCatalog(key);
    assert.equal(good.version, 2);

    // Corrupt the cached v2 snapshot's dimension so it no longer matches the
    // family's embedding contract (as if the family contract changed).
    const corrupted = cache.snapshots.get(`2\0${key}\0false`);
    cache.snapshots.set(`2\0${key}\0false`, {
      ...corrupted,
      embedding: { ...corrupted.embedding, dimensions: 999 },
    });

    fixture.calls.length = 0;
    const repaired = await client(fixture, { cache }).loadCatalog(key);
    assert.equal(repaired.version, 2);
    assert.equal(repaired.dimension, 2);
    assert(warnings.length > 0, "expected a warning about the incompatible cache entry");
  } finally {
    console.warn = originalWarn;
  }
});

await test("a stale cached snapshot older than the feed base falls back to a full rebuild", async () => {
  const { fixture, key } = await buildFixture({ withUpdates: true });
  const cache = new MemorySnapshotCache();
  // Seed the cache with a snapshot at a version that no longer exists on the
  // chain (e.g. a hard checkpoint moved the base forward).
  const stale = await client(fixture, { cache }).loadCatalog(key);
  cache.snapshots.set(`-1\0${key}\0false`, stale);
  cache.snapshots.delete(`0\0${key}\0false`);
  cache.snapshots.delete(`1\0${key}\0false`);
  cache.snapshots.delete(`2\0${key}\0false`);

  fixture.calls.length = 0;
  const rebuilt = await client(fixture, { cache }).loadCatalog(key);
  assert.equal(rebuilt.version, 2);
  assert(fixture.calls.some((url) => url.includes("version/0/base")));
});

await test("cache read/write failures are tolerated and do not block loading", async () => {
  const { fixture, key } = await buildFixture();
  const failingCache = {
    async get() {
      throw new Error("read unavailable");
    },
    async put() {
      throw new Error("quota exceeded");
    },
  };
  const warnings = [];
  const originalWarn = console.warn;
  console.warn = (...values) => warnings.push(values);
  try {
    const catalog = await client(fixture, { cache: failingCache }).loadCatalog(key);
    assert.equal(catalog.rows, 2);
    assert.equal(warnings.length, 2);
  } finally {
    console.warn = originalWarn;
  }
});

// ---------------------------------------------------------------------------
// Checksums, sizes, and transport
// ---------------------------------------------------------------------------

console.log("\nChecksums, sizes, and HTTPS");

await test("rejects an asset whose bytes do not match the declared size", async () => {
  const { fixture, key } = await buildFixture();
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  const embeddingsUrl = feed.families.fam1.catalogs["scryfall/mtg"].base.recognition.assets.embeddings.url;
  fixture.replace(embeddingsUrl, new Uint8Array([1, 2, 3]));
  await assert.rejects(() => client(fixture).loadCatalog(key), /size mismatch/);
});

await test("rejects an asset whose bytes do not match the declared sha256", async () => {
  const { fixture, key } = await buildFixture();
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  const idsRef = feed.families.fam1.catalogs["scryfall/mtg"].base.recognition.assets.identifiers;
  const original = fixture.files.get(idsRef.url);
  const tampered = new Uint8Array(original);
  tampered[tampered.length - 1] ^= 0xff; // flip a byte without changing length
  assert.equal(tampered.byteLength, idsRef.size, "tamper fixture must preserve the declared size");
  fixture.replace(idsRef.url, tampered);
  await assert.rejects(() => client(fixture).loadCatalog(key), /checksum mismatch/);
});

await test("rejects a non-HTTPS asset URL", async () => {
  const { fixture, key } = await buildFixture();
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  feed.families.fam1.catalogs["scryfall/mtg"].base.recognition.assets.identifiers.url =
    feed.families.fam1.catalogs["scryfall/mtg"].base.recognition.assets.identifiers.url.replace(
      "https://",
      "http://",
    );
  fixture.setJson("catalog-feed-v2.json", feed);
  await assert.rejects(() => client(fixture).loadCatalog(key), /https/);
});

// ---------------------------------------------------------------------------
// Malformed identities
// ---------------------------------------------------------------------------

console.log("\nMalformed identities");

async function buildBrokenBaseFixture(rows) {
  const fixture = new FeedFixture();
  const identifiers = await fixture.putRows("g/version/0/base/identifiers.jsonl.gz", rows);
  const embeddings = await fixture.putEmbeddings(
    "g/version/0/base/embeddings.f16.gz",
    rows.flatMap(() => [1, 0]),
  );
  const metadata = await fixture.putRows(
    "g/version/0/base/metadata.jsonl.gz",
    rows.map(() => null),
  );
  fixture.setJson("catalog-feed-v2.json", {
    checked_at: "2026-07-26T12:00:00Z",
    families: {
      fam1: {
        embedding: EMBEDDING,
        catalogs: {
          "scryfall/mtg": {
            public_name: "scryfall-mtg",
            descriptor: descriptor(),
            current_version: 0,
            rows: rows.length,
            source_updated_at: "2026-07-24T00:00:00Z",
            base: {
              version: 0,
              rows: rows.length,
              source_updated_at: "2026-07-24T00:00:00Z",
              recognition: { assets: { embeddings, identifiers } },
              metadata: { assets: { records: metadata } },
            },
            updates: {},
          },
        },
      },
    },
  });
  return fixture;
}

await test("rejects a duplicate (id, face_index) identity in the base", async () => {
  const fixture = await buildBrokenBaseFixture([
    { id: "dup", identifiers: {} },
    { id: "dup", identifiers: {} },
  ]);
  await assert.rejects(
    () => client(fixture).loadCatalog("fam1/scryfall/mtg"),
    /duplicate base row identity/,
  );
});

await test("rejects a base row that duplicates the primary id under identifiers", async () => {
  const fixture = await buildBrokenBaseFixture([
    { id: "card-a", identifiers: { scryfall_card: "card-a" } },
  ]);
  await assert.rejects(
    () => client(fixture).loadCatalog("fam1/scryfall/mtg"),
    /must not duplicate the primary id/,
  );
});

await test("rejects a base row with a negative face_index", async () => {
  const fixture = await buildBrokenBaseFixture([{ id: "card-a", identifiers: {}, face_index: -1 }]);
  await assert.rejects(
    () => client(fixture).loadCatalog("fam1/scryfall/mtg"),
    /invalid face_index/,
  );
});

await test("rejects a base row missing a non-empty id", async () => {
  const fixture = await buildBrokenBaseFixture([{ identifiers: {} }]);
  await assert.rejects(
    () => client(fixture).loadCatalog("fam1/scryfall/mtg"),
    CatalogV2Error,
  );
});

// ---------------------------------------------------------------------------
// Delta failures
// ---------------------------------------------------------------------------

console.log("\nDelta failures");

async function buildDeltaFailureFixture(mutateUpdate) {
  const { fixture, catalog } = await buildFixture({ withUpdates: true });
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  mutateUpdate(feed.families.fam1.catalogs["scryfall/mtg"].updates["1"], catalog);
  fixture.setJson("catalog-feed-v2.json", feed);
  return fixture;
}

await test("rejects a recognition delta whose operation count does not match the feed", async () => {
  const fixture = await buildDeltaFailureFixture((update) => {
    update.recognition.rows = 3;
  });
  await assert.rejects(
    () => client(fixture).loadCatalog("fam1/scryfall/mtg"),
    /operation count/,
  );
});

await test("rejects a recognition delta with an out-of-range embedding_index", async () => {
  const { fixture, catalog } = await buildFixture({ withUpdates: true });
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  const badIdentifiers = await fixture.putRows("scryfall-mtg/version/1/delta-from-0/bad-identifiers.jsonl.gz", [
    { op: "upsert", record: { id: "card-a", identifiers: {} }, embedding_index: 5 },
    { op: "upsert", record: { id: "card-c", identifiers: {} }, embedding_index: 1 },
  ]);
  feed.families.fam1.catalogs["scryfall/mtg"].updates["1"].recognition.assets.identifiers = badIdentifiers;
  fixture.setJson("catalog-feed-v2.json", feed);
  void catalog;
  await assert.rejects(
    () => client(fixture).loadCatalog("fam1/scryfall/mtg"),
    /invalid embedding_index/,
  );
});

await test("rejects a recognition delta with duplicate embedding_index assignments", async () => {
  const { fixture } = await buildFixture({ withUpdates: true });
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  const badIdentifiers = await fixture.putRows("scryfall-mtg/version/1/delta-from-0/dup-identifiers.jsonl.gz", [
    { op: "upsert", record: { id: "card-a", identifiers: {} }, embedding_index: 0 },
    { op: "upsert", record: { id: "card-c", identifiers: {} }, embedding_index: 0 },
  ]);
  feed.families.fam1.catalogs["scryfall/mtg"].updates["1"].recognition.assets.identifiers = badIdentifiers;
  fixture.setJson("catalog-feed-v2.json", feed);
  await assert.rejects(
    () => client(fixture).loadCatalog("fam1/scryfall/mtg"),
    /invalid embedding_index/,
  );
});

await test("rejects a recognition delta that deletes a row not present in the previous snapshot", async () => {
  const fixture = await buildDeltaFailureFixture(() => {});
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  const badIdentifiers = await fixture.putRows(
    "scryfall-mtg/version/1/delta-from-0/missing-delete.jsonl.gz",
    [
      { op: "delete", id: "card-does-not-exist" },
      { op: "upsert", record: { id: "card-c", identifiers: {} }, embedding_index: 0 },
    ],
  );
  const embeddings = await fixture.putEmbeddings(
    "scryfall-mtg/version/1/delta-from-0/missing-delete.f16.gz",
    [1, 0],
  );
  const entry = feed.families.fam1.catalogs["scryfall/mtg"].updates["1"];
  entry.recognition.assets.identifiers = badIdentifiers;
  entry.recognition.assets.embeddings = embeddings;
  entry.rows = { added: 1, updated: 0, deleted: 1 };
  fixture.setJson("catalog-feed-v2.json", feed);
  await assert.rejects(
    () => client(fixture).loadCatalog("fam1/scryfall/mtg"),
    /deletes a row that is not present/,
  );
});

await test("rejects a recognition delta whose row classification disagrees with the feed", async () => {
  const fixture = await buildDeltaFailureFixture((update) => {
    update.rows = { added: 0, updated: 2, deleted: 0 };
  });
  await assert.rejects(
    () => client(fixture).loadCatalog("fam1/scryfall/mtg"),
    /row classification/,
  );
});

await test("rejects an unsupported recognition delta operation", async () => {
  const { fixture } = await buildFixture({ withUpdates: true });
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  const badIdentifiers = await fixture.putRows(
    "scryfall-mtg/version/1/delta-from-0/bad-op.jsonl.gz",
    [
      { op: "replace", id: "card-a" },
      { op: "upsert", record: { id: "card-c", identifiers: {} }, embedding_index: 0 },
    ],
  );
  const badEmbeddings = await fixture.putEmbeddings(
    "scryfall-mtg/version/1/delta-from-0/bad-op.f16.gz",
    [0.5, 0.5],
  );
  const entry = feed.families.fam1.catalogs["scryfall/mtg"].updates["1"];
  entry.recognition.assets.identifiers = badIdentifiers;
  entry.recognition.assets.embeddings = badEmbeddings;
  fixture.setJson("catalog-feed-v2.json", feed);
  await assert.rejects(
    () => client(fixture).loadCatalog("fam1/scryfall/mtg"),
    /unsupported recognition delta operation/,
  );
});

await test("rejects a metadata delta upsert that targets a row absent from recognition", async () => {
  const { fixture } = await buildFixture({ withUpdates: true, withMetadata: true });
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  const badMetadata = await fixture.putRows(
    "scryfall-mtg/version/1/delta-from-0/bad-metadata.jsonl.gz",
    [{ op: "upsert", id: "card-does-not-exist", metadata: { name: "Ghost" } }],
  );
  const entry = feed.families.fam1.catalogs["scryfall/mtg"].updates["1"];
  entry.metadata.assets.records = badMetadata;
  entry.metadata.rows = 1;
  fixture.setJson("catalog-feed-v2.json", feed);
  await assert.rejects(
    () => client(fixture).loadCatalog("fam1/scryfall/mtg", { includeMetadata: true }),
    /metadata delta upserts a row that is not present/,
  );
});

await test("rejects a catalog whose update chain skips a version", async () => {
  const { fixture } = await buildFixture({ withUpdates: true });
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  delete feed.families.fam1.catalogs["scryfall/mtg"].updates["1"];
  fixture.setJson("catalog-feed-v2.json", feed);
  await assert.rejects(
    () => client(fixture).loadCatalog("fam1/scryfall/mtg"),
    /missing an update/,
  );
});

await test("rejects a catalog whose update declares the wrong exact-predecessor base", async () => {
  const { fixture } = await buildFixture({ withUpdates: true });
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  feed.families.fam1.catalogs["scryfall/mtg"].updates["2"].from_version = 0;
  fixture.setJson("catalog-feed-v2.json", feed);
  await assert.rejects(
    () => client(fixture).loadCatalog("fam1/scryfall/mtg"),
    /exact-predecessor delta/,
  );
});

// ---------------------------------------------------------------------------
// Descriptor discovery
// ---------------------------------------------------------------------------

console.log("\nDescriptor discovery");

async function buildMultiCatalogFixture() {
  const fixture = new FeedFixture();
  async function simpleCatalog(path, publicName, descriptorOverrides) {
    const identifiers = await fixture.putRows(`${path}/version/0/base/identifiers.jsonl.gz`, [
      { id: "x" },
    ]);
    const embeddings = await fixture.putEmbeddings(`${path}/version/0/base/embeddings.f16.gz`, [1, 0]);
    const metadata = await fixture.putRows(`${path}/version/0/base/metadata.jsonl.gz`, [null]);
    return {
      public_name: publicName,
      descriptor: descriptor(descriptorOverrides),
      current_version: 0,
      rows: 1,
      source_updated_at: "2026-07-24T00:00:00Z",
      base: {
        version: 0,
        rows: 1,
        source_updated_at: "2026-07-24T00:00:00Z",
        recognition: { assets: { embeddings, identifiers } },
        metadata: { assets: { records: metadata } },
      },
      updates: {},
    };
  }

  const scryfallMtg = await simpleCatalog("scryfall-mtg", "scryfall-mtg", {
    game: "magic-the-gathering",
    source: "scryfall",
    result_identifier: "scryfall_card",
    recommended: true,
  });
  const tcgplayerMtg = await simpleCatalog("tcgplayer-mtg", "tcgplayer-mtg", {
    game: "magic-the-gathering",
    source: "tcgplayer",
    result_identifier: "tcgplayer_product",
    recommended: true,
  });
  const tcgplayerPokemon = await simpleCatalog("tcgplayer-pokemon", "tcgplayer-pokemon", {
    game: "pokemon",
    source: "tcgplayer",
    result_identifier: "tcgplayer_product",
    recommended: true,
  });

  fixture.setJson("catalog-feed-v2.json", {
    checked_at: "2026-07-26T12:00:00Z",
    families: {
      fam1: {
        embedding: EMBEDDING,
        catalogs: {
          "scryfall/mtg": scryfallMtg,
          "tcgplayer/mtg": tcgplayerMtg,
        },
      },
      fam2: {
        embedding: EMBEDDING,
        catalogs: { "tcgplayer/pokemon": tcgplayerPokemon },
      },
    },
  });
  return fixture;
}

await test("defaults to the recommended Scryfall catalog for MTG", async () => {
  const fixture = await buildMultiCatalogFixture();
  const catalog = await client(fixture).loadGame("mtg");
  assert.equal(catalog.catalogKey, "fam1/scryfall/mtg");
  assert.equal(catalog.descriptor.source, "scryfall");
});

await test("defaults to the recommended TCGplayer catalog for other games", async () => {
  const fixture = await buildMultiCatalogFixture();
  const catalog = await client(fixture).loadGame("pokemon");
  assert.equal(catalog.catalogKey, "fam2/tcgplayer/pokemon");
  assert.equal(catalog.descriptor.source, "tcgplayer");
});

await test("an explicit source overrides the default and still finds the recommended descriptor", async () => {
  const fixture = await buildMultiCatalogFixture();
  const catalog = await client(fixture).loadGame("mtg", { source: "tcgplayer" });
  assert.equal(catalog.catalogKey, "fam1/tcgplayer/mtg");
  assert.equal(catalog.descriptor.result_identifier, "tcgplayer_product");
});

await test("rejects a game/source combination with no matching feed entry", async () => {
  const fixture = await buildMultiCatalogFixture();
  await assert.rejects(
    () => client(fixture).loadGame("yugioh"),
    /no Catalog v2 feed entry matches/,
  );
});

await test("rejects an ambiguous game/source combination with multiple recommended entries", async () => {
  const fixture = await buildMultiCatalogFixture();
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  feed.families.fam2.catalogs["tcgplayer/mtg-2"] = {
    ...feed.families.fam1.catalogs["tcgplayer/mtg"],
  };
  fixture.setJson("catalog-feed-v2.json", feed);
  await assert.rejects(
    () => client(fixture).loadGame("mtg", { source: "tcgplayer" }),
    /multiple recommended/,
  );
});

await test("loadCatalog() loads directly by full catalog key without game discovery", async () => {
  const fixture = await buildMultiCatalogFixture();
  const catalog = await client(fixture).loadCatalog("fam2/tcgplayer/pokemon");
  assert.equal(catalog.catalogKey, "fam2/tcgplayer/pokemon");
});

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

console.log(`\n${"=".repeat(50)}`);
console.log(`Results: ${_passed} passed, ${_failed} failed`);
if (_failures.length > 0) {
  for (const { label, err } of _failures) {
    console.error(`  FAIL  ${label}: ${err.message.split("\n")[0]}`);
  }
  process.exit(1);
}
