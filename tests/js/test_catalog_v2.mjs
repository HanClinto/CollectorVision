/**
 * Node.js regression tests for the CollectorVision Catalog v2 browser client.
 *
 * These tests build fully synthetic, deterministic feeds and gzip assets in
 * memory (no network, no real CollectorVisionCatalog data) that follow the
 * live combined-record `catalog-feed-v2.json` contract described in
 * HanClinto/CollectorVisionCatalog `docs/catalog-v2.md` and
 * `docs/versioning.md`: a feed of embedding families, each nesting catalogs
 * keyed by `family/local-key`, each catalog advertising a base snapshot
 * (`assets: {records, embeddings}`, every records row carrying a required
 * `metadata` field) and zero or more exact-predecessor updates through
 * `current_version` (`rows: {added, updated, deleted}`, `recognition_rows`,
 * `metadata_rows`, `assets: {records, embeddings?}`).
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
  CatalogV2IndexedDbCache,
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

  /** Writes a gzip JSON-lines records asset (combined base rows or delta
   * operations); base rows and upsert operations are auto-completed with a
   * `name` field via withTestName() so terse fixture literals stay readable. */
  async putRecords(path, values) {
    return this.#putGzip(path, jsonLines(values.map(withTestName)));
  }

  /** Like putRecords(), but writes rows verbatim without auto-completing a
   * `name` field; used by tests that specifically exercise the "name is
   * required" validation path. */
  async putRecordsRaw(path, values) {
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

// Real records rows/upserts must carry `name`; terse fixture literals lacking
// it are auto-completed from `id` for both base rows and delta `record`
// payloads (delete ops and bare identity targets are left untouched).
function withTestName(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return value;
  if (value.record?.id && typeof value.record.id === "string") {
    return { ...value, record: { name: value.record.id, ...value.record } };
  }
  if (!("op" in value) && typeof value.id === "string") {
    return { name: value.id, ...value };
  }
  return value;
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
    this.deleteCalls = 0;
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

  async delete(version, catalogKey, includeMetadata) {
    this.deleteCalls += 1;
    this.snapshots.delete(`${version}\0${catalogKey}\0${includeMetadata}`);
  }
}

/** A minimal, self-contained in-memory fake of the IndexedDB API surface used
 * by CatalogV2IndexedDbCache: object stores keyed by `keyPath`, a single
 * secondary index, cursors, and readwrite transactions. Real browsers behave
 * the same way; Node has no native `indexedDB` global to test against.
 * `pendingOps` tracks in-flight scheduled work so a transaction's
 * `oncomplete` never fires before every request/cursor step it spawned
 * (including cascading cursor `continue()` calls) has actually settled. */
function makeFakeIndexedDb() {
  const databases = new Map();
  let pendingOps = 0;

  function scheduleWork(fn) {
    pendingOps += 1;
    queueMicrotask(() => {
      try {
        fn();
      } finally {
        pendingOps -= 1;
      }
    });
  }

  class FakeRequest {
    constructor() {
      this.result = undefined;
      this.error = undefined;
      this.onsuccess = null;
      this.onerror = null;
    }
    _succeed(result) {
      this.result = result;
      scheduleWork(() => this.onsuccess?.());
    }
    _fail(error) {
      this.error = error;
      scheduleWork(() => this.onerror?.());
    }
  }

  class FakeIndex {
    constructor(store, keyPath) {
      this.store = store;
      this.keyPath = keyPath;
    }
    #indexKeyFor(record) {
      return JSON.stringify(this.keyPath.map((part) => record[part]));
    }
    openCursor(query) {
      const request = new FakeRequest();
      const targetKey = JSON.stringify(query);
      const matches = [...this.store.records.values()].filter(
        (record) => this.#indexKeyFor(record) === targetKey,
      );
      let index = 0;
      const advance = () => {
        if (index >= matches.length) {
          request._succeed(null);
          return;
        }
        const record = matches[index];
        index += 1;
        request._succeed({
          value: record,
          delete: () => this.store.records.delete(record.id),
          continue: () => scheduleWork(advance),
        });
      };
      scheduleWork(advance);
      return request;
    }
  }

  class FakeStore {
    constructor(keyPath) {
      this.keyPath = keyPath;
      this.records = new Map();
      this.indexes = new Map();
    }
    createIndex(name, keyPath) {
      this.indexes.set(name, new FakeIndex(this, keyPath));
    }
    index(name) {
      return this.indexes.get(name);
    }
    get indexNames() {
      const names = this.indexes;
      return { contains: (name) => names.has(name) };
    }
    get(key) {
      const request = new FakeRequest();
      scheduleWork(() => request._succeed(this.records.get(key)));
      return request;
    }
    put(record) {
      const request = new FakeRequest();
      this.records.set(record[this.keyPath], structuredClone(record));
      scheduleWork(() => request._succeed(record[this.keyPath]));
      return request;
    }
    delete(key) {
      const request = new FakeRequest();
      this.records.delete(key);
      scheduleWork(() => request._succeed(undefined));
      return request;
    }
  }

  class FakeTransaction {
    constructor(store) {
      this.store = store;
      this.oncomplete = null;
      this.onerror = null;
      const check = () => {
        if (pendingOps === 0) {
          queueMicrotask(() => this.oncomplete?.());
        } else {
          queueMicrotask(check);
        }
      };
      queueMicrotask(check);
    }
    objectStore() {
      return this.store;
    }
  }

  class FakeDatabase {
    constructor() {
      this.objectStoreMap = new Map();
      this.version = 0;
    }
    get objectStoreNames() {
      const names = this.objectStoreMap;
      return { contains: (name) => names.has(name) };
    }
    createObjectStore(name, { keyPath }) {
      const store = new FakeStore(keyPath);
      this.objectStoreMap.set(name, store);
      return store;
    }
    transaction(name) {
      return new FakeTransaction(this.objectStoreMap.get(name));
    }
  }

  return {
    open(name, version) {
      const request = new FakeRequest();
      let database = databases.get(name);
      const previousVersion = database ? database.version : 0;
      const isNew = !database;
      if (isNew) {
        database = new FakeDatabase();
        databases.set(name, database);
      }
      const needsUpgrade = version > previousVersion;
      queueMicrotask(() => {
        if (needsUpgrade) {
          database.version = version;
          request.result = database;
          request.transaction = database.transaction("catalogs");
          request.onupgradeneeded?.();
        }
        request._succeed(database);
      });
      return request;
    },
  };
}

/** Builds a single-family, single-catalog fixture with a base snapshot of two
 * rows ("card-a" front face, "card-b" back face) and, optionally, a chain of
 * updates through version 2. Each update stage's records asset carries one
 * combined operation per globally affected identity (never separate
 * recognition/metadata operations for the same id):
 *   v0->v1: upsert card-a (recognition + metadata), add card-c (recognition + metadata)
 *   v1->v2: delete card-b, upsert card-c (recognition + metadata)
 * `withMetadata` toggles whether card-b's base metadata is null (used to
 * exercise the explicit-null-metadata path) or a plain object. */
async function buildFixture({ withUpdates = false, withMetadata = false } = {}) {
  const fixture = new FeedFixture();
  const key = "milo1/scryfall/mtg";

  const baseRecords = await fixture.putRecords("scryfall-mtg/version/0/base/records.jsonl.gz", [
    { id: "card-a", identifiers: { scryfall_oracle: "oracle-a" }, metadata: { name: "Alpha" } },
    {
      id: "card-b",
      identifiers: {},
      face_index: 1,
      finishes: ["foil", "nonfoil"],
      metadata: withMetadata ? null : { name: "Beta" },
    },
  ]);
  const baseEmbeddings = await fixture.putEmbeddings("scryfall-mtg/version/0/base/embeddings.f16.gz", [
    1, 0, 0, 1,
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
      assets: { records: baseRecords, embeddings: baseEmbeddings },
    },
    updates: {},
  };

  if (withUpdates) {
    const v1Records = await fixture.putRecords(
      "scryfall-mtg/version/1/delta-from-0/records.jsonl.gz",
      [
        {
          op: "upsert",
          record: { id: "card-a", identifiers: { scryfall_oracle: "oracle-a-2" } },
          metadata: { name: "Alpha II" },
          embedding_index: 0,
        },
        {
          op: "upsert",
          record: { id: "card-c", identifiers: {} },
          metadata: { name: "Gamma" },
          embedding_index: 1,
        },
      ],
    );
    const v1Embeddings = await fixture.putEmbeddings(
      "scryfall-mtg/version/1/delta-from-0/embeddings.f16.gz",
      [-1, 0, 0.5, 0.5],
    );
    catalog.updates["1"] = {
      from_version: 0,
      to_version: 1,
      rows: { added: 1, updated: 1, deleted: 0 },
      source_updated_at: "2026-07-25T00:00:00Z",
      recognition_rows: 2,
      metadata_rows: 2,
      assets: { records: v1Records, embeddings: v1Embeddings },
    };

    // card-b's base metadata is only non-null when withMetadata is false, so
    // its deletion here only contributes to metadata_rows in that case.
    const v2MetadataRows = 1 + (withMetadata ? 0 : 1);
    const v2Records = await fixture.putRecords(
      "scryfall-mtg/version/2/delta-from-1/records.jsonl.gz",
      [
        { op: "delete", id: "card-b", face_index: 1 },
        {
          op: "upsert",
          record: { id: "card-c", identifiers: {} },
          metadata: { name: "Gamma II" },
          embedding_index: 0,
        },
      ],
    );
    const v2Embeddings = await fixture.putEmbeddings(
      "scryfall-mtg/version/2/delta-from-1/embeddings.f16.gz",
      [2, 0],
    );
    catalog.updates["2"] = {
      from_version: 1,
      to_version: 2,
      rows: { added: 0, updated: 1, deleted: 1 },
      source_updated_at: "2026-07-26T00:00:00Z",
      recognition_rows: 2,
      metadata_rows: v2MetadataRows,
      assets: { records: v2Records, embeddings: v2Embeddings },
    };
    catalog.current_version = 2;
    catalog.rows = 2; // card-a, card-c (card-b deleted)
  }

  fixture.setJson("catalog-feed-v2.json", {
    checked_at: "2026-07-26T12:00:00Z",
    families: {
      milo1: {
        embedding: EMBEDDING,
        catalogs: { "scryfall/mtg": catalog },
      },
    },
  });

  return { fixture, key, catalog };
}

async function promoteToRoutineCheckpoint(fixture, catalog) {
  const bridge = catalog.updates["1"];
  const following = catalog.updates["2"];
  const baseRecords = await fixture.putRecords(
    "scryfall-mtg/version/10/base/records.jsonl.gz",
    [
      {
        id: "card-a",
        identifiers: { scryfall_oracle: "oracle-a-2" },
        metadata: { name: "Alpha II" },
      },
      {
        id: "card-b",
        identifiers: { scryfall_oracle: "oracle-b" },
        face_index: 1,
        finishes: ["foil", "nonfoil"],
        metadata: { name: "Beta" },
      },
      { id: "card-c", identifiers: {}, metadata: { name: "Gamma" } },
    ],
  );
  const baseEmbeddings = await fixture.putEmbeddings(
    "scryfall-mtg/version/10/base/embeddings.f16.gz",
    [-1, 0, 0, 1, 0.5, 0.5],
  );
  catalog.base = {
    version: 10,
    rows: 3,
    source_updated_at: bridge.source_updated_at,
    assets: { records: baseRecords, embeddings: baseEmbeddings },
  };
  catalog.updates = {
    10: { ...bridge, from_version: 9, to_version: 10 },
    11: { ...following, from_version: 10, to_version: 11 },
  };
  catalog.current_version = 11;
  fixture.setJson("catalog-feed-v2.json", {
    checked_at: "2026-07-26T12:00:00Z",
    families: {
      milo1: {
        embedding: EMBEDDING,
        catalogs: { "scryfall/mtg": catalog },
      },
    },
  });
}

function client(fixture, extra = {}) {
  return new CatalogV2FeedClient({ fetchImpl: fixture.fetchImpl(), feedUrl: FEED_URL, ...extra });
}

// ---------------------------------------------------------------------------
// Base load
// ---------------------------------------------------------------------------

console.log("Base snapshot loading");

await test("binds fetch to the active browser or worker global scope", async () => {
  const { fixture } = await buildFixture();
  const fixtureFetch = fixture.fetchImpl();
  async function scopeCheckedFetch(url) {
    assert.equal(this, globalThis);
    return fixtureFetch(url);
  }
  const catalog = await new CatalogV2FeedClient({
    fetchImpl: scopeCheckedFetch,
    feedUrl: FEED_URL,
  }).loadGame("mtg");
  assert.equal(catalog.rows, 2);
});

await test("loads a base-only catalog via BrowserCatalogV2.forGame", async () => {
  const { fixture, key } = await buildFixture();
  const catalog = await BrowserCatalogV2.forGame("mtg", {
    fetchImpl: fixture.fetchImpl(),
    feedUrl: FEED_URL,
  });
  assert(catalog instanceof BrowserCatalogV2);
  assert.equal(catalog.catalogKey, key);
  assert.equal(catalog.familyKey, "milo1");
  assert.equal(catalog.version, 0);
  assert.equal(catalog.rows, 2);
  assert.equal(catalog.dimension, 2);
  assert.equal(catalog.metadataLoaded, true);
  assert.deepEqual(catalog.recordForIndex(0).metadata, { name: "Alpha" });
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
  assert.equal(a.key, "scryfall:card-a");
  assert.equal(a.id, "card-a");
  assert.equal(a.name, "card-a");
  assert.equal(a.card_id, "card-a");
  assert.equal(a.result_identifier, "scryfall_card");
  assert.equal(a.identifiers.scryfall_card, "card-a");
  assert.equal(a.identifiers.scryfall_oracle, "oracle-a");
  assert.equal(a.face_index, 0);
  assert.deepEqual(a.finishes, []);
  const b = catalog.recordForIndex(1);
  assert.equal(b.key, "scryfall:card-b:face:1");
  assert.equal(b.card_id, "card-b");
  assert.deepEqual(b.finishes, ["foil", "nonfoil"]);
  assert.equal(b.face_index, 1);
});

await test("search() dequantizes packed float16 embeddings and returns [score, card_id]", async () => {
  const { fixture } = await buildFixture();
  const catalog = await client(fixture).loadGame("mtg");
  assert.deepEqual(catalog.search(new Float32Array([0, 1]), 1), [[1, "card-b"]]);
  assert.deepEqual(catalog.search(new Float32Array([1, 0]), 1), [[1, "card-a"]]);
});

// ---------------------------------------------------------------------------
// Metadata optionality
// ---------------------------------------------------------------------------

console.log("\nMetadata optionality");

await test("recognition-only load always fetches combined records but discards metadata after parsing", async () => {
  const { fixture } = await buildFixture({ withMetadata: true });
  const catalog = await client(fixture).loadCatalog("milo1/scryfall/mtg", { includeMetadata: false });
  assert.equal(catalog.metadataLoaded, false);
  assert.equal("metadata" in catalog.recordForIndex(0), false);
  assert(
    fixture.calls.some((url) => url.includes("records.jsonl.gz")),
    "the combined records asset must still be fetched even when metadata is not wanted",
  );
});

await test("metadata load treats rows generically, including null and promo/layout fields", async () => {
  const { fixture } = await buildFixture({ withMetadata: true });
  const catalog = await client(fixture).loadCatalog("milo1/scryfall/mtg", { includeMetadata: true });
  assert.equal(catalog.metadataLoaded, true);
  assert.deepEqual(catalog.recordForIndex(0).metadata, { name: "Alpha" });
  assert.equal(
    "metadata" in catalog.recordForIndex(1),
    true,
    "metadata key must be present whenever metadataLoaded, even when there is no metadata",
  );
  assert.equal(
    catalog.recordForIndex(1).metadata,
    null,
    "rows without metadata expose an explicit null, not an omitted key",
  );
});

await test("metadata objects pass through opaque fields such as promo and layout untouched", async () => {
  const fixture = new FeedFixture();
  const records = await fixture.putRecords("g/version/0/base/records.jsonl.gz", [
    { id: "x", metadata: { name: "Wretched Gift", promo: true, layout: "normal", cmc: 2 } },
  ]);
  const embeddings = await fixture.putEmbeddings("g/version/0/base/embeddings.f16.gz", [1, 0]);
  fixture.setJson("catalog-feed-v2.json", {
    checked_at: "2026-07-26T12:00:00Z",
    families: {
      milo1: {
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
              assets: { records, embeddings },
            },
            updates: {},
          },
        },
      },
    },
  });
  const catalog = await client(fixture).loadCatalog("milo1/scryfall/mtg", { includeMetadata: true });
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
  const catalog = await client(fixture).loadCatalog("milo1/scryfall/mtg", { includeMetadata: true });
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
  const v1 = await feedClient.loadCatalog("milo1/scryfall/mtg", {
    previous: null,
  });
  // Force the intermediate snapshot down to v1 for this assertion by loading
  // a fixture whose current_version is 1 first is unnecessary: instead verify
  // that supplying the final v2 snapshot back in as `previous` short-circuits
  // all further network access.
  assert.equal(v1.version, 2);
  fixture.calls.length = 0;
  const replayed = await feedClient.loadCatalog("milo1/scryfall/mtg", { previous: v1 });
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
      milo1: {
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
  const v1 = await client(v0Fixture).loadCatalog("milo1/scryfall/mtg");
  assert.equal(v1.version, 1);

  fixture.calls.length = 0;
  const v2 = await client(fixture).loadCatalog("milo1/scryfall/mtg", { previous: v1 });
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

await test("loads a routine checkpoint base without replaying its bridge", async () => {
  const { fixture, catalog } = await buildFixture({ withUpdates: true });
  await promoteToRoutineCheckpoint(fixture, catalog);

  const loaded = await client(fixture).loadCatalog("milo1/scryfall/mtg");

  assert.equal(loaded.version, 11);
  assert(fixture.calls.some((url) => url.includes("version/10/base")));
  assert(
    fixture.calls.every((url) => !url.includes("version/1/delta-from-0")),
    "a fresh client must start at the checkpoint base rather than replaying its bridge",
  );
});

await test("uses a routine checkpoint bridge from a cached predecessor", async () => {
  const { fixture, key, catalog } = await buildFixture({ withUpdates: true });
  const cache = new MemorySnapshotCache();
  const versionNine = {
    ...catalog,
    current_version: 9,
    base: { ...catalog.base, version: 9 },
    updates: {},
  };
  fixture.setJson("catalog-feed-v2.json", {
    checked_at: "2026-07-24T12:00:00Z",
    families: { milo1: { embedding: EMBEDDING, catalogs: { "scryfall/mtg": versionNine } } },
  });
  const cached = await client(fixture, { cache }).loadCatalog(key);
  assert.equal(cached.version, 9);

  await promoteToRoutineCheckpoint(fixture, catalog);
  fixture.calls.length = 0;
  const upgraded = await client(fixture, { cache }).loadCatalog(key);

  assert.equal(upgraded.version, 11);
  assert(fixture.calls.some((url) => url.includes("version/1/delta-from-0")));
  assert(fixture.calls.some((url) => url.includes("version/2/delta-from-1")));
  assert(
    fixture.calls.every((url) => !url.includes("version/10/base")),
    "a cached predecessor must use the bridge instead of downloading the checkpoint base",
  );
});

// ---------------------------------------------------------------------------
// Combined-record delta semantics
// ---------------------------------------------------------------------------

console.log("\nCombined-record delta semantics");

/** A tcgplayer-MTG-style fixture whose single update stage upserts "card-x"
 * with both a recognition change and metadata in one combined operation, and
 * upserts "card-y" with a metadata-only operation (no embedding_index at
 * all), exercising the "metadata-only no-embedding-index upsert" contract
 * requirement using an existing, surviving row. */
async function buildOverlapFixture() {
  const fixture = new FeedFixture();
  const key = "milo1/tcgplayer/mtg";

  const baseRecords = await fixture.putRecords("tcgplayer-mtg/version/0/base/records.jsonl.gz", [
    { id: "card-x", identifiers: {}, metadata: { name: "X" } },
    { id: "card-y", identifiers: {}, metadata: { name: "Y" } },
  ]);
  const baseEmbeddings = await fixture.putEmbeddings("tcgplayer-mtg/version/0/base/embeddings.f16.gz", [
    1, 0, 0, 1,
  ]);

  const v1Records = await fixture.putRecords(
    "tcgplayer-mtg/version/1/delta-from-0/records.jsonl.gz",
    [
      {
        op: "upsert",
        record: { id: "card-x", identifiers: {} },
        metadata: { name: "X2" },
        embedding_index: 0,
      },
      { op: "upsert", record: { id: "card-y", identifiers: {} }, metadata: { name: "Y2" } },
    ],
  );
  const v1Embeddings = await fixture.putEmbeddings(
    "tcgplayer-mtg/version/1/delta-from-0/embeddings.f16.gz",
    [0.5, 0.5],
  );

  fixture.setJson("catalog-feed-v2.json", {
    checked_at: "2026-08-01T00:00:00Z",
    families: {
      milo1: {
        embedding: EMBEDDING,
        catalogs: {
          "tcgplayer/mtg": {
            public_name: "tcgplayer-mtg",
            descriptor: descriptor({ source: "tcgplayer", result_identifier: "tcgplayer_product" }),
            current_version: 1,
            rows: 2,
            source_updated_at: "2026-08-01T00:00:00Z",
            base: {
              version: 0,
              rows: 2,
              source_updated_at: "2026-07-31T00:00:00Z",
              assets: { records: baseRecords, embeddings: baseEmbeddings },
            },
            updates: {
              1: {
                from_version: 0,
                to_version: 1,
                rows: { added: 0, updated: 2, deleted: 0 },
                source_updated_at: "2026-08-01T00:00:00Z",
                recognition_rows: 1,
                metadata_rows: 2,
                assets: { records: v1Records, embeddings: v1Embeddings },
              },
            },
          },
        },
      },
    },
  });

  return { fixture, key };
}

await test("a metadata-only upsert without embedding_index changes only metadata for a surviving row", async () => {
  const { fixture, key } = await buildOverlapFixture();
  const catalog = await client(fixture).loadCatalog(key, { includeMetadata: true });
  assert.equal(catalog.version, 1);
  const byCardId = new Map(catalog.records.map((r, index) => [r.id, catalog.recordForIndex(index)]));
  assert.equal(byCardId.get("card-x").metadata.name, "X2");
  assert.equal(byCardId.get("card-y").metadata.name, "Y2");
  // card-y's embedding must be untouched by its metadata-only upsert.
  assert.deepEqual(catalog.search(new Float32Array([0, 1]), 1), [[1, "card-y"]]);
});

await test("recognition-only mode never rejects a row that was only metadata-updated", async () => {
  const { fixture, key } = await buildOverlapFixture();
  const catalog = await client(fixture).loadCatalog(key, { includeMetadata: false });
  assert.equal(catalog.version, 1);
  assert.equal(catalog.metadataLoaded, false);
});

await test("rejects a no-embedding-index upsert whose core record differs from its predecessor", async () => {
  const fixture = new FeedFixture();
  const baseRecords = await fixture.putRecords("g/version/0/base/records.jsonl.gz", [
    { id: "card-x", identifiers: {}, metadata: { name: "X" } },
  ]);
  const baseEmbeddings = await fixture.putEmbeddings("g/version/0/base/embeddings.f16.gz", [1, 0]);
  const v1Records = await fixture.putRecords("g/version/1/delta-from-0/records.jsonl.gz", [
    {
      op: "upsert",
      record: { id: "card-x", identifiers: { scryfall_oracle: "changed" } },
      metadata: { name: "X2" },
    },
  ]);
  fixture.setJson("catalog-feed-v2.json", {
    checked_at: "2026-08-01T00:00:00Z",
    families: {
      milo1: {
        embedding: EMBEDDING,
        catalogs: {
          "scryfall/mtg": {
            public_name: "scryfall-mtg",
            descriptor: descriptor(),
            current_version: 1,
            rows: 1,
            source_updated_at: "2026-08-01T00:00:00Z",
            base: {
              version: 0,
              rows: 1,
              source_updated_at: "2026-07-31T00:00:00Z",
              assets: { records: baseRecords, embeddings: baseEmbeddings },
            },
            updates: {
              1: {
                from_version: 0,
                to_version: 1,
                rows: { added: 0, updated: 1, deleted: 0 },
                source_updated_at: "2026-08-01T00:00:00Z",
                recognition_rows: 0,
                metadata_rows: 1,
                assets: { records: v1Records },
              },
            },
          },
        },
      },
    },
  });
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg", { includeMetadata: true }),
    /unchanged core record/,
  );
});

await test("explicit null metadata on an upsert removes the row's metadata", async () => {
  const fixture = new FeedFixture();
  const baseRecords = await fixture.putRecords("g/version/0/base/records.jsonl.gz", [
    { id: "card-x", identifiers: {}, metadata: { name: "X" } },
  ]);
  const baseEmbeddings = await fixture.putEmbeddings("g/version/0/base/embeddings.f16.gz", [1, 0]);
  const v1Records = await fixture.putRecords("g/version/1/delta-from-0/records.jsonl.gz", [
    { op: "upsert", record: { id: "card-x", identifiers: {} }, metadata: null },
  ]);
  fixture.setJson("catalog-feed-v2.json", {
    checked_at: "2026-08-01T00:00:00Z",
    families: {
      milo1: {
        embedding: EMBEDDING,
        catalogs: {
          "scryfall/mtg": {
            public_name: "scryfall-mtg",
            descriptor: descriptor(),
            current_version: 1,
            rows: 1,
            source_updated_at: "2026-08-01T00:00:00Z",
            base: {
              version: 0,
              rows: 1,
              source_updated_at: "2026-07-31T00:00:00Z",
              assets: { records: baseRecords, embeddings: baseEmbeddings },
            },
            updates: {
              1: {
                from_version: 0,
                to_version: 1,
                rows: { added: 0, updated: 1, deleted: 0 },
                source_updated_at: "2026-08-01T00:00:00Z",
                recognition_rows: 0,
                metadata_rows: 1,
                assets: { records: v1Records },
              },
            },
          },
        },
      },
    },
  });
  const catalog = await client(fixture).loadCatalog("milo1/scryfall/mtg", { includeMetadata: true });
  assert.equal(catalog.recordForIndex(0).metadata, null);
});

await test("an added records delta row must include embedding_index", async () => {
  const { fixture } = await buildFixture({ withUpdates: true });
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  const badRecords = await fixture.putRecords(
    "scryfall-mtg/version/1/delta-from-0/no-index-add.jsonl.gz",
    [
      {
        op: "upsert",
        record: { id: "card-a", identifiers: { scryfall_oracle: "oracle-a-2" } },
        metadata: { name: "Alpha II" },
        embedding_index: 0,
      },
      { op: "upsert", record: { id: "card-c", identifiers: {} }, metadata: { name: "Gamma" } },
    ],
  );
  const badEmbeddings = await fixture.putEmbeddings(
    "scryfall-mtg/version/1/delta-from-0/no-index-add.f16.gz",
    [-1, 0],
  );
  const entry = feed.families.milo1.catalogs["scryfall/mtg"].updates["1"];
  entry.assets = { records: badRecords, embeddings: badEmbeddings };
  entry.recognition_rows = 1;
  fixture.setJson("catalog-feed-v2.json", feed);
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg", { includeMetadata: true }),
    /added .* row must include embedding_index/,
  );
});

await test("rejects an upsert with neither embedding_index nor metadata", async () => {
  const fixture = new FeedFixture();
  const baseRecords = await fixture.putRecords("g/version/0/base/records.jsonl.gz", [
    { id: "card-x", identifiers: {}, metadata: { name: "X" } },
  ]);
  const baseEmbeddings = await fixture.putEmbeddings("g/version/0/base/embeddings.f16.gz", [1, 0]);
  const v1Records = await fixture.putRecords("g/version/1/delta-from-0/records.jsonl.gz", [
    { op: "upsert", record: { id: "card-x", identifiers: {} } },
  ]);
  fixture.setJson("catalog-feed-v2.json", {
    checked_at: "2026-08-01T00:00:00Z",
    families: {
      milo1: {
        embedding: EMBEDDING,
        catalogs: {
          "scryfall/mtg": {
            public_name: "scryfall-mtg",
            descriptor: descriptor(),
            current_version: 1,
            rows: 1,
            source_updated_at: "2026-08-01T00:00:00Z",
            base: {
              version: 0,
              rows: 1,
              source_updated_at: "2026-07-31T00:00:00Z",
              assets: { records: baseRecords, embeddings: baseEmbeddings },
            },
            updates: {
              1: {
                from_version: 0,
                to_version: 1,
                rows: { added: 0, updated: 1, deleted: 0 },
                source_updated_at: "2026-08-01T00:00:00Z",
                recognition_rows: 0,
                metadata_rows: 0,
                assets: { records: v1Records },
              },
            },
          },
        },
      },
    },
  });
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    /must change recognition or metadata/,
  );
});

await test("rejects an update that declares assets despite having no operations", async () => {
  const { fixture } = await buildFixture({ withUpdates: true });
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  const entry = feed.families.milo1.catalogs["scryfall/mtg"].updates["1"];
  entry.rows = { added: 0, updated: 0, deleted: 0 };
  entry.recognition_rows = 0;
  entry.metadata_rows = 0;
  // assets is left non-empty despite zero total operations: inconsistent and
  // must be rejected.
  fixture.setJson("catalog-feed-v2.json", feed);
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    /must not declare assets when it has no operations/,
  );
});

await test("rejects an update that omits its required records asset despite having operations", async () => {
  const { fixture } = await buildFixture({ withUpdates: true });
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  const entry = feed.families.milo1.catalogs["scryfall/mtg"].updates["1"];
  entry.assets = {};
  fixture.setJson("catalog-feed-v2.json", feed);
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    /records asset is missing/,
  );
});

await test("rejects a catalog whose declared row count disagrees with base + update arithmetic", async () => {
  const { fixture } = await buildFixture({ withUpdates: true });
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  feed.families.milo1.catalogs["scryfall/mtg"].rows = 999;
  fixture.setJson("catalog-feed-v2.json", feed);
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    /row count does not match its base and update row arithmetic/,
  );
});

await test("rejects a base row with an explicit face_index of 0", async () => {
  const fixture = await buildBrokenBaseFixture([{ id: "card-a", identifiers: {}, face_index: 0 }]);
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    /invalid face_index/,
  );
});

await test("rejects unsorted finishes in a base row", async () => {
  const fixture = await buildBrokenBaseFixture([
    { id: "card-a", identifiers: {}, finishes: ["nonfoil", "foil"] },
  ]);
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    /unsorted or duplicate finishes/,
  );
});

await test("rejects duplicate finishes in a base row", async () => {
  const fixture = await buildBrokenBaseFixture([
    { id: "card-a", identifiers: {}, finishes: ["foil", "foil"] },
  ]);
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    /unsorted or duplicate finishes/,
  );
});

// ---------------------------------------------------------------------------
// Persistent cache: latest hit and stale/incompatible fallback
// ---------------------------------------------------------------------------

console.log("\nPersistent cache");

await test("browser entry points use a shared IndexedDB cache by default", async () => {
  const previousIndexedDb = globalThis.indexedDB;
  const hadIndexedDb = Object.hasOwn(globalThis, "indexedDB");
  globalThis.indexedDB = makeFakeIndexedDb();
  try {
    const { fixture } = await buildFixture();
    await BrowserCatalogV2.forGame("mtg", {
      fetchImpl: fixture.fetchImpl(),
      feedUrl: FEED_URL,
    });

    fixture.calls.length = 0;
    await BrowserCatalogV2.forGame("mtg", {
      fetchImpl: fixture.fetchImpl(),
      feedUrl: FEED_URL,
    });
    assert.deepEqual(fixture.calls, [FEED_URL], "the second load should reuse the default cache");
  } finally {
    if (hadIndexedDb) globalThis.indexedDB = previousIndexedDb;
    else delete globalThis.indexedDB;
  }
});

await test("cache: null explicitly disables default IndexedDB persistence", async () => {
  const previousIndexedDb = globalThis.indexedDB;
  const hadIndexedDb = Object.hasOwn(globalThis, "indexedDB");
  globalThis.indexedDB = makeFakeIndexedDb();
  try {
    const { fixture } = await buildFixture();
    const options = {
      fetchImpl: fixture.fetchImpl(),
      feedUrl: FEED_URL,
      cache: null,
    };
    await BrowserCatalogV2.forGame("mtg", options);

    fixture.calls.length = 0;
    await BrowserCatalogV2.forGame("mtg", options);
    assert(
      fixture.calls.some((url) => url.includes("records.jsonl.gz")),
      "an opted-out second load should fetch catalog assets again",
    );
  } finally {
    if (hadIndexedDb) globalThis.indexedDB = previousIndexedDb;
    else delete globalThis.indexedDB;
  }
});

await test("a cached snapshot already at current_version skips every asset fetch", async () => {
  const { fixture } = await buildFixture({ withUpdates: true });
  const cache = new MemorySnapshotCache();
  const warm = await client(fixture, { cache }).loadCatalog("milo1/scryfall/mtg");
  assert.equal(warm.version, 2);
  assert.equal(cache.putCalls, 1, "only the final resolved snapshot should ever be persisted");
  assert.equal(cache.snapshots.size, 1);

  fixture.calls.length = 0;
  const cached = await client(fixture, { cache }).loadCatalog("milo1/scryfall/mtg");
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
    const corrupted = cache.snapshots.get(`2\0${key}\0true`);
    cache.snapshots.set(`2\0${key}\0true`, {
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
  cache.snapshots.set(`-1\0${key}\0true`, stale);
  cache.snapshots.delete(`0\0${key}\0true`);
  cache.snapshots.delete(`1\0${key}\0true`);
  cache.snapshots.delete(`2\0${key}\0true`);

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

await test("only the final snapshot is ever persisted across a base + multiple updates load", async () => {
  const { fixture, key } = await buildFixture({ withUpdates: true });
  const cache = new MemorySnapshotCache();
  const catalog = await client(fixture, { cache }).loadCatalog(key);
  assert.equal(catalog.version, 2);
  assert.equal(cache.putCalls, 1, "intermediate base/v1 stages must never be persisted");
  assert.equal(cache.snapshots.size, 1);
});

await test("advancing the feed prunes the now-stale cached version for the same catalog/mode", async () => {
  const { fixture, key, catalog: catalogV2 } = await buildFixture({ withUpdates: true });
  const cache = new MemorySnapshotCache();

  // First, load and cache only the base (current_version 0) snapshot.
  const v0Fixture = new FeedFixture();
  v0Fixture.files = new Map(fixture.files);
  v0Fixture.setJson("catalog-feed-v2.json", {
    checked_at: "2026-07-24T12:00:00Z",
    families: {
      milo1: {
        embedding: EMBEDDING,
        catalogs: { "scryfall/mtg": { ...catalogV2, current_version: 0, rows: 2, updates: {} } },
      },
    },
  });
  const v0 = await client(v0Fixture, { cache }).loadCatalog(key);
  assert.equal(v0.version, 0);
  assert.equal(cache.snapshots.size, 1);
  assert(cache.snapshots.has(`0\0${key}\0true`));

  // Now the feed has advanced to current_version 2; loading again with the
  // same cache must land on v2 and must not leave the stale v0 entry behind.
  const v2 = await client(fixture, { cache }).loadCatalog(key);
  assert.equal(v2.version, 2);
  assert.equal(cache.snapshots.size, 1, "the stale v0 entry must be pruned, not accumulated alongside v2");
  assert(cache.snapshots.has(`2\0${key}\0true`));
  assert(!cache.snapshots.has(`0\0${key}\0true`));
  assert(cache.deleteCalls > 0, "pruning must go through the cache's delete() method");
});

await test("CatalogV2IndexedDbCache keeps only one row per catalog/mode after multiple loads", async () => {
  const indexedDb = makeFakeIndexedDb();
  const { fixture, key, catalog: catalogV2 } = await buildFixture({ withUpdates: true });
  const cache = new CatalogV2IndexedDbCache({ indexedDb, databaseName: "test-db" });

  const v0Fixture = new FeedFixture();
  v0Fixture.files = new Map(fixture.files);
  v0Fixture.setJson("catalog-feed-v2.json", {
    checked_at: "2026-07-24T12:00:00Z",
    families: {
      milo1: {
        embedding: EMBEDDING,
        catalogs: { "scryfall/mtg": { ...catalogV2, current_version: 0, rows: 2, updates: {} } },
      },
    },
  });
  const v0 = await client(v0Fixture, { cache }).loadCatalog(key);
  assert.equal(v0.version, 0);

  const v2 = await client(fixture, { cache }).loadCatalog(key);
  assert.equal(v2.version, 2);

  // Re-fetch straight from the cache to confirm exactly the final version survives.
  const roundTripped = await cache.get(2, key, true);
  assert.equal(roundTripped.version, 2);
  const stale = await cache.get(0, key, true);
  assert.equal(stale, null, "the stale v0 row must have been pruned by put()'s index scan");
});

await test("CatalogV2IndexedDbCache.delete() removes a specific version without touching others", async () => {
  const indexedDb = makeFakeIndexedDb();
  const cache = new CatalogV2IndexedDbCache({ indexedDb, databaseName: "test-db-2" });
  const { fixture, key } = await buildFixture();
  const snapshot = await client(fixture, { cache }).loadCatalog(key);
  assert.equal((await cache.get(snapshot.version, key, true)).version, snapshot.version);
  await cache.delete(snapshot.version, key, true);
  assert.equal(await cache.get(snapshot.version, key, true), null);
});

await test("pruning is scoped per catalog/mode and never deletes unrelated cache entries", async () => {
  const indexedDb = makeFakeIndexedDb();
  const cache = new CatalogV2IndexedDbCache({ indexedDb, databaseName: "test-db-3" });
  const { fixture, key } = await buildFixture({ withUpdates: true });

  // Cache the same catalog in both recognition-only and metadata modes; the
  // recognition-only mode's own prune pass must not disturb the metadata
  // mode's cached row, and vice versa.
  await client(fixture, { cache }).loadCatalog(key, { includeMetadata: false });
  await client(fixture, { cache }).loadCatalog(key, { includeMetadata: true });

  const recognitionOnly = await cache.get(2, key, false);
  const withMetadata = await cache.get(2, key, true);
  assert.equal(recognitionOnly.version, 2);
  assert.equal(withMetadata.version, 2);
  assert.equal(withMetadata.metadataLoaded, true);
  assert.equal(recognitionOnly.metadataLoaded, false);
});

// ---------------------------------------------------------------------------
// Checksums, sizes, and transport
// ---------------------------------------------------------------------------

console.log("\nChecksums, sizes, and HTTPS");

await test("rejects an asset whose bytes do not match the declared size", async () => {
  const { fixture, key } = await buildFixture();
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  const embeddingsUrl = feed.families.milo1.catalogs["scryfall/mtg"].base.assets.embeddings.url;
  fixture.replace(embeddingsUrl, new Uint8Array([1, 2, 3]));
  await assert.rejects(() => client(fixture).loadCatalog(key), /size mismatch/);
});

await test("rejects an asset whose bytes do not match the declared sha256", async () => {
  const { fixture, key } = await buildFixture();
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  const recordsRef = feed.families.milo1.catalogs["scryfall/mtg"].base.assets.records;
  const original = fixture.files.get(recordsRef.url);
  const tampered = new Uint8Array(original);
  tampered[tampered.length - 1] ^= 0xff; // flip a byte without changing length
  assert.equal(tampered.byteLength, recordsRef.size, "tamper fixture must preserve the declared size");
  fixture.replace(recordsRef.url, tampered);
  await assert.rejects(() => client(fixture).loadCatalog(key), /checksum mismatch/);
});

await test("rejects a non-HTTPS asset URL", async () => {
  const { fixture, key } = await buildFixture();
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  feed.families.milo1.catalogs["scryfall/mtg"].base.assets.records.url =
    feed.families.milo1.catalogs["scryfall/mtg"].base.assets.records.url.replace("https://", "http://");
  fixture.setJson("catalog-feed-v2.json", feed);
  await assert.rejects(() => client(fixture).loadCatalog(key), /https/);
});

// ---------------------------------------------------------------------------
// Malformed identities
// ---------------------------------------------------------------------------

console.log("\nMalformed identities");

async function buildBrokenBaseFixture(rows, path = "g/version/0/base/records.jsonl.gz") {
  const fixture = new FeedFixture();
  const records = await fixture.putRecords(
    path,
    rows.map((row) => ({ metadata: null, ...row })),
  );
  const embeddings = await fixture.putEmbeddings(
    "g/version/0/base/embeddings.f16.gz",
    rows.flatMap(() => [1, 0]),
  );
  fixture.setJson("catalog-feed-v2.json", {
    checked_at: "2026-07-26T12:00:00Z",
    families: {
      milo1: {
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
              assets: { records, embeddings },
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
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    /duplicate base row identity/,
  );
});

await test("rejects a base row that duplicates the primary id under identifiers", async () => {
  const fixture = await buildBrokenBaseFixture([
    { id: "card-a", identifiers: { scryfall_card: "card-a" } },
  ]);
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    /must not duplicate the primary id/,
  );
});

await test("rejects a base row with a negative face_index", async () => {
  const fixture = await buildBrokenBaseFixture([{ id: "card-a", identifiers: {}, face_index: -1 }]);
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    /invalid face_index/,
  );
});

await test("rejects a base row missing a non-empty id", async () => {
  const fixture = await buildBrokenBaseFixture([{ identifiers: {} }]);
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    CatalogV2Error,
  );
});

await test("rejects a base row missing a non-empty name", async () => {
  const fixture = new FeedFixture();
  const records = await fixture.putRecordsRaw("g/version/0/base/missing-name.jsonl.gz", [
    { id: "card-a", identifiers: {}, metadata: null },
  ]);
  const embeddings = await fixture.putEmbeddings("g/version/0/base/embeddings.f16.gz", [1, 0]);
  fixture.setJson("catalog-feed-v2.json", {
    checked_at: "2026-07-26T12:00:00Z",
    families: {
      milo1: {
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
              assets: { records, embeddings },
            },
            updates: {},
          },
        },
      },
    },
  });
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    /name must be a non-empty string/,
  );
});

await test("rejects a base row missing its required metadata field", async () => {
  const fixture = new FeedFixture();
  // Deliberately bypass buildBrokenBaseFixture's auto-added `metadata: null`
  // default so the row is genuinely missing the field.
  const records = await fixture.putRecords("g/version/0/base/records.jsonl.gz", [
    { id: "card-a", name: "card-a", identifiers: {} },
  ]);
  const embeddings = await fixture.putEmbeddings("g/version/0/base/embeddings.f16.gz", [1, 0]);
  fixture.setJson("catalog-feed-v2.json", {
    checked_at: "2026-07-26T12:00:00Z",
    families: {
      milo1: {
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
              assets: { records, embeddings },
            },
            updates: {},
          },
        },
      },
    },
  });
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    /missing its required metadata field/,
  );
});

// ---------------------------------------------------------------------------
// Delta failures
// ---------------------------------------------------------------------------

console.log("\nDelta failures");

async function buildDeltaFailureFixture(mutateUpdate) {
  const { fixture, catalog } = await buildFixture({ withUpdates: true });
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  mutateUpdate(feed.families.milo1.catalogs["scryfall/mtg"].updates["1"], catalog);
  fixture.setJson("catalog-feed-v2.json", feed);
  return fixture;
}

await test("rejects a records delta whose operation count does not match the feed", async () => {
  const fixture = await buildDeltaFailureFixture((update) => {
    // added-deleted stays +1 (matches the feed's overall row arithmetic), but
    // added+updated+deleted (3) no longer matches the asset's 2 operations.
    update.rows = { added: 1, updated: 2, deleted: 0 };
  });
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    /operation count/,
  );
});

await test("rejects a records delta with an out-of-range embedding_index", async () => {
  const { fixture } = await buildFixture({ withUpdates: true });
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  const badRecords = await fixture.putRecords(
    "scryfall-mtg/version/1/delta-from-0/bad-records.jsonl.gz",
    [
      {
        op: "upsert",
        record: { id: "card-a", identifiers: {} },
        metadata: { name: "Alpha II" },
        embedding_index: 5,
      },
      { op: "upsert", record: { id: "card-c", identifiers: {} }, metadata: { name: "Gamma" }, embedding_index: 1 },
    ],
  );
  feed.families.milo1.catalogs["scryfall/mtg"].updates["1"].assets.records = badRecords;
  fixture.setJson("catalog-feed-v2.json", feed);
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    /indexes must be contiguous/,
  );
});

await test("rejects a records delta with duplicate embedding_index assignments", async () => {
  const { fixture } = await buildFixture({ withUpdates: true });
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  const badRecords = await fixture.putRecords(
    "scryfall-mtg/version/1/delta-from-0/dup-records.jsonl.gz",
    [
      {
        op: "upsert",
        record: { id: "card-a", identifiers: {} },
        metadata: { name: "Alpha II" },
        embedding_index: 0,
      },
      { op: "upsert", record: { id: "card-c", identifiers: {} }, metadata: { name: "Gamma" }, embedding_index: 0 },
    ],
  );
  feed.families.milo1.catalogs["scryfall/mtg"].updates["1"].assets.records = badRecords;
  fixture.setJson("catalog-feed-v2.json", feed);
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    /invalid embedding_index/,
  );
});

await test("rejects a records delta that deletes a row not present in the previous snapshot", async () => {
  const fixture = await buildDeltaFailureFixture(() => {});
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  const badRecords = await fixture.putRecords(
    "scryfall-mtg/version/1/delta-from-0/missing-delete.jsonl.gz",
    [
      { op: "delete", id: "card-does-not-exist" },
      { op: "upsert", record: { id: "card-c", identifiers: {} }, metadata: { name: "Gamma" }, embedding_index: 0 },
    ],
  );
  const embeddings = await fixture.putEmbeddings(
    "scryfall-mtg/version/1/delta-from-0/missing-delete.f16.gz",
    [1, 0],
  );
  const entry = feed.families.milo1.catalogs["scryfall/mtg"].updates["1"];
  entry.assets = { records: badRecords, embeddings };
  // Keep added - deleted == 1 and total ops == 2 (matching the asset's two
  // operations) so both the feed's row arithmetic and operation-count checks
  // still pass; the malformed delete must fail during operation processing.
  entry.rows = { added: 1, updated: 1, deleted: 0 };
  entry.recognition_rows = 1;
  entry.metadata_rows = 1;
  fixture.setJson("catalog-feed-v2.json", feed);
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    /missing or duplicated/,
  );
});

await test("rejects a records delta whose row classification disagrees with the feed", async () => {
  const fixture = new FeedFixture();
  const baseRecords = await fixture.putRecords("g/version/0/base/records.jsonl.gz", [
    { id: "card-a", identifiers: {}, metadata: null },
  ]);
  const baseEmbeddings = await fixture.putEmbeddings("g/version/0/base/embeddings.f16.gz", [1, 0]);
  const v1Records = await fixture.putRecords("g/version/1/delta-from-0/records.jsonl.gz", [
    { op: "upsert", record: { id: "card-b", identifiers: {} }, embedding_index: 0 },
  ]);
  const v1Embeddings = await fixture.putEmbeddings("g/version/1/delta-from-0/embeddings.f16.gz", [0, 1]);
  fixture.setJson("catalog-feed-v2.json", {
    checked_at: "2026-08-01T00:00:00Z",
    families: {
      milo1: {
        embedding: EMBEDDING,
        catalogs: {
          "scryfall/mtg": {
            public_name: "scryfall-mtg",
            descriptor: descriptor(),
            current_version: 1,
            // Declared to match the (wrong) rows.added == 0 below, so the
            // arithmetic check passes and only the classification check can
            // catch the actually-new "card-b" row being mislabeled.
            rows: 1,
            source_updated_at: "2026-08-01T00:00:00Z",
            base: {
              version: 0,
              rows: 1,
              source_updated_at: "2026-07-31T00:00:00Z",
              assets: { records: baseRecords, embeddings: baseEmbeddings },
            },
            updates: {
              1: {
                from_version: 0,
                to_version: 1,
                // "card-b" does not exist in the predecessor and must be
                // classified as added, but this declares added: 0.
                rows: { added: 0, updated: 1, deleted: 0 },
                source_updated_at: "2026-08-01T00:00:00Z",
                recognition_rows: 1,
                metadata_rows: 0,
                assets: { records: v1Records, embeddings: v1Embeddings },
              },
            },
          },
        },
      },
    },
  });
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    /row classification/,
  );
});

await test("rejects an unsupported records delta operation", async () => {
  const { fixture } = await buildFixture({ withUpdates: true });
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  const badRecords = await fixture.putRecords(
    "scryfall-mtg/version/1/delta-from-0/bad-op.jsonl.gz",
    [
      { op: "replace", id: "card-a" },
      { op: "upsert", record: { id: "card-c", identifiers: {} }, metadata: { name: "Gamma" }, embedding_index: 0 },
    ],
  );
  const badEmbeddings = await fixture.putEmbeddings(
    "scryfall-mtg/version/1/delta-from-0/bad-op.f16.gz",
    [0.5, 0.5],
  );
  const entry = feed.families.milo1.catalogs["scryfall/mtg"].updates["1"];
  entry.assets = { records: badRecords, embeddings: badEmbeddings };
  fixture.setJson("catalog-feed-v2.json", feed);
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    /unsupported records delta operation/,
  );
});

await test("rejects a catalog whose update chain skips a version", async () => {
  const { fixture } = await buildFixture({ withUpdates: true });
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  delete feed.families.milo1.catalogs["scryfall/mtg"].updates["1"];
  fixture.setJson("catalog-feed-v2.json", feed);
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    /missing an update/,
  );
});

await test("rejects a catalog whose update declares the wrong exact-predecessor base", async () => {
  const { fixture } = await buildFixture({ withUpdates: true });
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  feed.families.milo1.catalogs["scryfall/mtg"].updates["2"].from_version = 0;
  fixture.setJson("catalog-feed-v2.json", feed);
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    /exact-predecessor delta/,
  );
});

await test("rejects a malformed routine checkpoint bridge", async () => {
  const { fixture, catalog } = await buildFixture({ withUpdates: true });
  await promoteToRoutineCheckpoint(fixture, catalog);
  const feed = JSON.parse(new TextDecoder().decode(fixture.files.get(FEED_URL)));
  feed.families.milo1.catalogs["scryfall/mtg"].updates["10"].from_version = 8;
  fixture.setJson("catalog-feed-v2.json", feed);
  await assert.rejects(
    () => client(fixture).loadCatalog("milo1/scryfall/mtg"),
    /checkpoint bridge/,
  );
});

// ---------------------------------------------------------------------------
// Descriptor discovery
// ---------------------------------------------------------------------------

console.log("\nDescriptor discovery");

async function buildMultiCatalogFixture() {
  const fixture = new FeedFixture();
  async function simpleCatalog(path, publicName, descriptorOverrides) {
    const records = await fixture.putRecords(`${path}/version/0/base/records.jsonl.gz`, [
      { id: "x", metadata: null },
    ]);
    const embeddings = await fixture.putEmbeddings(`${path}/version/0/base/embeddings.f16.gz`, [1, 0]);
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
        assets: { records, embeddings },
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
  const scryfallMtgAlt = await simpleCatalog("scryfall-mtg-alt", "scryfall-mtg-alt", {
    game: "magic-the-gathering",
    source: "scryfall",
    profile: "alt",
    result_identifier: "scryfall_card",
    recommended: false,
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
  const tcgplayerDbs = await simpleCatalog("tcgplayer-dbs", "tcgplayer-dbs", {
    game: "dragon-ball-super-card-game",
    source: "tcgplayer",
    result_identifier: "tcgplayer_product",
    recommended: true,
  });
  const tcgplayerLorcana = await simpleCatalog("tcgplayer-lorcana", "tcgplayer-lorcana", {
    game: "lorcana",
    source: "tcgplayer",
    result_identifier: "tcgplayer_product",
    recommended: true,
  });

  fixture.setJson("catalog-feed-v2.json", {
    checked_at: "2026-07-26T12:00:00Z",
    families: {
      milo1: {
        embedding: EMBEDDING,
        catalogs: {
          "scryfall/mtg": scryfallMtg,
          "scryfall/mtg-alt": scryfallMtgAlt,
          "tcgplayer/mtg": tcgplayerMtg,
          "tcgplayer/pokemon": tcgplayerPokemon,
          "tcgplayer/dbs": tcgplayerDbs,
        },
      },
      milo2: {
        embedding: EMBEDDING,
        catalogs: { "tcgplayer/lorcana": tcgplayerLorcana },
      },
    },
  });
  return fixture;
}

await test("defaults to the recommended Scryfall catalog for MTG", async () => {
  const fixture = await buildMultiCatalogFixture();
  const catalog = await client(fixture).loadGame("mtg");
  assert.equal(catalog.catalogKey, "milo1/scryfall/mtg");
  assert.equal(catalog.familyKey, "milo1");
  assert.equal(catalog.descriptor.source, "scryfall");
});

await test("defaults to the recommended TCGplayer catalog for other games", async () => {
  const fixture = await buildMultiCatalogFixture();
  const catalog = await client(fixture).loadGame("pokemon");
  assert.equal(catalog.catalogKey, "milo1/tcgplayer/pokemon");
  assert.equal(catalog.descriptor.source, "tcgplayer");
});

await test("an explicit source overrides the default and still finds the recommended descriptor", async () => {
  const fixture = await buildMultiCatalogFixture();
  const catalog = await client(fixture).loadGame("mtg", { source: "tcgplayer" });
  assert.equal(catalog.catalogKey, "milo1/tcgplayer/mtg");
  assert.equal(catalog.descriptor.result_identifier, "tcgplayer_product");
});

await test("the dbs alias resolves to dragon-ball-super-card-game", async () => {
  const fixture = await buildMultiCatalogFixture();
  const catalog = await client(fixture).loadGame("dbs");
  assert.equal(catalog.catalogKey, "milo1/tcgplayer/dbs");
  assert.equal(catalog.descriptor.game, "dragon-ball-super-card-game");
  assert.equal(catalog.descriptor.source, "tcgplayer");
});

await test("the default family is milo1 and does not search other families", async () => {
  const fixture = await buildMultiCatalogFixture();
  await assert.rejects(
    () => client(fixture).loadGame("lorcana"),
    /no Catalog v2 feed entry matches/,
  );
});

await test("an explicit family selects a catalog outside the default milo1 family", async () => {
  const fixture = await buildMultiCatalogFixture();
  const catalog = await client(fixture).loadGame("lorcana", { family: "milo2" });
  assert.equal(catalog.catalogKey, "milo2/tcgplayer/lorcana");
});

await test("family: null searches every family in the feed", async () => {
  const fixture = await buildMultiCatalogFixture();
  const catalog = await client(fixture).loadGame("lorcana", { family: null });
  assert.equal(catalog.catalogKey, "milo2/tcgplayer/lorcana");
});

await test("an explicit profile selects a catalog regardless of its recommended flag", async () => {
  const fixture = await buildMultiCatalogFixture();
  const catalog = await client(fixture).loadGame("mtg", { profile: "alt" });
  assert.equal(catalog.catalogKey, "milo1/scryfall/mtg-alt");
  assert.equal(catalog.descriptor.profile, "alt");
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
  feed.families.milo1.catalogs["tcgplayer/mtg-2"] = {
    ...feed.families.milo1.catalogs["tcgplayer/mtg"],
  };
  fixture.setJson("catalog-feed-v2.json", feed);
  await assert.rejects(
    () => client(fixture).loadGame("mtg", { source: "tcgplayer" }),
    /multiple Catalog v2 feed entries match/,
  );
});

await test("loadCatalog() loads directly by full catalog key without game discovery", async () => {
  const fixture = await buildMultiCatalogFixture();
  const catalog = await client(fixture).loadCatalog("milo1/tcgplayer/pokemon");
  assert.equal(catalog.catalogKey, "milo1/tcgplayer/pokemon");
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
