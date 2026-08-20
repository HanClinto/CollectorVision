// Client for the live CollectorVisionCatalog v2 contract.
//
// The only document a normal client needs is the moving feed
// (`catalog-feed-v2.json`). It nests catalogs under immutable embedding
// families and advertises the newest usable base plus a retained route of
// exact-predecessor updates ending at `current_version`. The route may begin
// before the base so recent cached clients can cross a checkpoint without
// downloading it. Every asset URL, size, and SHA-256 is authoritative.

const DEFAULT_FEED_URL =
  "https://hanclinto.github.io/CollectorVisionCatalog/catalog-v2/catalog-feed-v2.json";

// The standard embedding family, matching Python's `Embedding.MILO` value.
// Passing `family: null` searches every family in the feed instead.
const DEFAULT_FAMILY = "milo1";

// Short aliases are a purely ergonomic convenience for `forGame`/`loadGame`.
// They are never used to construct a catalog path: the matching catalog is
// always discovered from the feed by descriptor.
const GAME_ALIASES = Object.freeze({
  mtg: "magic-the-gathering",
  pokemon: "pokemon",
  pokemonjapan: "pokemon-japan",
  "pokemon-japan": "pokemon-japan",
  yugioh: "yugioh",
  fab: "flesh-and-blood",
  lorcana: "lorcana",
  digimon: "digimon-card-game",
  onepiece: "one-piece",
  swu: "star-wars-unlimited",
  unionarena: "union-arena",
  "union-arena": "union-arena",
  gundam: "gundam-card-game",
  riftbound: "riftbound",
  dbs: "dragon-ball-super-card-game",
});

const DEFAULT_SOURCE_BY_GAME = Object.freeze({
  "magic-the-gathering": "scryfall",
});
const FALLBACK_SOURCE = "tcgplayer";
const FLOAT16_LOOKUP = createFloat16LookupTable();

export class CatalogV2Error extends Error {}

export class BrowserCatalogV2 {
  static async forGame(game, options = {}) {
    const {
      fetchImpl = globalThis.fetch,
      feedUrl = DEFAULT_FEED_URL,
      cache = defaultCatalogV2Cache(),
      ...selection
    } = options;
    const client = new CatalogV2FeedClient({ fetchImpl, feedUrl, cache });
    return client.loadGame(game, selection);
  }

  constructor({
    familyKey,
    catalogKey,
    publicName,
    descriptor,
    embedding,
    version,
    sourceUpdatedAt,
    records,
    embeddings,
    metadataLoaded,
  }) {
    this.familyKey = familyKey;
    this.catalogKey = catalogKey;
    this.publicName = publicName;
    this.descriptor = Object.freeze({ ...descriptor });
    this.embedding = Object.freeze({ ...embedding });
    this.dimension = embedding.dimensions;
    this.version = version;
    this.sourceUpdatedAt = sourceUpdatedAt;
    this.records = Object.freeze(records);
    this.embeddings = embeddings;
    this.metadataLoaded = metadataLoaded;
  }

  get rows() {
    return this.records.length;
  }

  search(query, topK = 5) {
    return this.searchRecords(query, topK).map(({ score, card_id }) => [score, card_id]);
  }

  searchRecords(query, topK = 5) {
    if (!(query instanceof Float32Array) || query.length !== this.dimension) {
      throw new TypeError(`query must be Float32Array(${this.dimension})`);
    }
    if (!Number.isInteger(topK) || topK <= 0) {
      throw new RangeError("topK must be a positive integer");
    }
    const resultCount = Math.min(topK, this.records.length);
    const bestScores = new Float64Array(resultCount);
    bestScores.fill(-Infinity);
    const bestIndexes = new Int32Array(resultCount);
    bestIndexes.fill(-1);
    for (let row = 0; row < this.records.length; row += 1) {
      let score = 0;
      const offset = row * this.dimension;
      for (let column = 0; column < this.dimension; column += 1) {
        score += FLOAT16_LOOKUP[this.embeddings[offset + column]] * query[column];
      }
      if (score <= bestScores[resultCount - 1]) continue;
      let position = resultCount - 1;
      while (position > 0 && score > bestScores[position - 1]) {
        bestScores[position] = bestScores[position - 1];
        bestIndexes[position] = bestIndexes[position - 1];
        position -= 1;
      }
      bestScores[position] = score;
      bestIndexes[position] = row;
    }
    return Array.from(
      { length: resultCount },
      (_, index) => this.recordForIndex(bestIndexes[index], bestScores[index]),
    );
  }

  recordForIndex(index, score = undefined) {
    const record = this.records[index];
    if (!record) throw new RangeError(`catalog row ${index} is out of range`);
    return this.#toPublicRecord(record, score);
  }

  #toPublicRecord(record, score) {
    const key =
      record.faceIndex === 0
        ? `${this.descriptor.source}:${record.id}`
        : `${this.descriptor.source}:${record.id}:face:${record.faceIndex}`;
    const result = {
      key,
      id: record.id,
      name: record.name,
      identifiers: {
        [this.descriptor.result_identifier]: record.id,
        ...record.identifiers,
      },
      face_index: record.faceIndex,
      result_identifier: this.descriptor.result_identifier,
      card_id: record.id,
      finishes: record.finishes ? [...record.finishes] : [],
    };
    if (this.metadataLoaded) {
      result.metadata = record.metadata !== undefined ? structuredClone(record.metadata) : null;
    }
    if (score !== undefined) result.score = score;
    return result;
  }
}

export class CatalogV2IndexedDbCache {
  constructor({
    indexedDb = globalThis.indexedDB,
    databaseName = "collectorvision-catalog-v2",
  } = {}) {
    if (!indexedDb) throw new TypeError("IndexedDB is not available");
    this.indexedDb = indexedDb;
    this.databaseName = databaseName;
    this.databasePromise = null;
  }

  async get(version, catalogKey, includeMetadata) {
    const database = await this.#database();
    const snapshot = await requestResult(
      database
        .transaction("catalogs", "readonly")
        .objectStore("catalogs")
        .get(snapshotKey(version, catalogKey, includeMetadata)),
    );
    if (!snapshot) return null;
    if (
      !isObject(snapshot.descriptor) ||
      !isObject(snapshot.embedding) ||
      !Array.isArray(snapshot.records) ||
      !(snapshot.embeddings instanceof ArrayBuffer)
    ) {
      throw new CatalogV2Error("invalid Catalog v2 snapshot in IndexedDB");
    }
    return new BrowserCatalogV2({
      familyKey: snapshot.familyKey,
      catalogKey: snapshot.catalogKey,
      publicName: snapshot.publicName,
      descriptor: snapshot.descriptor,
      embedding: snapshot.embedding,
      version: snapshot.version,
      sourceUpdatedAt: snapshot.sourceUpdatedAt,
      records: snapshot.records,
      embeddings: new Uint16Array(snapshot.embeddings),
      metadataLoaded: includeMetadata,
    });
  }

  async put(catalog) {
    if (!(catalog instanceof BrowserCatalogV2)) {
      throw new TypeError("Catalog v2 cache accepts BrowserCatalogV2 snapshots");
    }
    const database = await this.#database();
    const metadataMode = catalog.metadataLoaded ? "metadata" : "recognition";
    const snapshot = {
      id: snapshotKey(catalog.version, catalog.catalogKey, catalog.metadataLoaded),
      catalogKey: catalog.catalogKey,
      metadataMode,
      familyKey: catalog.familyKey,
      publicName: catalog.publicName,
      descriptor: structuredClone(catalog.descriptor),
      embedding: structuredClone(catalog.embedding),
      version: catalog.version,
      sourceUpdatedAt: catalog.sourceUpdatedAt,
      records: structuredClone(catalog.records),
      embeddings: catalog.embeddings.slice().buffer,
    };
    const transaction = database.transaction("catalogs", "readwrite");
    const store = transaction.objectStore("catalogs");
    store.put(snapshot);
    // Only one snapshot per (catalogKey, metadataMode) is ever useful: an
    // older version is superseded the moment a newer one is cached. Prune it
    // here so the store never grows unbounded across intermediate updates.
    await pruneOtherVersions(store, catalog.catalogKey, metadataMode, catalog.version);
    await transactionComplete(transaction);
  }

  async delete(version, catalogKey, includeMetadata) {
    const database = await this.#database();
    const transaction = database.transaction("catalogs", "readwrite");
    transaction.objectStore("catalogs").delete(snapshotKey(version, catalogKey, includeMetadata));
    await transactionComplete(transaction);
  }

  async #database() {
    if (!this.databasePromise) {
      this.databasePromise = new Promise((resolve, reject) => {
        const request = this.indexedDb.open(this.databaseName, 2);
        request.onupgradeneeded = () => {
          const database = request.result;
          const store = database.objectStoreNames.contains("catalogs")
            ? request.transaction.objectStore("catalogs")
            : database.createObjectStore("catalogs", { keyPath: "id" });
          if (!store.indexNames.contains("byCatalogMode")) {
            store.createIndex("byCatalogMode", ["catalogKey", "metadataMode"], { unique: false });
          }
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
        request.onblocked = () => reject(new CatalogV2Error("Catalog v2 IndexedDB is blocked"));
      });
    }
    return this.databasePromise;
  }
}

function pruneOtherVersions(store, catalogKey, metadataMode, keepVersion) {
  return new Promise((resolve, reject) => {
    const request = store.index("byCatalogMode").openCursor([catalogKey, metadataMode]);
    request.onsuccess = () => {
      const cursor = request.result;
      if (!cursor) {
        resolve();
        return;
      }
      if (cursor.value.version !== keepVersion) cursor.delete();
      cursor.continue();
    };
    request.onerror = () => reject(request.error);
  });
}

const DEFAULT_CACHES = new WeakMap();

function defaultCatalogV2Cache() {
  const indexedDb = globalThis.indexedDB;
  if (!indexedDb || (typeof indexedDb !== "object" && typeof indexedDb !== "function")) {
    return null;
  }
  let cache = DEFAULT_CACHES.get(indexedDb);
  if (!cache) {
    cache = new CatalogV2IndexedDbCache({ indexedDb });
    DEFAULT_CACHES.set(indexedDb, cache);
  }
  return cache;
}

export class CatalogV2FeedClient {
  constructor({
    fetchImpl = globalThis.fetch,
    feedUrl = DEFAULT_FEED_URL,
    cache = defaultCatalogV2Cache(),
  } = {}) {
    if (typeof fetchImpl !== "function") throw new TypeError("fetch implementation is required");
    this.fetchImpl = fetchImpl.bind(globalThis);
    this.feedUrl = feedUrl;
    this.cache = cache;
    if (
      this.cache !== null &&
      (typeof this.cache.get !== "function" || typeof this.cache.put !== "function")
    ) {
      throw new TypeError("cache must provide async get() and put() methods");
    }
  }

  async loadGame(
    game,
    { source = null, family = DEFAULT_FAMILY, profile = null, includeMetadata = true, previous = null } = {},
  ) {
    const normalizedGame = normalizeGame(game);
    const selectedSource = source ?? DEFAULT_SOURCE_BY_GAME[normalizedGame] ?? FALLBACK_SOURCE;
    const feed = await this.#fetchFeed();
    const resolved = discoverCatalog(feed, normalizedGame, selectedSource, { family, profile });
    return this.#loadResolvedCatalog(resolved, { includeMetadata, previous });
  }

  async loadCatalog(fullCatalogKey, { includeMetadata = true, previous = null } = {}) {
    const feed = await this.#fetchFeed();
    const resolved = resolveCatalogByKey(feed, fullCatalogKey);
    return this.#loadResolvedCatalog(resolved, { includeMetadata, previous });
  }

  async #loadResolvedCatalog(resolved, { includeMetadata, previous }) {
    const { catalog, versions } = resolved;
    const targetVersion = catalog.current_version;

    let snapshot = null;
    let startIndex = -1;
    let mutated = false;

    if (previous !== null && isCompatibleSnapshot(previous, resolved, includeMetadata)) {
      const index = versions.indexOf(previous.version);
      if (index !== -1) {
        snapshot = previous;
        startIndex = index;
      }
    }

    if (snapshot === null && this.cache !== null) {
      for (let index = versions.length - 1; index >= 0; index -= 1) {
        const candidate = await this.#cachedSnapshot(
          versions[index],
          resolved.fullCatalogKey,
          includeMetadata,
        );
        if (candidate === null) continue;
        if (isCompatibleSnapshot(candidate, resolved, includeMetadata) && candidate.version === versions[index]) {
          snapshot = candidate;
          startIndex = index;
          break;
        }
        console.warn("Ignoring incompatible Catalog v2 snapshot in persistent cache");
      }
    }

    if (snapshot === null) {
      snapshot = await this.#loadBase(resolved, includeMetadata);
      startIndex = versions.indexOf(catalog.base.version);
      mutated = true;
    }

    for (let index = startIndex + 1; index < versions.length; index += 1) {
      const toVersion = versions[index];
      const updateEntry = catalog.updates[String(toVersion)];
      snapshot = await this.#applyUpdate(resolved, snapshot, updateEntry, includeMetadata);
      mutated = true;
    }

    if (snapshot.version !== targetVersion || snapshot.records.length !== catalog.rows) {
      throw new CatalogV2Error("reconstructed catalog does not match the feed's current state");
    }

    // Only the final, fully-resolved snapshot is worth persisting: every
    // intermediate stage is superseded before the caller ever sees it.
    if (mutated) {
      await this.#persistSnapshot(snapshot);
      await this.#pruneCachedVersions(resolved, includeMetadata, snapshot.version, versions);
    }
    return snapshot;
  }

  async #pruneCachedVersions(resolved, includeMetadata, keepVersion, versions) {
    if (this.cache === null || typeof this.cache.delete !== "function") return;
    for (const version of versions) {
      if (version === keepVersion) continue;
      try {
        await this.cache.delete(version, resolved.fullCatalogKey, includeMetadata);
      } catch (error) {
        console.warn("Catalog v2 persistent cache prune failed; stale snapshot may remain", error);
      }
    }
  }

  async #loadBase(resolved, includeMetadata) {
    const { catalog, family } = resolved;
    const base = catalog.base;
    const dimension = family.embedding.dimensions;

    // The feed always ships combined records; metadata is only discarded
    // in-memory afterwards when the caller does not want it.
    const rows = parseJsonLines(
      await this.#fetchGzipAsset(base.assets.records, "base records"),
      "base records",
    );
    if (rows.length !== base.rows) {
      throw new CatalogV2Error("base record row count does not match the feed");
    }
    const embeddingBytes = await this.#fetchGzipAsset(base.assets.embeddings, "base embeddings");
    const matrix = parseFloat16Matrix(embeddingBytes, base.rows, dimension);

    const seen = new Map();
    const staged = rows.map((value, row) => {
      const record = parseBaseRecord(value, catalog.descriptor, "base record row");
      let faces = seen.get(record.id);
      if (!faces) {
        faces = new Set();
        seen.set(record.id, faces);
      }
      if (faces.has(record.faceIndex)) {
        throw new CatalogV2Error(
          `duplicate base row identity for id ${JSON.stringify(record.id)}`,
        );
      }
      faces.add(record.faceIndex);
      if (!includeMetadata) delete record.metadata;
      return { record, embedding: matrix.slice(row * dimension, (row + 1) * dimension) };
    });

    staged.sort((left, right) => compareIdentity(left.record, right.record));
    const records = staged.map(({ record }) => record);
    const embeddings = new Uint16Array(records.length * dimension);
    staged.forEach(({ embedding }, row) => embeddings.set(embedding, row * dimension));

    return new BrowserCatalogV2({
      familyKey: resolved.familyKey,
      catalogKey: resolved.fullCatalogKey,
      publicName: catalog.public_name,
      descriptor: catalog.descriptor,
      embedding: family.embedding,
      version: base.version,
      sourceUpdatedAt: base.source_updated_at,
      records,
      embeddings,
      metadataLoaded: includeMetadata,
    });
  }

  async #applyUpdate(resolved, snapshot, updateEntry, includeMetadata) {
    const { catalog, family } = resolved;
    const dimension = family.embedding.dimensions;
    if (snapshot.version !== updateEntry.from_version) {
      throw new CatalogV2Error("update does not extend the loaded snapshot's exact base version");
    }

    const totalOps = updateEntry.rows.added + updateEntry.rows.updated + updateEntry.rows.deleted;
    const operations =
      totalOps === 0
        ? []
        : parseJsonLines(
            await this.#fetchGzipAsset(updateEntry.assets.records, "records delta"),
            "records delta",
          );
    if (operations.length !== totalOps) {
      throw new CatalogV2Error("records delta operation count does not match the feed");
    }

    const byId = new Map();
    snapshot.records.forEach((record, row) => {
      let faces = byId.get(record.id);
      if (!faces) {
        faces = new Map();
        byId.set(record.id, faces);
      }
      faces.set(record.faceIndex, {
        record: { ...record, identifiers: { ...record.identifiers } },
        embedding: snapshot.embeddings.slice(row * dimension, (row + 1) * dimension),
      });
    });

    // First pass: parse and validate every operation against the
    // *unmutated* predecessor state (needed so a no-embedding upsert can be
    // checked against the row it is meant to leave untouched).
    const parsedOperations = [];
    const operatedKeys = new Set();
    const usedEmbeddingIndexes = new Set();
    const addedKeys = new Set();
    const changedKeys = new Set();
    const deletedKeys = new Set();
    for (const operation of operations) {
      if (operation.op === "delete") {
        const target = parseIdentityTarget(operation, "records delta delete");
        const key = identityKey(target.id, target.faceIndex);
        const faces = byId.get(target.id);
        if (operatedKeys.has(key) || !faces || !faces.has(target.faceIndex)) {
          throw new CatalogV2Error(
            `records delta deletes a row that is missing or duplicated: ${JSON.stringify(target)}`,
          );
        }
        operatedKeys.add(key);
        deletedKeys.add(key);
        parsedOperations.push({ kind: "delete", id: target.id, faceIndex: target.faceIndex });
      } else if (operation.op === "upsert") {
        const record = parseIdentityRecord(operation.record, catalog.descriptor, "records delta upsert");
        const key = identityKey(record.id, record.faceIndex);
        if (operatedKeys.has(key)) {
          throw new CatalogV2Error(
            `records delta upsert is duplicated for id ${JSON.stringify(record.id)}`,
          );
        }
        const hasIndex = Object.hasOwn(operation, "embedding_index");
        const hasMetadata = Object.hasOwn(operation, "metadata");
        if (!hasIndex && !hasMetadata) {
          throw new CatalogV2Error(
            `records delta upsert for id ${JSON.stringify(record.id)} must change recognition or metadata`,
          );
        }
        let embeddingIndex = null;
        if (hasIndex) {
          embeddingIndex = operation.embedding_index;
          if (
            !Number.isInteger(embeddingIndex) ||
            embeddingIndex < 0 ||
            usedEmbeddingIndexes.has(embeddingIndex)
          ) {
            throw new CatalogV2Error(
              `records delta upsert has an invalid embedding_index for id ${JSON.stringify(record.id)}`,
            );
          }
          usedEmbeddingIndexes.add(embeddingIndex);
        }
        let metadataValue;
        if (hasMetadata) {
          metadataValue = operation.metadata;
          if (metadataValue !== null && !isObject(metadataValue)) {
            throw new CatalogV2Error(
              `records delta metadata must be an object or null for id ${JSON.stringify(record.id)}`,
            );
          }
        }
        operatedKeys.add(key);
        const faces = byId.get(record.id);
        const existing = faces?.get(record.faceIndex);
        if (!existing) {
          if (!hasIndex) {
            throw new CatalogV2Error(
              `an added records delta row must include embedding_index for id ${JSON.stringify(record.id)}`,
            );
          }
          addedKeys.add(key);
        } else {
          if (!hasIndex && !coreRecordsEqual(existing.record, record)) {
            throw new CatalogV2Error(
              `records delta upsert without embedding_index must repeat the unchanged core record for id ${JSON.stringify(record.id)}`,
            );
          }
          changedKeys.add(key);
        }
        parsedOperations.push({
          kind: "upsert",
          record,
          hasMetadata,
          metadataValue,
          embeddingIndex,
        });
      } else {
        throw new CatalogV2Error(`unsupported records delta operation ${JSON.stringify(operation.op)}`);
      }
    }

    const sortedIndexes = [...usedEmbeddingIndexes].sort((left, right) => left - right);
    if (!sortedIndexes.every((value, index) => value === index)) {
      throw new CatalogV2Error("records delta embedding indexes must be contiguous and unique");
    }

    let deltaEmbeddings;
    if (usedEmbeddingIndexes.size > 0) {
      const asset = updateEntry.assets.embeddings;
      if (!asset) throw new CatalogV2Error("records delta embeddings asset is missing");
      deltaEmbeddings = parseFloat16Matrix(
        await this.#fetchGzipAsset(asset, "records delta embeddings"),
        usedEmbeddingIndexes.size,
        dimension,
      );
    } else {
      if (updateEntry.assets.embeddings) {
        throw new CatalogV2Error(
          "records delta without recognition changes must not declare an embeddings asset",
        );
      }
      deltaEmbeddings = new Uint16Array();
    }

    const recognitionChangedCount =
      deletedKeys.size +
      parsedOperations.filter((op) => op.kind === "upsert" && op.embeddingIndex !== null).length;
    if (recognitionChangedCount !== updateEntry.recognition_rows) {
      throw new CatalogV2Error("records delta recognition_rows does not match its operations");
    }

    const metadataFieldCount = parsedOperations.filter(
      (op) => op.kind === "upsert" && op.hasMetadata,
    ).length;

    // Second pass: apply the parsed operations to the mutable state.
    let deleteWithMetadataCount = 0;
    for (const op of parsedOperations) {
      if (op.kind === "delete") {
        const faces = byId.get(op.id);
        const prior = faces.get(op.faceIndex);
        if (includeMetadata && prior.record.metadata !== undefined && prior.record.metadata !== null) {
          deleteWithMetadataCount += 1;
        }
        faces.delete(op.faceIndex);
        if (faces.size === 0) byId.delete(op.id);
        continue;
      }
      const { record, hasMetadata, metadataValue, embeddingIndex } = op;
      let faces = byId.get(record.id);
      if (!faces) {
        faces = new Map();
        byId.set(record.id, faces);
      }
      const existing = faces.get(record.faceIndex);
      const existingMetadata = existing?.record.metadata;
      const newMetadata = hasMetadata ? metadataValue : existingMetadata;
      const embedding =
        embeddingIndex !== null
          ? deltaEmbeddings.slice(embeddingIndex * dimension, (embeddingIndex + 1) * dimension)
          : existing.embedding;
      const nextRecord = { ...record };
      if (includeMetadata) {
        if (newMetadata !== undefined) nextRecord.metadata = newMetadata;
      } else {
        delete nextRecord.metadata;
      }
      faces.set(record.faceIndex, { record: nextRecord, embedding });
    }

    if (includeMetadata) {
      if (metadataFieldCount + deleteWithMetadataCount !== updateEntry.metadata_rows) {
        throw new CatalogV2Error("records delta metadata_rows does not match its operations");
      }
    } else if (metadataFieldCount > updateEntry.metadata_rows) {
      // Recognition-only mode cannot see prior metadata state on deletes, so
      // only a loose subset bound is checkable, per the feed contract.
      throw new CatalogV2Error("records delta metadata_rows does not match its operations");
    }

    if (
      addedKeys.size !== updateEntry.rows.added ||
      deletedKeys.size !== updateEntry.rows.deleted ||
      changedKeys.size !== updateEntry.rows.updated
    ) {
      throw new CatalogV2Error("records delta row classification does not match the feed");
    }

    const flattened = [];
    for (const faces of byId.values()) {
      for (const entry of faces.values()) flattened.push(entry);
    }
    flattened.sort((left, right) => compareIdentity(left.record, right.record));

    const expectedTotal = snapshot.records.length + updateEntry.rows.added - updateEntry.rows.deleted;
    if (flattened.length !== expectedTotal) {
      throw new CatalogV2Error("records delta reconstructed an unexpected row total");
    }

    const records = flattened.map(({ record }) => record);
    const embeddings = new Uint16Array(records.length * dimension);
    flattened.forEach(({ embedding }, row) => embeddings.set(embedding, row * dimension));

    return new BrowserCatalogV2({
      familyKey: resolved.familyKey,
      catalogKey: resolved.fullCatalogKey,
      publicName: catalog.public_name,
      descriptor: catalog.descriptor,
      embedding: family.embedding,
      version: updateEntry.to_version,
      sourceUpdatedAt: updateEntry.source_updated_at,
      records,
      embeddings,
      metadataLoaded: includeMetadata,
    });
  }

  async #cachedSnapshot(version, catalogKey, includeMetadata) {
    if (this.cache === null) return null;
    try {
      return await this.cache.get(version, catalogKey, includeMetadata);
    } catch (error) {
      console.warn("Catalog v2 persistent cache read failed; using network assets", error);
      return null;
    }
  }

  async #persistSnapshot(catalog) {
    if (this.cache === null) return;
    try {
      await this.cache.put(catalog);
    } catch (error) {
      console.warn("Catalog v2 persistent cache write failed; catalog remains loaded", error);
    }
  }

  async #fetchFeed() {
    const feed = await this.#fetchJson(new URL(this.feedUrl));
    validateFeed(feed);
    return feed;
  }

  async #fetchGzipAsset(assetReference, label) {
    validateAssetReference(assetReference, label);
    const url = new URL(assetReference.url);
    const compressed = await this.#fetchBytes(url);
    if (compressed.byteLength !== assetReference.size) {
      throw new CatalogV2Error(`asset size mismatch for ${label}`);
    }
    await verifySha256(compressed, assetReference.sha256, label);
    return gunzip(compressed);
  }

  async #fetchJson(url) {
    return parseJsonObject(await this.#fetchBytes(url), url.pathname);
  }

  async #fetchBytes(url) {
    const response = await this.fetchImpl(url);
    if (!response.ok) throw new CatalogV2Error(`request failed (${response.status}): ${url}`);
    return new Uint8Array(await response.arrayBuffer());
  }
}

function normalizeGame(game) {
  const value = String(game).trim().toLowerCase();
  return GAME_ALIASES[value] ?? value;
}

function discoverCatalog(feed, game, source, { family = DEFAULT_FAMILY, profile = null } = {}) {
  const matches = [];
  for (const [familyKey, familyEntry] of Object.entries(feed.families)) {
    if (family !== null && familyKey !== family) continue;
    validateFamily(familyKey, familyEntry);
    for (const [localKey, catalog] of Object.entries(familyEntry.catalogs)) {
      if (catalog.descriptor.game !== game || catalog.descriptor.source !== source) continue;
      matches.push({ familyKey, localKey, family: familyEntry, catalog });
    }
  }
  const description = `game ${JSON.stringify(game)}, source ${JSON.stringify(source)}, family ${JSON.stringify(family)}${profile !== null ? `, profile ${JSON.stringify(profile)}` : ""}`;
  const pool =
    profile !== null
      ? matches.filter((match) => match.catalog.descriptor.profile === profile)
      : (() => {
          const recommended = matches.filter((match) => match.catalog.descriptor.recommended === true);
          return recommended.length > 0 ? recommended : matches;
        })();
  if (pool.length === 0) {
    throw new CatalogV2Error(`no Catalog v2 feed entry matches ${description}`);
  }
  if (pool.length > 1) {
    throw new CatalogV2Error(`multiple Catalog v2 feed entries match ${description}`);
  }
  return finalizeResolution(pool[0]);
}

function resolveCatalogByKey(feed, fullCatalogKey) {
  const separator = fullCatalogKey.indexOf("/");
  if (separator <= 0 || separator === fullCatalogKey.length - 1) {
    throw new CatalogV2Error(`invalid Catalog v2 catalog key ${JSON.stringify(fullCatalogKey)}`);
  }
  const familyKey = fullCatalogKey.slice(0, separator);
  const localKey = fullCatalogKey.slice(separator + 1);
  const family = feed.families?.[familyKey];
  if (!isObject(family)) {
    throw new CatalogV2Error(`unknown Catalog v2 family ${JSON.stringify(familyKey)}`);
  }
  validateFamily(familyKey, family);
  const catalog = family.catalogs?.[localKey];
  if (!isObject(catalog)) {
    throw new CatalogV2Error(`unknown Catalog v2 catalog ${JSON.stringify(fullCatalogKey)}`);
  }
  return finalizeResolution({ familyKey, localKey, family, catalog });
}

function finalizeResolution({ familyKey, localKey, family, catalog }) {
  validateCatalogEntry(catalog, `${familyKey}/${localKey}`);
  const versions = buildVersionChain(catalog);
  return {
    familyKey,
    localKey,
    fullCatalogKey: `${familyKey}/${localKey}`,
    family,
    catalog,
    versions,
  };
}

function buildVersionChain(catalog) {
  if (catalog.base.version > catalog.current_version) {
    throw new CatalogV2Error("catalog current_version is not reachable from its base version");
  }
  const updateEntries = Object.entries(catalog.updates ?? {})
    .map(([key, update]) => {
      if (!/^(0|[1-9]\d*)$/.test(key) || !isObject(update) || update.to_version !== Number(key)) {
        throw new CatalogV2Error(`catalog has an invalid update entry for version ${JSON.stringify(key)}`);
      }
      if (
        !Number.isInteger(update.from_version) ||
        update.from_version < 0 ||
        update.to_version !== update.from_version + 1
      ) {
        throw new CatalogV2Error(`catalog update to version ${key} is not an exact-predecessor delta`);
      }
      validateUpdateEntry(update, update.to_version);
      return update;
    })
    .sort((left, right) => left.to_version - right.to_version);
  if (updateEntries.length === 0) {
    if (catalog.base.version !== catalog.current_version) {
      throw new CatalogV2Error("catalog is missing updates after its base version");
    }
    return [catalog.base.version];
  }
  for (let index = 1; index < updateEntries.length; index += 1) {
    if (updateEntries[index].from_version !== updateEntries[index - 1].to_version) {
      throw new CatalogV2Error("catalog retained update route is not contiguous");
    }
  }
  const routeStart = updateEntries[0].from_version;
  if (routeStart > catalog.base.version || updateEntries.at(-1).to_version !== catalog.current_version) {
    throw new CatalogV2Error("catalog retained update route does not connect its base to current_version");
  }
  const versions = [];
  for (let version = routeStart; version <= catalog.current_version; version += 1) {
    versions.push(version);
  }
  let runningRows = catalog.base.rows;
  for (let toVersion = catalog.base.version + 1; toVersion <= catalog.current_version; toVersion += 1) {
    const update = catalog.updates[String(toVersion)];
    runningRows += update.rows.added - update.rows.deleted;
  }
  if (runningRows !== catalog.rows) {
    throw new CatalogV2Error(
      "catalog's declared row count does not match its base and update row arithmetic",
    );
  }
  return versions;
}

function isCompatibleSnapshot(snapshot, resolved, includeMetadata) {
  return (
    snapshot instanceof BrowserCatalogV2 &&
    snapshot.catalogKey === resolved.fullCatalogKey &&
    snapshot.familyKey === resolved.familyKey &&
    snapshot.metadataLoaded === includeMetadata &&
    snapshot.dimension === resolved.family.embedding.dimensions &&
    snapshot.embeddings.length === snapshot.records.length * snapshot.dimension &&
    deepEqual(snapshot.embedding, resolved.family.embedding) &&
    deepEqual(snapshot.descriptor, resolved.catalog.descriptor)
  );
}

function parseIdentityRecord(value, descriptor, label) {
  if (!isObject(value)) throw new CatalogV2Error(`${label} must be a JSON object`);
  const id = requiredString(value.id, `${label} id`);
  const name = requiredString(value.name, `${label} name`);
  if (!isObject(value.identifiers ?? {})) {
    throw new CatalogV2Error(`${label} identifiers must be an object`);
  }
  const identifiers = {};
  for (const [name, identifier] of Object.entries(value.identifiers ?? {})) {
    if (name === descriptor.result_identifier) {
      throw new CatalogV2Error(
        `${label} must not duplicate the primary id under identifiers.${name}`,
      );
    }
    identifiers[requiredString(name, `${label} identifier name`)] = requiredString(
      identifier,
      `${label} identifier value`,
    );
  }
  const faceIndex = parseFaceIndex(value.face_index, `${label} for id ${JSON.stringify(id)}`);
  let finishes;
  if (value.finishes !== undefined) {
    if (!Array.isArray(value.finishes) || value.finishes.length === 0) {
      throw new CatalogV2Error(`${label} for id ${JSON.stringify(id)} has invalid finishes`);
    }
    finishes = value.finishes.map((finish, index) =>
      requiredString(finish, `${label} finishes[${index}]`),
    );
    for (let index = 1; index < finishes.length; index += 1) {
      if (finishes[index] <= finishes[index - 1]) {
        throw new CatalogV2Error(
          `${label} for id ${JSON.stringify(id)} has unsorted or duplicate finishes`,
        );
      }
    }
  }
  return { id, name, faceIndex, identifiers, finishes };
}

// A combined base row is a core identity record plus a *required* metadata
// field (an object, or null when the row has no metadata).
function parseBaseRecord(value, descriptor, label) {
  if (!isObject(value)) throw new CatalogV2Error(`${label} must be a JSON object`);
  if (!Object.hasOwn(value, "metadata")) {
    throw new CatalogV2Error(
      `${label} for id ${JSON.stringify(value.id)} is missing its required metadata field`,
    );
  }
  const metadata = value.metadata;
  if (metadata !== null && !isObject(metadata)) {
    throw new CatalogV2Error(
      `${label} for id ${JSON.stringify(value.id)} metadata must be an object or null`,
    );
  }
  const { metadata: _metadata, ...core } = value;
  const record = parseIdentityRecord(core, descriptor, label);
  record.metadata = metadata;
  return record;
}

// Compares two identity records ignoring metadata, used to validate that a
// no-embedding-index upsert repeats its predecessor's unchanged core record.
function coreRecordsEqual(left, right) {
  return (
    left.id === right.id &&
    left.name === right.name &&
    left.faceIndex === right.faceIndex &&
    deepEqual(left.identifiers, right.identifiers) &&
    deepEqual(left.finishes ?? [], right.finishes ?? [])
  );
}

function parseIdentityTarget(value, label) {
  if (!isObject(value)) throw new CatalogV2Error(`${label} must be a JSON object`);
  const id = requiredString(value.id, `${label} id`);
  const faceIndex = parseFaceIndex(value.face_index, `${label} for id ${JSON.stringify(id)}`);
  return { id, faceIndex };
}

// `face_index` is omitted for front faces and defaults to 0; an explicit
// literal 0 in the wire data is a non-canonical, rejected representation.
function parseFaceIndex(rawFaceIndex, label) {
  if (rawFaceIndex === undefined) return 0;
  if (!Number.isInteger(rawFaceIndex) || rawFaceIndex <= 0) {
    throw new CatalogV2Error(`${label} has an invalid face_index`);
  }
  return rawFaceIndex;
}

function compareIdentity(left, right) {
  const leftKey = left.faceIndex === 0 ? left.id : `${left.id}:face:${left.faceIndex}`;
  const rightKey = right.faceIndex === 0 ? right.id : `${right.id}:face:${right.faceIndex}`;
  if (leftKey < rightKey) return -1;
  if (leftKey > rightKey) return 1;
  return 0;
}

function identityKey(id, faceIndex) {
  return JSON.stringify([id, faceIndex]);
}

function parseFloat16Matrix(bytes, rows, dimension) {
  if (bytes.byteLength !== rows * dimension * 2) {
    throw new CatalogV2Error("FP16 matrix size does not match the feed's declared shape");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = new Uint16Array(rows * dimension);
  for (let index = 0; index < values.length; index += 1) {
    values[index] = view.getUint16(index * 2, true);
  }
  return values;
}

function parseJsonLines(bytes, label) {
  const text = new TextDecoder().decode(bytes);
  if (!text) return [];
  return text
    .trimEnd()
    .split("\n")
    .map((line) => {
      const value = JSON.parse(line);
      if (!isObject(value)) throw new CatalogV2Error(`${label} must contain JSON objects`);
      return value;
    });
}

function parseJsonObject(bytes, label) {
  let value;
  try {
    value = JSON.parse(new TextDecoder().decode(bytes));
  } catch (error) {
    throw new CatalogV2Error(`invalid JSON in ${label}`, { cause: error });
  }
  if (!isObject(value)) throw new CatalogV2Error(`${label} must be a JSON object`);
  return value;
}

function validateFeed(feed) {
  if (!isObject(feed) || typeof feed.checked_at !== "string" || !isObject(feed.families)) {
    throw new CatalogV2Error("invalid Catalog v2 feed");
  }
}

function validateFamily(familyKey, family) {
  if (!isObject(family) || !isObject(family.embedding) || !isObject(family.catalogs)) {
    throw new CatalogV2Error(`invalid Catalog v2 family ${JSON.stringify(familyKey)}`);
  }
  const embedding = family.embedding;
  if (
    typeof embedding.model !== "string" ||
    embedding.model.length === 0 ||
    !Number.isInteger(embedding.dimensions) ||
    embedding.dimensions <= 0 ||
    embedding.dtype !== "float16" ||
    embedding.byte_order !== "little" ||
    embedding.layout !== "row-major"
  ) {
    throw new CatalogV2Error(`family ${JSON.stringify(familyKey)} has an unsupported embedding contract`);
  }
}

function validateCatalogEntry(catalog, fullCatalogKey) {
  if (
    !isObject(catalog) ||
    typeof catalog.public_name !== "string" ||
    catalog.public_name.length === 0 ||
    !isObject(catalog.descriptor) ||
    !Number.isInteger(catalog.current_version) ||
    catalog.current_version < 0 ||
    !Number.isInteger(catalog.rows) ||
    catalog.rows < 0 ||
    typeof catalog.source_updated_at !== "string" ||
    !isObject(catalog.base)
  ) {
    throw new CatalogV2Error(`invalid Catalog v2 catalog entry ${JSON.stringify(fullCatalogKey)}`);
  }
  validateDescriptor(catalog.descriptor, fullCatalogKey);
  validateBaseEntry(catalog.base, fullCatalogKey);
}

function validateDescriptor(descriptor, fullCatalogKey) {
  if (
    typeof descriptor.game !== "string" ||
    typeof descriptor.source !== "string" ||
    typeof descriptor.profile !== "string" ||
    typeof descriptor.description !== "string" ||
    typeof descriptor.result_identifier !== "string" ||
    descriptor.result_identifier.length === 0 ||
    typeof descriptor.recommended !== "boolean"
  ) {
    throw new CatalogV2Error(`catalog ${JSON.stringify(fullCatalogKey)} has an invalid descriptor`);
  }
}

function validateBaseEntry(base, fullCatalogKey) {
  if (
    !Number.isInteger(base.version) ||
    base.version < 0 ||
    !Number.isInteger(base.rows) ||
    base.rows < 0 ||
    typeof base.source_updated_at !== "string" ||
    !isObject(base.assets)
  ) {
    throw new CatalogV2Error(`catalog ${JSON.stringify(fullCatalogKey)} has an invalid base`);
  }
  const assetKeys = new Set(Object.keys(base.assets));
  if (assetKeys.size !== 2 || !assetKeys.has("records") || !assetKeys.has("embeddings")) {
    throw new CatalogV2Error(
      `catalog ${JSON.stringify(fullCatalogKey)} base assets must contain records and embeddings`,
    );
  }
  validateAssetReference(
    base.assets.records,
    `catalog ${JSON.stringify(fullCatalogKey)} base records asset`,
  );
  validateAssetReference(
    base.assets.embeddings,
    `catalog ${JSON.stringify(fullCatalogKey)} base embeddings asset`,
  );
}

function validateUpdateEntry(update, toVersion) {
  if (
    !isObject(update.rows) ||
    !Number.isInteger(update.rows.added) ||
    update.rows.added < 0 ||
    !Number.isInteger(update.rows.updated) ||
    update.rows.updated < 0 ||
    !Number.isInteger(update.rows.deleted) ||
    update.rows.deleted < 0 ||
    typeof update.source_updated_at !== "string" ||
    !Number.isInteger(update.recognition_rows) ||
    update.recognition_rows < 0 ||
    !Number.isInteger(update.metadata_rows) ||
    update.metadata_rows < 0
  ) {
    throw new CatalogV2Error(`update to version ${toVersion} has an invalid shape`);
  }
  const totalOps = update.rows.added + update.rows.updated + update.rows.deleted;
  if (update.recognition_rows > totalOps || update.metadata_rows > totalOps) {
    throw new CatalogV2Error(
      `update to version ${toVersion} recognition_rows and metadata_rows cannot exceed total operations`,
    );
  }
  validateUpdateAssets(update.assets, totalOps, `update ${toVersion}`);
}

// A combined update carries no assets at all when it has zero total
// operations; otherwise it must declare a `records` asset (and, only when
// any operation touches recognition, an `embeddings` asset).
function validateUpdateAssets(assets, totalOps, label) {
  if (!isObject(assets)) {
    throw new CatalogV2Error(`${label} assets must be an object`);
  }
  const allowed = new Set(["records", "embeddings"]);
  for (const key of Object.keys(assets)) {
    if (!allowed.has(key)) {
      throw new CatalogV2Error(`${label} has an unexpected asset ${JSON.stringify(key)}`);
    }
  }
  if (totalOps === 0) {
    if (Object.keys(assets).length > 0) {
      throw new CatalogV2Error(`${label} must not declare assets when it has no operations`);
    }
    return;
  }
  if (!("records" in assets)) {
    throw new CatalogV2Error(`${label} records asset is missing`);
  }
  validateAssetReference(assets.records, `${label} records asset`);
  if ("embeddings" in assets) {
    validateAssetReference(assets.embeddings, `${label} embeddings asset`);
  }
}

function validateAssetReference(reference, label) {
  if (!isObject(reference) || typeof reference.url !== "string") {
    throw new CatalogV2Error(`${label} is missing a valid url`);
  }
  const url = new URL(reference.url);
  if (url.protocol !== "https:") {
    throw new CatalogV2Error(`${label} must use https`);
  }
  if (!Number.isInteger(reference.size) || reference.size < 0) {
    throw new CatalogV2Error(`${label} has an invalid size`);
  }
  if (!isSha256(reference.sha256)) {
    throw new CatalogV2Error(`${label} has an invalid sha256`);
  }
}

async function verifySha256(bytes, expectedSha256, label) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  const actual = [...digest].map((value) => value.toString(16).padStart(2, "0")).join("");
  if (actual !== expectedSha256) throw new CatalogV2Error(`asset checksum mismatch: ${label}`);
}

async function gunzip(bytes) {
  if (typeof DecompressionStream !== "function") {
    throw new CatalogV2Error("this browser does not support gzip DecompressionStream");
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function isSha256(value) {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}

function requiredString(value, label) {
  if (typeof value !== "string" || value.length === 0) {
    throw new CatalogV2Error(`${label} must be a non-empty string`);
  }
  return value;
}

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function deepEqual(left, right) {
  if (left === right) return true;
  if (typeof left !== typeof right || left === null || right === null) return false;
  if (typeof left !== "object") return false;
  const leftKeys = Object.keys(left);
  const rightKeys = Object.keys(right);
  if (leftKeys.length !== rightKeys.length) return false;
  return leftKeys.every((key) => Object.hasOwn(right, key) && deepEqual(left[key], right[key]));
}

function snapshotKey(version, catalogKey, includeMetadata) {
  return `${version}\u0000${catalogKey}\u0000${includeMetadata ? "metadata" : "recognition"}`;
}

function requestResult(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function transactionComplete(transaction) {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () =>
      reject(transaction.error ?? new CatalogV2Error("Catalog v2 cache transaction aborted"));
  });
}

function float16ToNumber(value) {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >>> 10) & 0x1f;
  const fraction = value & 0x03ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function createFloat16LookupTable() {
  return Float32Array.from({ length: 65536 }, (_, value) => float16ToNumber(value));
}
