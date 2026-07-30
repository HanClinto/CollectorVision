// Client for the live CollectorVisionCatalog v2 contract.
//
// The only document a normal client needs is the moving feed
// (`catalog-feed-v2.json`). It nests catalogs under immutable embedding
// families and advertises the newest usable base plus each subsequent
// exact-predecessor update through `current_version`. There are no release
// tags, manifests, or per-release indexes: every asset URL, size, and
// SHA-256 in the feed is authoritative on its own.

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
  yugioh: "yugioh",
  fab: "flesh-and-blood",
  lorcana: "lorcana",
  digimon: "digimon-card-game",
  onepiece: "one-piece",
  swu: "star-wars-unlimited",
  dbs: "dragon-ball-super",
});

const DEFAULT_SOURCE_BY_GAME = Object.freeze({
  "magic-the-gathering": "scryfall",
});
const FALLBACK_SOURCE = "tcgplayer";

export class CatalogV2Error extends Error {}

export class BrowserCatalogV2 {
  static async forGame(game, options = {}) {
    const {
      fetchImpl = globalThis.fetch,
      feedUrl = DEFAULT_FEED_URL,
      cache = null,
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
    const best = [];
    for (let row = 0; row < this.records.length; row += 1) {
      let score = 0;
      const offset = row * this.dimension;
      for (let column = 0; column < this.dimension; column += 1) {
        score += float16ToNumber(this.embeddings[offset + column]) * query[column];
      }
      const candidate = { score, index: row };
      let position = best.findIndex((item) => score > item.score);
      if (position < 0) position = best.length;
      best.splice(position, 0, candidate);
      if (best.length > topK) best.pop();
    }
    return best.map(({ score, index }) => this.recordForIndex(index, score));
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
      identifiers: { ...record.identifiers },
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

export class CatalogV2FeedClient {
  constructor({ fetchImpl = globalThis.fetch, feedUrl = DEFAULT_FEED_URL, cache = null } = {}) {
    if (typeof fetchImpl !== "function") throw new TypeError("fetch implementation is required");
    this.fetchImpl = fetchImpl;
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
    { source = null, family = DEFAULT_FAMILY, profile = null, includeMetadata = false, previous = null } = {},
  ) {
    const normalizedGame = normalizeGame(game);
    const selectedSource = source ?? DEFAULT_SOURCE_BY_GAME[normalizedGame] ?? FALLBACK_SOURCE;
    const feed = await this.#fetchFeed();
    const resolved = discoverCatalog(feed, normalizedGame, selectedSource, { family, profile });
    return this.#loadResolvedCatalog(resolved, { includeMetadata, previous });
  }

  async loadCatalog(fullCatalogKey, { includeMetadata = false, previous = null } = {}) {
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
      startIndex = 0;
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

    const identifierRows = parseJsonLines(
      await this.#fetchGzipAsset(base.recognition.assets.identifiers, "base identifiers"),
      "base identifiers",
    );
    if (identifierRows.length !== base.rows) {
      throw new CatalogV2Error("base identifier row count does not match the feed");
    }
    const embeddingBytes = await this.#fetchGzipAsset(
      base.recognition.assets.embeddings,
      "base embeddings",
    );
    const matrix = parseFloat16Matrix(embeddingBytes, base.rows, dimension);

    const seen = new Map();
    const staged = identifierRows.map((value, row) => {
      const record = parseIdentityRecord(value, catalog.descriptor, "base identifier row");
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
      return { record, embedding: matrix.slice(row * dimension, (row + 1) * dimension) };
    });

    if (includeMetadata) {
      const metadataRows = parseJsonLinesAllowNull(
        await this.#fetchGzipAsset(base.metadata.assets.records, "base metadata"),
        "base metadata",
      );
      if (metadataRows.length !== base.rows) {
        throw new CatalogV2Error("base metadata row count does not match the feed");
      }
      metadataRows.forEach((value, row) => {
        if (value === null) return;
        if (!isObject(value)) throw new CatalogV2Error("base metadata rows must be objects or null");
        staged[row].record.metadata = value;
      });
    }

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

    const recognitionRows = updateEntry.recognition.rows;
    const operations =
      recognitionRows === 0
        ? []
        : parseJsonLines(
            await this.#fetchGzipAsset(
              updateEntry.recognition.assets.identifiers,
              "recognition delta",
            ),
            "recognition delta",
          );
    if (operations.length !== recognitionRows) {
      throw new CatalogV2Error("recognition delta operation count does not match the feed");
    }
    const upserts = operations.filter((operation) => operation.op === "upsert");
    const deltaEmbeddings =
      upserts.length === 0
        ? new Uint16Array()
        : parseFloat16Matrix(
            await this.#fetchGzipAsset(
              updateEntry.recognition.assets.embeddings,
              "recognition delta embeddings",
            ),
            upserts.length,
            dimension,
          );

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

    let added = 0;
    let deleted = 0;
    const recognitionAddedIds = new Set();
    const recognitionDeletedIds = new Set();
    const recognitionUpdatedIds = new Set();
    const usedEmbeddingIndexes = new Set();
    for (const operation of operations) {
      if (operation.op === "delete") {
        const target = parseIdentityTarget(operation, "recognition delta delete");
        const faces = byId.get(target.id);
        if (!faces || !faces.has(target.faceIndex)) {
          throw new CatalogV2Error(
            `recognition delta deletes a row that is not present: ${JSON.stringify(target)}`,
          );
        }
        faces.delete(target.faceIndex);
        if (faces.size === 0) byId.delete(target.id);
        deleted += 1;
        recognitionDeletedIds.add(identityKey(target.id, target.faceIndex));
      } else if (operation.op === "upsert") {
        const record = parseIdentityRecord(
          operation.record,
          catalog.descriptor,
          "recognition delta upsert",
        );
        const embeddingIndex = operation.embedding_index;
        if (
          !Number.isInteger(embeddingIndex) ||
          embeddingIndex < 0 ||
          embeddingIndex >= upserts.length ||
          usedEmbeddingIndexes.has(embeddingIndex)
        ) {
          throw new CatalogV2Error(
            `recognition delta upsert has an invalid embedding_index for id ${JSON.stringify(record.id)}`,
          );
        }
        usedEmbeddingIndexes.add(embeddingIndex);
        let faces = byId.get(record.id);
        if (!faces) {
          faces = new Map();
          byId.set(record.id, faces);
        }
        const existing = faces.get(record.faceIndex);
        faces.set(record.faceIndex, {
          record: { ...record, metadata: existing?.record.metadata },
          embedding: deltaEmbeddings.slice(
            embeddingIndex * dimension,
            (embeddingIndex + 1) * dimension,
          ),
        });
        const key = identityKey(record.id, record.faceIndex);
        if (existing) {
          recognitionUpdatedIds.add(key);
        } else {
          added += 1;
          recognitionAddedIds.add(key);
        }
      } else {
        throw new CatalogV2Error(`unsupported recognition delta operation ${JSON.stringify(operation.op)}`);
      }
    }
    if (usedEmbeddingIndexes.size !== upserts.length) {
      throw new CatalogV2Error("recognition delta embedding indexes must be contiguous and unique");
    }
    if (added !== updateEntry.rows.added || deleted !== updateEntry.rows.deleted) {
      throw new CatalogV2Error("recognition delta row classification does not match the feed");
    }

    // `rows.updated` is a *global* classification: it also counts rows whose
    // metadata changed with no corresponding recognition operation at all.
    // Rows that were added or deleted this stage are never also "updated".
    const metadataTouchedIds = new Set();
    if (includeMetadata) {
      const metadataRows = updateEntry.metadata.rows;
      const metadataOperations =
        metadataRows === 0
          ? []
          : parseJsonLines(
              await this.#fetchGzipAsset(updateEntry.metadata.assets.records, "metadata delta"),
              "metadata delta",
            );
      if (metadataOperations.length !== metadataRows) {
        throw new CatalogV2Error("metadata delta operation count does not match the feed");
      }
      for (const operation of metadataOperations) {
        const target = parseIdentityTarget(operation, "metadata delta");
        const key = identityKey(target.id, target.faceIndex);
        const faces = byId.get(target.id);
        const entry = faces?.get(target.faceIndex);
        const touchesSurvivingRow = !recognitionAddedIds.has(key) && !recognitionDeletedIds.has(key);
        if (operation.op === "delete") {
          if (entry) {
            delete entry.record.metadata;
            if (touchesSurvivingRow) metadataTouchedIds.add(key);
          }
        } else if (operation.op === "upsert") {
          if (!entry) {
            throw new CatalogV2Error(
              `metadata delta upserts a row that is not present: ${JSON.stringify(target)}`,
            );
          }
          if (!isObject(operation.metadata)) {
            throw new CatalogV2Error("metadata delta upsert requires a metadata object");
          }
          entry.record.metadata = structuredClone(operation.metadata);
          if (touchesSurvivingRow) metadataTouchedIds.add(key);
        } else {
          throw new CatalogV2Error(`unsupported metadata delta operation ${JSON.stringify(operation.op)}`);
        }
      }
    } else {
      for (const faces of byId.values()) {
        for (const entry of faces.values()) delete entry.record.metadata;
      }
    }

    const updatedUnion = new Set([...recognitionUpdatedIds, ...metadataTouchedIds]);
    if (includeMetadata) {
      if (updatedUnion.size !== updateEntry.rows.updated) {
        throw new CatalogV2Error("recognition delta row classification does not match the feed");
      }
    } else if (recognitionUpdatedIds.size > updateEntry.rows.updated) {
      // Recognition-only mode never fetches metadata deltas, so a metadata-only
      // touched row is invisible here; only a loose subset bound is checkable.
      throw new CatalogV2Error("recognition delta row classification does not match the feed");
    }

    const flattened = [];
    for (const faces of byId.values()) {
      for (const entry of faces.values()) flattened.push(entry);
    }
    flattened.sort((left, right) => compareIdentity(left.record, right.record));

    const expectedTotal = snapshot.records.length + updateEntry.rows.added - updateEntry.rows.deleted;
    if (flattened.length !== expectedTotal) {
      throw new CatalogV2Error("recognition delta reconstructed an unexpected row total");
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
  const versions = [];
  for (let version = catalog.base.version; version <= catalog.current_version; version += 1) {
    versions.push(version);
  }
  if (versions.length === 0 || versions.at(-1) !== catalog.current_version) {
    throw new CatalogV2Error("catalog current_version is not reachable from its base version");
  }
  for (let index = 1; index < versions.length; index += 1) {
    const toVersion = versions[index];
    const update = catalog.updates?.[String(toVersion)];
    if (!isObject(update)) {
      throw new CatalogV2Error(`catalog is missing an update to version ${toVersion}`);
    }
    if (update.from_version !== versions[index - 1] || update.to_version !== toVersion) {
      throw new CatalogV2Error(`catalog update to version ${toVersion} is not an exact-predecessor delta`);
    }
    validateUpdateEntry(update, toVersion);
  }
  const expectedKeys = new Set(versions.slice(1).map(String));
  for (const key of Object.keys(catalog.updates ?? {})) {
    if (!expectedKeys.has(key)) {
      throw new CatalogV2Error(`catalog has an unexpected update entry for version ${JSON.stringify(key)}`);
    }
  }
  let runningRows = catalog.base.rows;
  for (let index = 1; index < versions.length; index += 1) {
    const update = catalog.updates[String(versions[index])];
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
  return { id, faceIndex, identifiers, finishes };
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
  if (left.id < right.id) return -1;
  if (left.id > right.id) return 1;
  return left.faceIndex - right.faceIndex;
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

function parseJsonLinesAllowNull(bytes, label) {
  const text = new TextDecoder().decode(bytes);
  if (!text) return [];
  return text
    .trimEnd()
    .split("\n")
    .map((line) => {
      const value = JSON.parse(line);
      if (value !== null && !isObject(value)) {
        throw new CatalogV2Error(`${label} rows must be JSON objects or null`);
      }
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
    !isObject(base.recognition) ||
    !isObject(base.metadata)
  ) {
    throw new CatalogV2Error(`catalog ${JSON.stringify(fullCatalogKey)} has an invalid base`);
  }
  validateLayerAssets(base.recognition, base.rows, {
    required: ["embeddings", "identifiers"],
    label: `catalog ${JSON.stringify(fullCatalogKey)} base recognition`,
  });
  validateLayerAssets(base.metadata, base.rows, {
    required: ["records"],
    label: `catalog ${JSON.stringify(fullCatalogKey)} base metadata`,
  });
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
    !isObject(update.recognition) ||
    !Number.isInteger(update.recognition.rows) ||
    update.recognition.rows < 0 ||
    !isObject(update.metadata) ||
    !Number.isInteger(update.metadata.rows) ||
    update.metadata.rows < 0
  ) {
    throw new CatalogV2Error(`update to version ${toVersion} has an invalid shape`);
  }
  validateLayerAssets(update.recognition, update.recognition.rows, {
    required: ["identifiers"],
    optional: ["embeddings"],
    label: `update ${toVersion} recognition`,
  });
  validateLayerAssets(update.metadata, update.metadata.rows, {
    required: ["records"],
    label: `update ${toVersion} metadata`,
  });
}

// A layer with zero rows carries no assets at all (e.g. a metadata-only
// update stage has `recognition.rows === 0` and an empty `recognition.assets`
// object); a layer with rows > 0 must declare every required asset and no
// asset names beyond `required`/`optional`.
function validateLayerAssets(layer, rows, { required = [], optional = [], label }) {
  if (!isObject(layer.assets)) {
    throw new CatalogV2Error(`${label} assets must be an object`);
  }
  const allowed = new Set([...required, ...optional]);
  for (const key of Object.keys(layer.assets)) {
    if (!allowed.has(key)) {
      throw new CatalogV2Error(`${label} has an unexpected asset ${JSON.stringify(key)}`);
    }
  }
  if (rows === 0) {
    if (Object.keys(layer.assets).length > 0) {
      throw new CatalogV2Error(`${label} must not declare assets when rows is 0`);
    }
    return;
  }
  for (const name of required) {
    validateAssetReference(layer.assets[name], `${label} ${name} asset`);
  }
  for (const name of optional) {
    if (name in layer.assets) validateAssetReference(layer.assets[name], `${label} ${name} asset`);
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
