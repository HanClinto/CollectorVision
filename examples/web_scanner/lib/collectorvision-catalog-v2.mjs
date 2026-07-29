const BETA_TAG = /^catalog-v2-beta\.[1-9][0-9]*-[0-9]{4}-[0-9]{2}-[0-9]{2}$/;
const INDEX_FILENAME = "catalog-index-v2.json";
const FEED_FILENAME = "catalog-feed-v2.json";
const DEFAULT_RELEASE_BASE_URL =
  "https://hanclinto.github.io/CollectorVisionCatalog/catalog-v2/";
const GAME_NAMES = Object.freeze({
  mtg: "magic-the-gathering",
  pokemon: "pokemon",
  yugioh: "yugioh",
  fab: "flesh-and-blood",
  lorcana: "lorcana",
  digimon: "digimon-card-game",
  onepiece: "one-piece",
  swu: "star-wars-unlimited",
});
const PRIMARY_SOURCES = Object.freeze({
  mtg: "scryfall",
  pokemon: "tcgplayer",
  yugioh: "tcgplayer",
  fab: "tcgplayer",
  lorcana: "tcgplayer",
  digimon: "tcgplayer",
  onepiece: "tcgplayer",
  swu: "tcgplayer",
});

export class CatalogV2Error extends Error {}

export class BrowserCatalogV2 {
  static async forGame(game, options = {}) {
    const {
      fetchImpl = globalThis.fetch,
      releaseBaseUrl = DEFAULT_RELEASE_BASE_URL,
      cache = null,
      ...selection
    } = options;
    const client = new CatalogV2BrowserClient({ fetchImpl, releaseBaseUrl, cache });
    return client.loadGame(game, selection);
  }

  constructor({ manifest, records, embeddings, metadataLoaded = false }) {
    this.manifest = Object.freeze(structuredClone(manifest));
    this.catalogKey = manifest.catalog_key;
    this.version = manifest.version;
    this.embeddingModel = manifest.embedding_model;
    this.descriptor = Object.freeze({ ...manifest.descriptor });
    this.dimension = manifest.dim;
    this.records = Object.freeze(records);
    this.embeddings = embeddings;
    this.metadataLoaded = metadataLoaded;
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
    const result = {
      key: record.key,
      identifiers: { ...record.identifiers },
      face_index: record.face_index,
      result_identifier: this.descriptor.result_identifier,
      card_id: record.identifiers[this.descriptor.result_identifier],
    };
    if (record.metadata !== undefined) result.metadata = structuredClone(record.metadata);
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

  async get(tag, catalogKey, includeMetadata) {
    const database = await this.#database();
    const snapshot = await requestResult(
      database
        .transaction("catalogs", "readonly")
        .objectStore("catalogs")
        .get(snapshotKey(tag, catalogKey, includeMetadata)),
    );
    if (!snapshot) return null;
    if (
      !isObject(snapshot.manifest) ||
      !Array.isArray(snapshot.records) ||
      !(snapshot.embeddings instanceof ArrayBuffer)
    ) {
      throw new CatalogV2Error("invalid Catalog v2 snapshot in IndexedDB");
    }
    return new BrowserCatalogV2({
      manifest: snapshot.manifest,
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
    const snapshot = {
      id: snapshotKey(catalog.version, catalog.catalogKey, catalog.metadataLoaded),
      manifest: structuredClone(catalog.manifest),
      records: structuredClone(catalog.records),
      embeddings: catalog.embeddings.slice().buffer,
    };
    const transaction = database.transaction("catalogs", "readwrite");
    transaction.objectStore("catalogs").put(snapshot);
    await transactionComplete(transaction);
  }

  async #database() {
    if (!this.databasePromise) {
      this.databasePromise = new Promise((resolve, reject) => {
        const request = this.indexedDb.open(this.databaseName, 1);
        request.onupgradeneeded = () => {
          const database = request.result;
          if (!database.objectStoreNames.contains("catalogs")) {
            database.createObjectStore("catalogs", { keyPath: "id" });
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

export class CatalogV2BrowserClient {
  constructor({
    fetchImpl = globalThis.fetch,
    releaseBaseUrl = DEFAULT_RELEASE_BASE_URL,
    cache = null,
  } = {}) {
    if (typeof fetchImpl !== "function") throw new TypeError("fetch implementation is required");
    this.fetchImpl = fetchImpl;
    this.releaseBaseUrl = releaseBaseUrl;
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
    {
      source = null,
      includeMetadata = false,
      tag = null,
      previous = null,
    } = {},
  ) {
    const normalizedGame = normalizeGame(game);
    const selectedSource = source ?? PRIMARY_SOURCES[normalizedGame];
    if (selectedSource === "scryfall" && normalizedGame !== "mtg") {
      throw new CatalogV2Error("Scryfall Catalog v2 is only available for MTG");
    }
    const catalogName = selectedSource === "scryfall" ? "mtg" : GAME_NAMES[normalizedGame];
    const catalogKey = `milo1/${selectedSource}/${catalogName}`;
    return tag === null
      ? this.loadFromFeed(catalogKey, { includeMetadata, previous })
      : this.load(tag, catalogKey, { includeMetadata, previous });
  }

  async loadFromFeed(catalogKey, { includeMetadata = false, previous = null } = {}) {
    const feed = await this.#fetchJson(
      new URL(FEED_FILENAME, ensureTrailingSlash(this.releaseBaseUrl)),
    );
    const entry = validateFeedEntry(feed, catalogKey);
    let catalog = await this.#loadFeedStage(entry.base, catalogKey, includeMetadata, previous);
    for (const delta of entry.deltas) {
      if (catalog.version !== delta.from) {
        throw new CatalogV2Error("catalog feed delta chain does not match loaded base");
      }
      catalog = await this.#loadFeedStage(delta, catalogKey, includeMetadata, catalog);
    }
    return catalog;
  }

  async load(tag, catalogKey, { includeMetadata = false, previous = null } = {}) {
    validateTag(tag);
    if (!this.releaseBaseUrl) {
      throw new TypeError(
        "releaseBaseUrl must identify a same-origin or CORS-enabled Catalog v2 mirror",
      );
    }
    const baseUrl = new URL(
      `${encodeURIComponent(tag)}/`,
      ensureTrailingSlash(this.releaseBaseUrl),
    );
    const index = await this.#fetchJson(new URL(INDEX_FILENAME, baseUrl));
    validateIndex(index, tag);
    const entry = index.catalogs[catalogKey];
    if (
      !isObject(entry) ||
      !isSafeFilename(entry.manifest_filename) ||
      !isSha256(entry.sha256)
    ) {
      throw new CatalogV2Error(`catalog ${JSON.stringify(catalogKey)} is not valid in ${tag}`);
    }
    const manifestBytes = await this.#fetchBytes(new URL(entry.manifest_filename, baseUrl));
    await verifyBytes(entry.manifest_filename, manifestBytes, entry.sha256);
    const manifest = parseJsonObject(manifestBytes, "catalog manifest");
    validateManifest(manifest, tag, catalogKey);
    if (isCompatibleSnapshot(previous, manifest, includeMetadata)) {
      return previous;
    }

    const cached = await this.#cachedSnapshot(tag, catalogKey, includeMetadata);
    if (cached !== null && cached !== undefined) {
      if (isCompatibleSnapshot(cached, manifest, includeMetadata)) {
        return cached;
      }
      console.warn("Ignoring incompatible Catalog v2 snapshot in persistent cache");
    }
    if (previous === null && manifest.delta?.requires_exact_base === true) {
      previous = await this.#cachedSnapshot(
        manifest.delta.base_version,
        catalogKey,
        includeMetadata,
      );
    }
    let catalog;
    if (isExactCompatibleBase(previous, manifest, includeMetadata)) {
      catalog = await this.#loadDelta(
        baseUrl,
        manifest,
        previous,
        includeMetadata,
      );
    } else {
      catalog = await this.#loadFull(baseUrl, manifest, includeMetadata);
    }
    await this.#persistSnapshot(catalog);
    return catalog;
  }

  async #loadFeedStage(reference, catalogKey, includeMetadata, previous) {
    const tag = reference.version ?? reference.to;
    const manifestReference = reference.manifest;
    const manifestBytes = await this.#fetchBytes(new URL(manifestReference.url));
    await verifyBytes(
      new URL(manifestReference.url).pathname.split("/").at(-1),
      manifestBytes,
      manifestReference.sha256,
      manifestReference.size,
    );
    const manifest = parseJsonObject(manifestBytes, "catalog manifest");
    validateManifest(manifest, tag, catalogKey);
    validateFeedAssets(reference, manifest);
    if (isCompatibleSnapshot(previous, manifest, includeMetadata)) {
      return previous;
    }
    const cached = await this.#cachedSnapshot(tag, catalogKey, includeMetadata);
    if (cached !== null && cached !== undefined && isCompatibleSnapshot(cached, manifest, includeMetadata)) {
      return cached;
    }
    const baseUrl = new URL(".", manifestReference.url);
    let catalog;
    if (reference.to !== undefined) {
      if (!isExactCompatibleBase(previous, manifest, includeMetadata)) {
        throw new CatalogV2Error("catalog feed delta is missing its exact compatible base");
      }
      catalog = await this.#loadDelta(
        baseUrl,
        manifest,
        previous,
        includeMetadata,
        reference.assets,
      );
    } else {
      catalog = await this.#loadFull(baseUrl, manifest, includeMetadata, reference.assets);
    }
    await this.#persistSnapshot(catalog);
    return catalog;
  }

  async #cachedSnapshot(tag, catalogKey, includeMetadata) {
    if (this.cache === null) return null;
    try {
      return await this.cache.get(tag, catalogKey, includeMetadata);
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

  async #loadFull(baseUrl, manifest, includeMetadata, feedAssets = null) {
    const records = parseRecognitionRows(
      await this.#fetchGzipAsset(
        baseUrl,
        manifest.assets.recognition_rows,
        feedAssets?.recognition_rows,
      ),
      manifest,
    );
    const embeddings = parseFloat16Matrix(
      await this.#fetchGzipAsset(
        baseUrl,
        manifest.assets.recognition_matrix,
        feedAssets?.recognition_matrix,
      ),
      manifest.rows,
      manifest.dim,
    );
    let metadataLoaded = false;
    if (includeMetadata) {
      attachMetadata(
        records,
        parseJsonLines(
          await this.#fetchGzipAsset(
            baseUrl,
            manifest.assets.metadata_rows,
            feedAssets?.metadata_rows,
          ),
          "metadata rows",
        ),
      );
      metadataLoaded = true;
    }
    return new BrowserCatalogV2({ manifest, records, embeddings, metadataLoaded });
  }

  async #loadDelta(baseUrl, manifest, previous, includeMetadata, feedAssets = null) {
    const operations =
      manifest.delta.operations === 0
        ? []
        : parseJsonLines(
            await this.#fetchGzipAsset(
              baseUrl,
              manifest.assets.delta_operations,
              feedAssets?.delta_operations,
            ),
            "delta operations",
          );
    if (operations.length !== manifest.delta.operations) {
      throw new CatalogV2Error("delta operation count does not match manifest");
    }
    const upserts = operations.filter((operation) => operation.op === "upsert");
    const deltaEmbeddings =
      upserts.length === 0
        ? new Float32Array()
        : parseFloat16Matrix(
            await this.#fetchGzipAsset(
              baseUrl,
              manifest.assets.delta_matrix,
              feedAssets?.delta_matrix,
            ),
            upserts.length,
            manifest.dim,
          );
    const records = new Map(previous.records.map((record) => [record.key, structuredClone(record)]));
    if (!includeMetadata) {
      for (const record of records.values()) delete record.metadata;
    }
    const embeddings = new Map();
    previous.records.forEach((record, row) => {
      embeddings.set(
        record.key,
        previous.embeddings.slice(
          row * previous.dimension,
          (row + 1) * previous.dimension,
        ),
      );
    });
    const usedIndexes = new Set();
    const operatedKeys = new Set();
    for (const operation of operations) {
      if (operation.op === "delete") {
        const key = requiredString(operation.key, "delta delete key");
        if (operatedKeys.has(key) || !records.delete(key)) {
          throw new CatalogV2Error(`invalid delta delete for ${JSON.stringify(key)}`);
        }
        embeddings.delete(key);
        operatedKeys.add(key);
      } else if (operation.op === "upsert") {
        const record = parseRecognitionRecord(operation.record, manifest);
        const embeddingIndex = operation.embedding_index;
        if (
          operatedKeys.has(record.key) ||
          !Number.isInteger(embeddingIndex) ||
          embeddingIndex < 0 ||
          embeddingIndex >= upserts.length ||
          usedIndexes.has(embeddingIndex)
        ) {
          throw new CatalogV2Error(`invalid delta upsert for ${JSON.stringify(record.key)}`);
        }
        record.metadata = records.get(record.key)?.metadata;
        records.set(record.key, record);
        embeddings.set(
          record.key,
          deltaEmbeddings.slice(
            embeddingIndex * manifest.dim,
            (embeddingIndex + 1) * manifest.dim,
          ),
        );
        operatedKeys.add(record.key);
        usedIndexes.add(embeddingIndex);
      } else {
        throw new CatalogV2Error(`unsupported delta operation ${JSON.stringify(operation.op)}`);
      }
    }
    if (usedIndexes.size !== upserts.length) {
      throw new CatalogV2Error("delta embedding indexes must be contiguous and unique");
    }

    if (includeMetadata) {
      const metadataOperations =
        manifest.delta.metadata_operations === 0
          ? []
          : parseJsonLines(
              await this.#fetchGzipAsset(
                baseUrl,
                manifest.assets.metadata_delta,
                feedAssets?.metadata_delta,
              ),
              "metadata delta operations",
            );
      if (metadataOperations.length !== manifest.delta.metadata_operations) {
        throw new CatalogV2Error("metadata delta operation count does not match manifest");
      }
      for (const operation of metadataOperations) {
        const key = requiredString(operation.key, "metadata delta key");
        const record = records.get(key);
        if (operation.op === "delete") {
          if (record) delete record.metadata;
        } else if (operation.op === "upsert" && isObject(operation.metadata) && record) {
          record.metadata = structuredClone(operation.metadata);
        } else {
          throw new CatalogV2Error(`invalid metadata delta for ${JSON.stringify(key)}`);
        }
      }
    }

    const sortedRecords = [...records.values()].sort((left, right) =>
      compareStableKeys(left.key, right.key),
    );
    if (sortedRecords.length !== manifest.rows) {
      throw new CatalogV2Error("delta reconstructed an unexpected row count");
    }
    const matrix = new Uint16Array(manifest.rows * manifest.dim);
    sortedRecords.forEach((record, row) => {
      matrix.set(embeddings.get(record.key), row * manifest.dim);
    });
    return new BrowserCatalogV2({
      manifest,
      records: sortedRecords,
      embeddings: matrix,
      metadataLoaded: includeMetadata,
    });
  }

  async #fetchGzipAsset(baseUrl, asset, feedReference = null) {
    validateAsset(asset);
    if (
      feedReference !== null &&
      (feedReference.sha256 !== asset.sha256 || feedReference.size !== asset.size)
    ) {
      throw new Error(`Feed reference does not match manifest asset ${asset.filename}`);
    }
    const url =
      feedReference === null ? new URL(asset.filename, baseUrl) : new URL(feedReference.url);
    const compressed = await this.#fetchBytes(url);
    await verifyBytes(asset.filename, compressed, asset.sha256, asset.size);
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

function validateTag(tag) {
  if (!BETA_TAG.test(tag)) throw new TypeError("Catalog v2 requires an explicit immutable beta tag");
  const dateText = tag.slice(-10);
  const [year, month, day] = dateText.split("-").map(Number);
  const parsed = new Date(Date.UTC(year, month - 1, day));
  if (
    parsed.getUTCFullYear() !== year ||
    parsed.getUTCMonth() !== month - 1 ||
    parsed.getUTCDate() !== day
  ) {
    throw new TypeError("Catalog v2 beta tag contains an invalid date");
  }
}

function validateIndex(index, tag) {
  if (index.schema_version !== 2 || index.release_version !== tag || !isObject(index.catalogs)) {
    throw new CatalogV2Error("invalid Catalog v2 index");
  }
}

function validateFeedEntry(feed, catalogKey) {
  if (
    !isObject(feed) ||
    feed.schema_version !== 2 ||
    typeof feed.release_version !== "string" ||
    typeof feed.checked_at !== "string" ||
    typeof feed.source_updated_at !== "string" ||
    !isObject(feed.catalogs)
  ) {
    throw new CatalogV2Error("invalid Catalog v2 feed");
  }
  const entry = feed.catalogs[catalogKey];
  if (
    !isObject(entry) ||
    typeof entry.source_updated_at !== "string" ||
    !isObject(entry.base) ||
    !Array.isArray(entry.deltas)
  ) {
    throw new CatalogV2Error(`catalog ${JSON.stringify(catalogKey)} is not valid in the feed`);
  }
  validateFeedStage(entry.base, entry.base.version, false);
  let expected = entry.base.version;
  for (const delta of entry.deltas) {
    if (!isObject(delta) || delta.from !== expected) {
      throw new CatalogV2Error("catalog feed delta chain is not contiguous");
    }
    validateFeedStage(delta, delta.to, true);
    expected = delta.to;
  }
  return entry;
}

function validateFeedStage(reference, version, isDelta) {
  validateTag(version);
  validateFileReference(reference.manifest, version);
  if (!isObject(reference.assets)) {
    throw new CatalogV2Error("catalog feed stage assets must be an object");
  }
  if (isDelta && Object.keys(reference.assets).length === 0) {
    throw new CatalogV2Error("catalog feed delta must contain assets");
  }
  if (
    !isDelta &&
    (!("recognition_rows" in reference.assets) ||
      !("recognition_matrix" in reference.assets))
  ) {
    throw new CatalogV2Error("catalog feed base lacks recognition assets");
  }
  for (const asset of Object.values(reference.assets)) {
    validateFileReference(asset, version);
  }
}

function validateFileReference(reference, version) {
  if (!isObject(reference) || typeof reference.url !== "string") {
    throw new CatalogV2Error("catalog feed contains an invalid file reference");
  }
  const url = new URL(reference.url);
  const parts = url.pathname.split("/").filter(Boolean).map(decodeURIComponent);
  if (
    url.protocol !== "https:" ||
    parts.length < 2 ||
    parts.at(-2) !== version ||
    !isSafeFilename(parts.at(-1)) ||
    !isSha256(reference.sha256) ||
    !Number.isInteger(reference.size) ||
    reference.size < 0
  ) {
    throw new CatalogV2Error("catalog feed contains an invalid file reference");
  }
}

function validateFeedAssets(reference, manifest) {
  const names =
    reference.to === undefined
      ? ["recognition_rows", "recognition_matrix", "metadata_rows"]
      : ["delta_operations", "delta_matrix", "metadata_delta"];
  const expected = names.filter((name) => name in manifest.assets).sort();
  const actual = Object.keys(reference.assets).sort();
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new CatalogV2Error("catalog feed assets do not match the manifest");
  }
  for (const name of expected) {
    const feedAsset = reference.assets[name];
    const manifestAsset = manifest.assets[name];
    const filename = decodeURIComponent(new URL(feedAsset.url).pathname.split("/").at(-1));
    if (
      filename !== manifestAsset.filename ||
      feedAsset.sha256 !== manifestAsset.sha256 ||
      feedAsset.size !== manifestAsset.size
    ) {
      throw new CatalogV2Error(`catalog feed asset ${JSON.stringify(name)} is inconsistent`);
    }
  }
}

function normalizeGame(game) {
  const value = String(game).trim().toLowerCase();
  if (!(value in GAME_NAMES)) {
    throw new RangeError(
      `unknown game ${JSON.stringify(game)}; expected one of ${Object.keys(GAME_NAMES).join(", ")}`,
    );
  }
  return value;
}

function validateManifest(manifest, tag, catalogKey) {
  if (
    manifest.schema_version !== 2 ||
    manifest.version !== tag ||
    manifest.catalog_key !== catalogKey ||
    manifest.dtype !== "float16" ||
    !Number.isInteger(manifest.rows) ||
    manifest.rows < 0 ||
    !Number.isInteger(manifest.dim) ||
    manifest.dim <= 0 ||
    !isObject(manifest.assets) ||
    !isObject(manifest.descriptor) ||
    !requiredString(manifest.descriptor.result_identifier, "result identifier")
  ) {
    throw new CatalogV2Error("invalid Catalog v2 manifest");
  }
}

function parseRecognitionRows(bytes, manifest) {
  const values = parseJsonLines(bytes, "recognition rows");
  if (values.length !== manifest.rows) throw new CatalogV2Error("recognition row count mismatch");
  const keys = new Set();
  return values.map((value) => {
    const record = parseRecognitionRecord(value, manifest);
    if (keys.has(record.key)) throw new CatalogV2Error(`duplicate recognition key ${record.key}`);
    keys.add(record.key);
    return record;
  });
}

function parseRecognitionRecord(value, manifest) {
  if (!isObject(value) || !isObject(value.identifiers)) {
    throw new CatalogV2Error("invalid recognition record");
  }
  const key = requiredString(value.key, "recognition key");
  const identifiers = {};
  for (const [name, identifier] of Object.entries(value.identifiers)) {
    identifiers[requiredString(name, "identifier name")] = requiredString(
      identifier,
      "identifier value",
    );
  }
  if (!identifiers[manifest.descriptor.result_identifier]) {
    throw new CatalogV2Error(`recognition record ${JSON.stringify(key)} lacks result identifier`);
  }
  const faceIndex = value.face_index ?? 0;
  if (!Number.isInteger(faceIndex) || faceIndex < 0) {
    throw new CatalogV2Error(`recognition record ${JSON.stringify(key)} has invalid face_index`);
  }
  return { key, identifiers, face_index: faceIndex };
}

function attachMetadata(records, metadataRows) {
  const byKey = new Map(records.map((record) => [record.key, record]));
  const seen = new Set();
  for (const value of metadataRows) {
    const key = requiredString(value.key, "metadata key");
    if (seen.has(key) || !byKey.has(key) || !isObject(value.metadata)) {
      throw new CatalogV2Error(`invalid metadata record for ${JSON.stringify(key)}`);
    }
    seen.add(key);
    byKey.get(key).metadata = structuredClone(value.metadata);
  }
}

function parseFloat16Matrix(bytes, rows, dimension) {
  if (bytes.byteLength !== rows * dimension * 2) {
    throw new CatalogV2Error("FP16 matrix size does not match manifest");
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

function isExactCompatibleBase(previous, manifest, includeMetadata) {
  return (
    previous instanceof BrowserCatalogV2 &&
    manifest.delta?.requires_exact_base === true &&
    manifest.delta.base_version === previous.version &&
    previous.catalogKey === manifest.catalog_key &&
    previous.embeddingModel === manifest.embedding_model &&
    previous.dimension === manifest.dim &&
    JSON.stringify(previous.descriptor) === JSON.stringify(manifest.descriptor) &&
    (!includeMetadata || previous.metadataLoaded)
  );
}

function isCompatibleSnapshot(snapshot, manifest, includeMetadata) {
  return (
    snapshot instanceof BrowserCatalogV2 &&
    snapshot.version === manifest.version &&
    snapshot.catalogKey === manifest.catalog_key &&
    snapshot.embeddingModel === manifest.embedding_model &&
    snapshot.dimension === manifest.dim &&
    snapshot.records.length === manifest.rows &&
    snapshot.embeddings.length === manifest.rows * manifest.dim &&
    JSON.stringify(snapshot.descriptor) === JSON.stringify(manifest.descriptor) &&
    snapshot.metadataLoaded === includeMetadata
  );
}

async function verifyBytes(filename, bytes, expectedSha256, expectedSize = undefined) {
  if (expectedSize !== undefined && bytes.byteLength !== expectedSize) {
    throw new CatalogV2Error(`asset size mismatch: ${filename}`);
  }
  if (!isSha256(expectedSha256)) {
    throw new CatalogV2Error(`invalid asset checksum: ${filename}`);
  }
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  const actual = [...digest].map((value) => value.toString(16).padStart(2, "0")).join("");
  if (actual !== expectedSha256) throw new CatalogV2Error(`asset checksum mismatch: ${filename}`);
}

async function gunzip(bytes) {
  if (typeof DecompressionStream !== "function") {
    throw new CatalogV2Error("this browser does not support gzip DecompressionStream");
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function validateAsset(asset) {
  if (
    !isObject(asset) ||
    !isSafeFilename(asset.filename) ||
    !isSha256(asset.sha256) ||
    !Number.isInteger(asset.size) ||
    asset.size < 0
  ) {
    throw new CatalogV2Error("invalid Catalog v2 asset descriptor");
  }
}

function isSafeFilename(value) {
  return (
    typeof value === "string" &&
    /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(value) &&
    value !== "." &&
    value !== ".."
  );
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

function ensureTrailingSlash(value) {
  return value.endsWith("/") ? value : `${value}/`;
}

function snapshotKey(tag, catalogKey, includeMetadata) {
  return `${tag}\u0000${catalogKey}\u0000${includeMetadata ? "metadata" : "recognition"}`;
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

function compareStableKeys(left, right) {
  if (left < right) return -1;
  if (left > right) return 1;
  return 0;
}

function float16ToNumber(value) {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >>> 10) & 0x1f;
  const fraction = value & 0x03ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}
