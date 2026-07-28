const BETA_TAG = /^catalog-v2-beta\.[1-9][0-9]*-[0-9]{4}-[0-9]{2}-[0-9]{2}$/;
const INDEX_FILENAME = "catalog-index-v2.json";

export class CatalogV2Error extends Error {}

export class BrowserCatalogV2 {
  constructor({ manifest, records, embeddings, metadataLoaded = false }) {
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

export class CatalogV2BrowserClient {
  constructor({
    fetchImpl = globalThis.fetch,
    releaseBaseUrl = null,
  } = {}) {
    if (typeof fetchImpl !== "function") throw new TypeError("fetch implementation is required");
    this.fetchImpl = fetchImpl;
    this.releaseBaseUrl = releaseBaseUrl;
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

    if (isExactCompatibleBase(previous, manifest, includeMetadata)) {
      return this.#loadDelta(baseUrl, manifest, previous, includeMetadata);
    }
    return this.#loadFull(baseUrl, manifest, includeMetadata);
  }

  async #loadFull(baseUrl, manifest, includeMetadata) {
    const records = parseRecognitionRows(
      await this.#fetchGzipAsset(baseUrl, manifest.assets.recognition_rows),
      manifest,
    );
    const embeddings = parseFloat16Matrix(
      await this.#fetchGzipAsset(baseUrl, manifest.assets.recognition_matrix),
      manifest.rows,
      manifest.dim,
    );
    let metadataLoaded = false;
    if (includeMetadata) {
      attachMetadata(
        records,
        parseJsonLines(
          await this.#fetchGzipAsset(baseUrl, manifest.assets.metadata_rows),
          "metadata rows",
        ),
      );
      metadataLoaded = true;
    }
    return new BrowserCatalogV2({ manifest, records, embeddings, metadataLoaded });
  }

  async #loadDelta(baseUrl, manifest, previous, includeMetadata) {
    const operations = parseJsonLines(
      await this.#fetchGzipAsset(baseUrl, manifest.assets.delta_operations),
      "delta operations",
    );
    if (operations.length !== manifest.delta.operations) {
      throw new CatalogV2Error("delta operation count does not match manifest");
    }
    const upserts = operations.filter((operation) => operation.op === "upsert");
    const deltaEmbeddings = parseFloat16Matrix(
      await this.#fetchGzipAsset(baseUrl, manifest.assets.delta_matrix),
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
      const metadataOperations = parseJsonLines(
        await this.#fetchGzipAsset(baseUrl, manifest.assets.metadata_delta),
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

  async #fetchGzipAsset(baseUrl, asset) {
    validateAsset(asset);
    const compressed = await this.#fetchBytes(new URL(asset.filename, baseUrl));
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
