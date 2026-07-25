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
const key = "milo1/test/demo";

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
      game: "demo",
      source: "test",
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

const files = await fixture();
const client = new CatalogV2BrowserClient({
  releaseBaseUrl: "https://catalog.test/",
  fetchImpl: mockFetch(files),
});
const catalog = await client.load(tag, key, { includeMetadata: true });

assert(catalog instanceof BrowserCatalogV2);
assert.equal(catalog.records.length, 2);
assert.equal(catalog.recordForIndex(1).metadata.name, "Beta");
assert.deepEqual(catalog.search(new Float32Array([0, 1]), 1), [[1, "b"]]);
assert.equal(catalog.recordForIndex(1).face_index, 1);

const nextTag = "catalog-v2-beta.2-2026-07-25";
const operations = await asset(
  "demo.delta.jsonl.gz",
  jsonLines([
    { op: "delete", key: "card:a:face:0" },
    {
      op: "upsert",
      record: { key: "card:c:face:0", identifiers: { source_card: "c" } },
      embedding_index: 0,
    },
  ]),
);
const deltaMatrix = await asset("demo.delta.f16.gz", fp16([0x3c00, 0]));
const metadataDelta = await asset(
  "demo.metadata.delta.jsonl.gz",
  jsonLines([
    { op: "delete", key: "card:a:face:0" },
    { op: "upsert", key: "card:c:face:0", metadata: { name: "Gamma" } },
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

const updated = await client.load(nextTag, key, {
  includeMetadata: true,
  previous: catalog,
});
assert.deepEqual(updated.records.map((record) => record.key), [
  "card:b:face:1",
  "card:c:face:0",
]);
assert.deepEqual(updated.search(new Float32Array([1, 0]), 1), [[1, "c"]]);
assert.equal(updated.recordForIndex(1).metadata.name, "Gamma");

const updatedWithoutMetadata = await client.load(nextTag, key, {
  previous: catalog,
});
assert.equal("metadata" in updatedWithoutMetadata.recordForIndex(0), false);
assert.equal("metadata" in updatedWithoutMetadata.recordForIndex(1), false);

await assert.rejects(
  () => client.load("latest", key),
  /explicit immutable beta tag/,
);
await assert.rejects(
  () => new CatalogV2BrowserClient().load(tag, key),
  /CORS-enabled Catalog v2 mirror/,
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

console.log("Catalog v2 browser tests passed");
