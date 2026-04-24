/**
 * tests/js/bisect_webgpu_versions.mjs
 *
 * Automatically tests a range of onnxruntime-web versions to find when the
 * WebGPU all-zeros output bug was introduced or fixed.
 *
 * For each version it:
 *   1. Downloads the tarball from the npm registry
 *   2. Extracts dist/ into a temp directory using the system `tar`
 *   3. Serves the model + ort-web files via a local HTTP server
 *   4. Runs the Cornelius model under WebGPU and WASM in headless Chrome
 *   5. Reports PASS / FAIL / SKIP per version
 *
 * Usage
 * -----
 *   node bisect_webgpu_versions.mjs
 *   USE_SYSTEM_CHROME=1 node bisect_webgpu_versions.mjs
 *
 * Requires: Node 18+, Playwright Chromium installed, system `tar` + `curl`
 */

import { chromium }                                   from 'playwright';
import { readFileSync, existsSync, mkdirSync,
         readdirSync, writeFileSync, rmSync }          from 'fs';
import { resolve, dirname }                           from 'path';
import { fileURLToPath }                              from 'url';
import { createServer }                               from 'http';
import { tmpdir }                                     from 'os';
import { execFileSync }                               from 'child_process';
import { get as httpsGet }                            from 'https';

const __dirname    = dirname(fileURLToPath(import.meta.url));
const ROOT         = resolve(__dirname, '../..');
const CORNELIUS    = resolve(ROOT, 'collector_vision/weights/cornelius.onnx');

const DETECTOR_SIZE = 384;
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD  = [0.229, 0.224, 0.225];
const TOLERANCE     = 0.02;
const PAGE_TIMEOUT  = 90_000;

// Which WebGPU entrypoint to test:
//   'all'    → ort.all.min.mjs  (legacy JSEP backend, default)
//   'webgpu' → ort.webgpu.min.mjs (new WebGPU backend, available 1.24+)
// Override with: TEST_EP=webgpu node bisect_webgpu_versions.mjs
const TEST_EP = process.env.TEST_EP ?? 'all';
const ORT_MJS = TEST_EP === 'webgpu' ? 'ort.webgpu.min.mjs' : 'ort.all.min.mjs';

// Versions to test, oldest → newest.
// WebGPU JSEP first appeared in ort-web ~1.19; older versions will be SKIPped.
// 1.25.0-dev.20260209: last dev build BEFORE PR #27249 (merged Feb 11)
// 1.25.0-dev.20260212: first dev build AFTER PR #27249 merge
const ALL_VERSIONS = [
  '1.17.3',
  '1.18.0',
  '1.19.0',
  '1.20.0',
  '1.20.1',
  '1.21.0',
  '1.22.0',
  '1.23.0',
  '1.23.2',
  '1.24.1',
  '1.24.3',
  '1.25.0-dev.20260209-a3749f1353',   // pre-fix (last before PR #27249)
  '1.25.0-dev.20260212-1a71a5f46e',   // post-fix (first after PR #27249)
  '1.25.0-dev.20260303-e7e64dc112',   // post-fix
  '1.26.0-dev.20260410-5e55544225',   // latest dev
];

// Optionally filter via VERSIONS_ONLY=1.24.3,1.25.0-dev.20260212-1a71a5f46e
const VERSIONS = process.env.VERSIONS_ONLY
  ? process.env.VERSIONS_ONLY.split(',').map(v => v.trim())
  : ALL_VERSIONS;

// ─── synthetic input ──────────────────────────────────────────────────────

function makeSyntheticFloat32() {
  const N   = DETECTOR_SIZE * DETECTOR_SIZE;
  const f32 = new Float32Array(3 * N);
  let s = 0xdeadbeef >>> 0;
  for (let i = 0; i < f32.length; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const c   = Math.floor(i / N);
    const pix = (s >>> 24) / 255;
    f32[i]    = (pix - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
  }
  return f32;
}

// ─── npm download ─────────────────────────────────────────────────────────

function downloadToFile(url, destPath) {
  return new Promise((resolve_p, reject) => {
    const follow = (u) => {
      httpsGet(u, res => {
        if (res.statusCode === 301 || res.statusCode === 302) {
          return follow(res.headers.location);
        }
        if (res.statusCode !== 200) {
          return reject(new Error(`HTTP ${res.statusCode} for ${u}`));
        }
        const chunks = [];
        res.on('data', c => chunks.push(c));
        res.on('end', () => {
          writeFileSync(destPath, Buffer.concat(chunks));
          resolve_p();
        });
        res.on('error', reject);
      }).on('error', reject);
    };
    follow(url);
  });
}

// ─── HTTP server ──────────────────────────────────────────────────────────

function mimeOf(url) {
  if (url.endsWith('.mjs') || url.endsWith('.js')) return 'application/javascript';
  if (url.endsWith('.wasm'))                        return 'application/wasm';
  if (url.endsWith('.html'))                        return 'text/html; charset=utf-8';
  return 'application/octet-stream';
}

function startServer(staticFiles, inputData) {
  return new Promise(res => {
    const srv = createServer((req, resp) => {
      const url = (req.url ?? '/').split('?')[0];
      const hdr = {
        'Cross-Origin-Opener-Policy':   'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
        'Cross-Origin-Resource-Policy': 'cross-origin',
        'Content-Type':                 mimeOf(url),
      };
      if (url === '/input.f32') {
        resp.writeHead(200, hdr);
        resp.end(Buffer.from(inputData.buffer, inputData.byteOffset, inputData.byteLength));
      } else if (staticFiles[url]) {
        resp.writeHead(200, hdr);
        resp.end(staticFiles[url]());
      } else {
        resp.writeHead(404); resp.end('not found: ' + url);
      }
    });
    srv.listen(0, '127.0.0.1', () => res({ srv, port: srv.address().port }));
  });
}

// ─── test page ────────────────────────────────────────────────────────────

function buildPage(version) {
  return /* html */`<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>bisect ${version}</title></head><body>
<script type="module">
const DETECTOR_SIZE = ${DETECTOR_SIZE};
const TOLERANCE     = ${TOLERANCE};

async function run() {
  let ort;
  try { ort = await import('/${ORT_MJS}'); }
  catch(e) {
    window.__result = { version: '${version}', error: 'import: ' + e.message };
    window.__done = true; return;
  }

  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths  = location.origin + '/';

  const modelBuf = await (await fetch('/model.onnx')).arrayBuffer();
  const f32      = new Float32Array(await (await fetch('/input.f32')).arrayBuffer());

  async function infer(ep) {
    const sess   = await ort.InferenceSession.create(modelBuf,
                           { executionProviders: [ep] });
    const tensor = new ort.Tensor('float32', f32,
                                  [1, 3, DETECTOR_SIZE, DETECTOR_SIZE]);
    const t0   = performance.now();
    const out  = await sess.run({ [sess.inputNames[0]]: tensor });
    return {
      corners:   Array.from(out[sess.outputNames[0]].data).slice(0, 8),
      sharpness: sess.outputNames[2] ? out[sess.outputNames[2]].data[0] : null,
      ms:        Math.round(performance.now() - t0),
    };
  }

  const r = { version: '${version}', gpuAvail: !!navigator.gpu };

  try { r.wasm = await infer('wasm'); } catch(e) { r.wasmError = e.message; }

  if (!navigator.gpu) {
    r.webgpuSkip = 'no navigator.gpu';
  } else {
    try { r.webgpu = await infer('webgpu'); } catch(e) { r.webgpuError = e.message; }
  }

  if (r.wasm?.corners && r.webgpu?.corners) {
    let d = 0;
    for (let i = 0; i < 8; i++)
      d = Math.max(d, Math.abs(r.wasm.corners[i] - r.webgpu.corners[i]));
    r.maxDiff = d;
    r.pass    = d <= TOLERANCE;
  }

  window.__result = r;
  window.__done   = true;
}
run().catch(e => {
  window.__result = { version: '${version}', fatalError: e.message };
  window.__done   = true;
});
</script></body></html>`;
}

// ─── main ─────────────────────────────────────────────────────────────────

async function main() {
  console.log('='.repeat(62));
  console.log('  ort-web WebGPU bisect — Cornelius model (synthetic input)');
  console.log(`  EP entrypoint: ${ORT_MJS}  (TEST_EP=${TEST_EP})`);
  console.log(`  Versions: ${VERSIONS.join('  ')}`);
  console.log(`  Tolerance: ${TOLERANCE}`);
  console.log('='.repeat(62) + '\n');

  if (!existsSync(CORNELIUS)) {
    console.error('ERROR: model not found:', CORNELIUS); process.exit(1);
  }

  const input = makeSyntheticFloat32();

  const useSystemChrome = process.env.USE_SYSTEM_CHROME === '1';
  const browser = await chromium.launch({
    headless: false,
    args: [
      '--no-sandbox', '--disable-gpu-sandbox',
      '--enable-gpu', '--enable-webgpu',
      '--enable-webgpu-developer-features',
      '--use-angle=metal',
    ],
    ...(useSystemChrome ? { channel: 'chrome' } : {}),
  });

  const results = [];

  for (const version of VERSIONS) {
    console.log(`\n${'─'.repeat(52)}`);
    console.log(`ort-web ${version}`);

    // ── download tarball ──────────────────────────────────────────────
    const tmpDir  = resolve(tmpdir(), `ort-bisect-${version}`);
    const tgzPath = resolve(tmpDir, 'pkg.tgz');
    const pkgDir  = resolve(tmpDir, 'pkg');

    try {
      mkdirSync(pkgDir, { recursive: true });
      const tgzUrl = `https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-${version}.tgz`;
      process.stdout.write('  Downloading... ');
      await downloadToFile(tgzUrl, tgzPath);
      process.stdout.write('done\n');

      execFileSync('tar', ['-xzf', tgzPath, '-C', pkgDir, '--strip-components=1'],
                   { stdio: 'pipe' });
    } catch(e) {
      console.log(`  ERROR: ${e.message}`);
      results.push({ version, error: e.message });
      try { rmSync(tmpDir, { recursive: true, force: true }); } catch {}
      continue;
    }

    const distDir = resolve(pkgDir, 'dist');
    if (!existsSync(distDir)) {
      console.log('  SKIP: no dist/ dir in package');
      results.push({ version, skip: 'no dist/' });
      try { rmSync(tmpDir, { recursive: true, force: true }); } catch {}
      continue;
    }

    const distFiles = readdirSync(distDir);

    // Only ESM builds have WebGPU; older versions shipped UMD ort.all.min.js
    const mainMjs = distFiles.find(f => f === ORT_MJS)
                 ?? distFiles.find(f => f === 'ort.all.min.mjs');
    if (!mainMjs) {
      const hasJs = distFiles.find(f => f === 'ort.all.min.js');
      console.log(`  SKIP: no ESM build (found: ${hasJs ?? 'nothing'}) — WebGPU not in this version`);
      results.push({ version, skip: 'no ESM / WebGPU EP' });
      try { rmSync(tmpDir, { recursive: true, force: true }); } catch {}
      continue;
    }
    if (mainMjs !== ORT_MJS) {
      console.log(`  NOTE: ${ORT_MJS} not found — falling back to ${mainMjs}`);
    }

    // Build static file map: serve every .mjs + .wasm from dist/
    const staticFiles = { '/model.onnx': () => readFileSync(CORNELIUS) };
    for (const f of distFiles) {
      if (f.endsWith('.mjs') || f.endsWith('.wasm') || f.endsWith('.js')) {
        const fp = resolve(distDir, f);
        staticFiles[`/${f}`] = () => readFileSync(fp);
      }
    }
    // canonical entry point name the page uses
    if (!staticFiles[`/${ORT_MJS}`]) {
      staticFiles[`/${ORT_MJS}`] = staticFiles[`/${mainMjs}`];
    }

    const hasJsep = distFiles.some(f => f.includes('jsep'));
    console.log(`  ESM: ${mainMjs}   JSEP wasm: ${hasJsep ? 'yes' : 'NO — WebGPU likely unsupported'}`);

    // ── run test page ─────────────────────────────────────────────────
    const { srv, port } = await startServer(staticFiles, input);
    const BASE          = `http://127.0.0.1:${port}`;

    const page = await browser.newPage();
    const browserErrs = [];
    page.on('console', m => { if (m.type() === 'error') browserErrs.push(m.text()); });
    page.on('pageerror', e => browserErrs.push(e.message));

    // Write the test page as a served route so page.goto() establishes
    // a real HTTP base URL (needed for dynamic import('/ort.all.min.mjs')).
    staticFiles['/test.html'] = () => Buffer.from(buildPage(version));

    let raw;
    try {
      await page.goto(`${BASE}/test.html`);
      await page.waitForFunction(() => window.__done === true,
                                 { timeout: PAGE_TIMEOUT });
      raw = await page.evaluate(() => window.__result);
    } catch(e) {
      raw = { version, error: `timeout/error: ${e.message}` };
      browserErrs.slice(-5).forEach(l => console.log('   [browser]', l));
    } finally {
      await page.close();
      srv.close();
    }

    // ── print result ──────────────────────────────────────────────────
    if (raw.fatalError || raw.error) {
      const msg = raw.fatalError ?? raw.error;
      console.log(`  ERROR: ${msg}`);
      if (browserErrs.length) browserErrs.slice(-3).forEach(l => console.log('   ', l));
      results.push({ version, error: msg });
    } else if (raw.webgpuSkip) {
      console.log(`  SKIP (WebGPU unavailable): ${raw.webgpuSkip}`);
      results.push({ version, skip: raw.webgpuSkip });
    } else if (raw.webgpuError) {
      console.log(`  WebGPU EP error: ${raw.webgpuError}`);
      results.push({ version, webgpuError: raw.webgpuError });
    } else if (raw.maxDiff !== undefined) {
      const v = raw.pass ? 'PASS ✓' : 'FAIL ✗';
      console.log(`  ${v}  maxDiff=${raw.maxDiff.toFixed(5)}  wasm_sharp=${raw.wasm?.sharpness?.toFixed(4)}  webgpu_sharp=${raw.webgpu?.sharpness?.toFixed(4)}  wasm=${raw.wasm?.ms}ms webgpu=${raw.webgpu?.ms}ms`);
      results.push({
        version, pass: raw.pass, maxDiff: raw.maxDiff,
        wasmSharpness:   raw.wasm?.sharpness,
        webgpuSharpness: raw.webgpu?.sharpness,
      });
    } else {
      console.log('  Unexpected result:', JSON.stringify(raw).slice(0, 120));
      results.push({ version, error: 'unexpected shape' });
    }

    try { rmSync(tmpDir, { recursive: true, force: true }); } catch {}
  }

  await browser.close();

  // ── summary ───────────────────────────────────────────────────────────
  console.log('\n' + '='.repeat(62));
  console.log('  SUMMARY');
  console.log('='.repeat(62));
  console.log(`${'Version'.padEnd(10)}  Result`);
  console.log('─'.repeat(62));

  for (const r of results) {
    let line;
    if (r.error)        line = `ERROR      ${r.error.slice(0, 50)}`;
    else if (r.skip)    line = `SKIP       ${r.skip}`;
    else if (r.webgpuError) line = `EP-ERROR   ${r.webgpuError.slice(0, 50)}`;
    else if (r.pass)    line = `PASS ✓     diff=${r.maxDiff?.toFixed(5)}  wasm_sharp=${r.wasmSharpness?.toFixed(4)}  webgpu_sharp=${r.webgpuSharpness?.toFixed(4)}`;
    else                line = `FAIL ✗     diff=${r.maxDiff?.toFixed(5)}  wasm_sharp=${r.wasmSharpness?.toFixed(4)}  webgpu_sharp=${r.webgpuSharpness?.toFixed(4)}`;
    console.log(`${r.version.padEnd(10)}  ${line}`);
  }

  const passing = results.filter(r => r.pass === true);
  const failing = results.filter(r => r.pass === false);
  console.log(`\n${passing.length} pass, ${failing.length} fail, ${results.length - passing.length - failing.length} skip/error`);

  if (passing.length && failing.length) {
    const lastPass  = passing.at(-1).version;
    const firstFail = failing[0].version;
    if (passing.at(-1) !== results[results.length - 1]) {
      // Last pass comes before first fail in version order
      console.log(`\n→ Bug introduced between ${lastPass} (last PASS) and ${firstFail} (first FAIL)`);
    } else {
      console.log(`\n→ Bug appears fixed: first PASS after FAIL is ${passing.at(-1).version}`);
    }
  } else if (failing.length && !passing.length) {
    console.log('\n→ All tested versions FAIL — WebGPU broke before 1.17.3, or on this GPU/driver regardless of version');
  } else if (!failing.length && passing.length) {
    console.log('\n→ All tested versions PASS — bug not present on this machine');
  }
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });
