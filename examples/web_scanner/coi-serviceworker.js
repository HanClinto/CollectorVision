/**
 * coi-serviceworker.js — Cross-Origin Isolation via Service Worker
 *
 * Injects Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers
 * on every same-origin response so that `crossOriginIsolated` becomes true,
 * which enables SharedArrayBuffer and therefore multi-threaded WASM in ort-web.
 *
 * GitHub Pages (and most static hosts) cannot set these response headers
 * server-side.  This service worker is the standard workaround.
 *
 * First visit: SW installs + claims immediately, then sends a "reload" message
 * to the page so it reloads under the new headers.  All subsequent visits are
 * already cross-origin isolated with no visible flash.
 *
 * Based on the well-known coi-serviceworker pattern by Guido Zuidhof.
 * https://github.com/gzuidhof/coi-serviceworker
 */

/* eslint-env serviceworker */

const SW_VERSION = 2;
const NON_ISOLATED_DOCUMENTS = [
  /\/applet_example\.html$/,
];

function shouldSkipIsolation(request) {
  const url = new URL(request.url);
  return request.mode === "navigate"
    && NON_ISOLATED_DOCUMENTS.some((pattern) => pattern.test(url.pathname));
}

self.addEventListener("install", () => {
  // Skip waiting so the new SW takes over without requiring user navigation.
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  // Claim all existing clients immediately so they benefit from the headers
  // on the very next fetch, rather than waiting for a navigation.
  event.waitUntil(self.clients.claim());
});

self.addEventListener("message", (event) => {
  if (event.data === "sw-version") {
    event.source.postMessage({ type: "sw-version", version: SW_VERSION });
  }
});

self.addEventListener("fetch", (event) => {
  const { request } = event;

  // Only intercept same-origin requests.  Cross-origin requests (CDN fonts,
  // etc.) must never get COEP applied by us — any cross-origin resource
  // already needs a CORP header from its server to pass COEP, and we touch
  // only the response that the browser delivers to this origin's JS context.
  if (!request.url.startsWith(self.location.origin)) {
    return;
  }

  // range requests (e.g. video seeks) are not compatible with Response
  // construction from a cloned body; skip them.
  if (request.headers.get("range")) {
    return;
  }

  // The applet playground intentionally demonstrates arbitrary page behavior,
  // including loading public Scryfall card art.  Scryfall's image CDN does not
  // emit CORP/CORS headers, so a `require-corp` document would show only the
  // image alt text in Firefox.  The applet does not need SharedArrayBuffer, so
  // leave that document non-isolated.
  if (shouldSkipIsolation(request)) {
    return;
  }

  event.respondWith(
    fetch(request).then((response) => {
      // Opaque responses (cross-origin no-cors) cannot be cloned safely.
      if (response.type === "opaque" || response.type === "error") {
        return response;
      }

      const newHeaders = new Headers(response.headers);
      newHeaders.set("Cross-Origin-Opener-Policy", "same-origin");
      newHeaders.set("Cross-Origin-Embedder-Policy", "require-corp");
      newHeaders.set("Cross-Origin-Resource-Policy", "cross-origin");

      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: newHeaders,
      });
    }),
  );
});
