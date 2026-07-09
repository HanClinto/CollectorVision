# CollectorVision Screen Capture Monitor

Design sketch for a static, browser-only utility that watches a user-selected
screen, window, or browser tab and emits card detections from the captured video.

The goal is not another full scanner app. This example should be a focused
monitoring tool:

- capture any visible video source the browser can share
- optionally restrict detection to a drawn region of interest
- identify cards with the existing web scanner pipeline
- emit simple events for overlays, automation, and exports

## Why screen capture

Browser JavaScript cannot reliably read pixels from an embedded YouTube player.
Drawing a cross-origin YouTube frame into a canvas taints the canvas, blocking
`getImageData()`, `toBlob()`, and related APIs.

`navigator.mediaDevices.getDisplayMedia()` is the clean browser-native escape
hatch. The user grants permission to share a tab, window, or screen, and the app
receives a normal `MediaStream` that can be drawn into a canvas and processed
locally.

This makes the tool useful beyond YouTube:

- YouTube videos and live streams
- Twitch, Whatnot, SpellTable, Discord, or webcam windows
- OBS preview/program windows
- local media players
- any tab or app the user can share

## Product shape

The monitor should feel like a small utility panel, not a production dashboard.

Primary controls:

1. **Capture source**: start/stop screen capture.
2. **Region of interest**: show/hide and edit one crop rectangle.
3. **Detection**: threshold, confirmation frames, scan interval.
4. **Outputs**: event stream, overlay page, CSV/JSON export.

Avoid multiple modes that compete with each other. The core loop is always:

```text
display capture -> optional ROI crop -> CollectorVision worker -> confirmed event -> output adapters
```

## Suggested files

```text
examples/screen_capture_monitor/
  README.md
  index.html              # capture + ROI + detection console
  monitor.js              # UI and capture loop
  monitor.css
  overlay.html            # simple OBS/browser-source overlay
  overlay.js
  events.html             # optional raw event/debug viewer
```

The example should reuse the existing static web scanner assets instead of
copying model logic:

- `examples/web_scanner/scanner.worker.mjs`
- `examples/web_scanner/assets/manifest.json`
- `examples/web_scanner/assets/**`
- `examples/web_scanner/vendor/**`

For local development, either serve this folder beside `web_scanner` or resolve
asset URLs through configuration.

## Capture pipeline

1. User clicks **Start capture**.
2. App calls `navigator.mediaDevices.getDisplayMedia({ video: true, audio: false })`.
3. Captured stream is attached to a hidden or visible `<video>`.
4. A preview canvas draws the capture stream.
5. If ROI is enabled, the process canvas draws only the ROI rectangle.
6. `createImageBitmap(processCanvas)` is posted to `scanner.worker.mjs`.
7. Worker returns the normal result message.
8. Main thread updates the preview overlay, confirmation bucket, exports, and
   event outputs.

This is very close to `examples/web_scanner/app.js`; the main difference is that
the source is a display-capture stream instead of `getUserMedia()`.

## ROI behavior

Keep ROI deliberately simple:

- one rectangular region
- drag to move
- drag handles to resize
- double-click or button to reset to full frame
- persist normalized ROI in `localStorage`

Store ROI in normalized capture coordinates:

```json
{
  "x": 0.25,
  "y": 0.10,
  "width": 0.50,
  "height": 0.70
}
```

Before posting to the worker, draw the crop into the process canvas:

```js
processCtx.drawImage(
  captureVideo,
  roi.x * videoWidth,
  roi.y * videoHeight,
  roi.width * videoWidth,
  roi.height * videoHeight,
  0,
  0,
  processCanvas.width,
  processCanvas.height,
);
```

Worker results are in cropped-frame coordinates. The UI can map detected corners
back to full-preview coordinates by applying the ROI offset and scale.

## Detection policy

Use conservative defaults for a utility tool:

| Setting | Default | Reason |
| --- | ---: | --- |
| scan interval | `500 ms` | responsive without wasting CPU |
| match threshold | `0.50` | matches existing web scanner default |
| consecutive matches | `2` | avoids one-frame false positives |
| cooldown | `3000 ms` | prevents rapid duplicate events |
| min corner confidence | `0.02` | matches existing browser path |

Advanced model/runtime controls should be hidden behind a compact details panel.
The default experience should be: start capture, draw ROI, watch detections.

## Event model

Every confirmed card should produce one normalized event:

```json
{
  "type": "collectorvision.card",
  "version": 1,
  "timestamp": "2026-07-09T22:00:00.000Z",
  "source": {
    "kind": "display-capture",
    "label": "user-selected source",
    "videoWidth": 1920,
    "videoHeight": 1080,
    "roi": { "x": 0.25, "y": 0.10, "width": 0.50, "height": 0.70 }
  },
  "card": {
    "cardId": "scryfall-or-catalog-id",
    "secondaryId": "oracle-or-group-id",
    "secondaryIdField": "oracleId",
    "name": "optional enriched name",
    "setCode": "optional set",
    "score": 0.73,
    "orientation": "upright"
  },
  "detection": {
    "corners": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.9], [0.1, 0.9]],
    "sharpness": 0.04,
    "confidence": 0.04
  }
}
```

The event should be useful without Scryfall enrichment. Enrichment can update UI
labels and CSV fields, but detection should not wait on network requests.

## Output adapters

Keep outputs small and explicit:

### 1. Browser events

Dispatch a DOM event from the monitor page:

```js
window.dispatchEvent(new CustomEvent("collectorvision:card", { detail: event }));
```

This makes custom examples easy without committing to a larger plugin API.

### 2. BroadcastChannel

Publish confirmed events to same-origin pages:

```js
new BroadcastChannel("collectorvision-monitor").postMessage(event);
```

`overlay.html` can listen on the same channel and update a browser-source overlay
for OBS. This keeps the first version fully static and serverless.

### 3. Export

Maintain an in-memory event log with buttons for:

- copy card list
- download CSV
- download JSONL
- clear session

CSV should include timestamp, card ID, secondary ID, name, set, score,
sharpness, ROI, and source dimensions.

### 4. Optional webhook later

Do not include webhooks in the first version. Browser CORS, local servers, and
OBS websocket auth can make the tool feel confusing. If needed later, add a
single advanced "POST events to URL" adapter.

## Overlay page

`overlay.html` should be tiny:

- listens on `BroadcastChannel("collectorvision-monitor")`
- shows the most recent confirmed card
- optionally fades out after N seconds
- can be used as an OBS browser source

This mirrors the Nifty scanner use case, but avoids requiring a local FastAPI or
Socket.IO server.

Future server-backed overlays can still bridge the same event shape into the
existing Nifty routes if needed.

## Integration-testing workflow

The same tool can be used for model comparisons:

1. Open a test video or live feed.
2. Start display capture.
3. Draw ROI around the card area.
4. Run a pass with Cornelius 205.
5. Download JSONL.
6. Switch manifest/model bundle to Cornelius 210.
7. Run the same video/time range.
8. Compare event streams offline.

For deterministic tests, a later version can add:

- URL fields for known videos
- manual start timestamp markers
- "record all frame results" diagnostic mode
- side-by-side workers using different manifests

Those should stay out of the first UI unless the basic utility proves useful.

## Minimal implementation touch-points

The clean first implementation should touch only a few surfaces:

1. Add this example folder with static files.
2. Reuse `scanner.worker.mjs` as-is if possible.
3. Add a small shared confirmation bucket helper only if duplication with the
   applet becomes painful.
4. Add asset URL configuration only if relative paths prevent reuse from this
   folder.

If `scanner.worker.mjs` cannot load assets from outside `examples/web_scanner`,
the least invasive fix is to let the init message include an `assetBaseUrl`,
then resolve model/catalog/vendor fetches relative to that base. Avoid copying
worker code.

## Non-goals

- direct YouTube canvas capture
- multi-source monitoring
- multi-card detection in one frame
- server requirement
- OBS websocket integration in v1
- full dashboard analytics
- automatic video playback control

## MVP checklist

- Start/stop display capture.
- Preview captured source.
- Draw and persist one ROI rectangle.
- Feed ROI frames into existing CollectorVision worker.
- Show live corners and latest result.
- Confirm cards using threshold + consecutive-frame bucket.
- Dispatch DOM events and BroadcastChannel events.
- Provide `overlay.html`.
- Export CSV and JSONL.

