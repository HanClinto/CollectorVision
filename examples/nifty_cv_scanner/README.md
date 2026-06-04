# CollectorVision Nifty Scanner

Python/CollectorVision port of Paul Saunders' [`paul-lrr/nifty-recognizer`](https://github.com/paul-lrr/nifty-recognizer).

The original project used Node, Express, Socket.IO, and DeckedBuilder notifications to update a stream overlay. This example keeps the same `card-view.html` and `card-controller.html` browser behavior, but serves them with FastAPI and can recognize uploaded frames with CollectorVision's Python ONNX Runtime backend.

## Features

Includes all of the same OBS overlay pages as the original, along with a new scanner page that offers fast CollectorVision-based card recognition -- all controllable through the browser.

<img width="1536" height="768" alt="Screenshot 2026-06-03 at 12 17 47 PM" src="https://github.com/user-attachments/assets/93898f6f-3240-49d2-a5f5-341f40935052" />

<img width="3222" height="2122" alt="image" src="https://github.com/user-attachments/assets/89b66821-8191-4572-8245-f4621c730b1f" />



## Install

Use an example-local virtual environment so the FastAPI server dependencies stay
out of your main CollectorVision development environment:

```bash
cd examples/nifty_cv_scanner
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

With `uv`:

```bash
cd examples/nifty_cv_scanner
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

```bash
cd examples/nifty_cv_scanner
python server.py --hfd HanClinto/milo scryfall-mtg --host 127.0.0.1 --port 8000
```

Startup logs show progress while CollectorVision loads the catalog, corner
detector, and embedder models. First startup may take longer if the catalog must
be downloaded; later runs use the local CollectorVision cache.

Then open:

- Index: <http://127.0.0.1:8000/>
- Scanner: <http://127.0.0.1:8000/scanner.html>
- Overlay: <http://127.0.0.1:8000/card-view.html>
- Controller: <http://127.0.0.1:8000/card-controller.html>

Use `--catalog ./catalog.npz` instead of `--hfd` to run from a local catalog.

## Camera Scanner

`scanner.html` is a browser camera control surface for the Nifty overlay. Pick a
camera, start scanning, and confirmed matches are broadcast to `card-view.html`
through the same `card_image` channel used by the controller page.

The page sends frames to CollectorVision's Python ONNX Runtime backend and keeps
the client-side scanner loop simple:

- Capture controls tune the minimum frame start-to-start interval and JPEG
  quality. Set the interval to `0 ms` for free-running mode, where the next
  frame starts as soon as the previous request finishes.
- Corner detector controls tune the minimum sharpness gate.
- Recognition controls tune score threshold and frames-to-confirm. The score is CollectorVision's
  cosine-similarity retrieval score after the accepted card crop is embedded.
  The score meter shows the latest frame score against the active threshold, and
  the confirmation meter shows how quickly the best bucket is filling toward the
  recent bucket window, with the threshold marker at frames-to-confirm.
- Scan upside-down cards embeds both the dewarped crop and a 180-degree rotated
  copy, searches both, and keeps the orientation with the stronger top match.
- Multi-frame controls tune the rolling embedding buffer, prior similarity
  filter, and recent bucket window.
  Only frames that pass the score threshold enter the rolling embedding buffer,
  and bucket confirmation uses average score rather than summed scores.
  The prior similarity meter shows the latest best similarity between the
  current accepted embedding and the rolling buffer.
- Bucket by oracle ID groups different printings of the same card together when
  the catalog includes secondary/oracle IDs; turn it off to require exact card IDs.
  CollectorVision returns these IDs directly from the catalog for Scryfall-based
  catalogs, so grouping does not need to wait for Scryfall label lookups.
- Recent buckets load friendly Scryfall labels as `Name [SET]` after each card
  is looked up. The Broadcast section can optionally prefetch card images as
  soon as a card enters a bucket, and the camera stage shows the current
  recognition FPS.
- Broadcast target defaults to the same server that served `scanner.html`
  (`window.location.origin`, usually `http://127.0.0.1:8000`). Change it when
  the overlay is connected to a different Nifty server, host, or port. For an
  original Nifty/Socket.IO server, use the full websocket endpoint, for example
  `ws://localhost/socket.io/?type=cardImage`.

For camera permissions, use `http://127.0.0.1:8000/scanner.html` or another
secure/local origin accepted by your browser.

## Recognize A Frame

Send an image to CollectorVision and broadcast the recognized card into the overlay:

```bash
curl -X POST 'http://127.0.0.1:8000/recognize/upload?broadcast=true' \
  -F 'file=@../../examples/images/7286819f-6c57-4503-898c-528786ad86e9_sample.jpg'
```

For API-only use without updating the overlay:

```bash
curl -X POST 'http://127.0.0.1:8000/identify/upload' \
  -F 'file=@../../examples/images/7286819f-6c57-4503-898c-528786ad86e9_sample.jpg'
```

The JSON `/identify` endpoint also accepts CollectorVision's base64 shape:

```json
{
  "_base64": "<base64 JPEG or PNG>",
  "top_k": 5,
  "min_sharpness": 0.02,
  "min_prior_similarity": 0.7,
  "prior_embeddings": [],
  "broadcast": true
}
```

It also accepts the older record wrapper:

```json
{
  "records": [
    { "_base64": "<base64 JPEG or PNG>", "broadcast": true }
  ]
}
```

## Suggested Settings
Full explanation of all settings is outside of the scope of this document, but feel free to experiment to see what gives you the best results. YMMV, but these are some settings that seemed to work well for me in my testing:

<img width="3258" height="894" alt="image" src="https://github.com/user-attachments/assets/da9e712b-71a7-4a71-b578-5c19f8beb23b" />


## Compatibility Routes

- `GET /cardmatch/{id}` mirrors the original DeckedBuilder callback and broadcasts `card_image` to the overlay.
- Numeric IDs are displayed through Gatherer, matching the original project.
- Scryfall UUIDs from the default CollectorVision MTG catalog are displayed through Scryfall's image endpoint.
- Non-numeric custom IDs fall back to `/cards/{id}.jpg`; place custom images in `public/cards/` if you need that original orb-file behavior.

## Notes And Attribution

The public page layout and controller behavior are intentionally kept close to the original `nifty-recognizer` project. The Node server has been replaced with a FastAPI server and a tiny `/socket.io/socket.io.js` compatibility shim that uses native WebSockets for the local Python server and minimal Socket.IO websocket framing for external `/socket.io/` Nifty servers.

Original project: <https://github.com/paul-lrr/nifty-recognizer>

CollectorVision handles detection, dewarping, embedding, and nearest-neighbor search with Python ONNX Runtime so the model stays loaded and reused across requests.
