# Decipher Star Wars CCG Catalog Research

Status: research notes, 2026-05-06
Goal: identify the cleanest way to build a CollectorVision example catalog for Decipher's Star Wars Customizable Card Game.

## Recommendation

Use the SWCCG Players Committee JSON repository as the primary metadata source, and use the image URLs already embedded in that JSON as the first-pass reference image source. For the first catalog, include only currently active, physically printed cards. Exclude Legacy, PreErrata, and Virtual cards.

Primary source:

- Metadata: https://github.com/swccgpc/swccg-card-json
- Canonical image origin: https://github.com/swccgpc/holotable
- Hosted image CDN: `https://res.starwarsccg.org/cards/...`

This is the cleanest source pair because it is maintained by the active SWCCG community, contains machine-readable card records, includes image URLs per face, and is already used by SCOMP and the Comlink Android app. The README asks: "Please do not use this database without notifying the SWCCG Players Committee." Treat that as a project step before publishing or redistributing a built catalog.

Card Game Geek should not be used for this first pass. Its visible listing images are 250x350 thumbnails, direct CMS assets sampled at 350x490, and the terms reserve content rights and restrict copying/downloading beyond normal service use. The first implementation should stick to Players Committee data and Holotable images.

## Resolved First-Pass Scope

- Include `Light.json` and `Dark.json` only.
- Exclude `LightLegacy.json`, `DarkLegacy.json`, `LightPreErrata.json`, and `DarkPreErrata.json`.
- Exclude Virtual sets. Treat set IDs `>= 200`, non-integer set IDs, and legacy sets as out of scope for the first catalog.
- Prefer Holotable `hires/*.png` when a matching file exists, with fallback to each card face's JSON `large` image URL.
- Keep `metadata.jsonl` next to the embedding catalog so integer SWCCG IDs, GEMP IDs, set names, face labels, and source image provenance remain queryable outside the compact NPZ.
- Assume Players Committee permission is acceptable for prototyping, but ask before public publication.
- Do not rely on Card Game Geek for metadata or images in this version.

## Source Findings

### SWCCG Players Committee JSON

Repository: https://github.com/swccgpc/swccg-card-json

Relevant files:

- `Light.json`
- `Dark.json`
- `LightLegacy.json`
- `DarkLegacy.json`
- `LightPreErrata.json`
- `DarkPreErrata.json`
- `sets.json`
- `rarity.json`

Observed shape:

```json
{
  "cards": [
    {
      "front": {
        "imageUrl": "https://res.starwarsccg.org/cards/Premiere-Light/large/2x3kpr.gif",
        "title": "2X-3KPR (Tooex)",
        "type": "Character"
      },
      "gempId": "1_1",
      "id": 2,
      "legacy": false,
      "printings": [{ "set": "1" }],
      "rarity": "U1",
      "set": "1",
      "side": "Light"
    }
  ]
}
```

Quick counts sampled from `main` on 2026-05-06:

| File | Cards | Front image URLs | Two-sided cards | Back image URLs | Sets |
|---|---:|---:|---:|---:|---:|
| `Light.json` | 1,953 | 1,953 | 33 | 33 | 53 |
| `Dark.json` | 1,867 | 1,867 | 30 | 30 | 52 |
| `LightLegacy.json` | 596 | 596 | 11 | 11 | 10 |
| `DarkLegacy.json` | 607 | 607 | 7 | 7 | 10 |
| `LightPreErrata.json` | 120 | 120 | 12 | 12 | 30 |
| `DarkPreErrata.json` | 81 | 81 | 8 | 8 | 27 |

`sets.json` contained 68 set entries: 58 non-legacy and 10 legacy. The sum of positive non-legacy set sizes was 3,751, but the live non-legacy card files contain 3,820 card records because reprints, alternate records, virtual cards, errata, and set-size bookkeeping do not collapse to a single simple set-size total.

With the first-pass physical-only filter applied by `build_swccg_catalog.py`, the normalized target is 2,579 records across 24 sets, from Premiere (`1`) through Jabba's Palace Sealed Deck (`112`). This count includes front/back face records when a physical card has both faces.

License note: `swccg-card-json` is MIT licensed, but its card text and imagery are still Star Wars/Decipher/Lucasfilm-derived game content. Preserve attribution and contact the Players Committee before using their database in a public CollectorVision catalog.

### Holotable / `res.starwarsccg.org`

Holotable repository: https://github.com/swccgpc/holotable

Image documentation: https://github.com/swccgpc/holotable/blob/master/docs/card-images.md

Important findings from the image docs:

- Holotable is described as the source of truth for SWCCG card images.
- Images are used by Holotable, GEMP, SCOMP, and websites.
- Images are stored under `Images-HT/starwars/{Set}-{Side}/`.
- `large/*.gif` images are served from `res.starwarsccg.org/cards`.
- `hires/*.png` exists in the repository for some cards and is used to generate downstream GIFs.
- Documented dimensions:
  - Small `t_*.gif`: 67x87
  - Old large GIF: 350x490
  - Standard large GIF: 745x1039
  - High-resolution PNG: 703x980 or better

Observed sample:

- `https://res.starwarsccg.org/cards/Premiere-Light/large/2x3kpr.gif`
- Response: HTTP 200 from S3/CloudFront
- Dimensions: 350x490 GIF
- Size: about 88 KB

The CDN path for `Premiere-Light/hires/2x3kpr.png` returned 403, but the Holotable GitHub `hires` directory exists for some Premiere Light images. For older cards, high-resolution PNG coverage appears partial and biased toward errata/remastered images. For a first catalog, use the JSON `front.imageUrl` and `back.imageUrl` fields directly; optionally add a later enrichment pass that checks the Holotable GitHub `hires` tree for a matching PNG.

### Static Cardlists On `res.starwarsccg.org`

Example: https://res.starwarsccg.org/cardlists/PremiereType.html

This is accessible and easy to scrape, but it is inferior to the JSON repository as a primary source.

Useful observations:

- The pages link directly to card images such as `https://res.starwarsccg.org/cards/Premiere-Light/large/2x3kpr.gif`.
- Rows contain set, side, type, title, and rarity.
- The navigation exposes Decipher sets, virtual sets, and legacy virtual cards.

Limitations:

- HTML table parsing would be more brittle than consuming the Players Committee JSON.
- It has less rich metadata than the JSON files.
- It duplicates image URL information already present in JSON.

Use static cardlists only as a sanity-check source, or as a fallback if a particular JSON/image mapping looks wrong.

### SCOMP

URL: https://scomp.starwarsccg.org/

Findings:

- Simple fetch returned 403.
- Browser access hit a Cloudflare challenge.
- The Players Committee JSON README says SCOMP uses the JSON database.

Recommendation: do not use SCOMP as an automated source. Treat it as a human-facing validation/search interface over the same underlying data.

### SWCCGDB

URL: https://swccgdb.com/

API docs: https://swccgdb.com/api/doc

Public examples:

- `/api/public/sets/`
- `/api/public/cards/`
- `/api/public/cards/{set_code}`
- `/api/public/card/{card_code}`

Observed sample: `/api/public/card/01001` returns card metadata and an image URL such as:

```text
https://res.starwarsccg.org/cards/Images-HT/starwars/Premiere-Light/large/2x3kpr.gif
```

That URL variant works and points to the same old large 350x490 GIF. SWCCGDB is a useful secondary metadata/API source, especially for Decipher-era physical sets, but the Players Committee JSON is broader and more current for virtual/errata data.

### Card Game Geek

Game page: https://cardgamegeek.com/database/games/swccg
Cards page: https://cardgamegeek.com/database/games/swccg/cards

Findings:

- The page reports 4,281 SWCCG card results.
- The card list is accessible without login.
- Visible card-list media images were 250x350 PNG thumbnails.
- A direct CMS asset sample returned 350x490 JPEG/GIF-size content.
- A sampled `card-standard-p-100` CMS transform returned 350x490.
- The page advertises `https://api.cardgamegeek.com/schemas/swccg/card.json`, but that host did not resolve from this environment during testing.
- `robots.txt` returned 404.
- Terms of service reserve content rights and say users may not copy/download/share content except where explicitly permitted for personal, non-commercial use.

The interesting nuance: one direct JPEG sample carried EXIF dimensions around 1497x2073, but the actual decoded image served to the client was still 350x490. That hints CGG may have original uploads or retained source metadata internally, but I did not find a public, documented way to retrieve higher-resolution originals.

Recommendation: do not build the first public catalog from Card Game Geek. If CGG scans look meaningfully better by eye, contact Card Game Geek for permission/API access and ask whether full-resolution originals can be licensed or used for a derived embedding catalog. Without that, use it as a manual comparison source only.

## Proposed Minimal Catalog Build

For a clean example implementation, start with only current, non-legacy, non-virtual, physically printed cards:

1. Download `sets.json`, `Light.json`, and `Dark.json` from `swccgpc/swccg-card-json` at a pinned commit SHA.
2. Flatten each `cards` array into one record per card face image:
   - front face: `front.imageUrl`
   - back face, when present: `back.imageUrl`
3. Skip records whose set is legacy, Virtual, PreErrata-only, or not a currently active physical set.
4. Keep a compact metadata manifest outside the NPZ:
   - `id`
   - `gempId`
   - `side`
   - `set`
   - `set_name`
   - `rarity`
   - `face` (`front` or `back`)
   - `title`
   - `type`
   - `image_url`
  - `preferred_image_url`
  - `preferred_image_kind`
  - `local_image`
  - source ref or commit SHA
5. Use stable image filenames as catalog IDs, because SWCCG IDs are integers rather than UUIDs. For example, write `p_146.png` under the Premiere image folder and store `swccg_id = "p_146.png"` in metadata.
6. Download images politely with local caching, resume-friendly `.part` files, and good-neighbor throttling.
7. Run the generic image-directory builder to embed each downloaded image using the same embedder contract as other CollectorVision catalogs.
8. Write the catalog NPZ with:
   - `embeddings`
   - `card_ids`
   - `source = "swccgpc-json"`
   - `embedder_spec`
9. Publish the external metadata manifest next to the NPZ or in the catalog-building repo, not inside the runtime NPZ.

For a first milestone, avoid `Legacy`, `PreErrata`, and Virtual files/sets. Add them later as separate build options because they may confuse ordinary physical-card recognition and inflate duplicate/near-duplicate rows.

## Example Normalized Record

```json
{
  "swccg_id": "p_2.gif",
  "source": "swccgpc-json",
  "source_ref": "<pinned sha>",
  "source_file": "Light.json",
  "source_id": 2,
  "gemp_id": "1_1",
  "side": "Light",
  "set_id": "1",
  "set_name": "Premiere",
  "rarity": "U1",
  "face": "front",
  "title": "2X-3KPR (Tooex)",
  "type": "Character",
  "image_url": "https://res.starwarsccg.org/cards/Premiere-Light/large/2x3kpr.gif"
}
```

## Lift Estimate

SWCCG is a moderate-lift but good example for adding a new game.

Why it is attractive:

- Small enough to build quickly compared with MTG: roughly 2.6k physical/current first-pass records, or about 3.8k current non-legacy records if Virtual cards are included later.
- Metadata is open, JSON-shaped, and community-maintained.
- Image URLs are already present in metadata.
- The game has enough set, side, rarity, virtual-card, and errata complexity to exercise a real new-game pipeline without becoming enormous.

Complications:

- IDs are not UUIDs, so the catalog builder needs deterministic UUIDv5 IDs or the core NPZ loader needs to support non-UUID string IDs more intentionally.
- Some cards have front/back faces.
- Virtual, legacy, and pre-errata cards need clear inclusion policy.
- Image quality is inconsistent: older physical cards may only have 350x490 GIFs, while some newer/remastered images may be larger.
- Legal/provenance details should be handled politely with the SWCCG Players Committee before publishing.

Expected first implementation size:

- One source downloader/normalizer script.
- One metadata manifest JSONL writer.
- One image cache/download stage.
- One catalog embedding stage using existing CollectorVision embedder APIs.
- A few focused tests for ID stability, face flattening, and required fields.

## Better Small-Game Alternatives

If the goal is only to demonstrate the simplest possible new-game integration, SWCCG is not the smallest option because of virtual cards, errata, two-sided records, and image provenance. A simpler example would be a game with a modern public API and consistent high-resolution card images.

However, SWCCG is a strong example if the goal is to show how CollectorVision handles a beloved long-tail game with community-maintained data. It is small enough to be practical, old enough to expose messy image realities, and active enough that the catalog would be useful.

## Resolved Questions And Remaining Questions

- Resolved: first catalog includes only `Light.json` and `Dark.json`.
- Resolved: Legacy, PreErrata, and Virtual cards are excluded.
- Resolved: high-resolution Holotable `hires` PNGs are preferred when present, falling back to JSON `large` URLs.
- Resolved for prototyping: assume Players Committee usage is acceptable, but ask before public publication.
- Resolved: do not rely on Card Game Geek for now.
- Remaining: decide whether to publish a full SWCCG catalog in the main `HanClinto/milo` HuggingFace repo or in a separate catalog repo.
- Remaining: decide whether future Virtual, Legacy, and PreErrata variants should be separate catalog keys.

## Next Step

Build a tiny proof-of-concept against a pinned `swccg-card-json` commit using the two-part example flow:

```bash
python catalog/swccg/build_swccg_catalog.py --source-ref <commit-sha> --limit 25
python catalog/build_catalog_from_images.py \
  --image-dir catalog/swccg/build/images \
  --embedder milo1 \
  --game swccg \
  --primary-key-name swccg_id
```

For quick local experiments, `--source-ref` defaults to `main`. For reproducible catalog builds, pass a commit SHA.

- Normalize `Light.json` and `Dark.json` to `catalog/swccg/build/metadata.jsonl`.
- Download 25 representative images across Decipher full and premium physical sets.
- Run the generic image-directory builder to embed those 25 images and write a standard named local NPZ.
- Add a smoke test that loads the NPZ with `Catalog.load(...)`.

That proof should reveal whether ID handling or image preprocessing needs a core CollectorVision tweak before doing the full 2.6k-record physical-only catalog.

Sample proof result, using `--limit 25` and then `catalog/build_catalog_from_images.py`:

- Wrote 25 metadata rows to `catalog/swccg/build/metadata.jsonl`.
- Downloaded 25 images on the first run; a repeat run reused all 25 cached images and downloaded nothing.
- Preferred Holotable `hires` PNGs for 13 of 25 sample cards.
- Observed `hires` images at 745x1039 and fallback `large` GIFs at 350x490.
- The image staging script now writes files under set-name folders, such as `catalog/swccg/build/images/Premiere/p_146.png`.
- The generic builder wrote `catalog/swccg/build/milo1-swccg-YYYY-MM-DD.npz`.
- `Catalog.load(...)` loaded the sample NPZ as 25 rows with `(25, 128)` embeddings, string card IDs such as `p_146.png`, and `source = "swccg"`.
