# Building Custom Catalogs

CollectorVision recognizes cards by comparing a camera image against a catalog
file. A catalog is just a list of IDs and reference-image fingerprints.

If you run a store, collection site, inventory tool, grading workflow, or any
other system with its own card records, you may want your own catalog instead of
using a public one.

## Why Build Your Own Catalog?

Build your own catalog when you want CollectorVision to recognize only the cards
that exist in your system.

For example, imagine your database has these records:

```json
[
  {"inventory_id": "store-1001", "image": "images/store-1001.jpg"},
  {"inventory_id": "store-1002", "image": "images/store-1002.jpg"},
  {"inventory_id": "store-1003", "image": "images/store-1003.jpg"}
]
```

If you build a catalog from those images, CollectorVision will search only those
three cards. When it finds a match, it gives back IDs like `store-1001`, because
those are the IDs you put in.

That means:

- You control what can be recognized.
- You get back IDs your application already understands.
- You do not need to map from someone else's card IDs into your own database.
- You can build small, private catalogs for one store, one event, one collection,
  or one customer's inventory.

## The Simple Version

Put reference images in a folder. Name each image with the ID you want back.

```text
my-images/
  store-1001.jpg
  store-1002.jpg
  store-1003.jpg
```

Then build a catalog:

```bash
python catalog/build_catalog_from_images.py \
  --image-dir my-images \
  --embedder milo1 \
  --game my-inventory \
  --primary-key-name inventory_id
```

This writes files next to the image folder:

```text
milo1-my-inventory-YYYY-MM-DD.npz
milo1-my-inventory-YYYY-MM-DD.metadata.jsonl
```

The `.npz` file is the catalog that CollectorVision loads. The `.metadata.jsonl`
file is a small helper file that shows which ID came from which image path.

## What Is In The Catalog?

The catalog file is a compact NumPy archive. It is not meant to be edited by
hand, but conceptually it looks like this:

```json
{
  "embeddings": "one numeric fingerprint per image",
  "card_ids": ["store-1001.jpg", "store-1002.jpg", "store-1003.jpg"],
  "source": "my-inventory",
  "embedder_spec": {
    "kind": "neural",
    "algo_key": "milo1",
    "image_size": 448,
    "primary_key_name": "inventory_id"
  }
}
```

The companion metadata file is plain JSONL, one row per image:

```json
{"inventory_id": "store-1001.jpg", "relative_path": "store-1001.jpg"}
{"inventory_id": "store-1002.jpg", "relative_path": "store-1002.jpg"}
{"inventory_id": "store-1003.jpg", "relative_path": "store-1003.jpg"}
```

## Using The Catalog

Load the generated `.npz` file in your application:

```python
import collector_vision as cvg

catalog = cvg.Catalog.load("milo1-my-inventory-YYYY-MM-DD.npz")
```

After that, searches return your filenames as IDs. If your filenames are your
database keys, you can use the result directly.

## Adding Another Game

You can also use this folder as a starting point for adding a new game. The
experimental SWCCG example in `catalog/swccg/` shows the two-step shape:

1. A game-specific script downloads images and writes metadata.
2. `build_catalog_from_images.py` turns those images into a CollectorVision
   catalog.

This is still experimental. Milo was trained for Magic: The Gathering card
recognition, and there is no guarantee that the same embedder will work well for
other games. Some games may work surprisingly well, some may need better images,
and some may need a future model trained for that kind of card.

The safest first use case is a custom catalog for cards that look like the cards
Milo already understands. New games are worth exploring, but treat results as a
prototype until you have tested them with real camera images.
