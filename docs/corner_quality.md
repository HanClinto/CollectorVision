# Corner Quality Metric

CollectorVision records a `corner_quality` score for every Cornelius corner detection. The score is a deterministic geometric plausibility check over the four predicted corners. It is not a learned model output today, and the default threshold is `0`, meaning the score is observational by default and does not reject detections before dewarp.

## Why It Exists

Cornelius can occasionally return a high-sharpness detection whose four points are not a plausible card quadrilateral: duplicated corners, triangle-like shapes, concave quads, or extreme proportions. `corner_quality` makes those cases visible and lets tools filter them without discarding the raw detection.

## Output Shape

Python exposes the metric in `DetectionResult.extra`:

- `corner_quality`: scalar score in `[0, 1]`
- `corner_quality_accepted`: whether `corner_quality >= min_corner_quality`
- `corner_quality_reason`: weakest failing subscore when rejected by a positive threshold
- `corner_quality_metrics`: diagnostic geometry and subscore details
- `detector_card_present`: the original sharpness/presence decision before the corner-quality gate

The web scanner mirrors these fields as `cornerQuality`, `cornerQualityReason`, and `cornerQualityMetrics`.

## Calculation

The final score is the minimum of several normalized subscores. This is intentionally conservative: a single fatal geometry issue should make the whole quadrilateral suspicious.

```text
corner_quality = min(
  edge,
  convexity,
  area_fill,
  opposite_edges,
  diagonals,
  aspect,
  angles,
)
```

Each subscore is in `[0, 1]`, where `1` is ideal and `0` is implausible.

### Edge

Rejects duplicated or nearly duplicated adjacent corners.

```text
min_edge_ratio = shortest_edge_px / max(image_width, image_height)
edge = clamp01((min_edge_ratio - 0.005) / 0.015)
```

This reaches `0` when the shortest edge is at or below `0.5%` of the larger frame dimension, and reaches `1` around `2%`. A duplicated corner scores `0`.

### Convexity

Rejects concave, self-crossing, or nearly collinear four-point shapes.

```text
convexity = 1 if all four edge cross products have the same non-zero sign else 0
```

A real card projection should be a convex quadrilateral.

### Area Fill

Rejects triangle-like or collapsed quads that occupy too little of their own bounding box.

This is not a screen-size check. A tiny but well-shaped card and a large well-shaped card can both score well. The denominator is the bounding box around the four predicted points, not the full image area.

```text
area_ratio = quadrilateral_area / bounding_box_area
area_fill = clamp01((area_ratio - 0.20) / 0.40)
```

This scores `0` at `20%` fill and reaches `1` around `60%` fill.

This subscore is meant to catch cases where the four points form something closer to a triangle, sliver, or folded shape than a perspective projection of a rectangular card. It should be treated as a tunable heuristic rather than proof that a card is absent.

### Opposite Edges

Checks whether top/bottom and left/right side lengths are wildly mismatched.

```text
top_bottom_ratio = max(top, bottom) / min(top, bottom)
side_ratio = max(right, left) / min(right, left)
opposite_edges = min(
  ramp_down(top_bottom_ratio, ideal=1.0, worst=2.5),
  ramp_down(side_ratio, ideal=1.0, worst=2.5),
)
```

Perspective can change opposite edge lengths, so this allows meaningful skew. It only collapses to `0` when an opposite pair differs by `2.5x` or more.

### Diagonals

Checks for severe projective distortion or collapsed shapes.

```text
diagonal_ratio = max(diagonal_a, diagonal_b) / min(diagonal_a, diagonal_b)
diagonals = ramp_down(diagonal_ratio, ideal=1.0, worst=2.5)
```

A perspective view may have unequal diagonals, but a very large mismatch is suspicious.

### Aspect

Checks whether the quadrilateral has a plausible long/short side ratio for a CCG card under perspective.

```text
aspect_ratio = average_long_side / average_short_side
aspect = triangular_score(aspect_ratio, target=1.40, min=1.10, max=2.20)
```

The target `1.40` is near a standard Magic/card aspect ratio. The broad `[1.10, 2.20]` range is intentionally permissive for perspective, camera crop, and model noise.

### Angles

Checks for extremely acute or obtuse corners.

```text
max_angle_deviation = max(abs(angle - 90 degrees) for angle in interior_angles)
angles = ramp_down(max_angle_deviation, ideal=0, worst=55)
```

This permits interior angles from about `35` to `145` degrees before the angle subscore reaches `0`.

## Threshold Guidance

The default `min_corner_quality` is `0`, so no detections are rejected early. This is best for evaluation because downstream tools can adjust the threshold without rerunning detection.

Suggested use:

- `0.00`: record quality only; process every detector-positive quadrilateral.
- `0.20`: very light filtering for obviously broken geometry.
- `0.35`: practical filtering for duplicated corners, concave quads, and severe shape collapse.
- `>0.50`: stricter filtering; use only after reviewing false rejects on real capture data.

The E2E viewer applies its corner-quality threshold interactively. Rerun the E2E harness with `--min-corner-quality 0` when you want threshold changes in the GUI to be reversible.

## Model Output Option

This metric can be moved into an exported ONNX graph as a derived tensor output because it is deterministic math over the predicted corners. That would let runtimes read `corner_quality` the same way they read Cornelius `sharpness`, without duplicating post-processing code.

That would not require retraining. A separate learned quality head could be trained later, but it would need labeled examples of usable and unusable corner geometry.
