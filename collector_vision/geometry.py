"""Geometry quality helpers for detected card quadrilaterals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QuadQuality:
    """Normalized plausibility score for a detected card quadrilateral."""

    score: float
    accepted: bool
    reason: str | None
    metrics: dict[str, Any]


DEFAULT_MIN_CORNER_QUALITY = 0.0


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _ramp_down(value: float, ideal: float, worst: float) -> float:
    if worst <= ideal:
        return 0.0
    return _clamp01((worst - value) / (worst - ideal))


def _aspect_score(aspect_ratio: float, target: float, min_value: float, max_value: float) -> float:
    if aspect_ratio < min_value or aspect_ratio > max_value:
        return 0.0
    if aspect_ratio <= target:
        return _clamp01((aspect_ratio - min_value) / (target - min_value))
    return _clamp01((max_value - aspect_ratio) / (max_value - target))


def quad_quality(
    corners: np.ndarray | None,
    image_shape: tuple[int, ...] | None = None,
    *,
    min_score: float = DEFAULT_MIN_CORNER_QUALITY,
) -> QuadQuality:
    """Return a single plausibility score for a TL/TR/BR/BL card quad.

    The score is the minimum of independent normalized checks: finite points,
    non-duplicate edges, convexity, area fill, opposite-edge balance, diagonal
    balance, card-like aspect ratio, and interior angle sanity. A score near 1
    is a plausible card rectangle under perspective; 0 means at least one fatal
    geometric property failed.
    """
    base_metrics: dict[str, Any] = {}
    if corners is None:
        return QuadQuality(0.0, False, "missing_corners", base_metrics)

    pts = np.asarray(corners, dtype=np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        return QuadQuality(0.0, False, "invalid_corners", base_metrics)

    if image_shape is None:
        scale = np.array([1.0, 1.0], dtype=np.float32)
        frame_span = 1.0
    else:
        height, width = image_shape[:2]
        scale = np.array([width, height], dtype=np.float32)
        frame_span = float(max(width, height))
    pts = pts * scale

    edges = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
    diagonals = np.array([np.linalg.norm(pts[0] - pts[2]), np.linalg.norm(pts[1] - pts[3])], dtype=np.float32)
    min_edge = float(edges.min())
    max_edge = float(edges.max())
    min_diagonal = float(diagonals.min())
    max_diagonal = float(diagonals.max())

    shoelace = float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
    area = abs(shoelace) / 2.0
    bbox_w = float(pts[:, 0].max() - pts[:, 0].min())
    bbox_h = float(pts[:, 1].max() - pts[:, 1].min())
    bbox_area = bbox_w * bbox_h

    angles: list[float] = []
    for index, point in enumerate(pts):
        prev_vec = pts[(index - 1) % 4] - point
        next_vec = pts[(index + 1) % 4] - point
        prev_len = float(np.linalg.norm(prev_vec))
        next_len = float(np.linalg.norm(next_vec))
        if prev_len <= 1e-6 or next_len <= 1e-6:
            angle = 0.0
        else:
            cosine = float(np.dot(prev_vec, next_vec) / (prev_len * next_len))
            angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        angles.append(angle)

    crosses = []
    for index in range(4):
        a = pts[(index + 1) % 4] - pts[index]
        b = pts[(index + 2) % 4] - pts[(index + 1) % 4]
        crosses.append(float(np.cross(a, b)))
    nonzero_crosses = [value for value in crosses if abs(value) > 1e-6]
    convex = len(nonzero_crosses) == 4 and (
        all(value > 0 for value in nonzero_crosses) or all(value < 0 for value in nonzero_crosses)
    )

    top_bottom_ratio = float(max(edges[0], edges[2]) / max(min(edges[0], edges[2]), 1e-6))
    side_ratio = float(max(edges[1], edges[3]) / max(min(edges[1], edges[3]), 1e-6))
    aspect_ratio = float(((edges[1] + edges[3]) / 2.0) / max((edges[0] + edges[2]) / 2.0, 1e-6))
    if aspect_ratio < 1.0:
        aspect_ratio = 1.0 / max(aspect_ratio, 1e-6)
    area_ratio = float(area / max(bbox_area, 1e-6))
    diagonal_ratio = float(max_diagonal / max(min_diagonal, 1e-6))
    min_edge_ratio = min_edge / max(frame_span, 1e-6)
    max_angle_deviation = max(abs(angle - 90.0) for angle in angles)

    subscores = {
        "edge": _clamp01((min_edge_ratio - 0.005) / 0.015),
        "convexity": 1.0 if convex else 0.0,
        "area_fill": _clamp01((area_ratio - 0.20) / 0.40),
        "opposite_edges": min(_ramp_down(top_bottom_ratio, 1.0, 2.5), _ramp_down(side_ratio, 1.0, 2.5)),
        "diagonals": _ramp_down(diagonal_ratio, 1.0, 2.5),
        "aspect": _aspect_score(aspect_ratio, 1.40, 1.10, 2.20),
        "angles": _ramp_down(max_angle_deviation, 0.0, 55.0),
    }
    score = float(min(subscores.values()))
    weakest = min(subscores, key=subscores.get)
    accepted = score >= min_score
    metrics = {
        "edge_lengths": [float(value) for value in edges],
        "min_edge": min_edge,
        "max_edge": max_edge,
        "top_bottom_ratio": top_bottom_ratio,
        "side_ratio": side_ratio,
        "aspect_ratio": aspect_ratio,
        "diagonal_ratio": diagonal_ratio,
        "area": area,
        "area_ratio": area_ratio,
        "angles_deg": angles,
        "min_angle_deg": float(min(angles)),
        "max_angle_deg": float(max(angles)),
        "convex": convex,
        "subscores": subscores,
    }
    return QuadQuality(score, accepted, None if accepted else weakest, metrics)
