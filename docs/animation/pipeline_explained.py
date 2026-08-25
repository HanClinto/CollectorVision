"""CollectorVision — Pipeline Explained (3Blue1Brown-style manim animation).

Render:
    cd /path/to/collector_vision
    manim -ql docs/animation/pipeline_explained.py PipelineExplained   # low quality, fast
    manim -qh docs/animation/pipeline_explained.py PipelineExplained   # high quality

Individual scenes (for iteration):
    manim -ql docs/animation/pipeline_explained.py TitleScene
    manim -ql docs/animation/pipeline_explained.py CornerScene
    manim -ql docs/animation/pipeline_explained.py DewarpScene
    manim -ql docs/animation/pipeline_explained.py EmbedScene
    manim -ql docs/animation/pipeline_explained.py SearchScene
    manim -ql docs/animation/pipeline_explained.py ResultScene
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import *

# ---------------------------------------------------------------------------
# Constants / palette
# ---------------------------------------------------------------------------

BG_COLOR = ManimColor("#1C1C2E")
ACCENT = ManimColor("#58C4DD")
HIGHLIGHT = ManimColor("#00E678")
DIM_WHITE = ManimColor("#AAAACC")

CORNER_COLORS = {
    "TL": ManimColor("#FF4444"),
    "TR": ManimColor("#44FFFF"),
    "BR": ManimColor("#FFFF44"),
    "BL": ManimColor("#FF44FF"),
}
CORNER_LABELS = ["TL", "TR", "BR", "BL"]

_HERE = Path(__file__).parent
_ROOT = _HERE.parent.parent

SAMPLE_IMAGE = _ROOT / "examples/images/7286819f-6c57-4503-898c-528786ad86e9_sample.jpg"
DEWARPED_IMAGE = _ROOT / "docs/scrying_glass_dewarped.jpg"
ORIGINAL_IMAGE = _ROOT / "docs/scrying_glass_original.jpg"


def _load_data() -> dict:
    p = _HERE / "precomputed.json"
    if p.exists():
        return json.loads(p.read_text())
    return {
        "image_size": [1280, 960],
        "corners_normalized": [[0.310, 0.104], [0.662, 0.115], [0.672, 0.810], [0.295, 0.819]],
        "sharpness": 0.0603,
        "query_embedding": [(-1 + 2 * (i % 17) / 17) for i in range(128)],
        "top_hits": [
            (0.846, "7286819f-6c57-4503-898c-528786ad86e9"),
            (0.538, "0d8bdf32-a644-4931-8b5e-b379530e165b"),
            (0.537, "8ae591d2-b9d3-4bc5-bcec-5d3d79a13b41"),
            (0.528, "d39c8166-9e63-4b02-af7b-4caf14ca73ac"),
            (0.521, "2f6e5575-b004-417f-9366-6ba7840a79e7"),
        ],
        "hit_embeddings": {},
    }


DATA = _load_data()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_barcode(
    values: list[float], width: float = 5.0, height: float = 0.4, n_show: int = 64
) -> VGroup:
    """Render a float list as a horizontal strip of blue→red colored cells."""
    vals = np.array(values[:n_show], dtype=float)
    normed = np.clip((vals + 1.0) / 2.0, 0.0, 1.0)
    cell_w = width / len(vals)
    rects = VGroup()
    for i, v in enumerate(normed):
        r = int(255 * v)
        b = int(255 * (1.0 - v))
        g = int(60 * (1.0 - abs(2 * v - 1.0)))
        color = ManimColor.from_rgb((r / 255, g / 255, b / 255))
        rect = Rectangle(
            width=cell_w, height=height, fill_color=color, fill_opacity=1.0, stroke_width=0
        )
        rect.shift(RIGHT * (i * cell_w - width / 2 + cell_w / 2))
        rects.add(rect)
    border = Rectangle(
        width=width, height=height, stroke_color=WHITE, stroke_width=1, fill_opacity=0
    )
    return VGroup(rects, border)


def image_corner_to_manim(cx: float, cy: float, img_mob) -> np.ndarray:
    """Convert normalized image coords [0,1] to manim point on ImageMobject."""
    w = img_mob.width
    h = img_mob.height
    center = img_mob.get_center()
    return np.array([center[0] + (cx - 0.5) * w, center[1] + (0.5 - cy) * h, 0])


def step_badge(n: int, text: str) -> VGroup:
    circle = Circle(radius=0.28, color=ACCENT, fill_color=ACCENT, fill_opacity=0.15, stroke_width=2)
    num = Text(str(n), font_size=20, color=ACCENT).move_to(circle)
    label = Text(text, font_size=22, color=WHITE).next_to(circle, RIGHT, buff=0.15)
    return VGroup(circle, num, label)


def _img_or_rect(path, height, fill_color=BLUE_D):
    if path.exists():
        mob = ImageMobject(str(path))
        mob.height = height
        return mob
    aspect = 9 / 6.3 if height > 3 else 1
    return Rectangle(
        width=height / aspect,
        height=height,
        fill_color=fill_color,
        fill_opacity=0.4,
        stroke_color=WHITE,
        stroke_width=1,
    )


# ---------------------------------------------------------------------------
# Per-scene animation functions  (each accepts `s` = the active Scene)
# ---------------------------------------------------------------------------


def play_title(s: Scene) -> None:
    title = Text(
        "How does a computer\nrecognize a trading card?",
        font_size=44,
        color=WHITE,
        line_spacing=1.3,
    )
    subtitle = Text("CollectorVision — a three-step pipeline", font_size=24, color=ACCENT)
    subtitle.next_to(title, DOWN, buff=0.5)

    steps = (
        VGroup(
            step_badge(1, "Detect corners"),
            step_badge(2, "Dewarp (flatten)"),
            step_badge(3, "Embed & search"),
        )
        .arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        .next_to(subtitle, DOWN, buff=0.6)
    )

    s.play(Write(title), run_time=1.5)
    s.play(FadeIn(subtitle, shift=UP * 0.2), run_time=0.8)
    s.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.3) for b in steps], lag_ratio=0.3), run_time=1.2)
    s.wait(1.5)


def play_corner(s: Scene) -> None:
    corners_norm = DATA["corners_normalized"]
    sharpness = DATA["sharpness"]

    title = Text("Step 1 — Corner Detection", font_size=32, color=ACCENT)
    title.to_edge(UP, buff=0.3)
    s.play(Write(title), run_time=0.7)

    img = _img_or_rect(SAMPLE_IMAGE, 4.2)
    img.to_edge(LEFT, buff=0.5).shift(DOWN * 0.3)
    s.play(FadeIn(img), run_time=0.6)

    caption = Text(
        "A neural network looks at the full image...", font_size=22, color=DIM_WHITE
    ).to_edge(DOWN, buff=0.4)
    s.play(FadeIn(caption), run_time=0.5)
    s.wait(0.3)

    # Heatmap blobs at corner positions
    blobs = VGroup(
        *[
            Circle(
                radius=0.45,
                color=CORNER_COLORS[label],
                fill_color=CORNER_COLORS[label],
                fill_opacity=0.25,
                stroke_width=2,
            ).move_to(image_corner_to_manim(cx, cy, img))
            for (cx, cy), label in zip(corners_norm, CORNER_LABELS)
        ]
    )
    s.play(LaggedStart(*[GrowFromCenter(b) for b in blobs], lag_ratio=0.2), run_time=1.0)

    s.play(
        Transform(
            caption,
            Text(
                "...and outputs probability distributions for each corner",
                font_size=22,
                color=DIM_WHITE,
            ).to_edge(DOWN, buff=0.4),
        ),
        run_time=0.5,
    )
    s.wait(0.3)

    # Dots snap to peak
    dots = VGroup(
        *[
            Dot(point=image_corner_to_manim(cx, cy, img), radius=0.09, color=CORNER_COLORS[label])
            for (cx, cy), label in zip(corners_norm, CORNER_LABELS)
        ]
    )
    dot_labels = VGroup(
        *[
            Text(label, font_size=16, color=CORNER_COLORS[label]).next_to(
                Dot(point=image_corner_to_manim(cx, cy, img)), UP * 0.4 + RIGHT * 0.3, buff=0.05
            )
            for (cx, cy), label in zip(corners_norm, CORNER_LABELS)
        ]
    )
    s.play(*[Transform(b, d) for b, d in zip(blobs, dots)], run_time=0.8)
    s.play(LaggedStart(*[FadeIn(l) for l in dot_labels], lag_ratio=0.15), run_time=0.6)

    polygon = Polygon(
        *[image_corner_to_manim(cx, cy, img) for cx, cy in corners_norm],
        color=HIGHLIGHT,
        stroke_width=2.5,
        fill_opacity=0,
    )
    s.play(Create(polygon), run_time=0.8)

    s.play(
        Transform(
            caption,
            Text("Corners located — card boundary defined", font_size=22, color=DIM_WHITE).to_edge(
                DOWN, buff=0.4
            ),
        ),
        run_time=0.4,
    )

    # Sharpness readout
    readout = VGroup(
        Text("Sharpness", font_size=20, color=DIM_WHITE),
        Text(f"{sharpness:.3f}", font_size=32, color=HIGHLIGHT),
        Text("(confidence gate)", font_size=16, color=DIM_WHITE),
    ).arrange(DOWN, buff=0.1)
    readout.next_to(img, RIGHT, buff=0.5).shift(UP * 0.8)

    gauge_bg = Rectangle(
        width=2.2, height=0.22, fill_color=DARK_GRAY, fill_opacity=1, stroke_width=0
    )
    fill_frac = min(sharpness / 0.10, 1.0)
    gauge_fill = Rectangle(
        width=2.2 * fill_frac, height=0.22, fill_color=HIGHLIGHT, fill_opacity=1, stroke_width=0
    )
    gauge_fill.align_to(gauge_bg, LEFT)
    threshold = Line(
        gauge_bg.get_left() + RIGHT * 2.2 * 0.2,
        gauge_bg.get_left() + RIGHT * 2.2 * 0.2 + UP * 0.38,
        color=YELLOW,
        stroke_width=1.5,
    )
    thresh_lbl = Text("min", font_size=12, color=YELLOW).next_to(threshold, UP, buff=0.04)

    gauge_group = VGroup(gauge_bg, gauge_fill, threshold, thresh_lbl)
    gauge_group.next_to(readout, DOWN, buff=0.3)

    s.play(FadeIn(readout), run_time=0.5)
    s.play(FadeIn(gauge_bg), GrowFromEdge(gauge_fill, LEFT), run_time=0.6)
    s.play(Create(threshold), FadeIn(thresh_lbl), run_time=0.4)
    s.wait(1.2)


def play_dewarp(s: Scene) -> None:
    corners_norm = DATA["corners_normalized"]

    title = Text("Step 2 — Perspective Dewarp", font_size=32, color=ACCENT)
    title.to_edge(UP, buff=0.3)
    s.play(Write(title), run_time=0.7)

    scale = 3.8
    aspect = 960 / 1280

    def norm_to_left(cx, cy):
        return np.array([-3.5 + cx * scale, 1.5 - cy * scale * aspect, 0])

    quad_pts = [norm_to_left(cx, cy) for cx, cy in corners_norm]
    polygon = Polygon(
        *quad_pts, stroke_color=HIGHLIGHT, stroke_width=2.5, fill_color=BLUE_E, fill_opacity=0.15
    )
    corner_dots = VGroup(
        *[
            Dot(norm_to_left(cx, cy), radius=0.09, color=CORNER_COLORS[lbl])
            for (cx, cy), lbl in zip(corners_norm, CORNER_LABELS)
        ]
    )
    corner_lbls = VGroup(
        *[
            Text(lbl, font_size=15, color=CORNER_COLORS[lbl]).next_to(
                Dot(norm_to_left(cx, cy)), UP * 0.3, buff=0.05
            )
            for (cx, cy), lbl in zip(corners_norm, CORNER_LABELS)
        ]
    )

    dewarped = _img_or_rect(DEWARPED_IMAGE, 3.8)
    dewarped.move_to(RIGHT * 3.0)

    card_label = Text(
        "448 × 448 px\nembedder input", font_size=18, color=DIM_WHITE, line_spacing=1.2
    )
    card_label.next_to(dewarped, DOWN, buff=0.2)

    arrow = Arrow(
        LEFT * 0.4, RIGHT * 1.2, color=ACCENT, stroke_width=3, max_tip_length_to_length_ratio=0.15
    )
    xform_text = Text("Perspective Transform", font_size=20, color=DIM_WHITE)
    xform_text.next_to(arrow, UP, buff=0.1)

    s.play(Create(polygon), Create(corner_dots), run_time=0.8)
    s.play(FadeIn(corner_lbls), run_time=0.4)
    s.wait(0.3)

    s.play(GrowArrow(arrow), FadeIn(xform_text), run_time=0.6)

    target = Rectangle(
        width=dewarped.width,
        height=dewarped.height,
        stroke_color=HIGHLIGHT,
        stroke_width=2.5,
        fill_color=BLUE_E,
        fill_opacity=0.15,
    ).move_to(dewarped.get_center())

    s.play(Transform(polygon, target), FadeOut(corner_dots), FadeOut(corner_lbls), run_time=1.2)
    s.play(FadeIn(dewarped), run_time=0.6)
    s.play(FadeIn(card_label), run_time=0.4)

    caption = Text(
        "Card is now flat, front-facing, and ready for recognition", font_size=22, color=DIM_WHITE
    ).to_edge(DOWN, buff=0.4)
    s.play(FadeIn(caption), run_time=0.5)
    s.wait(1.5)


def play_embed(s: Scene) -> None:
    query_emb = DATA["query_embedding"]

    title = Text("Step 3 — Embedding", font_size=32, color=ACCENT)
    title.to_edge(UP, buff=0.3)
    s.play(Write(title), run_time=0.7)

    card_img = _img_or_rect(DEWARPED_IMAGE, 2.8)
    card_img.move_to(LEFT * 4.5 + UP * 0.3)
    card_lbl = Text(
        "Card image\n448 × 448 px", font_size=17, color=DIM_WHITE, line_spacing=1.2
    ).next_to(card_img, DOWN, buff=0.15)
    s.play(FadeIn(card_img), FadeIn(card_lbl), run_time=0.6)

    # Neural net funnel
    funnel = (
        VGroup(
            *[
                Rectangle(
                    width=w,
                    height=0.28,
                    fill_color=interpolate_color(BLUE_D, BLUE_A, i / 4),
                    fill_opacity=0.7,
                    stroke_color=ACCENT,
                    stroke_width=1,
                )
                for i, w in enumerate([1.8, 1.4, 1.0, 0.7, 0.45])
            ]
        )
        .arrange(DOWN, buff=0.15)
        .move_to(LEFT * 1.8 + UP * 0.3)
    )
    nn_lbl = Text("MobileViT-XXS\n(Milo)", font_size=18, color=DIM_WHITE, line_spacing=1.2).next_to(
        funnel, DOWN, buff=0.2
    )

    arrow_in = Arrow(
        card_img.get_right(),
        funnel.get_left(),
        color=ACCENT,
        stroke_width=2.5,
        max_tip_length_to_length_ratio=0.12,
    )
    s.play(GrowArrow(arrow_in), run_time=0.5)
    s.play(LaggedStart(*[FadeIn(l) for l in funnel], lag_ratio=0.1), FadeIn(nn_lbl), run_time=0.8)
    s.wait(0.3)

    output_lbl = Text("128 numbers", font_size=28, color=HIGHLIGHT).move_to(RIGHT * 2.0 + UP * 1.6)
    dim_note = Text("(L2-normalised float16 vector)", font_size=17, color=DIM_WHITE).next_to(
        output_lbl, DOWN, buff=0.1
    )
    arrow_out = Arrow(
        funnel.get_right(),
        output_lbl.get_left() + LEFT * 0.2,
        color=ACCENT,
        stroke_width=2.5,
        max_tip_length_to_length_ratio=0.12,
    )
    s.play(GrowArrow(arrow_out), run_time=0.5)
    s.play(Write(output_lbl), FadeIn(dim_note), run_time=0.6)
    s.wait(0.3)

    barcode = make_barcode(query_emb, width=5.5, height=0.45, n_show=64)
    barcode.move_to(RIGHT * 2.0 + UP * 0.3)
    bc_lbl = Text('Embedding "fingerprint"', font_size=19, color=DIM_WHITE).next_to(
        barcode, DOWN, buff=0.18
    )
    arrow_bc = Arrow(
        output_lbl.get_bottom() + DOWN * 0.05,
        barcode.get_top() + UP * 0.05,
        color=DIM_WHITE,
        stroke_width=1.5,
        max_tip_length_to_length_ratio=0.12,
    )
    s.play(GrowArrow(arrow_bc), run_time=0.4)
    s.play(
        LaggedStart(*[FadeIn(c) for c in barcode[0]], lag_ratio=0.01),
        FadeIn(barcode[1]),
        run_time=1.0,
    )
    s.play(FadeIn(bc_lbl), run_time=0.4)

    insight = Text(
        "Same card → similar fingerprint,\nregardless of lighting or angle",
        font_size=21,
        color=YELLOW,
        line_spacing=1.3,
    ).to_edge(DOWN, buff=0.4)
    s.play(Write(insight), run_time=0.8)
    s.wait(1.5)


def play_search(s: Scene) -> None:
    query_emb = DATA["query_embedding"]
    top_hits = DATA["top_hits"]
    hit_embs = DATA.get("hit_embeddings", {})

    title = Text("Step 3b — Nearest-Neighbor Search", font_size=29, color=ACCENT)
    title.to_edge(UP, buff=0.3)
    caption = Text(
        "Query fingerprint vs. 108,000 catalog embeddings", font_size=20, color=DIM_WHITE
    ).next_to(title, DOWN, buff=0.2)
    s.play(Write(title), FadeIn(caption), run_time=0.8)

    # Query barcode — left panel
    q_bc = make_barcode(query_emb, width=1.8, height=0.35, n_show=64)
    q_bc.move_to(LEFT * 5.3 + DOWN * 0.2)
    q_lbl = Text("Query", font_size=18, color=WHITE).next_to(q_bc, UP, buff=0.12)
    s.play(
        LaggedStart(*[FadeIn(c) for c in q_bc[0]], lag_ratio=0.005),
        FadeIn(q_bc[1]),
        FadeIn(q_lbl),
        run_time=0.7,
    )

    # Arrow from query panel
    arrow = CurvedArrow(
        q_bc.get_right() + RIGHT * 0.05, LEFT * 1.8 + UP * 1.6, color=ACCENT, angle=-TAU / 8
    )
    s.play(Create(arrow), run_time=0.5)

    # Catalog rows
    row_start_y = 1.6
    row_height = 0.9

    for i, (score, cid) in enumerate(top_hits):
        y = row_start_y - i * row_height
        is_top = i == 0

        if cid in hit_embs:
            cat_emb = hit_embs[cid]
        else:
            rng = np.random.default_rng(abs(hash(cid)) % 2**31)
            cat_emb = np.array(query_emb[:128]) * (0.9 if is_top else 0.3) + rng.normal(0, 0.5, 128)
            cat_emb = (cat_emb / (np.linalg.norm(cat_emb) + 1e-9)).tolist()

        q_strip = make_barcode(query_emb, width=3.5, height=0.14, n_show=64).move_to(
            RIGHT * 0.3 + UP * (y + 0.17)
        )
        c_strip = make_barcode(cat_emb, width=3.5, height=0.14, n_show=64).move_to(
            RIGHT * 0.3 + UP * (y - 0.01)
        )

        score_color = HIGHLIGHT if is_top else WHITE
        score_text = Text(
            f"{score:.3f}", font_size=16 if is_top else 14, color=score_color
        ).move_to(RIGHT * 2.4 + UP * y)
        rank_text = Text(f"#{i + 1}", font_size=14, color=DIM_WHITE).move_to(LEFT * 1.55 + UP * y)
        id_text = Text(cid[:8] + "…", font_size=11, color=DIM_WHITE).move_to(LEFT * 0.6 + UP * y)

        anims = [
            FadeIn(q_strip),
            FadeIn(c_strip),
            FadeIn(score_text),
            FadeIn(rank_text),
            FadeIn(id_text),
        ]

        if is_top:
            bg = Rectangle(
                width=9.0,
                height=row_height - 0.1,
                fill_color=ManimColor("#002200"),
                fill_opacity=0.7,
                stroke_color=HIGHLIGHT,
                stroke_width=1.5,
            ).move_to(RIGHT * 0.5 + UP * y)
            s.play(FadeIn(bg), run_time=0.15)

        s.play(*anims, run_time=0.25)

    s.wait(0.4)
    top_score = top_hits[0][0]
    note = Text(
        f"Score: {top_score:.3f}  ← clear winner  (gap to #2: {top_score - top_hits[1][0]:.3f})",
        font_size=20,
        color=HIGHLIGHT,
    ).to_edge(DOWN, buff=0.4)
    s.play(Write(note), run_time=0.7)
    s.wait(1.5)


def play_result(s: Scene) -> None:
    top_score = DATA["top_hits"][0][0] if DATA["top_hits"] else 0.846

    orig = _img_or_rect(
        ORIGINAL_IMAGE if ORIGINAL_IMAGE.exists() else SAMPLE_IMAGE, 3.4, fill_color=GRAY
    )
    orig.move_to(LEFT * 4.2)

    arrow = Arrow(
        LEFT * 1.8, RIGHT * 0.2, color=ACCENT, stroke_width=3, max_tip_length_to_length_ratio=0.1
    )

    card_name = Text("Scrying Glass", font_size=40, color=HIGHLIGHT, weight=BOLD)
    card_meta = Text("Urza's Destiny  ·  rk post", font_size=22, color=DIM_WHITE)
    score_grp = VGroup(
        Text("Cosine similarity", font_size=16, color=DIM_WHITE),
        Text(f"{top_score:.3f}", font_size=36, color=HIGHLIGHT),
    ).arrange(DOWN, buff=0.05)

    result = VGroup(card_name, card_meta, score_grp).arrange(DOWN, buff=0.25).move_to(RIGHT * 2.5)

    s.play(FadeIn(orig), run_time=0.6)
    s.play(GrowArrow(arrow), run_time=0.5)
    s.play(Write(card_name), run_time=0.7)
    s.play(FadeIn(card_meta), FadeIn(score_grp), run_time=0.5)
    s.wait(0.5)

    timing = Text("Identified in < 1 second on CPU", font_size=20, color=DIM_WHITE).to_edge(
        DOWN, buff=0.6
    )
    footer = Text(
        "collector_vision · pip install collector-vision", font_size=18, color=ACCENT
    ).next_to(timing, DOWN, buff=0.2)
    s.play(FadeIn(timing), run_time=0.4)
    s.play(Write(footer), run_time=0.6)
    s.wait(2.0)


# ---------------------------------------------------------------------------
# Individual Scene classes (for iterating on single scenes)
# ---------------------------------------------------------------------------


class _Base(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        self._play_fn(self)


class TitleScene(_Base):
    _play_fn = staticmethod(play_title)


class CornerScene(_Base):
    _play_fn = staticmethod(play_corner)


class DewarpScene(_Base):
    _play_fn = staticmethod(play_dewarp)


class EmbedScene(_Base):
    _play_fn = staticmethod(play_embed)


class SearchScene(_Base):
    _play_fn = staticmethod(play_search)


class ResultScene(_Base):
    _play_fn = staticmethod(play_result)


# ---------------------------------------------------------------------------
# Full pipeline: all scenes in sequence
# ---------------------------------------------------------------------------


class PipelineExplained(Scene):
    """Full ~90-second pipeline animation."""

    def construct(self):
        self.camera.background_color = BG_COLOR

        for fn in [play_title, play_corner, play_dewarp, play_embed, play_search, play_result]:
            fn(self)
            self.play(FadeOut(*self.mobjects), run_time=0.4)
