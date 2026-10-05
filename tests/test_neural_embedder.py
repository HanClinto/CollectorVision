from __future__ import annotations

from PIL import Image

from collector_vision.embedders.neural import _ensure_portrait, _preprocess_pil


def test_ensure_portrait_rotates_landscape_counterclockwise() -> None:
    image = Image.new("RGB", (3, 2))
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((2, 0), (0, 255, 0))
    image.putpixel((0, 1), (0, 0, 255))
    image.putpixel((2, 1), (255, 255, 0))

    portrait = _ensure_portrait(image)
    try:
        assert portrait.size == (2, 3)
        assert portrait.getpixel((0, 0)) == (0, 255, 0)
        assert portrait.getpixel((1, 2)) == (0, 0, 255)
    finally:
        portrait.close()
        image.close()


def test_ensure_portrait_preserves_portrait_and_square_images() -> None:
    portrait = Image.new("RGB", (2, 3))
    square = Image.new("RGB", (2, 2))
    try:
        assert _ensure_portrait(portrait) is portrait
        assert _ensure_portrait(square) is square
    finally:
        portrait.close()
        square.close()


def test_preprocess_keeps_caller_owned_image_open() -> None:
    image = Image.new("RGB", (3, 2), (10, 20, 30))
    try:
        output = _preprocess_pil(image, 2)

        assert output.shape == (1, 3, 2, 2)
        assert image.getpixel((0, 0)) == (10, 20, 30)
    finally:
        image.close()
