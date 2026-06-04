import pytest

pytest.importorskip("PIL")

from PIL import Image

from plant_classifier.preprocessing.views import LeafBoundingBoxCrop, LeafInteriorCrop, center_crop_box


def test_leaf_bbox_crop_removes_thin_stem_before_bbox() -> None:
    image = Image.new("RGB", (30, 30), "white")
    pixels = image.load()
    for x in range(10, 20):
        for y in range(8, 18):
            pixels[x, y] = (0, 0, 0)
    for y in range(18, 29):
        pixels[15, y] = (0, 0, 0)

    cropped = LeafBoundingBoxCrop(padding=0, morphology_size=5)(image)

    assert cropped.size == (10, 10)


def test_center_crop_box_uses_smallest_side_when_crop_is_larger_than_image() -> None:
    assert center_crop_box(width=20, height=10, crop_size=32) == (5, 0, 15, 10)


def test_leaf_interior_crop_uses_border_background_to_avoid_empty_center() -> None:
    image = Image.new("RGB", (96, 96), (226, 226, 218))
    pixels = image.load()
    for x in range(8, 44):
        for y in range(32, 68):
            pixels[x, y] = (35, 135, 45)

    cropped = LeafInteriorCrop(crop_size=32, step=4)(image)

    assert _green_fraction(cropped) > 0.80


def test_leaf_interior_crop_prefers_leaf_patch_over_thin_stem() -> None:
    image = Image.new("RGB", (96, 96), (226, 226, 218))
    pixels = image.load()
    for y in range(0, 96):
        for x in range(47, 51):
            pixels[x, y] = (45, 115, 40)
    for x in range(8, 44):
        for y in range(32, 68):
            pixels[x, y] = (35, 135, 45)

    cropped = LeafInteriorCrop(crop_size=32, step=4)(image)

    assert _green_fraction(cropped) > 0.80


def _green_fraction(image: Image.Image) -> float:
    pixels = list(image.convert("RGB").getdata())
    green_pixels = sum(1 for red, green, blue in pixels if green > red + 40 and green > blue + 40)
    return green_pixels / len(pixels)
