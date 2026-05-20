import pytest

pytest.importorskip("PIL")

from PIL import Image

from plant_classifier.preprocessing.views import LeafBoundingBoxCrop, center_crop_box


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
