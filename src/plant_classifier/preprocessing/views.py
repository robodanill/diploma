from __future__ import annotations


def center_crop_box(width: int, height: int, crop_size: int) -> tuple[int, int, int, int]:
    """Return a centered square crop box as left, top, right, bottom."""

    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if crop_size <= 0:
        raise ValueError("Crop size must be positive")

    side = min(width, height, crop_size)
    left = (width - side) // 2
    top = (height - side) // 2
    return left, top, left + side, top + side

