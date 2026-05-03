from __future__ import annotations

from PIL import Image, ImageFilter, ImageOps


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


class LeafBoundingBoxCrop:
    """Crop the image to a foreground leaf mask estimated with Otsu thresholding."""

    def __init__(self, padding: int = 4, morphology_size: int = 5) -> None:
        self.padding = max(0, int(padding))
        size = max(1, int(morphology_size))
        self.morphology_size = size if size % 2 == 1 else size + 1

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        mask = _leaf_mask(rgb, self.morphology_size)
        bbox = mask.getbbox()
        if bbox is None:
            return rgb
        left, top, right, bottom = bbox
        left = max(0, left - self.padding)
        top = max(0, top - self.padding)
        right = min(rgb.width, right + self.padding)
        bottom = min(rgb.height, bottom + self.padding)
        if right <= left or bottom <= top:
            return rgb
        return rgb.crop((left, top, right, bottom))


def _leaf_mask(image: Image.Image, morphology_size: int) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    threshold = _otsu_threshold(grayscale.histogram())
    dark_foreground = grayscale.point(lambda value: 255 if value <= threshold else 0, mode="L")
    light_foreground = grayscale.point(lambda value: 255 if value > threshold else 0, mode="L")
    mask = _choose_foreground_mask(dark_foreground, light_foreground, grayscale.width * grayscale.height)
    if morphology_size > 1:
        # Opening removes small disconnected objects before computing the leaf bbox.
        mask = mask.filter(ImageFilter.MinFilter(morphology_size))
        mask = mask.filter(ImageFilter.MaxFilter(morphology_size))
    return mask


def _choose_foreground_mask(dark_mask: Image.Image, light_mask: Image.Image, pixel_count: int) -> Image.Image:
    dark_count = _foreground_count(dark_mask)
    light_count = _foreground_count(light_mask)
    candidates = [
        (dark_mask, dark_count),
        (light_mask, light_count),
    ]
    valid = [
        (mask, count)
        for mask, count in candidates
        if 0.01 * pixel_count <= count <= 0.95 * pixel_count
    ]
    if valid:
        return min(valid, key=lambda item: item[1])[0]
    return dark_mask if dark_count <= light_count else light_mask


def _foreground_count(mask: Image.Image) -> int:
    histogram = mask.histogram()
    return histogram[255] if len(histogram) > 255 else 0


def _otsu_threshold(histogram: list[int]) -> int:
    total = sum(histogram)
    if total == 0:
        return 0

    sum_total = sum(index * count for index, count in enumerate(histogram))
    weight_background = 0
    sum_background = 0.0
    best_threshold = 0
    best_variance = -1.0

    for threshold, count in enumerate(histogram):
        weight_background += count
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += threshold * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold

    return best_threshold
