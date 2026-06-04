from __future__ import annotations

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps


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
    """Crop to the Otsu foreground after top-hat-style morphology cleanup."""

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


class LeafInteriorCrop:
    """Select a local crop that lies inside the detected leaf region."""

    def __init__(
        self,
        crop_size: int = 32,
        step: int = 8,
        morphology_size: int = 5,
        boundary_margin: int = 3,
    ) -> None:
        if crop_size <= 0:
            raise ValueError("Crop size must be positive")
        self.crop_size = int(crop_size)
        self.step = max(1, int(step))
        size = max(1, int(morphology_size))
        self.morphology_size = size if size % 2 == 1 else size + 1
        self.boundary_margin = max(0, int(boundary_margin))

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        crop_size = min(self.crop_size, rgb.width, rgb.height)
        if crop_size <= 0:
            return rgb

        center_box = center_crop_box(rgb.width, rgb.height, crop_size)
        leaf_mask, green_mask, background_mask, otsu_mask = _leaf_region_masks(
            rgb,
            morphology_size=self.morphology_size,
            boundary_margin=self.boundary_margin,
        )
        interior_mask = _erode_mask(leaf_mask, self.boundary_margin)
        integrals = {
            "leaf": _integral_mask(leaf_mask),
            "interior": _integral_mask(interior_mask),
            "green": _integral_mask(green_mask),
            "background": _integral_mask(background_mask),
            "otsu": _integral_mask(otsu_mask),
        }
        center_left, center_top = center_box[0], center_box[1]

        candidates: list[tuple[int, int, dict[str, float]]] = []
        max_left = rgb.width - crop_size
        max_top = rgb.height - crop_size
        for top in range(0, max_top + 1, self.step):
            for left in range(0, max_left + 1, self.step):
                score = _local_crop_score(integrals, left, top, crop_size)
                if score["background_fraction"] >= 0.95 and score["leaf_fraction"] < 0.25:
                    continue
                center_distance = (
                    (left - center_left) ** 2 + (top - center_top) ** 2
                ) ** 0.5
                candidates.append(
                    (left, top, {**score, "score": score["score"] - 0.0005 * center_distance})
                )

        if not candidates:
            best_left, best_top = center_box[0], center_box[1]
        else:
            best_left, best_top, _best_score = _select_best_local_crop(candidates)

        return rgb.crop((best_left, best_top, best_left + crop_size, best_top + crop_size))


def _leaf_mask(image: Image.Image, morphology_size: int) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    threshold = _otsu_threshold(grayscale.histogram())
    dark_foreground = grayscale.point(lambda value: 255 if value <= threshold else 0, mode="L")
    light_foreground = grayscale.point(lambda value: 255 if value > threshold else 0, mode="L")
    mask = _choose_foreground_mask(
        dark_foreground,
        light_foreground,
        grayscale.width * grayscale.height,
    )
    if morphology_size > 1:
        mask = _remove_tophat_artifacts(mask, morphology_size)
    return mask


def _remove_tophat_artifacts(mask: Image.Image, morphology_size: int) -> Image.Image:
    opened = mask.filter(ImageFilter.MinFilter(morphology_size))
    opened = opened.filter(ImageFilter.MaxFilter(morphology_size))
    top_hat = ImageChops.subtract(mask, opened)
    return ImageChops.subtract(mask, top_hat)


def _choose_foreground_mask(
    dark_mask: Image.Image,
    light_mask: Image.Image,
    pixel_count: int,
) -> Image.Image:
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


def _leaf_region_masks(
    image: Image.Image,
    morphology_size: int,
    boundary_margin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(image)
    white_mask = _near_white_mask(array)
    green_candidate_mask = _green_leaf_mask(array)
    border_background_mask = _border_background_mask(array) & ~green_candidate_mask
    background_mask = white_mask | border_background_mask
    green_mask = green_candidate_mask & ~background_mask
    otsu_mask = np.asarray(_leaf_mask(image, morphology_size)) > 0
    leaf_mask = (otsu_mask & ~background_mask) | green_mask

    leaf_image = _mask_to_image(leaf_mask)
    if boundary_margin > 0:
        close_size = 2 * boundary_margin + 1
        leaf_image = leaf_image.filter(ImageFilter.MaxFilter(close_size))
        leaf_image = leaf_image.filter(ImageFilter.MinFilter(close_size))
    leaf_mask = np.asarray(leaf_image) > 0

    if leaf_mask.mean() < 0.01:
        leaf_mask = green_mask if green_mask.mean() >= 0.01 else ~background_mask
    return leaf_mask, green_mask, background_mask, otsu_mask


def _near_white_mask(array: np.ndarray) -> np.ndarray:
    return (array[:, :, 0] >= 245) & (array[:, :, 1] >= 245) & (array[:, :, 2] >= 245)


def _border_background_mask(array: np.ndarray) -> np.ndarray:
    height, width, _channels = array.shape
    border_width = max(1, min(height, width) // 40)
    border = np.concatenate(
        [
            array[:border_width, :, :].reshape(-1, 3),
            array[-border_width:, :, :].reshape(-1, 3),
            array[:, :border_width, :].reshape(-1, 3),
            array[:, -border_width:, :].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.int16)
    background_color = np.median(border, axis=0)
    border_distance = np.linalg.norm(border - background_color, axis=1)
    threshold = float(np.clip(np.percentile(border_distance, 80) + 4.0, 18.0, 45.0))
    distance = np.linalg.norm(array.astype(np.int16) - background_color, axis=2)
    return distance <= threshold


def _green_leaf_mask(array: np.ndarray) -> np.ndarray:
    rgb = array.astype(np.int16)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    excess_green = 2 * green - red - blue
    return (excess_green > 10) & (green > 35) & (green >= red - 5) & (green >= blue - 15)


def _erode_mask(mask: np.ndarray, margin: int) -> np.ndarray:
    if margin <= 0:
        return mask
    return np.asarray(_mask_to_image(mask).filter(ImageFilter.MinFilter(2 * margin + 1))) > 0


def _mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def _integral_mask(mask: np.ndarray) -> np.ndarray:
    integral = mask.astype(np.float32).cumsum(axis=0).cumsum(axis=1)
    return np.pad(integral, ((1, 0), (1, 0)), mode="constant")


def _window_sum(integral: np.ndarray, left: int, top: int, size: int) -> float:
    right = left + size
    bottom = top + size
    return float(
        integral[bottom, right]
        - integral[top, right]
        - integral[bottom, left]
        + integral[top, left]
    )


def _local_crop_score(
    integrals: dict[str, np.ndarray],
    left: int,
    top: int,
    size: int,
) -> dict[str, float]:
    area = float(size * size)
    leaf_fraction = _window_sum(integrals["leaf"], left, top, size) / area
    interior_fraction = _window_sum(integrals["interior"], left, top, size) / area
    green_fraction = _window_sum(integrals["green"], left, top, size) / area
    background_fraction = _window_sum(integrals["background"], left, top, size) / area
    otsu_fraction = _window_sum(integrals["otsu"], left, top, size) / area
    otsu_homogeneity = max(otsu_fraction, 1.0 - otsu_fraction)
    score = (
        1.20 * green_fraction
        + 1.05 * leaf_fraction
        + 0.95 * interior_fraction
        + 0.55 * otsu_homogeneity
        - 1.20 * background_fraction
    )
    return {
        "score": score,
        "leaf_fraction": leaf_fraction,
        "interior_fraction": interior_fraction,
        "green_fraction": green_fraction,
        "background_fraction": background_fraction,
        "white_fraction": background_fraction,
    }


def _select_best_local_crop(
    candidates: list[tuple[int, int, dict[str, float]]],
) -> tuple[int, int, dict[str, float]]:
    max_leaf_fraction = max(score["leaf_fraction"] for _left, _top, score in candidates)
    max_interior_fraction = max(score["interior_fraction"] for _left, _top, score in candidates)
    leaf_floor = max(0.40, min(0.92, max_leaf_fraction - 0.05))
    interior_floor = max(0.05, min(0.55, max_interior_fraction - 0.10))
    preferred = [
        candidate
        for candidate in candidates
        if candidate[2]["leaf_fraction"] >= leaf_floor
        and candidate[2]["interior_fraction"] >= interior_floor
        and candidate[2]["background_fraction"] <= 0.45
    ]
    if not preferred:
        preferred = [
            candidate
            for candidate in candidates
            if candidate[2]["leaf_fraction"] >= leaf_floor
            and candidate[2]["background_fraction"] <= 0.65
        ]
    if not preferred:
        preferred = candidates
    return max(preferred, key=lambda item: item[2]["score"])


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
