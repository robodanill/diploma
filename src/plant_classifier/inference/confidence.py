from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from typing import TypeVar

LabelKey = TypeVar("LabelKey", bound=Hashable)


def normalize_confidences(
    scores: Mapping[LabelKey, float],
    temperature: float = 2.0,
) -> dict[LabelKey, float]:
    """Convert relative ranking scores into a deterministic confidence distribution.

    The result is rank-preserving and sums to one, but it is not a statistically
    calibrated probability estimate.
    """

    if temperature <= 0:
        raise ValueError("Confidence temperature must be greater than zero")
    if not scores:
        return {}

    values = [float(score) for score in scores.values()]
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    standard_deviation = math.sqrt(variance)
    if standard_deviation <= 1e-12:
        uniform = 1.0 / count
        return {key: uniform for key in scores}

    maximum = max(values)
    scale = standard_deviation * temperature
    weights = {
        key: math.exp(max(-50.0, (float(score) - maximum) / scale))
        for key, score in scores.items()
    }
    total = sum(weights.values())
    return {key: weight / total for key, weight in weights.items()}
