from __future__ import annotations

from pathlib import Path
from typing import Protocol

from plant_classifier.inference.types import ImagePrediction


class Predictor(Protocol):
    """Common interface for stub and trained model predictors."""

    def predict_many(self, image_paths: list[Path], top_k: int = 5) -> list[ImagePrediction]:
        """Return ranked predictions for each image path."""

