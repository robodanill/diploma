from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PredictionLabel:
    """Single ranked class hypothesis."""

    family: str
    genus: str
    species: str
    score: float

    @property
    def display_name(self) -> str:
        if self.species.lower().startswith(self.genus.lower()):
            return self.species
        return f"{self.genus} {self.species}"


@dataclass(frozen=True)
class ImagePrediction:
    """Prediction result for one image."""

    image_path: Path
    labels: tuple[PredictionLabel, ...]
    error: str | None = None
    genus_labels: tuple[PredictionLabel, ...] = ()

    @property
    def top_label(self) -> PredictionLabel | None:
        return self.labels[0] if self.labels else None
