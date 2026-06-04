from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROBABILITY_TEMPERATURE = 4.0


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


def normalize_label_scores(
    labels: tuple[PredictionLabel, ...] | list[PredictionLabel],
    temperature: float = PROBABILITY_TEMPERATURE,
) -> tuple[PredictionLabel, ...]:
    """Convert positive ranking scores to display probabilities without changing rank order."""

    if not labels:
        return ()
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    adjusted_scores = [
        max(label.score, 0.0) ** (1.0 / temperature)
        for label in labels
    ]
    total = sum(adjusted_scores)
    if total <= 0:
        probability = 1.0 / len(labels)
        return tuple(
            PredictionLabel(
                family=label.family,
                genus=label.genus,
                species=label.species,
                score=probability,
            )
            for label in labels
        )

    return tuple(
        PredictionLabel(
            family=label.family,
            genus=label.genus,
            species=label.species,
            score=score / total,
        )
        for label, score in zip(labels, adjusted_scores, strict=True)
    )
