from __future__ import annotations

import hashlib
from pathlib import Path

from plant_classifier.inference.types import ImagePrediction, PredictionLabel


class StubPredictor:
    """Deterministic predictor used until the trained S-CNN is connected."""

    _classes: tuple[tuple[str, str, str], ...] = (
        ("Rosaceae", "Prunus", "avium"),
        ("Rosaceae", "Prunus", "cerasus"),
        ("Rosaceae", "Crataegus", "monogyna"),
        ("Rosaceae", "Sorbus", "torminalis"),
        ("Salicaceae", "Populus", "alba"),
        ("Salicaceae", "Populus", "nigra"),
        ("Salicaceae", "Salix", "cinerea"),
        ("Fagaceae", "Quercus", "cerris"),
        ("Fagaceae", "Quercus", "rubra"),
        ("Betulaceae", "Betula", "pendula"),
    )

    def predict_many(self, image_paths: list[Path], top_k: int = 5) -> list[ImagePrediction]:
        return [self._predict_one(path, top_k=top_k) for path in image_paths]

    def _predict_one(self, image_path: Path, top_k: int) -> ImagePrediction:
        if not image_path.exists():
            return ImagePrediction(image_path=image_path, labels=(), error="Файл не существует")

        digest = hashlib.sha256(str(image_path.resolve()).encode("utf-8")).digest()
        start = digest[0] % len(self._classes)
        labels: list[PredictionLabel] = []

        for rank in range(min(top_k, len(self._classes))):
            family, genus, species = self._classes[(start + rank) % len(self._classes)]
            score = max(0.05, 0.92 - rank * 0.13 - (digest[rank + 1] % 7) / 100)
            labels.append(
                PredictionLabel(
                    family=family,
                    genus=genus,
                    species=species,
                    score=round(score, 3),
                )
            )

        genus_labels: list[PredictionLabel] = []
        seen_genera: set[str] = set()
        for label in labels:
            if label.genus in seen_genera:
                continue
            seen_genera.add(label.genus)
            genus_labels.append(
                PredictionLabel(
                    family=label.family,
                    genus=label.genus,
                    species="",
                    score=label.score,
                )
            )

        return ImagePrediction(
            image_path=image_path,
            labels=tuple(labels),
            genus_labels=tuple(genus_labels),
        )
