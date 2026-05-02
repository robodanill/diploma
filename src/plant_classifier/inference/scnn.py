from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor

from plant_classifier.inference.types import ImagePrediction, PredictionLabel
from plant_classifier.models.siamese import BackboneSpec, SiameseNetwork, build_siamese_network
from plant_classifier.training.image_pairs import build_image_transform


@dataclass(frozen=True)
class ReferenceEmbedding:
    image_path: Path
    family: str
    genus: str
    species: str
    global_embedding: Tensor
    local_embedding: Tensor


class TwoStageSiamesePredictor:
    """Two-view genus-to-species predictor based on trained Siamese networks."""

    def __init__(
        self,
        genus_model: SiameseNetwork,
        species_model: SiameseNetwork,
        references: list[ReferenceEmbedding],
        genus_candidates: int = 30,
        top_k: int = 5,
        image_size: int = 224,
        local_crop_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.genus_model = genus_model.to(self.device).eval()
        self.species_model = species_model.to(self.device).eval()
        self.references = references
        self.genus_candidates = genus_candidates
        self.top_k = top_k
        self.global_transform = build_image_transform("global", image_size=image_size, crop_size=local_crop_size)
        self.local_transform = build_image_transform("local", image_size=image_size, crop_size=local_crop_size)

    @classmethod
    def from_artifacts(
        cls,
        genus_checkpoint: Path,
        species_checkpoint: Path,
        reference_index: Path,
        backbone: str = "vgg16",
        pretrained: bool = False,
        **kwargs,
    ) -> "TwoStageSiamesePredictor":
        device = torch.device(kwargs.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        genus_model = build_siamese_network(BackboneSpec(name=backbone, pretrained=pretrained))
        species_model = build_siamese_network(BackboneSpec(name=backbone, pretrained=pretrained))
        genus_model.load_state_dict(torch.load(genus_checkpoint, map_location=device))
        species_model.load_state_dict(torch.load(species_checkpoint, map_location=device))
        references = load_reference_index(reference_index, map_location=device)
        return cls(genus_model=genus_model, species_model=species_model, references=references, **kwargs)

    def predict_many(self, image_paths: list[Path], top_k: int | None = None) -> list[ImagePrediction]:
        return [self._predict_one(path, top_k=top_k or self.top_k) for path in image_paths]

    @torch.inference_mode()
    def _predict_one(self, image_path: Path, top_k: int) -> ImagePrediction:
        if not image_path.exists():
            return ImagePrediction(image_path=image_path, labels=(), error="File does not exist")

        try:
            image = _load_rgb(image_path)
            global_query = self.genus_model.embed(_prepare(image, self.global_transform, self.device))
            local_query = self.species_model.embed(_prepare(image, self.local_transform, self.device))

            genus_scores = self._rank_genus_references(global_query)
            selected_genus = [reference.genus for reference, _ in genus_scores[: self.genus_candidates]]
            genus_weights = Counter(selected_genus)
            candidate_genera = set(genus_weights)

            species_scores: dict[tuple[str, str, str], float] = {}
            for reference in self.references:
                if reference.genus not in candidate_genera:
                    continue
                score = self._similarity(
                    self.species_model,
                    local_query,
                    reference.local_embedding.to(self.device).unsqueeze(0),
                )
                weighted_score = score * genus_weights[reference.genus] / max(1, self.genus_candidates)
                key = (reference.family, reference.genus, reference.species)
                species_scores[key] = max(species_scores.get(key, 0.0), weighted_score)

            labels = [
                PredictionLabel(family=family, genus=genus, species=species, score=score)
                for (family, genus, species), score in sorted(
                    species_scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:top_k]
            ]
            return ImagePrediction(image_path=image_path, labels=tuple(labels))
        except Exception as exc:  # pragma: no cover - inference boundary
            return ImagePrediction(image_path=image_path, labels=(), error=str(exc))

    def _rank_genus_references(self, query: Tensor) -> list[tuple[ReferenceEmbedding, float]]:
        scores = [
            (
                reference,
                self._similarity(
                    self.genus_model,
                    query,
                    reference.global_embedding.to(self.device).unsqueeze(0),
                ),
            )
            for reference in self.references
        ]
        return sorted(scores, key=lambda item: item[1], reverse=True)

    @staticmethod
    def _similarity(model: SiameseNetwork, query: Tensor, reference: Tensor) -> float:
        distance = torch.abs(query - reference)
        return float(model.comparator(distance).flatten().item())


def save_reference_index(references: list[ReferenceEmbedding], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        [
            {
                "image_path": str(reference.image_path),
                "family": reference.family,
                "genus": reference.genus,
                "species": reference.species,
                "global_embedding": reference.global_embedding.cpu(),
                "local_embedding": reference.local_embedding.cpu(),
            }
            for reference in references
        ],
        output_path,
    )


def load_reference_index(reference_index: Path, map_location: torch.device | str = "cpu") -> list[ReferenceEmbedding]:
    payload = torch.load(reference_index, map_location=map_location)
    return [
        ReferenceEmbedding(
            image_path=Path(item["image_path"]),
            family=item["family"],
            genus=item["genus"],
            species=item["species"],
            global_embedding=item["global_embedding"],
            local_embedding=item["local_embedding"],
        )
        for item in payload
    ]


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _prepare(image: Image.Image, transform, device: torch.device) -> Tensor:
    return transform(image).unsqueeze(0).to(device)

