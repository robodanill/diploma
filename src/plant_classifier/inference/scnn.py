from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

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
        genus_references: list[ReferenceEmbedding] | None = None,
        genus_candidates: int = 30,
        top_k: int = 5,
        image_size: int = 224,
        local_crop_size: int = 32,
        local_crop_position: str = "center",
        preprocessing: bool = False,
        genus_score_mode: str = "comparator",
        species_score_mode: str = "comparator",
        species_aggregation: str = "max",
        genus_candidate_mode: str = "reference",
        genus_weight_mode: str = "frequency",
        device: str | None = None,
    ) -> None:
        _validate_score_mode(genus_score_mode)
        _validate_score_mode(species_score_mode)
        _validate_species_aggregation(species_aggregation)
        _validate_genus_candidate_mode(genus_candidate_mode)
        _validate_genus_weight_mode(genus_weight_mode)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.genus_model = genus_model.to(self.device).eval()
        self.species_model = species_model.to(self.device).eval()
        self.references = references
        self.species_references = references
        self.genus_references = genus_references or references
        self.genus_candidates = genus_candidates
        self.top_k = top_k
        self.genus_score_mode = genus_score_mode
        self.species_score_mode = species_score_mode
        self.species_aggregation = species_aggregation
        self.genus_candidate_mode = genus_candidate_mode
        self.genus_weight_mode = genus_weight_mode
        self.global_transform = build_image_transform(
            "global",
            image_size=image_size,
            crop_size=local_crop_size,
            preprocessing=preprocessing,
        )
        self.local_transform = build_image_transform(
            "local",
            image_size=image_size,
            crop_size=local_crop_size,
            crop_position=local_crop_position,
            preprocessing=preprocessing,
        )

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
        genus_references, species_references = load_two_stage_reference_index(
            reference_index,
            map_location=device,
        )
        return cls(
            genus_model=genus_model,
            species_model=species_model,
            references=species_references,
            genus_references=genus_references,
            **kwargs,
        )

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
            genus_weights = _select_genus_weights(
                ranked_genus_scores=genus_scores,
                genus_candidates=self.genus_candidates,
                candidate_mode=self.genus_candidate_mode,
                weight_mode=self.genus_weight_mode,
            )
            candidate_genera = set(genus_weights)

            species_reference_scores: dict[tuple[str, str, str], list[float]] = {}
            for reference in self.species_references:
                if reference.genus not in candidate_genera:
                    continue
                score = self._score(
                    self.species_model,
                    local_query,
                    reference.local_embedding.to(self.device).unsqueeze(0),
                    self.species_score_mode,
                )
                key = (reference.family, reference.genus, reference.species)
                species_reference_scores.setdefault(key, []).append(score)

            weight_denominator = max(1, sum(genus_weights.values()))
            species_scores = {
                key: (
                    _aggregate_scores(scores, self.species_aggregation)
                    * genus_weights[key[1]]
                    / weight_denominator
                )
                for key, scores in species_reference_scores.items()
            }

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
                self._score(
                    self.genus_model,
                    query,
                    reference.global_embedding.to(self.device).unsqueeze(0),
                    self.genus_score_mode,
                ),
            )
            for reference in self.genus_references
        ]
        return sorted(scores, key=lambda item: item[1], reverse=True)

    @staticmethod
    def _score(model: SiameseNetwork, query: Tensor, reference: Tensor, score_mode: str) -> float:
        distance = torch.abs(query - reference)
        if score_mode == "l1":
            return 1.0 / (1.0 + float(distance.sum().item()))
        return float(model.comparator(distance).flatten().item())


def _validate_score_mode(score_mode: str) -> None:
    if score_mode not in {"comparator", "l1"}:
        raise ValueError(f"Unsupported score mode: {score_mode}")


def _validate_species_aggregation(aggregation: str) -> None:
    if aggregation not in {"max", "mean", "sum"}:
        raise ValueError(f"Unsupported species aggregation: {aggregation}")


def _validate_genus_candidate_mode(mode: str) -> None:
    if mode not in {"reference", "unique"}:
        raise ValueError(f"Unsupported genus candidate mode: {mode}")


def _validate_genus_weight_mode(mode: str) -> None:
    if mode not in {"frequency", "score", "uniform"}:
        raise ValueError(f"Unsupported genus weight mode: {mode}")


def _select_genus_weights(
    ranked_genus_scores: list[tuple[ReferenceEmbedding, float]],
    genus_candidates: int,
    candidate_mode: str,
    weight_mode: str,
) -> dict[str, float]:
    if genus_candidates <= 0:
        return {}

    if candidate_mode == "reference":
        selected = ranked_genus_scores[:genus_candidates]
    elif candidate_mode == "unique":
        selected = []
        seen: set[str] = set()
        for reference, score in ranked_genus_scores:
            if reference.genus in seen:
                continue
            selected.append((reference, score))
            seen.add(reference.genus)
            if len(selected) >= genus_candidates:
                break
    else:
        raise ValueError(f"Unsupported genus candidate mode: {candidate_mode}")

    if weight_mode == "frequency":
        return dict(Counter(reference.genus for reference, _score in selected))
    if weight_mode == "uniform":
        return {reference.genus: 1.0 for reference, _score in selected}
    if weight_mode == "score":
        weights: dict[str, float] = {}
        for reference, score in selected:
            weights[reference.genus] = max(weights.get(reference.genus, 0.0), float(score))
        return weights
    raise ValueError(f"Unsupported genus weight mode: {weight_mode}")


def _aggregate_scores(scores: list[float], aggregation: str) -> float:
    if aggregation == "max":
        return max(scores)
    if aggregation == "mean":
        return mean(scores)
    if aggregation == "sum":
        return sum(scores)
    raise ValueError(f"Unsupported species aggregation: {aggregation}")


def save_reference_index(
    references: list[ReferenceEmbedding],
    output_path: Path,
    genus_references: list[ReferenceEmbedding] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialize_references(references)
    if genus_references is not None:
        payload = {
            "format": "plant-classifier-two-stage-reference-index-v1",
            "genus_references": _serialize_references(genus_references),
            "species_references": _serialize_references(references),
        }
    torch.save(payload, output_path)


def load_reference_index(reference_index: Path, map_location: torch.device | str = "cpu") -> list[ReferenceEmbedding]:
    payload = torch.load(reference_index, map_location=map_location)
    if isinstance(payload, dict) and "species_references" in payload:
        payload = payload["species_references"]
    return _deserialize_references(payload)


def load_two_stage_reference_index(
    reference_index: Path,
    map_location: torch.device | str = "cpu",
) -> tuple[list[ReferenceEmbedding], list[ReferenceEmbedding]]:
    payload = torch.load(reference_index, map_location=map_location)
    if isinstance(payload, dict) and "species_references" in payload:
        genus_payload = payload.get("genus_references") or payload["species_references"]
        return _deserialize_references(genus_payload), _deserialize_references(payload["species_references"])
    references = _deserialize_references(payload)
    return references, references


def _serialize_references(references: list[ReferenceEmbedding]) -> list[dict]:
    return [
        {
            "image_path": str(reference.image_path),
            "family": reference.family,
            "genus": reference.genus,
            "species": reference.species,
            "global_embedding": reference.global_embedding.cpu(),
            "local_embedding": reference.local_embedding.cpu(),
        }
        for reference in references
    ]


def _deserialize_references(payload) -> list[ReferenceEmbedding]:
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
