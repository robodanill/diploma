from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
from PIL import Image

from plant_classifier.data import ImageRecord
from plant_classifier.inference.scnn import ReferenceEmbedding, TwoStageSiamesePredictor
from plant_classifier.models.siamese import SiameseNetwork
from plant_classifier.training.image_pairs import build_image_transform


@dataclass(frozen=True)
class SpeciesEvalResult:
    accuracies: dict[int, float]
    hits: dict[int, int]
    mean_inverse_rank: float
    queries: int
    references: int
    top_ks: tuple[int, ...]


def select_species_references(
    records: list[ImageRecord],
    references_per_species: int,
    allowed_species: set[str] | None = None,
) -> list[ImageRecord]:
    grouped: dict[str, list[ImageRecord]] = {}
    for record in sorted(records, key=lambda item: (item.species, str(item.image_path))):
        if allowed_species is not None and record.species not in allowed_species:
            continue
        grouped.setdefault(record.species, []).append(record)

    references: list[ImageRecord] = []
    for species in sorted(grouped):
        references.extend(grouped[species][:references_per_species])
    return references


@torch.inference_mode()
def build_reference_embeddings(
    genus_model: SiameseNetwork,
    species_model: SiameseNetwork,
    references: list[ImageRecord],
    image_size: int,
    crop_size: int,
    preprocessing: bool,
    device: torch.device,
) -> list[ReferenceEmbedding]:
    global_transform = build_image_transform(
        "global",
        image_size=image_size,
        crop_size=crop_size,
        preprocessing=preprocessing,
    )
    local_transform = build_image_transform(
        "local",
        image_size=image_size,
        crop_size=crop_size,
        preprocessing=preprocessing,
    )

    embeddings: list[ReferenceEmbedding] = []
    for record in references:
        with Image.open(record.image_path) as image:
            rgb = image.convert("RGB")
        global_tensor = global_transform(rgb).unsqueeze(0).to(device)
        local_tensor = local_transform(rgb).unsqueeze(0).to(device)
        embeddings.append(
            ReferenceEmbedding(
                image_path=record.image_path,
                family=record.family,
                genus=record.genus,
                species=record.species,
                global_embedding=genus_model.embed(global_tensor).squeeze(0).cpu(),
                local_embedding=species_model.embed(local_tensor).squeeze(0).cpu(),
            )
        )
    return embeddings


def evaluate_species_retrieval(
    predictor: TwoStageSiamesePredictor,
    queries: list[ImageRecord],
    top_ks: tuple[int, ...],
) -> SpeciesEvalResult:
    top_ks = tuple(sorted(set(top_ks)))
    max_top_k = max(top_ks)
    hits = {top_k: 0 for top_k in top_ks}
    inverse_rank_sum = 0.0

    for query in queries:
        prediction = predictor.predict_many([query.image_path], top_k=max_top_k)[0]
        ranked_species = [label.species for label in prediction.labels]
        rank = _rank_of(query.species, ranked_species)
        if rank is not None:
            inverse_rank_sum += 1.0 / rank
        for top_k in top_ks:
            if rank is not None and rank <= top_k:
                hits[top_k] += 1

    query_count = len(queries)
    return SpeciesEvalResult(
        accuracies={top_k: hits[top_k] / query_count for top_k in top_ks},
        hits=hits,
        mean_inverse_rank=inverse_rank_sum / query_count,
        queries=query_count,
        references=len(predictor.references),
        top_ks=top_ks,
    )


def describe_species_distribution(records: list[ImageRecord], limit: int = 10) -> dict[str, int]:
    return dict(Counter(record.species for record in records).most_common(limit))


def _rank_of(expected_species: str, ranked_species: list[str]) -> int | None:
    for index, species in enumerate(ranked_species, start=1):
        if species == expected_species:
            return index
    return None
