from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from plant_classifier.data import ImageRecord, limit_records_by_species
from plant_classifier.training.image_pairs import build_image_transform


@dataclass(frozen=True)
class GenusEvalResult:
    accuracy: float
    hits: int
    queries: int
    references: int
    top_k: int


def prepare_genus_eval_records(
    records: list[ImageRecord],
    max_species: int | None,
    references_per_genus: int,
    queries_per_genus: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    if max_species:
        records = limit_records_by_species(
            records,
            max_species=max_species,
            min_images_per_species=references_per_genus + queries_per_genus,
            max_images_per_species=references_per_genus + queries_per_genus,
        )
    return split_references_and_queries(records, references_per_genus, queries_per_genus)


def split_references_and_queries(
    records: list[ImageRecord],
    references_per_genus: int,
    queries_per_genus: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in sorted(records, key=lambda item: (item.genus, item.species, str(item.image_path))):
        grouped[record.genus].append(record)

    references: list[ImageRecord] = []
    queries: list[ImageRecord] = []
    for genus in sorted(grouped):
        items = grouped[genus]
        needed = references_per_genus + queries_per_genus
        if len(items) < needed:
            continue
        references.extend(items[:references_per_genus])
        queries.extend(items[references_per_genus:needed])
    return references, queries


@torch.inference_mode()
def evaluate_genus_retrieval(
    model,
    references: list[ImageRecord],
    queries: list[ImageRecord],
    image_size: int,
    crop_size: int,
    top_k: int,
    device: torch.device,
) -> GenusEvalResult:
    if not references or not queries:
        raise ValueError("Not enough records to evaluate genus retrieval")

    was_training = model.training
    model.eval()
    transform = build_image_transform("global", image_size=image_size, crop_size=crop_size)
    reference_embeddings = [
        (record, embed_image(model, record.image_path, transform, device)) for record in references
    ]

    hits = 0
    for query in queries:
        query_embedding = embed_image(model, query.image_path, transform, device)
        ranked = rank_references(model, query_embedding, reference_embeddings)
        top_genera = [record.genus for record, _ in ranked[:top_k]]
        if query.genus in top_genera:
            hits += 1

    if was_training:
        model.train()

    return GenusEvalResult(
        accuracy=hits / len(queries),
        hits=hits,
        queries=len(queries),
        references=len(references),
        top_k=top_k,
    )


def describe_genus_distribution(records: list[ImageRecord], limit: int = 10) -> dict[str, int]:
    return dict(Counter(record.genus for record in records).most_common(limit))


def embed_image(model, image_path: Path, transform, device: torch.device):
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    return model.embed(tensor)


def rank_references(model, query_embedding, reference_embeddings):
    ranked = []
    for record, reference_embedding in reference_embeddings:
        distance = torch.abs(query_embedding - reference_embedding)
        score = float(model.comparator(distance).flatten().item())
        ranked.append((record, score))
    return sorted(ranked, key=lambda item: item[1], reverse=True)
