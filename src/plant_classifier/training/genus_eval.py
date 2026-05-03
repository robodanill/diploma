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
    accuracies: dict[int, float]
    hits: dict[int, int]
    queries: int
    references: int
    top_ks: tuple[int, ...]

    @property
    def primary_top_k(self) -> int:
        return max(self.top_ks)

    @property
    def primary_accuracy(self) -> float:
        return self.accuracies[self.primary_top_k]


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


def select_reference_records(
    records: list[ImageRecord],
    taxonomic_level: str,
    references_per_label: int,
) -> list[ImageRecord]:
    if taxonomic_level == "genus":
        return select_genus_references(records, references_per_genus=references_per_label)

    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in sorted(
        records,
        key=lambda item: (item.label_for(taxonomic_level), str(item.image_path)),
    ):
        grouped[record.label_for(taxonomic_level)].append(record)

    references: list[ImageRecord] = []
    for label in sorted(grouped):
        references.extend(grouped[label][:references_per_label])
    return references


def select_genus_references(
    records: list[ImageRecord],
    references_per_genus: int,
) -> list[ImageRecord]:
    """Select up to N references per genus while covering its species evenly."""

    by_genus: dict[str, dict[str, list[ImageRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in sorted(records, key=lambda item: (item.genus, item.species, str(item.image_path))):
        by_genus[record.genus][record.species].append(record)

    references: list[ImageRecord] = []
    for genus in sorted(by_genus):
        species_groups = by_genus[genus]
        cursors = {species: 0 for species in species_groups}
        selected_for_genus: list[ImageRecord] = []
        while len(selected_for_genus) < references_per_genus:
            added = False
            for species in sorted(species_groups):
                cursor = cursors[species]
                species_records = species_groups[species]
                if cursor >= len(species_records):
                    continue
                selected_for_genus.append(species_records[cursor])
                cursors[species] = cursor + 1
                added = True
                if len(selected_for_genus) >= references_per_genus:
                    break
            if not added:
                break
        references.extend(selected_for_genus)
    return references


@torch.inference_mode()
def evaluate_genus_retrieval(
    model,
    references: list[ImageRecord],
    queries: list[ImageRecord],
    image_size: int,
    crop_size: int,
    top_ks: tuple[int, ...],
    device: torch.device,
    preprocessing: bool = False,
) -> GenusEvalResult:
    if not references or not queries:
        raise ValueError("Not enough records to evaluate genus retrieval")

    was_training = model.training
    model.eval()
    transform = build_image_transform(
        "global",
        image_size=image_size,
        crop_size=crop_size,
        preprocessing=preprocessing,
    )
    top_ks = tuple(sorted(set(top_ks)))
    max_top_k = max(top_ks)
    reference_embeddings = [
        (record, embed_image(model, record.image_path, transform, device)) for record in references
    ]

    hits = {top_k: 0 for top_k in top_ks}
    for query in queries:
        query_embedding = embed_image(model, query.image_path, transform, device)
        ranked = rank_references(model, query_embedding, reference_embeddings)
        top_genera = [record.genus for record, _ in ranked[:max_top_k]]
        for top_k in top_ks:
            if query.genus in top_genera[:top_k]:
                hits[top_k] += 1

    if was_training:
        model.train()

    return GenusEvalResult(
        accuracies={top_k: hits[top_k] / len(queries) for top_k in top_ks},
        hits=hits,
        queries=len(queries),
        references=len(references),
        top_ks=top_ks,
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
