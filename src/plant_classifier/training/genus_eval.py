from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
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
    score_mode: str = "comparator"
    pair_loss: float | None = None

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
    seed: int | None = None,
) -> list[ImageRecord]:
    if taxonomic_level == "genus":
        return select_genus_references(
            records,
            references_per_genus=references_per_label,
            seed=seed,
        )

    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in sorted(
        records,
        key=lambda item: (item.label_for(taxonomic_level), str(item.image_path)),
    ):
        grouped[record.label_for(taxonomic_level)].append(record)

    rng = random.Random(seed) if seed is not None else None
    references: list[ImageRecord] = []
    for label in sorted(grouped):
        label_records = list(grouped[label])
        if rng is not None:
            rng.shuffle(label_records)
        references.extend(label_records[:references_per_label])
    return references


def select_genus_references(
    records: list[ImageRecord],
    references_per_genus: int,
    seed: int | None = None,
) -> list[ImageRecord]:
    """Select up to N references per genus while covering its species evenly."""

    by_genus: dict[str, dict[str, list[ImageRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in sorted(records, key=lambda item: (item.genus, item.species, str(item.image_path))):
        by_genus[record.genus][record.species].append(record)

    rng = random.Random(seed) if seed is not None else None
    references: list[ImageRecord] = []
    for genus in sorted(by_genus):
        species_groups = by_genus[genus]
        species_order = sorted(species_groups)
        if rng is not None:
            rng.shuffle(species_order)
            for species in species_order:
                rng.shuffle(species_groups[species])
        cursors = {species: 0 for species in species_groups}
        selected_for_genus: list[ImageRecord] = []
        while len(selected_for_genus) < references_per_genus:
            added = False
            for species in species_order:
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
    score_mode: str = "comparator",
) -> GenusEvalResult:
    if not references or not queries:
        raise ValueError("Not enough records to evaluate genus retrieval")
    _validate_score_mode(score_mode)

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
    pair_loss_sum = 0.0
    pair_loss_count = 0
    for query in queries:
        query_embedding = embed_image(model, query.image_path, transform, device)
        ranked = []
        for reference, reference_embedding in reference_embeddings:
            distance = torch.abs(query_embedding - reference_embedding)
            comparator_score = model.comparator(distance).flatten()
            target_value = 1.0 if query.genus == reference.genus else 0.0
            target = torch.full_like(comparator_score, target_value)
            pair_loss_sum += float(
                F.binary_cross_entropy(comparator_score, target, reduction="sum").item()
            )
            pair_loss_count += int(target.numel())
            if score_mode == "comparator":
                score = float(comparator_score.item())
            else:
                score = float(distance.sum().item())
            ranked.append((reference, score))
        ranked = sorted(ranked, key=lambda item: item[1], reverse=score_mode == "comparator")
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
        score_mode=score_mode,
        pair_loss=pair_loss_sum / max(1, pair_loss_count),
    )


def describe_genus_distribution(records: list[ImageRecord], limit: int = 10) -> dict[str, int]:
    return dict(Counter(record.genus for record in records).most_common(limit))


def describe_species_per_genus(records: list[ImageRecord], limit: int = 10) -> dict[str, int]:
    species_by_genus: dict[str, set[str]] = defaultdict(set)
    for record in records:
        species_by_genus[record.genus].add(record.species)
    ranked = sorted(
        species_by_genus.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    return {genus: len(species) for genus, species in ranked[:limit]}


def describe_genus_reference_coverage(
    reference_pool: list[ImageRecord],
    references: list[ImageRecord],
    limit: int = 10,
) -> dict[str, str]:
    pool_species_by_genus: dict[str, set[str]] = defaultdict(set)
    reference_species_by_genus: dict[str, set[str]] = defaultdict(set)
    for record in reference_pool:
        pool_species_by_genus[record.genus].add(record.species)
    for record in references:
        reference_species_by_genus[record.genus].add(record.species)

    ranked = sorted(
        pool_species_by_genus,
        key=lambda genus: (
            len(reference_species_by_genus[genus]) / max(1, len(pool_species_by_genus[genus])),
            -len(pool_species_by_genus[genus]),
            genus,
        ),
    )
    return {
        genus: f"{len(reference_species_by_genus[genus])}/{len(pool_species_by_genus[genus])}"
        for genus in ranked[:limit]
    }


@torch.inference_mode()
def embed_image(model, image_path: Path, transform, device: torch.device):
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    return model.embed(tensor)


@torch.inference_mode()
def rank_references(model, query_embedding, reference_embeddings, score_mode: str = "comparator"):
    _validate_score_mode(score_mode)
    ranked = []
    for record, reference_embedding in reference_embeddings:
        distance = torch.abs(query_embedding - reference_embedding)
        if score_mode == "comparator":
            score = float(model.comparator(distance).flatten().item())
        else:
            score = float(distance.sum().item())
        ranked.append((record, score))
    return sorted(ranked, key=lambda item: item[1], reverse=score_mode == "comparator")


def _validate_score_mode(score_mode: str) -> None:
    if score_mode not in {"comparator", "l1"}:
        raise ValueError(f"Unsupported score mode: {score_mode}")
