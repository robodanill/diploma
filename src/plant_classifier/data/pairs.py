from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    family: str
    genus: str
    species: str

    def label_for(self, level: str) -> str:
        if level == "family":
            return self.family
        if level == "genus":
            return self.genus
        if level == "species":
            return self.species
        raise ValueError(f"Unsupported taxonomic level: {level}")


@dataclass(frozen=True)
class PairRecord:
    left: Path
    right: Path
    label: int
    taxonomic_level: str


def sample_pairs(
    records: Iterable[ImageRecord],
    taxonomic_level: str,
    positive_count: int,
    negative_count: int,
    seed: int = 42,
) -> list[PairRecord]:
    """Sample positive and negative image pairs for Siamese metric learning."""

    rng = random.Random(seed)
    grouped = _group_by_label(records, taxonomic_level)
    positives = _sample_positive_pairs(grouped, positive_count, taxonomic_level, rng)
    negatives = _sample_negative_pairs(grouped, negative_count, taxonomic_level, rng)
    pairs = positives + negatives
    rng.shuffle(pairs)
    return pairs


def _group_by_label(
    records: Iterable[ImageRecord],
    taxonomic_level: str,
) -> dict[str, list[ImageRecord]]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label_for(taxonomic_level)].append(record)
    return dict(grouped)


def _sample_positive_pairs(
    grouped: dict[str, list[ImageRecord]],
    count: int,
    taxonomic_level: str,
    rng: random.Random,
) -> list[PairRecord]:
    eligible_labels = [label for label, items in grouped.items() if len(items) >= 2]
    if not eligible_labels:
        raise ValueError("At least one class must contain two images for positive pairs")

    pairs: list[PairRecord] = []
    for _ in range(count):
        label = rng.choice(eligible_labels)
        left, right = rng.sample(grouped[label], 2)
        pairs.append(
            PairRecord(
                left=left.image_path,
                right=right.image_path,
                label=1,
                taxonomic_level=taxonomic_level,
            )
        )
    return pairs


def _sample_negative_pairs(
    grouped: dict[str, list[ImageRecord]],
    count: int,
    taxonomic_level: str,
    rng: random.Random,
) -> list[PairRecord]:
    labels = [label for label, items in grouped.items() if items]
    if len(labels) < 2:
        raise ValueError("At least two classes are required for negative pairs")

    pairs: list[PairRecord] = []
    for _ in range(count):
        left_label, right_label = rng.sample(labels, 2)
        left = rng.choice(grouped[left_label])
        right = rng.choice(grouped[right_label])
        pairs.append(
            PairRecord(
                left=left.image_path,
                right=right.image_path,
                label=0,
                taxonomic_level=taxonomic_level,
            )
        )
    return pairs

