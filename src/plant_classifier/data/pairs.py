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
    split: str = ""

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
    hard_negative_ratio: float = 0.0,
    targeted_negative_ratio: float = 0.0,
    targeted_negative_label_pairs: Iterable[tuple[str, str]] | None = None,
    strategy: str = "label_uniform",
) -> list[PairRecord]:
    """Sample positive and negative image pairs for Siamese metric learning."""

    rng = random.Random(seed)
    records = list(records)
    _validate_strategy(strategy)
    grouped = _group_by_label(records, taxonomic_level)
    positives = _sample_positive_pairs(
        records,
        grouped,
        positive_count,
        taxonomic_level,
        rng,
        strategy,
    )
    hard_negative_ratio = max(0.0, min(1.0, hard_negative_ratio))
    targeted_negative_ratio = max(0.0, min(1.0, targeted_negative_ratio))
    targeted_negative_count = int(negative_count * targeted_negative_ratio)
    remaining_negative_count = negative_count - targeted_negative_count
    hard_negative_count = int(remaining_negative_count * hard_negative_ratio)
    easy_negative_count = remaining_negative_count - hard_negative_count
    negatives = _sample_targeted_negative_pairs(
        grouped,
        targeted_negative_count,
        taxonomic_level,
        rng,
        targeted_negative_label_pairs or [],
    )
    easy_negative_count += targeted_negative_count - len(negatives)
    negatives.extend(
        _sample_hard_negative_pairs(
            records,
            grouped,
            hard_negative_count,
            taxonomic_level,
            rng,
            strategy,
        )
    )
    negatives.extend(
        _sample_negative_pairs(
            records,
            grouped,
            easy_negative_count,
            taxonomic_level,
            rng,
            strategy,
        )
    )
    pairs = positives + negatives
    rng.shuffle(pairs)
    return pairs


def _sample_targeted_negative_pairs(
    grouped: dict[str, list[ImageRecord]],
    count: int,
    taxonomic_level: str,
    rng: random.Random,
    label_pairs: Iterable[tuple[str, str]],
) -> list[PairRecord]:
    if count <= 0:
        return []

    eligible_pairs = [
        (left_label, right_label)
        for left_label, right_label in label_pairs
        if left_label != right_label and grouped.get(left_label) and grouped.get(right_label)
    ]
    if not eligible_pairs:
        return []

    pairs: list[PairRecord] = []
    for _ in range(count):
        left_label, right_label = rng.choice(eligible_pairs)
        if rng.random() < 0.5:
            left_label, right_label = right_label, left_label
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


def _validate_strategy(strategy: str) -> None:
    if strategy not in {"label_uniform", "pair_uniform"}:
        raise ValueError(f"Unsupported pair sampling strategy: {strategy}")


def _group_by_label(
    records: Iterable[ImageRecord],
    taxonomic_level: str,
) -> dict[str, list[ImageRecord]]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label_for(taxonomic_level)].append(record)
    return dict(grouped)


def _sample_positive_pairs(
    records: list[ImageRecord],
    grouped: dict[str, list[ImageRecord]],
    count: int,
    taxonomic_level: str,
    rng: random.Random,
    strategy: str,
) -> list[PairRecord]:
    if count <= 0:
        return []

    if strategy == "pair_uniform":
        return _sample_positive_pairs_uniformly_by_pair(records, count, taxonomic_level, rng)

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
    records: list[ImageRecord],
    grouped: dict[str, list[ImageRecord]],
    count: int,
    taxonomic_level: str,
    rng: random.Random,
    strategy: str,
) -> list[PairRecord]:
    if count <= 0:
        return []

    if strategy == "pair_uniform":
        return _sample_negative_pairs_uniformly_by_pair(records, count, taxonomic_level, rng)

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


def _sample_hard_negative_pairs(
    records: list[ImageRecord],
    grouped: dict[str, list[ImageRecord]],
    count: int,
    taxonomic_level: str,
    rng: random.Random,
    strategy: str,
) -> list[PairRecord]:
    if count <= 0:
        return []

    if strategy == "pair_uniform":
        pairs = _sample_hard_negative_pairs_uniformly_by_pair(records, count, taxonomic_level, rng)
        if pairs:
            return pairs
        return _sample_negative_pairs(records, grouped, count, taxonomic_level, rng, strategy)

    hard_groups = _group_by_hard_negative_context(records, taxonomic_level)
    eligible_contexts = [
        context
        for context, items in hard_groups.items()
        if len({item.label_for(taxonomic_level) for item in items}) >= 2
    ]
    if not eligible_contexts:
        return _sample_negative_pairs(records, grouped, count, taxonomic_level, rng, strategy)

    pairs: list[PairRecord] = []
    for _ in range(count):
        context = rng.choice(eligible_contexts)
        context_items = hard_groups[context]
        left = rng.choice(context_items)
        candidate_items = [
            item
            for item in context_items
            if item.label_for(taxonomic_level) != left.label_for(taxonomic_level)
        ]
        right = rng.choice(candidate_items)
        pairs.append(
            PairRecord(
                left=left.image_path,
                right=right.image_path,
                label=0,
                taxonomic_level=taxonomic_level,
            )
        )
    return pairs


def _sample_positive_pairs_uniformly_by_pair(
    records: list[ImageRecord],
    count: int,
    taxonomic_level: str,
    rng: random.Random,
) -> list[PairRecord]:
    candidates: list[tuple[ImageRecord, ImageRecord]] = []
    grouped = _group_by_label(records, taxonomic_level)
    for items in grouped.values():
        candidates.extend(_record_pairs(items))
    if not candidates:
        raise ValueError("At least one class must contain two images for positive pairs")

    return [
        _make_pair(left, right, label=1, taxonomic_level=taxonomic_level)
        for left, right in _sample_with_replacement(candidates, count, rng)
    ]


def _sample_negative_pairs_uniformly_by_pair(
    records: list[ImageRecord],
    count: int,
    taxonomic_level: str,
    rng: random.Random,
) -> list[PairRecord]:
    labels = {record.label_for(taxonomic_level) for record in records}
    if len(labels) < 2:
        raise ValueError("At least two classes are required for negative pairs")

    pairs: list[PairRecord] = []
    while len(pairs) < count:
        left, right = rng.sample(records, 2)
        if left.label_for(taxonomic_level) == right.label_for(taxonomic_level):
            continue
        pairs.append(_make_pair(left, right, label=0, taxonomic_level=taxonomic_level))
    return pairs


def _sample_hard_negative_pairs_uniformly_by_pair(
    records: list[ImageRecord],
    count: int,
    taxonomic_level: str,
    rng: random.Random,
) -> list[PairRecord]:
    candidates: list[tuple[ImageRecord, ImageRecord]] = []
    grouped = _group_by_hard_negative_context(records, taxonomic_level)
    for items in grouped.values():
        for left, right in _record_pairs(items):
            if left.label_for(taxonomic_level) != right.label_for(taxonomic_level):
                candidates.append((left, right))
    return [
        _make_pair(left, right, label=0, taxonomic_level=taxonomic_level)
        for left, right in _sample_with_replacement(candidates, count, rng)
    ]


def _record_pairs(items: list[ImageRecord]) -> list[tuple[ImageRecord, ImageRecord]]:
    return [
        (left, right)
        for left_index, left in enumerate(items)
        for right in items[left_index + 1 :]
    ]


def _sample_with_replacement(
    candidates: list[tuple[ImageRecord, ImageRecord]],
    count: int,
    rng: random.Random,
) -> list[tuple[ImageRecord, ImageRecord]]:
    if not candidates:
        return []
    return [rng.choice(candidates) for _ in range(count)]


def _make_pair(
    left: ImageRecord,
    right: ImageRecord,
    label: int,
    taxonomic_level: str,
) -> PairRecord:
    return PairRecord(
        left=left.image_path,
        right=right.image_path,
        label=label,
        taxonomic_level=taxonomic_level,
    )


def _group_by_hard_negative_context(
    records: Iterable[ImageRecord],
    taxonomic_level: str,
) -> dict[str, list[ImageRecord]]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        context = _hard_negative_context(record, taxonomic_level)
        if context:
            grouped[context].append(record)
    return dict(grouped)


def _hard_negative_context(record: ImageRecord, taxonomic_level: str) -> str:
    if taxonomic_level == "genus":
        return record.family
    if taxonomic_level == "species":
        return record.genus
    return ""
