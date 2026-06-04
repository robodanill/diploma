from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from plant_classifier.data.pairs import ImageRecord


def load_metadata_csv(
    metadata_path: Path,
    dataset_root: Path,
    image_column: str = "image_path",
    family_column: str = "family",
    genus_column: str = "genus",
    species_column: str = "species",
    split_column: str = "split",
) -> list[ImageRecord]:
    """Load image records from a normalized metadata CSV file."""

    records: list[ImageRecord] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {image_column, family_column, genus_column, species_column}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Metadata CSV is missing required columns: {missing}")

        for row in reader:
            image_path = Path(row[image_column])
            if not image_path.is_absolute():
                image_path = dataset_root / image_path
            records.append(
                ImageRecord(
                    image_path=image_path,
                    family=row[family_column],
                    genus=row[genus_column],
                    species=row[species_column],
                    split=row.get(split_column, ""),
                )
            )
    return records


def filter_records_by_split(records: list[ImageRecord], split: str | None) -> list[ImageRecord]:
    """Return records for a split, falling back to all records when split labels are absent."""

    if not split:
        return records
    if not any(record.split for record in records):
        return records
    return [record for record in records if record.split == split]


def create_species_stratified_split(
    records: list[ImageRecord],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> list[ImageRecord]:
    """Create a deterministic image-level train/val/test split inside each species."""

    if round(train_ratio + val_ratio + test_ratio, 8) != 1.0:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    rng = random.Random(seed)
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.species].append(record)

    split_records: list[ImageRecord] = []
    for species in sorted(grouped):
        species_records = sorted(grouped[species], key=lambda item: str(item.image_path))
        rng.shuffle(species_records)
        split_names = _split_names_for_count(len(species_records), train_ratio, val_ratio)
        for record, split in zip(species_records, split_names, strict=True):
            split_records.append(
                ImageRecord(
                    image_path=record.image_path,
                    family=record.family,
                    genus=record.genus,
                    species=record.species,
                    split=split,
                )
            )
    return sorted(split_records, key=lambda item: str(item.image_path))


def write_metadata_csv(records: list[ImageRecord], output_path: Path, dataset_root: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "family", "genus", "species", "split"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_path": _safe_relative(record.image_path, dataset_root),
                    "family": record.family,
                    "genus": record.genus,
                    "species": record.species,
                    "split": record.split,
                }
            )


def _split_names_for_count(count: int, train_ratio: float, val_ratio: float) -> list[str]:
    if count <= 2:
        return ["train"] * count
    train_count = max(1, int(count * train_ratio))
    val_count = max(1, int(count * val_ratio))
    if train_count + val_count >= count:
        train_count = max(1, count - 2)
        val_count = 1
    test_count = count - train_count - val_count
    return ["train"] * train_count + ["val"] * val_count + ["test"] * test_count


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def limit_records_by_species(
    records: list[ImageRecord],
    max_species: int | None = None,
    min_images_per_species: int = 1,
    max_images_per_species: int | None = None,
    seed: int | None = None,
) -> list[ImageRecord]:
    """Select a species-balanced subset, optionally sampling records by seed."""

    grouped: dict[str, list[ImageRecord]] = {}
    for record in sorted(records, key=lambda item: (item.species, str(item.image_path))):
        grouped.setdefault(record.species, []).append(record)

    rng = random.Random(seed) if seed is not None else None
    selected: list[ImageRecord] = []
    species_seen = 0
    for species in sorted(grouped):
        species_records = list(grouped[species])
        if len(species_records) < min_images_per_species:
            continue
        if rng is not None:
            rng.shuffle(species_records)
        limit = max_images_per_species or len(species_records)
        selected.extend(species_records[:limit])
        species_seen += 1
        if max_species is not None and species_seen >= max_species:
            break

    return selected


def limit_records_by_genus(
    records: list[ImageRecord],
    max_genera: int | None = None,
    min_images_per_genus: int = 1,
    max_images_per_genus: int | None = None,
    seed: int | None = None,
    cover_species: bool = True,
) -> list[ImageRecord]:
    """Select a genus-balanced subset, optionally covering species inside each genus."""

    if not cover_species:
        grouped: dict[str, list[ImageRecord]] = {}
        for record in sorted(records, key=lambda item: (item.genus, item.species, str(item.image_path))):
            grouped.setdefault(record.genus, []).append(record)
        return _limit_grouped_records(
            grouped,
            max_labels=max_genera,
            min_images_per_label=min_images_per_genus,
            max_images_per_label=max_images_per_genus,
            seed=seed,
        )

    by_genus: dict[str, dict[str, list[ImageRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in sorted(records, key=lambda item: (item.genus, item.species, str(item.image_path))):
        by_genus[record.genus][record.species].append(record)

    rng = random.Random(seed) if seed is not None else None
    selected: list[ImageRecord] = []
    genera_seen = 0
    for genus in sorted(by_genus):
        species_groups = by_genus[genus]
        total_genus_records = sum(len(items) for items in species_groups.values())
        if total_genus_records < min_images_per_genus:
            continue
        limit = max_images_per_genus or total_genus_records
        species_order = sorted(species_groups)
        if rng is not None:
            rng.shuffle(species_order)
            for species in species_order:
                rng.shuffle(species_groups[species])
        cursors = {species: 0 for species in species_order}
        selected_for_genus: list[ImageRecord] = []
        while len(selected_for_genus) < limit:
            added = False
            for species in species_order:
                species_records = species_groups[species]
                cursor = cursors[species]
                if cursor >= len(species_records):
                    continue
                selected_for_genus.append(species_records[cursor])
                cursors[species] = cursor + 1
                added = True
                if len(selected_for_genus) >= limit:
                    break
            if not added:
                break
        selected.extend(selected_for_genus)
        genera_seen += 1
        if max_genera is not None and genera_seen >= max_genera:
            break

    return selected


def _limit_grouped_records(
    grouped: dict[str, list[ImageRecord]],
    max_labels: int | None,
    min_images_per_label: int,
    max_images_per_label: int | None,
    seed: int | None,
) -> list[ImageRecord]:
    rng = random.Random(seed) if seed is not None else None
    selected: list[ImageRecord] = []
    labels_seen = 0
    for label in sorted(grouped):
        label_records = list(grouped[label])
        if len(label_records) < min_images_per_label:
            continue
        if rng is not None:
            rng.shuffle(label_records)
        limit = max_images_per_label or len(label_records)
        selected.extend(label_records[:limit])
        labels_seen += 1
        if max_labels is not None and labels_seen >= max_labels:
            break
    return selected
