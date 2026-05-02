from __future__ import annotations

import csv
from pathlib import Path

from plant_classifier.data.pairs import ImageRecord


def load_metadata_csv(
    metadata_path: Path,
    dataset_root: Path,
    image_column: str = "image_path",
    family_column: str = "family",
    genus_column: str = "genus",
    species_column: str = "species",
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
                )
            )
    return records

