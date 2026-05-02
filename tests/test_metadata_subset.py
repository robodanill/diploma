from pathlib import Path

from plant_classifier.data import ImageRecord, limit_records_by_species


def test_limit_records_by_species_selects_deterministic_balanced_subset() -> None:
    records = [
        ImageRecord(Path("b1.jpg"), "F", "G", "B"),
        ImageRecord(Path("a2.jpg"), "F", "G", "A"),
        ImageRecord(Path("a1.jpg"), "F", "G", "A"),
        ImageRecord(Path("b2.jpg"), "F", "G", "B"),
        ImageRecord(Path("c1.jpg"), "F", "G", "C"),
    ]

    subset = limit_records_by_species(
        records,
        max_species=2,
        min_images_per_species=2,
        max_images_per_species=1,
    )

    assert [record.image_path for record in subset] == [
        Path("a1.jpg"),
        Path("b1.jpg"),
    ]
