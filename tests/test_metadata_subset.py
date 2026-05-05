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


def test_limit_records_by_species_can_sample_records_by_seed() -> None:
    records = [
        ImageRecord(Path(f"a{index}.jpg"), "F", "G", "A")
        for index in range(6)
    ] + [
        ImageRecord(Path(f"b{index}.jpg"), "F", "G", "B")
        for index in range(6)
    ]

    first = limit_records_by_species(
        records,
        min_images_per_species=6,
        max_images_per_species=2,
        seed=42,
    )
    second = limit_records_by_species(
        records,
        min_images_per_species=6,
        max_images_per_species=2,
        seed=42,
    )

    assert first == second
    assert len(first) == 4
    assert [record.species for record in first].count("A") == 2
    assert [record.species for record in first].count("B") == 2
