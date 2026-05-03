from pathlib import Path

from plant_classifier.data import (
    ImageRecord,
    create_species_stratified_split,
    filter_records_by_split,
)


def test_create_species_stratified_split_assigns_all_splits() -> None:
    records = [
        ImageRecord(Path(f"a{i}.jpg"), "F", "G", "A") for i in range(10)
    ] + [
        ImageRecord(Path(f"b{i}.jpg"), "F", "G", "B") for i in range(10)
    ]

    split_records = create_species_stratified_split(records, seed=1)

    assert len(split_records) == len(records)
    assert {record.split for record in split_records} == {"train", "val", "test"}
    assert len(filter_records_by_split(split_records, "train")) == 14
    assert len(filter_records_by_split(split_records, "val")) == 2
    assert len(filter_records_by_split(split_records, "test")) == 4


def test_filter_records_by_split_falls_back_when_no_split_labels() -> None:
    records = [ImageRecord(Path("a.jpg"), "F", "G", "A")]

    assert filter_records_by_split(records, "train") == records
