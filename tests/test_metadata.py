from pathlib import Path

from plant_classifier.data import load_metadata_csv


def test_load_metadata_csv_resolves_relative_image_paths(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "image_path,family,genus,species\nimages/a.jpg,F1,G1,S1\n",
        encoding="utf-8",
    )

    records = load_metadata_csv(metadata, dataset_root)

    assert len(records) == 1
    assert records[0].image_path == dataset_root / "images/a.jpg"
    assert records[0].family == "F1"
    assert records[0].genus == "G1"
    assert records[0].species == "S1"
