from pathlib import Path

import pytest

from plant_classifier.data import ImageRecord

pytest.importorskip("torch")

from plant_classifier.training.genus_eval import select_reference_records


def test_select_genus_references_balances_species_inside_genus() -> None:
    records = [
        ImageRecord(Path("prunus_a2.jpg"), "F", "Prunus", "Prunus alpha"),
        ImageRecord(Path("prunus_a1.jpg"), "F", "Prunus", "Prunus alpha"),
        ImageRecord(Path("prunus_b1.jpg"), "F", "Prunus", "Prunus beta"),
        ImageRecord(Path("prunus_b2.jpg"), "F", "Prunus", "Prunus beta"),
        ImageRecord(Path("prunus_c1.jpg"), "F", "Prunus", "Prunus gamma"),
        ImageRecord(Path("prunus_c2.jpg"), "F", "Prunus", "Prunus gamma"),
        ImageRecord(Path("acer_a1.jpg"), "F", "Acer", "Acer alpha"),
        ImageRecord(Path("acer_a2.jpg"), "F", "Acer", "Acer alpha"),
    ]

    references = select_reference_records(records, "genus", references_per_label=6)

    assert [record.image_path for record in references] == [
        Path("acer_a1.jpg"),
        Path("acer_a2.jpg"),
        Path("prunus_a1.jpg"),
        Path("prunus_b1.jpg"),
        Path("prunus_c1.jpg"),
        Path("prunus_a2.jpg"),
        Path("prunus_b2.jpg"),
        Path("prunus_c2.jpg"),
    ]


def test_select_species_references_keeps_per_species_count() -> None:
    records = [
        ImageRecord(Path("a1.jpg"), "F", "G", "S1"),
        ImageRecord(Path("a2.jpg"), "F", "G", "S1"),
        ImageRecord(Path("b1.jpg"), "F", "G", "S2"),
        ImageRecord(Path("b2.jpg"), "F", "G", "S2"),
    ]

    references = select_reference_records(records, "species", references_per_label=1)

    assert [record.image_path for record in references] == [Path("a1.jpg"), Path("b1.jpg")]
