from pathlib import Path

import pytest

from plant_classifier.data import ImageRecord

pytest.importorskip("torch")

import torch

from plant_classifier.training.genus_eval import rank_references, select_reference_records


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


def test_seeded_genus_references_are_reproducible_and_cover_species() -> None:
    records = [
        ImageRecord(Path(f"prunus_a{index}.jpg"), "F", "Prunus", "Prunus alpha")
        for index in range(4)
    ] + [
        ImageRecord(Path(f"prunus_b{index}.jpg"), "F", "Prunus", "Prunus beta")
        for index in range(4)
    ] + [
        ImageRecord(Path(f"prunus_c{index}.jpg"), "F", "Prunus", "Prunus gamma")
        for index in range(4)
    ]

    first = select_reference_records(records, "genus", references_per_label=3, seed=123)
    second = select_reference_records(records, "genus", references_per_label=3, seed=123)

    assert first == second
    assert {record.species for record in first} == {
        "Prunus alpha",
        "Prunus beta",
        "Prunus gamma",
    }


def test_rank_references_can_use_raw_l1_distance() -> None:
    class DistanceIncreasingComparator:
        def comparator(self, distance):
            return distance.sum(dim=1, keepdim=True)

    query = torch.tensor([[0.0, 0.0]])
    close = ImageRecord(Path("close.jpg"), "F", "G", "S1")
    far = ImageRecord(Path("far.jpg"), "F", "G", "S2")
    references = [
        (far, torch.tensor([[10.0, 0.0]])),
        (close, torch.tensor([[1.0, 0.0]])),
    ]

    comparator_ranked = rank_references(DistanceIncreasingComparator(), query, references)
    l1_ranked = rank_references(DistanceIncreasingComparator(), query, references, score_mode="l1")

    assert comparator_ranked[0][0] == far
    assert l1_ranked[0][0] == close
