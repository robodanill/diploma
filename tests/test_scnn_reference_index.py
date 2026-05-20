from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from plant_classifier.inference.scnn import (
    ReferenceEmbedding,
    load_reference_index,
    load_two_stage_reference_index,
    save_reference_index,
)


def test_two_stage_reference_index_keeps_separate_reference_sets(tmp_path: Path) -> None:
    genus_reference = _reference("genus.jpg", "Acer", "Acer alpha")
    species_reference = _reference("species.jpg", "Quercus", "Quercus beta")
    output_path = tmp_path / "reference_index.pt"

    save_reference_index(
        [species_reference],
        output_path,
        genus_references=[genus_reference],
    )

    loaded_species = load_reference_index(output_path)
    loaded_genus, loaded_species_two_stage = load_two_stage_reference_index(output_path)

    assert [reference.image_path for reference in loaded_species] == [Path("species.jpg")]
    assert [reference.image_path for reference in loaded_species_two_stage] == [Path("species.jpg")]
    assert [reference.image_path for reference in loaded_genus] == [Path("genus.jpg")]


def test_legacy_reference_index_is_used_for_both_stages(tmp_path: Path) -> None:
    reference = _reference("legacy.jpg", "Acer", "Acer alpha")
    output_path = tmp_path / "legacy_reference_index.pt"

    save_reference_index([reference], output_path)

    loaded_genus, loaded_species = load_two_stage_reference_index(output_path)

    assert loaded_genus == loaded_species
    assert [item.image_path for item in loaded_species] == [Path("legacy.jpg")]


def _reference(image_path: str, genus: str, species: str) -> ReferenceEmbedding:
    return ReferenceEmbedding(
        image_path=Path(image_path),
        family="Sapindaceae",
        genus=genus,
        species=species,
        global_embedding=torch.tensor([1.0, 2.0]),
        local_embedding=torch.tensor([3.0, 4.0]),
    )
