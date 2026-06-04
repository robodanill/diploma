from pathlib import Path

import pytest

from plant_classifier.inference.factory import (
    ModelArtifacts,
    _validate_artifacts,
    artifacts_from_environment,
)


def test_model_artifacts_defaults_match_final_paper60_inference_profile(tmp_path: Path) -> None:
    artifacts = ModelArtifacts(
        genus_checkpoint=tmp_path / "genus.pt",
        species_checkpoint=tmp_path / "species.pt",
        reference_index=tmp_path / "index.pt",
    )

    assert artifacts.preprocessing is True
    assert artifacts.local_crop_position == "leaf_interior"
    assert artifacts.genus_candidates == 30
    assert artifacts.genus_score_mode == "l1"
    assert artifacts.species_score_mode == "l1"
    assert artifacts.species_aggregation == "max"
    assert artifacts.genus_candidate_mode == "unique"
    assert artifacts.genus_weight_mode == "score"
    assert artifacts.require_two_stage_reference_index is True


def test_environment_artifacts_use_final_paper60_profile_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PLANT_CLASSIFIER_GENUS_CHECKPOINT", str(tmp_path / "genus.pt"))
    monkeypatch.setenv("PLANT_CLASSIFIER_SPECIES_CHECKPOINT", str(tmp_path / "species.pt"))
    monkeypatch.setenv("PLANT_CLASSIFIER_REFERENCE_INDEX", str(tmp_path / "index.pt"))

    artifacts = artifacts_from_environment()

    assert artifacts is not None
    assert artifacts.preprocessing is True
    assert artifacts.local_crop_position == "leaf_interior"
    assert artifacts.genus_score_mode == "l1"
    assert artifacts.species_score_mode == "l1"
    assert artifacts.genus_candidate_mode == "unique"
    assert artifacts.genus_weight_mode == "score"
    assert artifacts.require_two_stage_reference_index is True


def test_artifact_validation_rejects_reusing_same_file_for_two_roles(tmp_path: Path) -> None:
    shared = tmp_path / "shared.pt"
    species = tmp_path / "species.pt"
    shared.touch()
    species.touch()

    artifacts = ModelArtifacts(
        genus_checkpoint=shared,
        species_checkpoint=species,
        reference_index=shared,
    )

    with pytest.raises(ValueError, match="three different files"):
        _validate_artifacts(artifacts)
