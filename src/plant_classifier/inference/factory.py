from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from plant_classifier.inference.predictor import Predictor
from plant_classifier.inference.stub import StubPredictor

SUPPORTED_BACKBONES = (
    "vgg16",
    "alexnet",
    "googlenet",
    "efficientnet_b3",
    "mobilenet_v3_large",
)


@dataclass(frozen=True)
class ModelArtifacts:
    genus_checkpoint: Path
    species_checkpoint: Path
    reference_index: Path
    backbone: str = "vgg16"
    local_crop_position: str = "leaf_interior"
    preprocessing: bool = True
    genus_candidates: int = 30
    genus_score_mode: str = "l1"
    species_score_mode: str = "l1"
    species_aggregation: str = "max"
    genus_candidate_mode: str = "unique"
    genus_weight_mode: str = "score"
    require_two_stage_reference_index: bool = True


def create_predictor(artifacts: ModelArtifacts | None = None) -> Predictor:
    """Create a real S-CNN predictor when artifacts are provided, otherwise use stub."""

    resolved_artifacts = artifacts or artifacts_from_environment()
    if resolved_artifacts is None:
        return StubPredictor()
    _validate_artifacts(resolved_artifacts)

    from plant_classifier.inference.scnn import TwoStageSiamesePredictor

    return TwoStageSiamesePredictor.from_artifacts(
        genus_checkpoint=resolved_artifacts.genus_checkpoint,
        species_checkpoint=resolved_artifacts.species_checkpoint,
        reference_index=resolved_artifacts.reference_index,
        backbone=resolved_artifacts.backbone,
        pretrained=False,
        local_crop_position=resolved_artifacts.local_crop_position,
        preprocessing=resolved_artifacts.preprocessing,
        genus_candidates=resolved_artifacts.genus_candidates,
        genus_score_mode=resolved_artifacts.genus_score_mode,
        species_score_mode=resolved_artifacts.species_score_mode,
        species_aggregation=resolved_artifacts.species_aggregation,
        genus_candidate_mode=resolved_artifacts.genus_candidate_mode,
        genus_weight_mode=resolved_artifacts.genus_weight_mode,
        require_two_stage_reference_index=resolved_artifacts.require_two_stage_reference_index,
    )


def artifacts_from_environment() -> ModelArtifacts | None:
    genus = os.getenv("PLANT_CLASSIFIER_GENUS_CHECKPOINT")
    species = os.getenv("PLANT_CLASSIFIER_SPECIES_CHECKPOINT")
    references = os.getenv("PLANT_CLASSIFIER_REFERENCE_INDEX")
    if not (genus and species and references):
        return None
    return ModelArtifacts(
        genus_checkpoint=Path(genus),
        species_checkpoint=Path(species),
        reference_index=Path(references),
        backbone=os.getenv("PLANT_CLASSIFIER_BACKBONE", "vgg16"),
        local_crop_position=os.getenv("PLANT_CLASSIFIER_LOCAL_CROP_POSITION", "leaf_interior"),
        preprocessing=_env_flag("PLANT_CLASSIFIER_PREPROCESSING", default=True),
        genus_candidates=int(os.getenv("PLANT_CLASSIFIER_GENUS_CANDIDATES", "30")),
        genus_score_mode=os.getenv("PLANT_CLASSIFIER_GENUS_SCORE_MODE", "l1"),
        species_score_mode=os.getenv("PLANT_CLASSIFIER_SPECIES_SCORE_MODE", "l1"),
        species_aggregation=os.getenv("PLANT_CLASSIFIER_SPECIES_AGGREGATION", "max"),
        genus_candidate_mode=os.getenv("PLANT_CLASSIFIER_GENUS_CANDIDATE_MODE", "unique"),
        genus_weight_mode=os.getenv("PLANT_CLASSIFIER_GENUS_WEIGHT_MODE", "score"),
        require_two_stage_reference_index=_env_flag(
            "PLANT_CLASSIFIER_REQUIRE_TWO_STAGE_REFERENCE_INDEX",
            default=True,
        ),
    )


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "")
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _validate_artifacts(artifacts: ModelArtifacts) -> None:
    paths = (
        artifacts.genus_checkpoint,
        artifacts.species_checkpoint,
        artifacts.reference_index,
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Model artifact does not exist: " + ", ".join(missing))

    resolved_paths = [path.resolve() for path in paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError(
            "Genus checkpoint, species checkpoint, and reference index must be three different files"
        )
