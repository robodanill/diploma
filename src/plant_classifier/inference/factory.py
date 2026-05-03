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


def create_predictor(artifacts: ModelArtifacts | None = None) -> Predictor:
    """Create a real S-CNN predictor when artifacts are provided, otherwise use stub."""

    resolved_artifacts = artifacts or artifacts_from_environment()
    if resolved_artifacts is None:
        return StubPredictor()

    from plant_classifier.inference.scnn import TwoStageSiamesePredictor

    return TwoStageSiamesePredictor.from_artifacts(
        genus_checkpoint=resolved_artifacts.genus_checkpoint,
        species_checkpoint=resolved_artifacts.species_checkpoint,
        reference_index=resolved_artifacts.reference_index,
        backbone=resolved_artifacts.backbone,
        pretrained=False,
        preprocessing=_env_flag("PLANT_CLASSIFIER_PREPROCESSING"),
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
    )


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}
