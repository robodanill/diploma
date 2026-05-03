"""Inference contracts and predictor implementations."""

from plant_classifier.inference.factory import SUPPORTED_BACKBONES, ModelArtifacts, create_predictor
from plant_classifier.inference.predictor import Predictor
from plant_classifier.inference.stub import StubPredictor
from plant_classifier.inference.types import ImagePrediction, PredictionLabel

__all__ = [
    "ImagePrediction",
    "ModelArtifacts",
    "PredictionLabel",
    "Predictor",
    "StubPredictor",
    "SUPPORTED_BACKBONES",
    "create_predictor",
]
