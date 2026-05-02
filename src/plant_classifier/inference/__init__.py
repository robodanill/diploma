"""Inference contracts and predictor implementations."""

from plant_classifier.inference.factory import ModelArtifacts, create_predictor
from plant_classifier.inference.predictor import Predictor
from plant_classifier.inference.stub import StubPredictor
from plant_classifier.inference.types import ImagePrediction, PredictionLabel

__all__ = [
    "ImagePrediction",
    "ModelArtifacts",
    "PredictionLabel",
    "Predictor",
    "StubPredictor",
    "create_predictor",
]
