"""Inference contracts and predictor implementations."""

from plant_classifier.inference.predictor import Predictor
from plant_classifier.inference.scnn import TwoStageSiamesePredictor
from plant_classifier.inference.stub import StubPredictor
from plant_classifier.inference.types import ImagePrediction, PredictionLabel

__all__ = [
    "ImagePrediction",
    "PredictionLabel",
    "Predictor",
    "StubPredictor",
    "TwoStageSiamesePredictor",
]
