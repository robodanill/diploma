from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torchvision import models


@dataclass(frozen=True)
class ClassifierSpec:
    name: str = "vgg16"
    pretrained: bool = True
    num_classes: int = 1


def build_image_classifier(spec: ClassifierSpec) -> nn.Module:
    """Build a supervised image classifier for the baseline experiments."""

    name = spec.name.lower()
    if name != "vgg16":
        raise ValueError(f"Unsupported classifier backbone: {spec.name}")

    weights = models.VGG16_Weights.DEFAULT if spec.pretrained else None
    model = models.vgg16(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, spec.num_classes)
    return model
