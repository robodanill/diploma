from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torchvision import models


@dataclass(frozen=True)
class BackboneSpec:
    name: str = "vgg16"
    pretrained: bool = True


class SiameseNetwork(nn.Module):
    """Siamese CNN with shared backbone and an L1-distance comparison head."""

    def __init__(self, backbone: nn.Module, embedding_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.comparator = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid(),
        )

    def embed(self, image: Tensor) -> Tensor:
        return self.backbone(image)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        left_embedding = self.embed(left)
        right_embedding = self.embed(right)
        distance = torch.abs(left_embedding - right_embedding)
        return self.comparator(distance).flatten()


def build_siamese_network(spec: BackboneSpec) -> SiameseNetwork:
    backbone, embedding_dim = build_backbone(spec)
    return SiameseNetwork(backbone=backbone, embedding_dim=embedding_dim)


def build_backbone(spec: BackboneSpec) -> tuple[nn.Module, int]:
    name = spec.name.lower()
    if name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if spec.pretrained else None
        model = models.vgg16(weights=weights)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        return model, 4096

    if name == "alexnet":
        weights = models.AlexNet_Weights.DEFAULT if spec.pretrained else None
        model = models.alexnet(weights=weights)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        return model, 4096

    if name == "googlenet":
        weights = models.GoogLeNet_Weights.DEFAULT if spec.pretrained else None
        model = models.googlenet(weights=weights, aux_logits=False)
        model.fc = nn.Identity()
        return model, 1024

    if name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if spec.pretrained else None
        model = models.efficientnet_b3(weights=weights)
        embedding_dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
        return model, embedding_dim

    if name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if spec.pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        return model, 1280

    raise ValueError(f"Unsupported backbone: {spec.name}")
