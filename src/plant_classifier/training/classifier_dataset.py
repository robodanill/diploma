from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from plant_classifier.data import ImageRecord
from plant_classifier.training.image_pairs import build_image_transform


class ClassificationImageDataset(Dataset[tuple[Tensor, int]]):
    """Torch dataset that loads single images and species class ids."""

    def __init__(
        self,
        records: list[ImageRecord],
        class_to_idx: dict[str, int],
        image_size: int = 224,
        crop_size: int = 32,
        preprocessing: bool = False,
    ) -> None:
        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = build_image_transform(
            view="global",
            image_size=image_size,
            crop_size=crop_size,
            preprocessing=preprocessing,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record = self.records[index]
        return self.transform(_load_rgb(record.image_path)), self.class_to_idx[record.species]


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")
