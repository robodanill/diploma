from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from plant_classifier.data import PairRecord


class PairImageDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Torch dataset that loads Siamese image pairs."""

    def __init__(
        self,
        pairs: list[PairRecord],
        view: str,
        image_size: int = 224,
        crop_size: int = 32,
    ) -> None:
        self.pairs = pairs
        self.view = view
        self.transform = build_image_transform(view=view, image_size=image_size, crop_size=crop_size)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        pair = self.pairs[index]
        left = _load_rgb(pair.left)
        right = _load_rgb(pair.right)
        label = Tensor([float(pair.label)]).squeeze(0)
        return self.transform(left), self.transform(right), label


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def build_image_transform(view: str, image_size: int, crop_size: int) -> transforms.Compose:
    steps: list[object] = []
    if view == "local":
        steps.extend(
            [
                transforms.CenterCrop(crop_size),
                transforms.Resize((image_size, image_size)),
            ]
        )
    elif view == "global":
        steps.append(transforms.Resize((image_size, image_size)))
    else:
        raise ValueError(f"Unsupported image view: {view}")

    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(steps)
