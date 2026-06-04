from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from plant_classifier.data import PairRecord
from plant_classifier.preprocessing.views import LeafBoundingBoxCrop, LeafInteriorCrop


class PairImageDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Torch dataset that loads Siamese image pairs."""

    def __init__(
        self,
        pairs: list[PairRecord],
        view: str,
        image_size: int = 224,
        crop_size: int = 32,
        crop_position: str = "center",
        preprocessing: bool = False,
    ) -> None:
        self.pairs = pairs
        self.view = view
        self._tensor_cache: dict[Path, Tensor] = {}
        self.transform = build_image_transform(
            view=view,
            image_size=image_size,
            crop_size=crop_size,
            crop_position=crop_position,
            preprocessing=preprocessing,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        pair = self.pairs[index]
        label = Tensor([float(pair.label)]).squeeze(0)
        return self._load_transformed(pair.left), self._load_transformed(pair.right), label

    def _load_transformed(self, path: Path) -> Tensor:
        tensor = self._tensor_cache.get(path)
        if tensor is None:
            tensor = self.transform(_load_rgb(path))
            self._tensor_cache[path] = tensor
        return tensor


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def build_image_transform(
    view: str,
    image_size: int,
    crop_size: int,
    crop_position: str = "center",
    preprocessing: bool = False,
) -> transforms.Compose:
    steps: list[object] = []
    if preprocessing:
        steps.extend(
            [
                LeafBoundingBoxCrop(),
                transforms.Resize((image_size, image_size)),
            ]
        )

    if view == "local":
        crop_position = crop_position.lower()
        if crop_position == "center":
            local_crop = transforms.CenterCrop(crop_size)
        elif crop_position in {"leaf_interior", "green_aware"}:
            local_crop = LeafInteriorCrop(crop_size=crop_size)
        else:
            raise ValueError(f"Unsupported local crop position: {crop_position}")
        steps.extend(
            [
                local_crop,
                transforms.Resize((image_size, image_size)),
            ]
        )
    elif view == "global":
        if not preprocessing:
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
