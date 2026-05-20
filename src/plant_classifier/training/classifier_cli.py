from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from plant_classifier.data import (
    filter_records_by_split,
    limit_records_by_species,
    load_metadata_csv,
)
from plant_classifier.models.classifier import ClassifierSpec, build_image_classifier
from plant_classifier.training.classifier_dataset import ClassificationImageDataset
from plant_classifier.training.classifier_loop import train_classifier
from plant_classifier.training.validation import validate_records_exist


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a supervised VGG16 species baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/leafscan_paper60_vgg16_baseline.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = _load_config(args.config)
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    dataset_config = config["dataset"]
    records = load_metadata_csv(
        metadata_path=Path(dataset_config["metadata"]),
        dataset_root=Path(dataset_config["root"]),
        image_column=dataset_config["image_column"],
        family_column=dataset_config["family_column"],
        genus_column=dataset_config["genus_column"],
        species_column=dataset_config["species_column"],
    )
    validate_records_exist(records)
    train_records = filter_records_by_split(records, config["training"].get("split", "train"))
    train_records = _apply_subset(train_records, dataset_config)
    validate_records_exist(train_records)

    class_to_idx = build_species_class_mapping(train_records)
    print(
        f"training supervised classifier records={len(train_records)} "
        f"classes={len(class_to_idx)}",
        flush=True,
    )

    model_config = config["model"]
    model = build_image_classifier(
        ClassifierSpec(
            name=model_config["backbone"],
            pretrained=bool(model_config["pretrained"]),
            num_classes=len(class_to_idx),
        )
    )
    image_size = int(config["views"]["global"]["image_size"])
    crop_size = int(config["views"].get("local", {}).get("crop_size", 32))
    preprocessing = _preprocessing_enabled(config)
    dataset = ClassificationImageDataset(
        records=train_records,
        class_to_idx=class_to_idx,
        image_size=image_size,
        crop_size=crop_size,
        preprocessing=preprocessing,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 2)),
        generator=torch.Generator().manual_seed(seed),
    )
    train_classifier(
        model=model,
        dataloader=dataloader,
        checkpoint_path=args.output,
        class_to_idx=class_to_idx,
        backbone=model_config["backbone"],
        image_size=image_size,
        crop_size=crop_size,
        preprocessing=preprocessing,
        epochs=int(config["training"]["epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
        momentum=float(config["training"]["momentum"]),
        lr_decay_step=int(config["training"].get("lr_decay_step", 0)),
        lr_decay_gamma=float(config["training"].get("lr_decay_gamma", 0.5)),
        max_iterations=_optional_int(config["training"].get("max_iterations")),
        progress_every=int(config["training"].get("progress_every_batches", 10)),
    )
    return 0


def build_species_class_mapping(records) -> dict[str, int]:
    return {species: index for index, species in enumerate(sorted({record.species for record in records}))}


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _apply_subset(records: list, dataset_config: dict) -> list:
    subset = dataset_config.get("subset")
    if not subset:
        return records
    limited = limit_records_by_species(
        records,
        max_species=subset.get("max_species"),
        min_images_per_species=int(subset.get("min_images_per_species", 1)),
        max_images_per_species=subset.get("max_images_per_species"),
        seed=subset.get("seed"),
    )
    print(
        f"using subset: {len(limited)} images from {len({record.species for record in limited})} species",
        flush=True,
    )
    return limited


def _preprocessing_enabled(config: dict) -> bool:
    preprocessing = config.get("preprocessing", {})
    return bool(preprocessing.get("enabled", preprocessing.get("leaf_bbox", False)))


def _optional_int(value) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
