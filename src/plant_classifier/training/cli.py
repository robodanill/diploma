from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from plant_classifier.data import load_metadata_csv, sample_pairs
from plant_classifier.models.siamese import BackboneSpec, build_siamese_network
from plant_classifier.training.image_pairs import PairImageDataset
from plant_classifier.training.loop import train_siamese


def main() -> int:
    parser = argparse.ArgumentParser(description="Train one stage of the Siamese plant classifier.")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--stage", choices=("genus", "species"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = _load_config(args.config)
    dataset_config = config["dataset"]
    records = load_metadata_csv(
        metadata_path=Path(dataset_config["metadata"]),
        dataset_root=Path(dataset_config["root"]),
        image_column=dataset_config["image_column"],
        family_column=dataset_config["family_column"],
        genus_column=dataset_config["genus_column"],
        species_column=dataset_config["species_column"],
    )

    stage = args.stage
    view = "global" if stage == "genus" else "local"
    pairs = sample_pairs(
        records=records,
        taxonomic_level=stage,
        positive_count=int(config["pair_sampling"]["positive_per_epoch"][stage]),
        negative_count=int(config["pair_sampling"]["negative_per_epoch"][stage]),
        seed=int(config["seed"]),
    )
    dataset = PairImageDataset(
        pairs=pairs,
        view=view,
        image_size=int(config["views"][view]["image_size"]),
        crop_size=int(config["views"].get("local", {}).get("crop_size", 32)),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=2,
    )

    model_config = config["model"]
    model = build_siamese_network(
        BackboneSpec(
            name=model_config["backbone"],
            pretrained=bool(model_config["pretrained"]),
            embedding_dim=int(model_config["embedding_dim"]),
        )
    )
    train_siamese(
        model=model,
        dataloader=dataloader,
        checkpoint_path=args.output,
        epochs=int(config["training"]["epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
        momentum=float(config["training"]["momentum"]),
    )
    return 0


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    raise SystemExit(main())

