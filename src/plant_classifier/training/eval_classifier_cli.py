from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from plant_classifier.data import filter_records_by_split, load_metadata_csv
from plant_classifier.models.classifier import ClassifierSpec, build_image_classifier
from plant_classifier.training.classifier_dataset import ClassificationImageDataset
from plant_classifier.training.validation import validate_records_exist


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a supervised VGG16 species baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/leafscan_paper60_vgg16_baseline.yaml"))
    parser.add_argument("--query-config", type=Path, default=Path("configs/leafscan_test.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--query-split", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()

    checkpoint = _load_checkpoint(args.checkpoint)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {index: species for species, index in class_to_idx.items()}
    config = _load_config(args.config)
    query_config = _load_config(args.query_config)
    query_records = filter_records_by_split(_load_records(query_config["dataset"]), args.query_split)
    validate_records_exist(query_records)
    _validate_query_species(query_records, class_to_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_image_classifier(
        ClassifierSpec(
            name=checkpoint.get("backbone", config["model"]["backbone"]),
            pretrained=False,
            num_classes=len(class_to_idx),
        )
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image_size = int(checkpoint.get("image_size", config["views"]["global"]["image_size"]))
    crop_size = int(checkpoint.get("crop_size", config["views"].get("local", {}).get("crop_size", 32)))
    preprocessing = bool(checkpoint.get("preprocessing", _preprocessing_enabled(config)))
    dataset = ClassificationImageDataset(
        records=query_records,
        class_to_idx=class_to_idx,
        image_size=image_size,
        crop_size=crop_size,
        preprocessing=preprocessing,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    result = evaluate_classifier(
        model=model,
        dataloader=dataloader,
        idx_to_class=idx_to_class,
        records=query_records,
        top_ks=tuple(args.top_k),
        device=device,
    )

    print(
        f"queries={result['queries']} classes={len(class_to_idx)} "
        f"top_k={list(result['top_ks'])}"
    )
    for top_k in result["top_ks"]:
        print(
            f"top{top_k}_species_accuracy={result['accuracies'][top_k]:.3f} "
            f"({result['hits'][top_k]}/{result['queries']})"
        )
    print(f"mean_inverse_rank_at_{max(result['top_ks'])}={result['mean_inverse_rank']:.3f}")
    return 0


@torch.inference_mode()
def evaluate_classifier(
    model,
    dataloader: DataLoader,
    idx_to_class: dict[int, str],
    records,
    top_ks: tuple[int, ...],
    device: torch.device,
) -> dict:
    top_ks = tuple(sorted(set(top_ks)))
    max_top_k = max(top_ks)
    hits = {top_k: 0 for top_k in top_ks}
    inverse_rank_sum = 0.0
    offset = 0
    for images, _labels in dataloader:
        images = images.to(device)
        logits = model(images)
        _, predicted = logits.topk(max_top_k, dim=1)
        for row in predicted.cpu().tolist():
            expected = records[offset].species
            ranked_species = [idx_to_class[index] for index in row]
            rank = _rank_of(expected, ranked_species)
            if rank is not None:
                inverse_rank_sum += 1.0 / rank
            for top_k in top_ks:
                if rank is not None and rank <= top_k:
                    hits[top_k] += 1
            offset += 1
    query_count = len(records)
    return {
        "top_ks": top_ks,
        "hits": hits,
        "accuracies": {top_k: hits[top_k] / query_count for top_k in top_ks},
        "mean_inverse_rank": inverse_rank_sum / query_count,
        "queries": query_count,
    }


def _load_records(dataset_config: dict):
    return load_metadata_csv(
        metadata_path=Path(dataset_config["metadata"]),
        dataset_root=Path(dataset_config["root"]),
        image_column=dataset_config["image_column"],
        family_column=dataset_config["family_column"],
        genus_column=dataset_config["genus_column"],
        species_column=dataset_config["species_column"],
    )


def _validate_query_species(query_records, class_to_idx: dict[str, int]) -> None:
    missing = sorted({record.species for record in query_records if record.species not in class_to_idx})
    if missing:
        raise ValueError(
            "Query species missing from classifier training classes: " + ", ".join(missing)
        )


def _rank_of(expected_species: str, ranked_species: list[str]) -> int | None:
    for index, species in enumerate(ranked_species, start=1):
        if species == expected_species:
            return index
    return None


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return torch.load(path, map_location="cpu")


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _preprocessing_enabled(config: dict) -> bool:
    preprocessing = config.get("preprocessing", {})
    return bool(preprocessing.get("enabled", preprocessing.get("leaf_bbox", False)))


if __name__ == "__main__":
    raise SystemExit(main())
