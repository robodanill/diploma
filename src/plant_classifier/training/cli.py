from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from plant_classifier.data import (
    filter_records_by_split,
    limit_records_by_species,
    load_metadata_csv,
    sample_pairs,
)
from plant_classifier.models.siamese import BackboneSpec, build_siamese_network
from plant_classifier.training.genus_eval import evaluate_genus_retrieval, prepare_genus_eval_records
from plant_classifier.training.image_pairs import PairImageDataset
from plant_classifier.training.loop import train_siamese, train_siamese_with_dynamic_pairs
from plant_classifier.training.validation import validate_records_exist


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
    validate_records_exist(records)
    train_records = filter_records_by_split(records, config["training"].get("split", "train"))
    train_records = _apply_subset(train_records, dataset_config)
    validate_records_exist(train_records)
    print(
        f"training split={config['training'].get('split', 'train')} "
        f"records={len(train_records)}",
        flush=True,
    )

    stage = args.stage
    view = "global" if stage == "genus" else "local"
    model_config = config["model"]
    print(
        f"building model backbone={model_config['backbone']} pretrained={bool(model_config['pretrained'])}",
        flush=True,
    )
    model = build_siamese_network(
        BackboneSpec(
            name=model_config["backbone"],
            pretrained=bool(model_config["pretrained"]),
        )
    )
    print("model ready", flush=True)
    dynamic_pairs = bool(config["training"].get("dynamic_pairs", True))
    if dynamic_pairs:
        train_siamese_with_dynamic_pairs(
            model=model,
            records=train_records,
            taxonomic_level=stage,
            view=view,
            checkpoint_path=args.output,
            positive_count=int(config["pair_sampling"]["positive_per_epoch"][stage]),
            negative_count=int(config["pair_sampling"]["negative_per_epoch"][stage]),
            hard_negative_ratio=float(config["pair_sampling"].get("hard_negative_ratio", 0.0)),
            image_size=int(config["views"][view]["image_size"]),
            crop_size=int(config["views"].get("local", {}).get("crop_size", 32)),
            preprocessing=_preprocessing_enabled(config),
            batch_size=int(config["training"]["batch_size"]),
            epochs=int(config["training"]["epochs"]),
            learning_rate=float(config["training"]["learning_rate"]),
            momentum=float(config["training"]["momentum"]),
            lr_decay_step=int(config["training"].get("lr_decay_step", 0)),
            lr_decay_gamma=float(config["training"].get("lr_decay_gamma", 0.5)),
            max_iterations=_optional_int(config["training"].get("max_iterations")),
            num_workers=int(config["training"].get("num_workers", 2)),
            seed=int(config["seed"]),
            eval_fn=_build_eval_fn(config, records, stage),
            progress_every=int(config["training"].get("progress_every_batches", 5)),
        )
    else:
        _train_static_pairs(config, train_records, stage, view, model, args.output)
    return 0


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


def _train_static_pairs(config: dict, records: list, stage: str, view: str, model, output: Path) -> None:
    pairs = sample_pairs(
        records=records,
        taxonomic_level=stage,
        positive_count=int(config["pair_sampling"]["positive_per_epoch"][stage]),
        negative_count=int(config["pair_sampling"]["negative_per_epoch"][stage]),
        hard_negative_ratio=float(config["pair_sampling"].get("hard_negative_ratio", 0.0)),
        seed=int(config["seed"]),
    )
    dataset = PairImageDataset(
        pairs=pairs,
        view=view,
        image_size=int(config["views"][view]["image_size"]),
        crop_size=int(config["views"].get("local", {}).get("crop_size", 32)),
        preprocessing=_preprocessing_enabled(config),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 2)),
    )
    train_siamese(
        model=model,
        dataloader=dataloader,
        checkpoint_path=output,
        epochs=int(config["training"]["epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
        momentum=float(config["training"]["momentum"]),
        lr_decay_step=int(config["training"].get("lr_decay_step", 0)),
        lr_decay_gamma=float(config["training"].get("lr_decay_gamma", 0.5)),
        max_iterations=_optional_int(config["training"].get("max_iterations")),
        progress_every=int(config["training"].get("progress_every_batches", 5)),
    )


def _build_eval_fn(config: dict, records: list, stage: str):
    evaluation_config = config.get("evaluation", {})
    if stage != "genus" or not evaluation_config.get("enabled", False):
        return None

    references, queries = prepare_genus_eval_records(
        records=filter_records_by_split(records, evaluation_config.get("split", "val")),
        max_species=evaluation_config.get("max_species"),
        references_per_genus=int(evaluation_config.get("references_per_genus", 2)),
        queries_per_genus=int(evaluation_config.get("queries_per_genus", 2)),
    )
    if not references or not queries:
        print(
            "genus eval skipped: not enough records to create references and queries "
            f"for split={evaluation_config.get('split', 'val')}",
            flush=True,
        )
        return None

    top_ks = _parse_top_ks(evaluation_config.get("top_k", [1, 3, 5]))
    score_mode = str(evaluation_config.get("score_mode", "comparator"))
    image_size = int(config["views"]["global"]["image_size"])
    crop_size = int(config["views"]["local"]["crop_size"])
    preprocessing = _preprocessing_enabled(config)
    print(
        f"genus eval enabled: references={len(references)} "
        f"queries={len(queries)} top_k={top_ks} score_mode={score_mode}",
        flush=True,
    )

    def eval_fn(model, device):
        return evaluate_genus_retrieval(
            model=model,
            references=references,
            queries=queries,
            image_size=image_size,
            crop_size=crop_size,
            preprocessing=preprocessing,
            top_ks=top_ks,
            device=device,
            score_mode=score_mode,
        )

    return eval_fn


def _parse_top_ks(value) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    return tuple(int(item) for item in value)


def _preprocessing_enabled(config: dict) -> bool:
    preprocessing = config.get("preprocessing", {})
    return bool(preprocessing.get("enabled", preprocessing.get("leaf_bbox", False)))


def _optional_int(value) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
