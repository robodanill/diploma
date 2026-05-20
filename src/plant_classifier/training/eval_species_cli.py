from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from plant_classifier.data import ImageRecord, filter_records_by_split, load_metadata_csv
from plant_classifier.inference.scnn import TwoStageSiamesePredictor
from plant_classifier.models.siamese import BackboneSpec, build_siamese_network
from plant_classifier.training.species_eval import (
    build_reference_embeddings,
    describe_species_distribution,
    evaluate_species_retrieval,
    select_species_references,
)
from plant_classifier.training.validation import validate_records_exist


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate full two-stage S-CNN species retrieval.")
    parser.add_argument("--config", type=Path, default=Path("configs/leafscan_paper60_training.yaml"))
    parser.add_argument("--query-config", type=Path, default=Path("configs/leafscan_test.yaml"))
    parser.add_argument("--genus-checkpoint", type=Path, required=True)
    parser.add_argument("--species-checkpoint", type=Path, required=True)
    parser.add_argument("--references-per-species", type=int, default=6)
    parser.add_argument("--genus-candidates", type=int, default=30)
    parser.add_argument("--reference-split", default="train")
    parser.add_argument("--query-split", default="")
    parser.add_argument(
        "--all-reference-species",
        action="store_true",
        help="Use references for all train species instead of restricting to query species.",
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()

    _require_checkpoint(args.genus_checkpoint)
    _require_checkpoint(args.species_checkpoint)

    config = _load_config(args.config)
    query_config = _load_config(args.query_config)
    reference_records = filter_records_by_split(_load_records(config["dataset"]), args.reference_split)
    query_records = filter_records_by_split(_load_records(query_config["dataset"]), args.query_split)
    validate_records_exist(reference_records)
    validate_records_exist(query_records)
    if not query_records:
        print("species eval skipped: no query records")
        return 0

    allowed_species = None if args.all_reference_species else {record.species for record in query_records}
    references = select_species_references(
        reference_records,
        references_per_species=args.references_per_species,
        allowed_species=allowed_species,
    )
    validate_records_exist(references)
    if not references:
        print("species eval skipped: no reference records")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = config["model"]
    genus_model = build_siamese_network(
        BackboneSpec(name=model_config["backbone"], pretrained=False)
    ).to(device)
    species_model = build_siamese_network(
        BackboneSpec(name=model_config["backbone"], pretrained=False)
    ).to(device)
    genus_model.load_state_dict(torch.load(args.genus_checkpoint, map_location=device))
    species_model.load_state_dict(torch.load(args.species_checkpoint, map_location=device))
    genus_model.eval()
    species_model.eval()

    image_size = int(config["views"]["global"]["image_size"])
    crop_size = int(config["views"]["local"]["crop_size"])
    preprocessing = _preprocessing_enabled(config)
    embeddings = build_reference_embeddings(
        genus_model=genus_model,
        species_model=species_model,
        references=references,
        image_size=image_size,
        crop_size=crop_size,
        preprocessing=preprocessing,
        device=device,
    )
    predictor = TwoStageSiamesePredictor(
        genus_model=genus_model,
        species_model=species_model,
        references=embeddings,
        genus_candidates=args.genus_candidates,
        top_k=max(args.top_k),
        image_size=image_size,
        local_crop_size=crop_size,
        preprocessing=preprocessing,
        device=str(device),
    )

    result = evaluate_species_retrieval(
        predictor=predictor,
        queries=query_records,
        top_ks=tuple(args.top_k),
    )
    print(
        f"references={result.references} queries={result.queries} "
        f"top_k={list(result.top_ks)} genus_candidates={args.genus_candidates}"
    )
    for top_k in result.top_ks:
        print(
            f"top{top_k}_species_accuracy={result.accuracies[top_k]:.3f} "
            f"({result.hits[top_k]}/{result.queries})"
        )
    print(f"mean_inverse_rank_at_{max(result.top_ks)}={result.mean_inverse_rank:.3f}")
    print("query species distribution:", describe_species_distribution(query_records))
    print("reference species distribution:", describe_species_distribution(references))
    return 0


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _load_records(dataset_config: dict) -> list[ImageRecord]:
    return load_metadata_csv(
        metadata_path=Path(dataset_config["metadata"]),
        dataset_root=Path(dataset_config["root"]),
        image_column=dataset_config["image_column"],
        family_column=dataset_config["family_column"],
        genus_column=dataset_config["genus_column"],
        species_column=dataset_config["species_column"],
    )


def _preprocessing_enabled(config: dict) -> bool:
    preprocessing = config.get("preprocessing", {})
    return bool(preprocessing.get("enabled", preprocessing.get("leaf_bbox", False)))


def _require_checkpoint(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {path}. Run both S-CNN training stages first."
        )


if __name__ == "__main__":
    raise SystemExit(main())
