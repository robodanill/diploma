from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from plant_classifier.data import ImageRecord, filter_records_by_split, load_metadata_csv
from plant_classifier.inference import ModelArtifacts, create_predictor
from plant_classifier.inference.scnn import TwoStageSiamesePredictor
from plant_classifier.training.species_eval import evaluate_species_retrieval
from plant_classifier.training.validation import validate_records_exist


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the exact model artifacts and inference profile used by the app."
    )
    parser.add_argument("--query-config", type=Path, default=Path("configs/leafscan_test.yaml"))
    parser.add_argument(
        "--genus-checkpoint",
        type=Path,
        default=Path("weights/scnn_genus_vgg16.pt"),
    )
    parser.add_argument(
        "--species-checkpoint",
        type=Path,
        default=Path("weights/scnn_species_vgg16.pt"),
    )
    parser.add_argument(
        "--reference-index",
        type=Path,
        default=Path("weights/reference_index_leafscan_vgg16.pt"),
    )
    parser.add_argument("--query-split", default="")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()

    query_config = _load_config(args.query_config)
    queries = filter_records_by_split(_load_records(query_config["dataset"]), args.query_split)
    validate_records_exist(queries)
    if not queries:
        print("artifact eval skipped: no query records")
        return 0

    artifacts = ModelArtifacts(
        genus_checkpoint=args.genus_checkpoint,
        species_checkpoint=args.species_checkpoint,
        reference_index=args.reference_index,
    )
    predictor = create_predictor(artifacts)
    if not isinstance(predictor, TwoStageSiamesePredictor):
        raise TypeError("Expected a TwoStageSiamesePredictor for artifact evaluation")

    ranking_limit = len({reference.species for reference in predictor.references})
    result = evaluate_species_retrieval(
        predictor=predictor,
        queries=queries,
        top_ks=tuple(args.top_k),
        ranking_limit=ranking_limit,
    )
    print(
        f"genus_references={result.genus_references} species_references={result.references} "
        f"queries={result.queries} top_k={list(result.top_ks)} "
        f"preprocessing={artifacts.preprocessing} "
        f"local_crop_position={artifacts.local_crop_position} "
        f"genus_score_mode={artifacts.genus_score_mode} "
        f"species_score_mode={artifacts.species_score_mode} "
        f"species_aggregation={artifacts.species_aggregation} "
        f"genus_candidate_mode={artifacts.genus_candidate_mode} "
        f"genus_weight_mode={artifacts.genus_weight_mode} "
        f"confidence_temperature={artifacts.confidence_temperature}"
    )
    for top_k in result.top_ks:
        print(
            f"top{top_k}_species_accuracy={result.accuracies[top_k]:.3f} "
            f"({result.hits[top_k]}/{result.queries})"
        )
    print(f"plantclef_s={result.plantclef_s:.3f}")
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


if __name__ == "__main__":
    raise SystemExit(main())
