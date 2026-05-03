from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from plant_classifier.data import (
    ImageRecord,
    filter_records_by_split,
    limit_records_by_species,
    load_metadata_csv,
)
from plant_classifier.models.siamese import BackboneSpec, build_siamese_network
from plant_classifier.training.genus_eval import (
    describe_genus_distribution,
    evaluate_genus_retrieval,
    select_reference_records,
    split_references_and_queries,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Quickly evaluate S-CNN(A) genus retrieval.")
    parser.add_argument("--config", type=Path, default=Path("configs/leaf_training.yaml"))
    parser.add_argument("--query-config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-species", type=int, default=40)
    parser.add_argument("--references-per-genus", type=int, default=2)
    parser.add_argument("--queries-per-genus", type=int, default=2)
    parser.add_argument("--reference-level", choices=("genus", "species"), default="genus")
    parser.add_argument("--reference-split", default="train")
    parser.add_argument("--query-split", default="")
    parser.add_argument(
        "--all-reference-species",
        action="store_true",
        help="Use references for all train species instead of restricting to query species.",
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {args.checkpoint}. "
            "Run plant-classifier-train first and check that it finished successfully."
        )

    config = _load_config(args.config)
    dataset_config = config["dataset"]
    records = _load_records(dataset_config)
    references, queries = _build_eval_sets(records, args)
    if not references or not queries:
        print(
            "genus eval skipped: not enough records to create references and queries "
            f"(references={len(references)} queries={len(queries)})"
        )
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = config["model"]
    model = build_siamese_network(
        BackboneSpec(name=model_config["backbone"], pretrained=False)
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    result = evaluate_genus_retrieval(
        model=model,
        references=references,
        queries=queries,
        image_size=int(config["views"]["global"]["image_size"]),
        crop_size=int(config["views"]["local"]["crop_size"]),
        top_ks=tuple(args.top_k),
        device=device,
        preprocessing=_preprocessing_enabled(config),
    )

    print(f"references={result.references} queries={result.queries} top_k={list(result.top_ks)}")
    for top_k in result.top_ks:
        print(
            f"top{top_k}_genus_accuracy={result.accuracies[top_k]:.3f} "
            f"({result.hits[top_k]}/{result.queries})"
        )
    print("query genus distribution:", describe_genus_distribution(queries))
    print("reference genus distribution:", describe_genus_distribution(references))
    return 0


def _build_eval_sets(
    records: list[ImageRecord],
    args,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    if args.query_config:
        reference_records = filter_records_by_split(records, args.reference_split)
        query_config = _load_config(args.query_config)
        query_records = _load_records(query_config["dataset"])
        query_records = filter_records_by_split(query_records, args.query_split)
        if not args.all_reference_species:
            query_species = {record.species for record in query_records}
            reference_records = [
                record for record in reference_records if record.species in query_species
            ]
        if args.max_species:
            reference_records = limit_records_by_species(
                reference_records,
                max_species=args.max_species,
                min_images_per_species=args.references_per_genus,
                max_images_per_species=args.references_per_genus,
            )
        references = select_reference_records(
            reference_records,
            taxonomic_level=args.reference_level,
            references_per_label=args.references_per_genus,
        )
        return references, query_records

    if args.max_species:
        records = limit_records_by_species(
            records,
            max_species=args.max_species,
            min_images_per_species=3,
            max_images_per_species=args.references_per_genus + args.queries_per_genus,
        )

    return split_references_and_queries(
        records,
        references_per_genus=args.references_per_genus,
        queries_per_genus=args.queries_per_genus,
    )


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


if __name__ == "__main__":
    raise SystemExit(main())
