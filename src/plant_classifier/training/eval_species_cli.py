from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import torch
import yaml

from plant_classifier.data import (
    ImageRecord,
    filter_records_by_split,
    limit_records_by_species,
    load_metadata_csv,
)
from plant_classifier.inference.scnn import TwoStageSiamesePredictor
from plant_classifier.models.siamese import BackboneSpec, build_siamese_network
from plant_classifier.training.genus_eval import (
    describe_genus_reference_coverage,
    describe_species_per_genus,
    select_genus_references,
)
from plant_classifier.training.species_eval import (
    SpeciesEvalItem,
    SpeciesEvalResult,
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
    parser.add_argument("--genus-references-per-genus", type=int, default=6)
    parser.add_argument("--references-per-species", type=int, default=6)
    parser.add_argument("--genus-candidates", type=int, default=30)
    parser.add_argument("--reference-split", default="train")
    parser.add_argument("--query-split", default="")
    parser.add_argument("--reference-seed", type=int)
    parser.add_argument(
        "--genus-score-mode",
        choices=("comparator", "l1"),
        default="l1",
        help="Scoring used by S-CNN(A) for the coarse genus reference ranking.",
    )
    parser.add_argument(
        "--species-score-mode",
        choices=("comparator", "l1"),
        default="comparator",
        help="Scoring used by S-CNN(B) for local species reference comparisons.",
    )
    parser.add_argument(
        "--species-aggregation",
        choices=("max", "mean", "sum"),
        default="max",
        help="How to combine multiple reference scores for the same species.",
    )
    parser.add_argument(
        "--genus-candidate-mode",
        choices=("reference", "unique"),
        default="reference",
        help=(
            "How the genus gate is built: top reference images, or first occurrences "
            "of unique genera in the ranked reference list."
        ),
    )
    parser.add_argument(
        "--genus-weight-mode",
        choices=("frequency", "score", "uniform"),
        default="frequency",
        help="How selected genus candidates weight species scores.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write predictions, confusion matrix, and diagnostic plots to this directory.",
    )
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
    reference_records = _apply_subset(reference_records, config["dataset"])
    query_records = filter_records_by_split(_load_records(query_config["dataset"]), args.query_split)
    validate_records_exist(reference_records)
    validate_records_exist(query_records)
    if not query_records:
        print("species eval skipped: no query records")
        return 0

    allowed_species = None if args.all_reference_species else {record.species for record in query_records}
    selected_reference_records = _filter_allowed_species(reference_records, allowed_species)
    genus_references = select_genus_references(
        selected_reference_records,
        references_per_genus=args.genus_references_per_genus,
        seed=args.reference_seed,
    )
    species_references = select_species_references(
        reference_records,
        references_per_species=args.references_per_species,
        allowed_species=allowed_species,
    )
    validate_records_exist(genus_references)
    validate_records_exist(species_references)
    if not genus_references or not species_references:
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
    crop_position = _local_crop_position(config)
    preprocessing = _preprocessing_enabled(config)
    genus_embeddings = build_reference_embeddings(
        genus_model=genus_model,
        species_model=species_model,
        references=genus_references,
        image_size=image_size,
        crop_size=crop_size,
        crop_position=crop_position,
        preprocessing=preprocessing,
        device=device,
    )
    species_embeddings = build_reference_embeddings(
        genus_model=genus_model,
        species_model=species_model,
        references=species_references,
        image_size=image_size,
        crop_size=crop_size,
        crop_position=crop_position,
        preprocessing=preprocessing,
        device=device,
    )
    predictor = TwoStageSiamesePredictor(
        genus_model=genus_model,
        species_model=species_model,
        references=species_embeddings,
        genus_references=genus_embeddings,
        genus_candidates=args.genus_candidates,
        top_k=max(args.top_k),
        image_size=image_size,
        local_crop_size=crop_size,
        local_crop_position=crop_position,
        preprocessing=preprocessing,
        genus_score_mode=args.genus_score_mode,
        species_score_mode=args.species_score_mode,
        species_aggregation=args.species_aggregation,
        genus_candidate_mode=args.genus_candidate_mode,
        genus_weight_mode=args.genus_weight_mode,
        device=str(device),
    )

    ranking_limit = len({record.species for record in species_references})
    result = evaluate_species_retrieval(
        predictor=predictor,
        queries=query_records,
        top_ks=tuple(args.top_k),
        ranking_limit=ranking_limit,
    )
    print(
        f"genus_references={result.genus_references} species_references={result.references} "
        f"queries={result.queries} top_k={list(result.top_ks)} "
        f"genus_candidates={args.genus_candidates} "
        f"genus_score_mode={args.genus_score_mode} "
        f"species_score_mode={args.species_score_mode} "
        f"species_aggregation={args.species_aggregation} "
        f"genus_candidate_mode={args.genus_candidate_mode} "
        f"genus_weight_mode={args.genus_weight_mode} "
        f"reference_seed={args.reference_seed}"
    )
    for top_k in result.top_ks:
        print(
            f"top{top_k}_species_accuracy={result.accuracies[top_k]:.3f} "
            f"({result.hits[top_k]}/{result.queries})"
        )
    print(f"plantclef_s={result.plantclef_s:.3f}")
    print(f"mean_inverse_rank={result.mean_inverse_rank:.3f}")
    print("query species distribution:", describe_species_distribution(query_records))
    print("genus reference distribution:", _describe_genus_distribution(genus_references))
    print("species reference distribution:", describe_species_distribution(species_references))
    print("reference pool species per genus:", describe_species_per_genus(selected_reference_records))
    print(
        "genus reference species coverage:",
        describe_genus_reference_coverage(selected_reference_records, genus_references),
    )
    if args.output_dir:
        _write_eval_artifacts(result, args.output_dir)
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


def _apply_subset(records: list[ImageRecord], dataset_config: dict) -> list[ImageRecord]:
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
        f"using reference subset: {len(limited)} images "
        f"from {len({record.species for record in limited})} species",
        flush=True,
    )
    return limited


def _filter_allowed_species(
    records: list[ImageRecord],
    allowed_species: set[str] | None,
) -> list[ImageRecord]:
    if allowed_species is None:
        return records
    return [record for record in records if record.species in allowed_species]


def _describe_genus_distribution(records: list[ImageRecord], limit: int = 10) -> dict[str, int]:
    return dict(Counter(record.genus for record in records).most_common(limit))


def _preprocessing_enabled(config: dict) -> bool:
    preprocessing = config.get("preprocessing", {})
    return bool(preprocessing.get("enabled", preprocessing.get("leaf_bbox", False)))


def _local_crop_position(config: dict) -> str:
    return str(config.get("views", {}).get("local", {}).get("crop_position", "center"))


def _require_checkpoint(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {path}. Run both S-CNN training stages first."
        )


def _write_eval_artifacts(result: SpeciesEvalResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_predictions_csv(result.items, output_dir / "species_predictions.csv", result.top_ks)
    labels, matrix = _confusion_matrix(result.items)
    _write_confusion_csv(labels, matrix, output_dir / "species_confusion_matrix.csv")
    _write_summary_csv(result, output_dir / "species_eval_summary.csv")
    _plot_topk_accuracy(result, output_dir / "topk_species_accuracy.png")
    _plot_rank_histogram(result.items, output_dir / "species_rank_histogram.png")
    _plot_per_species_accuracy(result.items, output_dir / "per_species_top1_accuracy.png")
    _plot_confusion_matrix(labels, matrix, output_dir / "species_confusion_matrix.png")
    print(f"saved eval artifacts to {output_dir}")


def _write_predictions_csv(
    items: tuple[SpeciesEvalItem, ...],
    output_path: Path,
    top_ks: tuple[int, ...],
) -> None:
    fieldnames = [
        "image_path",
        "expected_family",
        "expected_genus",
        "expected_species",
        "predicted_family",
        "predicted_genus",
        "predicted_species",
        "predicted_score",
        "rank",
        *[f"hit_top{top_k}" for top_k in top_ks],
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            row = {
                "image_path": item.image_path,
                "expected_family": item.expected_family,
                "expected_genus": item.expected_genus,
                "expected_species": item.expected_species,
                "predicted_family": item.predicted_family,
                "predicted_genus": item.predicted_genus,
                "predicted_species": item.predicted_species,
                "predicted_score": item.predicted_score,
                "rank": item.rank or "",
            }
            for top_k in top_ks:
                row[f"hit_top{top_k}"] = int(item.rank is not None and item.rank <= top_k)
            writer.writerow(row)


def _write_summary_csv(result: SpeciesEvalResult, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerow(["queries", result.queries])
        writer.writerow(["genus_references", result.genus_references])
        writer.writerow(["species_references", result.references])
        writer.writerow(["plantclef_s", f"{result.plantclef_s:.6f}"])
        for top_k in result.top_ks:
            writer.writerow([f"top{top_k}_species_accuracy", f"{result.accuracies[top_k]:.6f}"])


def _confusion_matrix(items: tuple[SpeciesEvalItem, ...]) -> tuple[list[str], list[list[int]]]:
    labels = sorted(
        {
            label
            for item in items
            for label in (item.expected_species, item.predicted_species or "<none>")
        }
    )
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for item in items:
        expected_index = label_to_index[item.expected_species]
        predicted_index = label_to_index[item.predicted_species or "<none>"]
        matrix[expected_index][predicted_index] += 1
    return labels, matrix


def _write_confusion_csv(labels: list[str], matrix: list[list[int]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["expected\\predicted", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])


def _plot_topk_accuracy(result: SpeciesEvalResult, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    top_ks = list(result.top_ks)
    values = [result.accuracies[top_k] for top_k in top_ks]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar([f"top-{top_k}" for top_k in top_ks], values, color="#3b82f6")
    axis.set_ylim(0, 1)
    axis.set_ylabel("accuracy")
    axis.set_title("Species top-k accuracy")
    for index, value in enumerate(values):
        axis.text(index, value + 0.02, f"{value:.3f}", ha="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_rank_histogram(items: tuple[SpeciesEvalItem, ...], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    ranks = [item.rank for item in items if item.rank is not None]
    missed = sum(1 for item in items if item.rank is None)
    figure, axis = plt.subplots(figsize=(7, 4))
    if ranks:
        axis.hist(ranks, bins=range(1, max(ranks) + 2), color="#10b981", edgecolor="white")
    axis.set_xlabel("rank of correct species")
    axis.set_ylabel("queries")
    axis.set_title(f"Correct species rank distribution; missed={missed}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_per_species_accuracy(items: tuple[SpeciesEvalItem, ...], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    totals: Counter[str] = Counter(item.expected_species for item in items)
    hits: Counter[str] = Counter(
        item.expected_species
        for item in items
        if item.rank is not None and item.rank == 1
    )
    species = sorted(totals, key=lambda label: (hits[label] / totals[label], totals[label]))
    values = [hits[label] / totals[label] for label in species]
    figure, axis = plt.subplots(figsize=(max(10, len(species) * 0.22), 5))
    axis.bar(range(len(species)), values, color="#f59e0b")
    axis.set_ylim(0, 1)
    axis.set_ylabel("top-1 accuracy")
    axis.set_title("Per-species top-1 accuracy")
    axis.set_xticks(range(len(species)))
    axis.set_xticklabels([_short_label(label) for label in species], rotation=90, fontsize=6)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_confusion_matrix(labels: list[str], matrix: list[list[int]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure_size = max(10, len(labels) * 0.22)
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title("Species confusion matrix")
    axis.set_xlabel("predicted")
    axis.set_ylabel("expected")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    short_labels = [_short_label(label) for label in labels]
    axis.set_xticklabels(short_labels, rotation=90, fontsize=5)
    axis.set_yticklabels(short_labels, fontsize=5)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _short_label(label: str) -> str:
    parts = label.replace("(", "").replace(")", "").split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {' '.join(parts[1:3])}"
    return label[:18]


if __name__ == "__main__":
    raise SystemExit(main())
