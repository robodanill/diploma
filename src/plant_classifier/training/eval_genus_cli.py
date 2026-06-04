from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
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
    describe_genus_reference_coverage,
    describe_species_per_genus,
    evaluate_genus_retrieval,
    embed_image,
    rank_references,
    select_reference_records,
    split_references_and_queries,
)


@dataclass(frozen=True)
class GenusEvalItem:
    image_path: str
    expected_genus: str
    expected_species: str
    predicted_genus: str
    predicted_species: str
    predicted_score: float
    rank: int | None


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
    parser.add_argument(
        "--reference-seed",
        type=int,
        help=(
            "Seed random reference selection. Use this for paper-style random reference "
            "sets while keeping runs reproducible."
        ),
    )
    parser.add_argument("--query-split", default="")
    parser.add_argument(
        "--score-mode",
        choices=("comparator", "l1"),
        default="comparator",
        help="Use the learned S-CNN comparator or raw L1 embedding distance for diagnostics.",
    )
    parser.add_argument(
        "--all-reference-species",
        action="store_true",
        help="Use references for all train species instead of restricting to query species.",
    )
    parser.add_argument(
        "--use-full-reference-pool",
        action="store_true",
        help="Do not apply the config six-shot subset to reference records.",
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write genus predictions, confusion matrix, and diagnostic plots.",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {args.checkpoint}. "
            "Run plant-classifier-train first and check that it finished successfully."
        )

    config = _load_config(args.config)
    dataset_config = config["dataset"]
    records = _load_records(dataset_config)
    if not args.use_full_reference_pool:
        records = _apply_subset(records, dataset_config)
    references, queries, reference_pool = _build_eval_sets(records, args)
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
        score_mode=args.score_mode,
    )

    print(
        f"references={result.references} queries={result.queries} "
        f"top_k={list(result.top_ks)} score_mode={result.score_mode} "
        f"reference_seed={args.reference_seed}"
    )
    for top_k in result.top_ks:
        print(
            f"top{top_k}_genus_accuracy={result.accuracies[top_k]:.3f} "
            f"({result.hits[top_k]}/{result.queries})"
        )
    print("query genus distribution:", describe_genus_distribution(queries))
    print("reference genus distribution:", describe_genus_distribution(references))
    print("reference pool species per genus:", describe_species_per_genus(reference_pool))
    print(
        "genus reference species coverage:",
        describe_genus_reference_coverage(reference_pool, references),
    )
    if args.output_dir:
        _write_eval_artifacts(
            model=model,
            references=references,
            queries=queries,
            image_size=int(config["views"]["global"]["image_size"]),
            crop_size=int(config["views"]["local"]["crop_size"]),
            top_ks=tuple(args.top_k),
            device=device,
            preprocessing=_preprocessing_enabled(config),
            score_mode=args.score_mode,
            output_dir=args.output_dir,
        )
    return 0


def _build_eval_sets(
    records: list[ImageRecord],
    args,
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
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
            seed=args.reference_seed,
        )
        return references, query_records, reference_records

    if args.max_species:
        records = limit_records_by_species(
            records,
            max_species=args.max_species,
            min_images_per_species=3,
            max_images_per_species=args.references_per_genus + args.queries_per_genus,
        )

    references, queries = split_references_and_queries(
        records,
        references_per_genus=args.references_per_genus,
        queries_per_genus=args.queries_per_genus,
    )
    return references, queries, records


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


def _preprocessing_enabled(config: dict) -> bool:
    preprocessing = config.get("preprocessing", {})
    return bool(preprocessing.get("enabled", preprocessing.get("leaf_bbox", False)))


@torch.inference_mode()
def _write_eval_artifacts(
    model,
    references: list[ImageRecord],
    queries: list[ImageRecord],
    image_size: int,
    crop_size: int,
    top_ks: tuple[int, ...],
    device: torch.device,
    preprocessing: bool,
    score_mode: str,
    output_dir: Path,
) -> None:
    from plant_classifier.training.image_pairs import build_image_transform

    output_dir.mkdir(parents=True, exist_ok=True)
    top_ks = tuple(sorted(set(top_ks)))
    transform = build_image_transform(
        "global",
        image_size=image_size,
        crop_size=crop_size,
        preprocessing=preprocessing,
    )
    reference_embeddings = [
        (record, embed_image(model, record.image_path, transform, device)) for record in references
    ]
    items: list[GenusEvalItem] = []
    hits = {top_k: 0 for top_k in top_ks}
    for query in queries:
        query_embedding = embed_image(model, query.image_path, transform, device)
        ranked = rank_references(
            model,
            query_embedding,
            reference_embeddings,
            score_mode=score_mode,
        )
        rank = _rank_of_genus(query.genus, [record.genus for record, _score in ranked])
        for top_k in top_ks:
            if rank is not None and rank <= top_k:
                hits[top_k] += 1
        top_record, top_score = ranked[0]
        items.append(
            GenusEvalItem(
                image_path=str(query.image_path),
                expected_genus=query.genus,
                expected_species=query.species,
                predicted_genus=top_record.genus,
                predicted_species=top_record.species,
                predicted_score=float(top_score),
                rank=rank,
            )
        )

    labels, matrix = _confusion_matrix(items)
    _write_predictions_csv(items, output_dir / "genus_predictions.csv", top_ks)
    _write_confusion_csv(labels, matrix, output_dir / "genus_confusion_matrix.csv")
    _write_summary_csv(
        hits=hits,
        top_ks=top_ks,
        queries=len(queries),
        references=len(references),
        score_mode=score_mode,
        output_path=output_dir / "genus_eval_summary.csv",
    )
    _plot_topk_accuracy(hits, top_ks, len(queries), output_dir / "topk_genus_accuracy.png")
    _plot_rank_histogram(items, output_dir / "genus_rank_histogram.png")
    _plot_per_genus_accuracy(items, output_dir / "per_genus_top1_accuracy.png")
    _plot_confusion_matrix(labels, matrix, output_dir / "genus_confusion_matrix.png")
    print(f"saved genus eval artifacts to {output_dir}")


def _rank_of_genus(expected_genus: str, ranked_genera: list[str]) -> int | None:
    for index, genus in enumerate(ranked_genera, start=1):
        if genus == expected_genus:
            return index
    return None


def _write_predictions_csv(
    items: list[GenusEvalItem],
    output_path: Path,
    top_ks: tuple[int, ...],
) -> None:
    fieldnames = [
        "image_path",
        "expected_genus",
        "expected_species",
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
                "expected_genus": item.expected_genus,
                "expected_species": item.expected_species,
                "predicted_genus": item.predicted_genus,
                "predicted_species": item.predicted_species,
                "predicted_score": item.predicted_score,
                "rank": item.rank or "",
            }
            for top_k in top_ks:
                row[f"hit_top{top_k}"] = int(item.rank is not None and item.rank <= top_k)
            writer.writerow(row)


def _confusion_matrix(items: list[GenusEvalItem]) -> tuple[list[str], list[list[int]]]:
    labels = sorted(
        {
            label
            for item in items
            for label in (item.expected_genus, item.predicted_genus or "<none>")
        }
    )
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for item in items:
        expected_index = label_to_index[item.expected_genus]
        predicted_index = label_to_index[item.predicted_genus or "<none>"]
        matrix[expected_index][predicted_index] += 1
    return labels, matrix


def _write_confusion_csv(labels: list[str], matrix: list[list[int]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["expected\\predicted", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])


def _write_summary_csv(
    hits: dict[int, int],
    top_ks: tuple[int, ...],
    queries: int,
    references: int,
    score_mode: str,
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerow(["queries", queries])
        writer.writerow(["references", references])
        writer.writerow(["score_mode", score_mode])
        for top_k in top_ks:
            writer.writerow([f"top{top_k}_genus_accuracy", f"{hits[top_k] / queries:.6f}"])


def _plot_topk_accuracy(
    hits: dict[int, int],
    top_ks: tuple[int, ...],
    queries: int,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    values = [hits[top_k] / queries for top_k in top_ks]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar([f"top-{top_k}" for top_k in top_ks], values, color="#2563eb")
    axis.set_ylim(0, 1)
    axis.set_ylabel("accuracy")
    axis.set_title("Genus top-k accuracy")
    for index, value in enumerate(values):
        axis.text(index, value + 0.02, f"{value:.3f}", ha="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_rank_histogram(items: list[GenusEvalItem], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    ranks = [item.rank for item in items if item.rank is not None]
    missed = sum(1 for item in items if item.rank is None)
    figure, axis = plt.subplots(figsize=(7, 4))
    if ranks:
        axis.hist(ranks, bins=range(1, max(ranks) + 2), color="#059669", edgecolor="white")
    axis.set_xlabel("rank of first correct-genus reference")
    axis.set_ylabel("queries")
    axis.set_title(f"Correct genus rank distribution; missed={missed}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_per_genus_accuracy(items: list[GenusEvalItem], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    totals: Counter[str] = Counter(item.expected_genus for item in items)
    hits: Counter[str] = Counter(
        item.expected_genus
        for item in items
        if item.rank is not None and item.rank == 1
    )
    genera = sorted(totals, key=lambda label: (hits[label] / totals[label], totals[label], label))
    values = [hits[label] / totals[label] for label in genera]
    figure, axis = plt.subplots(figsize=(max(9, len(genera) * 0.24), 5))
    axis.bar(range(len(genera)), values, color="#f59e0b")
    axis.set_ylim(0, 1)
    axis.set_ylabel("top-1 accuracy")
    axis.set_title("Per-genus top-1 accuracy")
    axis.set_xticks(range(len(genera)))
    axis.set_xticklabels(genera, rotation=90, fontsize=6)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_confusion_matrix(labels: list[str], matrix: list[list[int]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure_size = max(9, len(labels) * 0.24)
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title("Genus confusion matrix")
    axis.set_xlabel("predicted")
    axis.set_ylabel("expected")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=90, fontsize=6)
    axis.set_yticklabels(labels, fontsize=6)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
