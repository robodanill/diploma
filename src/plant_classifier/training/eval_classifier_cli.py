from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from plant_classifier.data import filter_records_by_split, load_metadata_csv
from plant_classifier.models.classifier import ClassifierSpec, build_image_classifier
from plant_classifier.training.classifier_dataset import ClassificationImageDataset
from plant_classifier.training.validation import validate_records_exist


@dataclass(frozen=True)
class ClassifierEvalItem:
    image_path: str
    expected_family: str
    expected_genus: str
    expected_species: str
    predicted_species: str
    predicted_score: float
    rank: int | None


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a supervised VGG16 species baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/leafscan_paper60_vgg16_baseline.yaml"))
    parser.add_argument("--query-config", type=Path, default=Path("configs/leafscan_test.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--query-split", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write predictions, confusion matrix, and diagnostic plots to this directory.",
    )
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
    print(f"plantclef_s={result['mean_inverse_rank']:.3f}")
    if args.output_dir:
        _write_eval_artifacts(result, args.output_dir)
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
    max_top_k = min(max(top_ks), len(idx_to_class))
    ranking_limit = max(max_top_k, len(idx_to_class))
    hits = {top_k: 0 for top_k in top_ks}
    inverse_rank_sum = 0.0
    offset = 0
    items: list[ClassifierEvalItem] = []
    for images, _labels in dataloader:
        images = images.to(device)
        logits = model(images)
        scores, predicted = logits.softmax(dim=1).topk(ranking_limit, dim=1)
        for row, score_row in zip(predicted.cpu().tolist(), scores.cpu().tolist(), strict=True):
            record = records[offset]
            expected = record.species
            ranked_species = [idx_to_class[index] for index in row]
            rank = _rank_of(expected, ranked_species)
            if rank is not None:
                inverse_rank_sum += 1.0 / rank
            for top_k in top_ks:
                if rank is not None and rank <= top_k:
                    hits[top_k] += 1
            items.append(
                ClassifierEvalItem(
                    image_path=str(record.image_path),
                    expected_family=record.family,
                    expected_genus=record.genus,
                    expected_species=record.species,
                    predicted_species=ranked_species[0] if ranked_species else "",
                    predicted_score=score_row[0] if score_row else 0.0,
                    rank=rank,
                )
            )
            offset += 1
    query_count = len(records)
    plantclef_s = inverse_rank_sum / query_count
    return {
        "top_ks": top_ks,
        "hits": hits,
        "accuracies": {top_k: hits[top_k] / query_count for top_k in top_ks},
        "mean_inverse_rank": plantclef_s,
        "plantclef_s": plantclef_s,
        "queries": query_count,
        "items": tuple(items),
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


def _write_eval_artifacts(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = result["items"]
    _write_predictions_csv(items, output_dir / "classifier_predictions.csv", result["top_ks"])
    labels, matrix = _confusion_matrix(items)
    _write_confusion_csv(labels, matrix, output_dir / "classifier_confusion_matrix.csv")
    _write_summary_csv(result, output_dir / "classifier_eval_summary.csv")
    _plot_topk_accuracy(result, output_dir / "classifier_topk_accuracy.png")
    _plot_rank_histogram(items, output_dir / "classifier_rank_histogram.png")
    _plot_per_species_accuracy(items, output_dir / "classifier_per_species_top1_accuracy.png")
    _plot_confusion_matrix(labels, matrix, output_dir / "classifier_confusion_matrix.png")
    print(f"saved classifier eval artifacts to {output_dir}")


def _write_predictions_csv(
    items: tuple[ClassifierEvalItem, ...],
    output_path: Path,
    top_ks: tuple[int, ...],
) -> None:
    fieldnames = [
        "image_path",
        "expected_family",
        "expected_genus",
        "expected_species",
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
                "predicted_species": item.predicted_species,
                "predicted_score": item.predicted_score,
                "rank": item.rank or "",
            }
            for top_k in top_ks:
                row[f"hit_top{top_k}"] = int(item.rank is not None and item.rank <= top_k)
            writer.writerow(row)


def _write_summary_csv(result: dict, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerow(["queries", result["queries"]])
        writer.writerow(["plantclef_s", f"{result['plantclef_s']:.6f}"])
        for top_k in result["top_ks"]:
            writer.writerow([f"top{top_k}_species_accuracy", f"{result['accuracies'][top_k]:.6f}"])


def _confusion_matrix(
    items: tuple[ClassifierEvalItem, ...],
) -> tuple[list[str], list[list[int]]]:
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


def _plot_topk_accuracy(result: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    top_ks = list(result["top_ks"])
    values = [result["accuracies"][top_k] for top_k in top_ks]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar([f"top-{top_k}" for top_k in top_ks], values, color="#3b82f6")
    axis.set_ylim(0, 1)
    axis.set_ylabel("accuracy")
    axis.set_title("Classifier species top-k accuracy")
    for index, value in enumerate(values):
        axis.text(index, value + 0.02, f"{value:.3f}", ha="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_rank_histogram(items: tuple[ClassifierEvalItem, ...], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    ranks = [item.rank for item in items if item.rank is not None]
    missed = sum(1 for item in items if item.rank is None)
    figure, axis = plt.subplots(figsize=(7, 4))
    if ranks:
        axis.hist(ranks, bins=range(1, max(ranks) + 2), color="#10b981", edgecolor="white")
    axis.set_xlabel("rank of correct species")
    axis.set_ylabel("queries")
    axis.set_title(f"Classifier rank distribution; missed={missed}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_per_species_accuracy(
    items: tuple[ClassifierEvalItem, ...],
    output_path: Path,
) -> None:
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
    axis.set_title("Classifier per-species top-1 accuracy")
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
    axis.set_title("Classifier species confusion matrix")
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
