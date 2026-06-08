from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an expanded PlantCLEF paper60 metadata file for the S-CNN(B) "
            "species stage from honest two-stage prediction errors."
        )
    )
    parser.add_argument(
        "--source-metadata",
        type=Path,
        default=Path("data/plantclef2015/leafscan_paper60_metadata.csv"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="species_predictions.csv produced by eval_species_cli.",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=Path("data/plantclef2015/leafscan_paper60_scnn2_boost_metadata.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("data/plantclef2015/leafscan_paper60_scnn2_boost_summary.csv"),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/leafscan_paper60_training.yaml"),
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("configs/leafscan_paper60_scnn2_boost_generated.yaml"),
    )
    parser.add_argument("--base-images-per-species", type=int, default=6)
    parser.add_argument("--boost-images-per-species", type=int, default=18)
    parser.add_argument("--max-boost-species", type=int, default=24)
    parser.add_argument("--targeted-negative-pairs", type=int, default=24)
    parser.add_argument("--targeted-negative-ratio", type=float, default=0.20)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_rows, fieldnames = _read_rows(args.source_metadata)
    predictions = _read_prediction_rows(args.predictions)
    stats = _prediction_stats(predictions)
    boost_species = _select_boost_species(stats, args.max_boost_species)
    selected_rows = _select_metadata_rows(
        source_rows=source_rows,
        boost_species=boost_species,
        base_images_per_species=args.base_images_per_species,
        boost_images_per_species=args.boost_images_per_species,
        seed=args.seed,
    )
    _write_rows(args.output_metadata, selected_rows, fieldnames)
    confusion_pairs = _select_confusion_pairs(
        predictions=predictions,
        limit=args.targeted_negative_pairs,
    )
    _write_summary(
        output_path=args.summary_csv,
        source_rows=source_rows,
        selected_rows=selected_rows,
        stats=stats,
        boost_species=boost_species,
        confusion_pairs=confusion_pairs,
    )
    _write_training_config(
        base_config_path=args.base_config,
        output_config_path=args.output_config,
        metadata_path=args.output_metadata,
        confusion_pairs=confusion_pairs,
        targeted_negative_ratio=args.targeted_negative_ratio,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
    )
    print(f"source rows: {len(source_rows)}")
    print(f"selected rows: {len(selected_rows)}")
    print(f"boosted species: {len(boost_species)}")
    for species in boost_species:
        item = stats[species]
        print(
            "boost:",
            species,
            f"queries={item['queries']}",
            f"top1_misses={item['top1_misses']}",
            f"same_genus_top1_misses={item['same_genus_top1_misses']}",
            f"top5_misses={item['top5_misses']}",
        )
    print(f"targeted species-negative pairs: {len(confusion_pairs)}")
    print(f"metadata: {args.output_metadata}")
    print(f"summary: {args.summary_csv}")
    print(f"config: {args.output_config}")
    return 0


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Metadata is empty: {path}")
    for column in ("image_path", "family", "genus", "species"):
        if column not in fieldnames:
            raise ValueError(f"Metadata {path} misses required column: {column}")
    return rows, fieldnames


def _read_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Predictions are empty: {path}")
    for column in ("expected_species", "expected_genus", "predicted_species", "predicted_genus"):
        if column not in rows[0]:
            raise ValueError(f"Predictions {path} miss required column: {column}")
    return rows


def _prediction_stats(predictions: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "queries": 0,
            "top1_hits": 0,
            "top1_misses": 0,
            "top5_hits": 0,
            "top5_misses": 0,
            "same_genus_top1_misses": 0,
            "cross_genus_top1_misses": 0,
        }
    )
    for row in predictions:
        species = row["expected_species"]
        item = stats[species]
        item["queries"] += 1
        rank = _rank(row.get("rank", ""))
        top1_hit = rank == 1
        top5_hit = rank is not None and rank <= 5
        if top1_hit:
            item["top1_hits"] += 1
        else:
            item["top1_misses"] += 1
            if row.get("predicted_genus") == row.get("expected_genus"):
                item["same_genus_top1_misses"] += 1
            else:
                item["cross_genus_top1_misses"] += 1
        if top5_hit:
            item["top5_hits"] += 1
        else:
            item["top5_misses"] += 1
    return dict(stats)


def _rank(value: str) -> int | None:
    value = str(value).strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _select_boost_species(stats: dict[str, dict[str, int]], limit: int) -> list[str]:
    def score(item: tuple[str, dict[str, int]]) -> tuple[int, int, int, int, str]:
        species, values = item
        return (
            values["same_genus_top1_misses"] * 4
            + values["top5_misses"] * 3
            + values["top1_misses"],
            values["same_genus_top1_misses"],
            values["top5_misses"],
            values["queries"],
            species,
        )

    candidates = [
        item
        for item in stats.items()
        if item[1]["top1_misses"] > 0
        and (item[1]["same_genus_top1_misses"] > 0 or item[1]["top5_misses"] > 0)
    ]
    if len(candidates) < limit:
        seen = {species for species, _values in candidates}
        candidates.extend(
            item for item in stats.items() if item[0] not in seen and item[1]["top1_misses"] > 0
        )
    return [species for species, _values in sorted(candidates, key=score, reverse=True)[:limit]]


def _select_metadata_rows(
    source_rows: list[dict[str, str]],
    boost_species: list[str],
    base_images_per_species: int,
    boost_images_per_species: int,
    seed: int,
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        grouped[row["species"]].append(row)

    boosted = set(boost_species)
    selected: list[dict[str, str]] = []
    for species in sorted(grouped):
        species_rows = sorted(grouped[species], key=lambda row: row["image_path"])
        rng = random.Random(_stable_seed(seed, species))
        rng.shuffle(species_rows)
        limit = boost_images_per_species if species in boosted else base_images_per_species
        selected.extend(species_rows[: min(limit, len(species_rows))])
    return sorted(selected, key=lambda row: (row["species"], row["image_path"]))


def _stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _select_confusion_pairs(
    predictions: list[dict[str, str]],
    limit: int,
) -> list[tuple[str, str]]:
    same_genus_counts: Counter[tuple[str, str]] = Counter()
    other_counts: Counter[tuple[str, str]] = Counter()
    for row in predictions:
        expected = row.get("expected_species", "")
        predicted = row.get("predicted_species", "")
        if not expected or not predicted or expected == predicted:
            continue
        pair = (expected, predicted)
        if row.get("expected_genus") == row.get("predicted_genus"):
            same_genus_counts[pair] += 1
        else:
            other_counts[pair] += 1

    pairs = [pair for pair, _count in same_genus_counts.most_common(limit)]
    if len(pairs) < limit:
        seen = set(pairs)
        pairs.extend(
            pair
            for pair, _count in other_counts.most_common(limit - len(pairs))
            if pair not in seen
        )
    return pairs[:limit]


def _write_summary(
    output_path: Path,
    source_rows: list[dict[str, str]],
    selected_rows: list[dict[str, str]],
    stats: dict[str, dict[str, int]],
    boost_species: list[str],
    confusion_pairs: list[tuple[str, str]],
) -> None:
    source_counts = Counter(row["species"] for row in source_rows)
    selected_counts = Counter(row["species"] for row in selected_rows)
    boosted = set(boost_species)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "species",
                "boosted",
                "source_images",
                "selected_images",
                "queries",
                "top1_hits",
                "top1_misses",
                "top5_hits",
                "top5_misses",
                "same_genus_top1_misses",
                "cross_genus_top1_misses",
            ],
        )
        writer.writeheader()
        for species in sorted(source_counts):
            item = stats.get(species, {})
            writer.writerow(
                {
                    "species": species,
                    "boosted": int(species in boosted),
                    "source_images": source_counts[species],
                    "selected_images": selected_counts[species],
                    "queries": item.get("queries", 0),
                    "top1_hits": item.get("top1_hits", 0),
                    "top1_misses": item.get("top1_misses", 0),
                    "top5_hits": item.get("top5_hits", 0),
                    "top5_misses": item.get("top5_misses", 0),
                    "same_genus_top1_misses": item.get("same_genus_top1_misses", 0),
                    "cross_genus_top1_misses": item.get("cross_genus_top1_misses", 0),
                }
            )
    pairs_path = output_path.with_name(f"{output_path.stem}_targeted_pairs.csv")
    with pairs_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["expected_species", "predicted_species"])
        writer.writerows(confusion_pairs)


def _write_training_config(
    base_config_path: Path,
    output_config_path: Path,
    metadata_path: Path,
    confusion_pairs: list[tuple[str, str]],
    targeted_negative_ratio: float,
    checkpoint_every_epochs: int,
) -> None:
    config = _read_yaml(base_config_path)
    config["dataset"]["name"] = "PlantCLEF2015LeafScanPaper60SCNN2Boost"
    config["dataset"]["metadata"] = str(metadata_path)
    config["dataset"].pop("subset", None)

    pair_sampling = config.setdefault("pair_sampling", {})
    targeted_ratio = _stage_mapping(pair_sampling.get("targeted_negative_ratio", 0.0))
    targeted_ratio["species"] = targeted_negative_ratio if confusion_pairs else 0.0
    pair_sampling["targeted_negative_ratio"] = targeted_ratio

    targeted_pairs = pair_sampling.get("targeted_negative_label_pairs") or {}
    if not isinstance(targeted_pairs, dict):
        targeted_pairs = {"default": targeted_pairs}
    targeted_pairs["species"] = [list(pair) for pair in confusion_pairs]
    pair_sampling["targeted_negative_label_pairs"] = targeted_pairs

    checkpoint_map = _stage_mapping(config.setdefault("training", {}).get("checkpoint_every_epochs", 0))
    checkpoint_map["species"] = checkpoint_every_epochs
    config["training"]["checkpoint_every_epochs"] = checkpoint_map

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    with output_config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _stage_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {"default": value}


if __name__ == "__main__":
    raise SystemExit(main())
