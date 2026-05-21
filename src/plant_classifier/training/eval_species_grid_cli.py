from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import yaml
from PIL import Image

from plant_classifier.data import (
    ImageRecord,
    filter_records_by_split,
    limit_records_by_species,
    load_metadata_csv,
)
from plant_classifier.models.siamese import BackboneSpec, SiameseNetwork, build_siamese_network
from plant_classifier.training.genus_eval import (
    describe_genus_reference_coverage,
    describe_species_per_genus,
    select_genus_references,
)
from plant_classifier.training.image_pairs import build_image_transform
from plant_classifier.training.species_eval import (
    describe_species_distribution,
    select_species_references,
)
from plant_classifier.training.validation import validate_records_exist


@dataclass(frozen=True)
class ReferenceEmbeddings:
    records: tuple[ImageRecord, ...]
    global_embeddings: torch.Tensor
    local_embeddings: torch.Tensor


@dataclass(frozen=True)
class QueryEmbeddings:
    records: tuple[ImageRecord, ...]
    global_embeddings: torch.Tensor
    local_embeddings: torch.Tensor


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a grid of two-stage S-CNN species retrieval settings while "
            "reusing precomputed query/reference embeddings."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/leafscan_paper60_training.yaml"),
    )
    parser.add_argument("--query-config", type=Path, default=Path("configs/leafscan_test.yaml"))
    parser.add_argument("--genus-checkpoint", type=Path, required=True)
    parser.add_argument("--species-checkpoint", type=Path, required=True)
    parser.add_argument("--genus-references-per-genus", type=int, default=6)
    parser.add_argument("--references-per-species", type=int, default=6)
    parser.add_argument("--genus-candidates", type=int, nargs="+", default=[30])
    parser.add_argument("--reference-split", default="train")
    parser.add_argument("--query-split", default="")
    parser.add_argument("--reference-seed", type=int)
    parser.add_argument(
        "--genus-score-modes",
        nargs="+",
        choices=("comparator", "l1"),
        default=["l1"],
    )
    parser.add_argument(
        "--species-score-modes",
        nargs="+",
        choices=("comparator", "l1"),
        default=["comparator"],
    )
    parser.add_argument(
        "--species-aggregations",
        nargs="+",
        choices=("max", "mean", "sum"),
        default=["max"],
    )
    parser.add_argument("--all-reference-species", action="store_true")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--print-top", type=int, default=20)
    args = parser.parse_args()

    _require_checkpoint(args.genus_checkpoint)
    _require_checkpoint(args.species_checkpoint)

    config = _load_config(args.config)
    query_config = _load_config(args.query_config)
    reference_records = filter_records_by_split(
        _load_records(config["dataset"]),
        args.reference_split,
    )
    reference_records = _apply_subset(reference_records, config["dataset"])
    query_records = filter_records_by_split(
        _load_records(query_config["dataset"]),
        args.query_split,
    )
    validate_records_exist(reference_records)
    validate_records_exist(query_records)

    allowed_species = (
        None
        if args.all_reference_species
        else {record.species for record in query_records}
    )
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
    print(
        f"genus_references={len(genus_references)} species_references={len(species_references)} "
        f"queries={len(query_records)} device={device}",
        flush=True,
    )
    print("query species distribution:", describe_species_distribution(query_records))
    print(
        "reference pool species per genus:",
        describe_species_per_genus(selected_reference_records),
    )
    print(
        "genus reference species coverage:",
        describe_genus_reference_coverage(selected_reference_records, genus_references),
    )

    global_transform = build_image_transform(
        "global",
        image_size=image_size,
        crop_size=crop_size,
        preprocessing=preprocessing,
    )
    local_transform = build_image_transform(
        "local",
        image_size=image_size,
        crop_size=crop_size,
        preprocessing=preprocessing,
    )
    print("embedding references and queries once", flush=True)
    genus_reference_embeddings = _embed_references(
        genus_model,
        species_model,
        genus_references,
        global_transform,
        local_transform,
        device,
    )
    species_reference_embeddings = _embed_references(
        genus_model,
        species_model,
        species_references,
        global_transform,
        local_transform,
        device,
    )
    query_embeddings = _embed_queries(
        genus_model,
        species_model,
        query_records,
        global_transform,
        local_transform,
        device,
    )

    print("precomputing genus and species scores", flush=True)
    genus_scores = {
        mode: _score_matrix(
            genus_model,
            query_embeddings.global_embeddings,
            genus_reference_embeddings.global_embeddings,
            mode,
            device,
        )
        for mode in args.genus_score_modes
    }
    species_scores = {
        mode: _score_matrix(
            species_model,
            query_embeddings.local_embeddings,
            species_reference_embeddings.local_embeddings,
            mode,
            device,
        )
        for mode in args.species_score_modes
    }

    rows: list[dict[str, object]] = []
    top_ks = tuple(sorted(set(args.top_k)))
    for genus_score_mode in args.genus_score_modes:
        for genus_candidates in args.genus_candidates:
            for species_score_mode in args.species_score_modes:
                for species_aggregation in args.species_aggregations:
                    row = _evaluate_combo(
                        query_records=tuple(query_records),
                        genus_references=genus_reference_embeddings.records,
                        species_references=species_reference_embeddings.records,
                        genus_scores=genus_scores[genus_score_mode],
                        species_scores=species_scores[species_score_mode],
                        genus_candidates=genus_candidates,
                        genus_score_mode=genus_score_mode,
                        species_score_mode=species_score_mode,
                        species_aggregation=species_aggregation,
                        top_ks=top_ks,
                    )
                    rows.append(row)

    rows = sorted(
        rows,
        key=lambda row: (
            float(row["plantclef_s"]),
            float(row[f"top{max(top_ks)}"]),
            float(row[f"top{min(top_ks)}"]),
        ),
        reverse=True,
    )
    _print_rows(rows, top_ks, limit=args.print_top)
    if args.output_csv:
        _write_csv(rows, args.output_csv)
        print(f"saved grid summary to {args.output_csv}", flush=True)
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


@torch.inference_mode()
def _embed_references(
    genus_model: SiameseNetwork,
    species_model: SiameseNetwork,
    records: list[ImageRecord],
    global_transform,
    local_transform,
    device: torch.device,
) -> ReferenceEmbeddings:
    global_embeddings = []
    local_embeddings = []
    for record in records:
        with Image.open(record.image_path) as image:
            rgb = image.convert("RGB")
        global_embeddings.append(
            genus_model.embed(global_transform(rgb).unsqueeze(0).to(device))
            .squeeze(0)
            .cpu()
        )
        local_embeddings.append(
            species_model.embed(local_transform(rgb).unsqueeze(0).to(device))
            .squeeze(0)
            .cpu()
        )
    return ReferenceEmbeddings(
        records=tuple(records),
        global_embeddings=torch.stack(global_embeddings),
        local_embeddings=torch.stack(local_embeddings),
    )


@torch.inference_mode()
def _embed_queries(
    genus_model: SiameseNetwork,
    species_model: SiameseNetwork,
    records: list[ImageRecord],
    global_transform,
    local_transform,
    device: torch.device,
) -> QueryEmbeddings:
    global_embeddings = []
    local_embeddings = []
    for record in records:
        with Image.open(record.image_path) as image:
            rgb = image.convert("RGB")
        global_embeddings.append(
            genus_model.embed(global_transform(rgb).unsqueeze(0).to(device))
            .squeeze(0)
            .cpu()
        )
        local_embeddings.append(
            species_model.embed(local_transform(rgb).unsqueeze(0).to(device))
            .squeeze(0)
            .cpu()
        )
    return QueryEmbeddings(
        records=tuple(records),
        global_embeddings=torch.stack(global_embeddings),
        local_embeddings=torch.stack(local_embeddings),
    )


@torch.inference_mode()
def _score_matrix(
    model: SiameseNetwork,
    queries: torch.Tensor,
    references: torch.Tensor,
    score_mode: str,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    reference_embeddings = references.to(device)
    for query in queries:
        distance = torch.abs(reference_embeddings - query.to(device).unsqueeze(0))
        if score_mode == "l1":
            scores = 1.0 / (1.0 + distance.sum(dim=1))
        else:
            scores = model.comparator(distance).flatten()
        rows.append(scores.cpu())
    return torch.stack(rows)


def _evaluate_combo(
    query_records: tuple[ImageRecord, ...],
    genus_references: tuple[ImageRecord, ...],
    species_references: tuple[ImageRecord, ...],
    genus_scores: torch.Tensor,
    species_scores: torch.Tensor,
    genus_candidates: int,
    genus_score_mode: str,
    species_score_mode: str,
    species_aggregation: str,
    top_ks: tuple[int, ...],
) -> dict[str, object]:
    hits = {top_k: 0 for top_k in top_ks}
    inverse_rank_sum = 0.0
    genus_gate_hits = 0
    misses_without_candidate_genus = 0

    for query_index, query in enumerate(query_records):
        ranked_genus_indices = torch.argsort(genus_scores[query_index], descending=True).tolist()
        candidate_indices = ranked_genus_indices[:genus_candidates]
        selected_genera = [genus_references[index].genus for index in candidate_indices]
        genus_weights = Counter(selected_genera)
        candidate_genera = set(genus_weights)
        if query.genus in candidate_genera:
            genus_gate_hits += 1
        else:
            misses_without_candidate_genus += 1

        species_score_buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for reference_index, reference in enumerate(species_references):
            if reference.genus not in candidate_genera:
                continue
            key = (reference.family, reference.genus, reference.species)
            species_score_buckets[key].append(float(species_scores[query_index, reference_index]))

        weight_denominator = max(1, sum(genus_weights.values()))
        ranked_species = [
            key[2]
            for key, _score in sorted(
                (
                    (
                        key,
                        _aggregate(scores, species_aggregation)
                        * genus_weights[key[1]]
                        / weight_denominator,
                    )
                    for key, scores in species_score_buckets.items()
                ),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        rank = _rank_of(query.species, ranked_species)
        if rank is not None:
            inverse_rank_sum += 1.0 / rank
        for top_k in top_ks:
            if rank is not None and rank <= top_k:
                hits[top_k] += 1

    query_count = len(query_records)
    row: dict[str, object] = {
        "genus_candidates": genus_candidates,
        "genus_score_mode": genus_score_mode,
        "species_score_mode": species_score_mode,
        "species_aggregation": species_aggregation,
        "queries": query_count,
        "genus_references": len(genus_references),
        "species_references": len(species_references),
        "genus_gate_accuracy": genus_gate_hits / query_count,
        "genus_gate_hits": genus_gate_hits,
        "misses_without_candidate_genus": misses_without_candidate_genus,
        "plantclef_s": inverse_rank_sum / query_count,
        "mean_inverse_rank": inverse_rank_sum / query_count,
    }
    for top_k in top_ks:
        row[f"top{top_k}"] = hits[top_k] / query_count
        row[f"top{top_k}_hits"] = hits[top_k]
    return row


def _aggregate(scores: Iterable[float], aggregation: str) -> float:
    values = list(scores)
    if aggregation == "max":
        return max(values)
    if aggregation == "mean":
        return sum(values) / len(values)
    if aggregation == "sum":
        return sum(values)
    raise ValueError(f"Unsupported species aggregation: {aggregation}")


def _rank_of(expected_species: str, ranked_species: list[str]) -> int | None:
    for index, species in enumerate(ranked_species, start=1):
        if species == expected_species:
            return index
    return None


def _print_rows(rows: list[dict[str, object]], top_ks: tuple[int, ...], limit: int) -> None:
    print("=== best grid rows ===", flush=True)
    for row in rows[:limit]:
        topk_text = " ".join(f"top{top_k}={float(row[f'top{top_k}']):.3f}" for top_k in top_ks)
        print(
            f"genus_candidates={row['genus_candidates']} "
            f"genus_score_mode={row['genus_score_mode']} "
            f"species_score_mode={row['species_score_mode']} "
            f"species_aggregation={row['species_aggregation']} "
            f"genus_gate={float(row['genus_gate_accuracy']):.3f} "
            f"{topk_text} "
            f"plantclef_s={float(row['plantclef_s']):.3f}",
            flush=True,
        )


def _write_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _preprocessing_enabled(config: dict) -> bool:
    preprocessing = config.get("preprocessing", {})
    return bool(preprocessing.get("enabled", preprocessing.get("leaf_bbox", False)))


def _require_checkpoint(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


if __name__ == "__main__":
    raise SystemExit(main())
