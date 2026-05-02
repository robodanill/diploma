from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml
from PIL import Image

from plant_classifier.data import ImageRecord, limit_records_by_species, load_metadata_csv
from plant_classifier.models.siamese import BackboneSpec, build_siamese_network
from plant_classifier.training.image_pairs import build_image_transform


def main() -> int:
    parser = argparse.ArgumentParser(description="Quickly evaluate S-CNN(A) genus retrieval.")
    parser.add_argument("--config", type=Path, default=Path("configs/leaf_training.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-species", type=int, default=40)
    parser.add_argument("--references-per-genus", type=int, default=2)
    parser.add_argument("--queries-per-genus", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    config = _load_config(args.config)
    dataset_config = config["dataset"]
    records = _load_records(dataset_config)
    if args.max_species:
        records = limit_records_by_species(
            records,
            max_species=args.max_species,
            min_images_per_species=3,
            max_images_per_species=args.references_per_genus + args.queries_per_genus,
        )

    references, queries = _split_references_and_queries(
        records,
        references_per_genus=args.references_per_genus,
        queries_per_genus=args.queries_per_genus,
    )
    if not references or not queries:
        raise ValueError("Not enough records to create references and queries")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = config["model"]
    model = build_siamese_network(
        BackboneSpec(name=model_config["backbone"], pretrained=False)
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    transform = build_image_transform(
        "global",
        image_size=int(config["views"]["global"]["image_size"]),
        crop_size=int(config["views"]["local"]["crop_size"]),
    )

    with torch.inference_mode():
        reference_embeddings = [
            (record, _embed(model, record.image_path, transform, device)) for record in references
        ]

        hits = 0
        for query in queries:
            query_embedding = _embed(model, query.image_path, transform, device)
            ranked = _rank_references(model, query_embedding, reference_embeddings)
            top_genera = [record.genus for record, _ in ranked[: args.top_k]]
            if query.genus in top_genera:
                hits += 1

        accuracy = hits / len(queries)

    print(f"references={len(references)} queries={len(queries)} top_k={args.top_k}")
    print(f"top{args.top_k}_genus_accuracy={accuracy:.3f} ({hits}/{len(queries)})")
    print("query genus distribution:", dict(Counter(record.genus for record in queries).most_common(10)))
    print("reference genus distribution:", dict(Counter(record.genus for record in references).most_common(10)))
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


def _split_references_and_queries(
    records: list[ImageRecord],
    references_per_genus: int,
    queries_per_genus: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in sorted(records, key=lambda item: (item.genus, item.species, str(item.image_path))):
        grouped[record.genus].append(record)

    references: list[ImageRecord] = []
    queries: list[ImageRecord] = []
    for genus in sorted(grouped):
        items = grouped[genus]
        needed = references_per_genus + queries_per_genus
        if len(items) < needed:
            continue
        references.extend(items[:references_per_genus])
        queries.extend(items[references_per_genus:needed])
    return references, queries


def _embed(model, image_path: Path, transform, device: torch.device):
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    return model.embed(tensor)


def _rank_references(model, query_embedding, reference_embeddings):
    ranked = []
    for record, reference_embedding in reference_embeddings:
        distance = torch.abs(query_embedding - reference_embedding)
        score = float(model.comparator(distance).flatten().item())
        ranked.append((record, score))
    return sorted(ranked, key=lambda item: item[1], reverse=True)


if __name__ == "__main__":
    raise SystemExit(main())
