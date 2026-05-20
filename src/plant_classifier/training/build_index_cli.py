from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from plant_classifier.data import (
    ImageRecord,
    filter_records_by_split,
    limit_records_by_species,
    load_metadata_csv,
)
from plant_classifier.inference.scnn import ReferenceEmbedding, save_reference_index
from plant_classifier.models.siamese import BackboneSpec, build_siamese_network
from plant_classifier.training.genus_eval import select_genus_references
from plant_classifier.training.image_pairs import build_image_transform


def main() -> int:
    parser = argparse.ArgumentParser(description="Build reference embeddings for two-stage inference.")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--genus-checkpoint", type=Path, required=True)
    parser.add_argument("--species-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = _load_config(args.config)
    records = _load_records(config)
    records = filter_records_by_split(
        records,
        config.get("inference", {}).get("reference_split", config["training"].get("split", "train")),
    )
    records = _apply_subset(records, config["dataset"])
    references_per_class = int(config["inference"]["references_per_class"])
    genus_references_per_genus = int(
        config["inference"].get("genus_references_per_genus", references_per_class)
    )
    genus_references = select_genus_references(
        records=records,
        references_per_genus=genus_references_per_genus,
    )
    species_references = _select_references(
        records=records,
        references_per_class=references_per_class,
    )

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

    global_transform = build_image_transform(
        "global",
        image_size=int(config["views"]["global"]["image_size"]),
        crop_size=int(config["views"]["local"]["crop_size"]),
        preprocessing=_preprocessing_enabled(config),
    )
    local_transform = build_image_transform(
        "local",
        image_size=int(config["views"]["local"]["image_size"]),
        crop_size=int(config["views"]["local"]["crop_size"]),
        preprocessing=_preprocessing_enabled(config),
    )

    with torch.inference_mode():
        genus_embeddings = _build_embeddings(
            records=genus_references,
            genus_model=genus_model,
            species_model=species_model,
            global_transform=global_transform,
            local_transform=local_transform,
            device=device,
        )
        species_embeddings = _build_embeddings(
            records=species_references,
            genus_model=genus_model,
            species_model=species_model,
            global_transform=global_transform,
            local_transform=local_transform,
            device=device,
        )

    save_reference_index(species_embeddings, args.output, genus_references=genus_embeddings)
    print(
        f"saved {len(genus_embeddings)} genus and "
        f"{len(species_embeddings)} species reference embeddings to {args.output}"
    )
    return 0


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _load_records(config: dict) -> list[ImageRecord]:
    dataset_config = config["dataset"]
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
    print(f"using subset: {len(limited)} images from {len({record.species for record in limited})} species")
    return limited


def _select_references(records: list[ImageRecord], references_per_class: int) -> list[ImageRecord]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.species].append(record)

    references: list[ImageRecord] = []
    for species in sorted(grouped):
        references.extend(sorted(grouped[species], key=lambda item: str(item.image_path))[:references_per_class])
    return references


def _build_embeddings(
    records: list[ImageRecord],
    genus_model,
    species_model,
    global_transform,
    local_transform,
    device: torch.device,
) -> list[ReferenceEmbedding]:
    embeddings: list[ReferenceEmbedding] = []
    for record in records:
        image = _load_rgb(record.image_path)
        global_tensor = global_transform(image).unsqueeze(0).to(device)
        local_tensor = local_transform(image).unsqueeze(0).to(device)
        embeddings.append(
            ReferenceEmbedding(
                image_path=record.image_path,
                family=record.family,
                genus=record.genus,
                species=record.species,
                global_embedding=genus_model.embed(global_tensor).squeeze(0).cpu(),
                local_embedding=species_model.embed(local_tensor).squeeze(0).cpu(),
            )
        )
    return embeddings


def _preprocessing_enabled(config: dict) -> bool:
    preprocessing = config.get("preprocessing", {})
    return bool(preprocessing.get("enabled", preprocessing.get("leaf_bbox", False)))


def _load_rgb(path: Path):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


if __name__ == "__main__":
    raise SystemExit(main())
