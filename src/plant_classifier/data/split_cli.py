from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from plant_classifier.data import (
    create_species_stratified_split,
    load_metadata_csv,
    write_metadata_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a stratified train/val/test metadata split.")
    parser.add_argument("--metadata", type=Path, default=Path("data/plantclef2015/metadata.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/plantclef2015/leaf"))
    parser.add_argument("--output", type=Path, default=Path("data/plantclef2015/metadata_split.csv"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_metadata_csv(args.metadata, args.dataset_root)
    split_records = create_species_stratified_split(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    write_metadata_csv(split_records, args.output, args.dataset_root)
    counts = Counter(record.split for record in split_records)
    print(f"saved {len(split_records)} rows to {args.output}")
    print("split counts:", dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
