from __future__ import annotations

import argparse
from pathlib import Path

from plant_classifier.data.plantclef import build_plantclef_metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert PlantCLEF XML annotations to normalized metadata.csv."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/plantclef2015/metadata.csv"))
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--relative-to", type=Path)
    parser.add_argument("--content", default="leaf")
    parser.add_argument("--all-content", action="store_true")
    args = parser.parse_args()

    rows = build_plantclef_metadata(
        source_root=args.source_root,
        output_csv=args.output,
        image_root=args.image_root,
        relative_to=args.relative_to,
        content_filter=None if args.all_content else args.content,
    )
    print(f"saved {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

