#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import tarfile
import tempfile
from collections import Counter
from pathlib import Path

from plant_classifier.data.plantclef import parse_plantclef_xml_directory


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a compact PlantCLEF content archive.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--content", default="LeafScan")
    parser.add_argument("--bundle-dir-name", default="")
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()

    source_root = args.source_root
    image_root = args.image_root or source_root
    bundle_dir_name = args.bundle_dir_name or _slug(args.content)
    rows = parse_plantclef_xml_directory(
        source_root=source_root,
        image_root=image_root,
        content_filter=args.content,
    )

    with _work_dir(args.work_dir) as work_dir:
        bundle_root = work_dir / bundle_dir_name
        bundle_root.mkdir(parents=True, exist_ok=True)
        metadata_path = bundle_root / "metadata.csv"
        _copy_rows(rows, image_root=image_root, bundle_root=bundle_root, metadata_path=metadata_path)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.output, "w:gz") as archive:
            archive.add(bundle_root, arcname=bundle_dir_name)

    print(f"content={args.content} rows={len(rows)} archive={args.output}")
    print("content counts:", dict(Counter(row.content for row in rows)))
    print("genera:", len({row.genus for row in rows}))
    print("species:", len({row.species for row in rows}))
    return 0


def _copy_rows(rows, image_root: Path, bundle_root: Path, metadata_path: Path) -> None:
    fieldnames = ["image_path", "family", "genus", "species", "content", "split", "source_xml"]
    with metadata_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            rel_image = _safe_relative(row.image_path, image_root)
            destination = bundle_root / rel_image
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row.image_path, destination)
            writer.writerow(
                {
                    "image_path": str(rel_image),
                    "family": row.family,
                    "genus": row.genus,
                    "species": row.species,
                    "content": row.content,
                    "split": row.split,
                    "source_xml": str(_safe_relative(row.source_xml, image_root))
                    if row.source_xml
                    else "",
                }
            )


def _work_dir(path: Path | None):
    if path:
        path.mkdir(parents=True, exist_ok=True)
        if any(path.iterdir()):
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        return _PersistentWorkDir(path)
    return _TemporaryWorkDir()


class _TemporaryWorkDir:
    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()

    def __enter__(self) -> Path:
        return Path(self._temporary.__enter__())

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._temporary.__exit__(exc_type, exc, traceback)


class _PersistentWorkDir:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> Path:
        return self.path

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


def _safe_relative(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def _slug(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


if __name__ == "__main__":
    raise SystemExit(main())
