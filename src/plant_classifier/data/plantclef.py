from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PlantClefMetadataRow:
    image_path: Path
    family: str
    genus: str
    species: str
    content: str = ""
    split: str = ""
    source_xml: Path | None = None


def build_plantclef_metadata(
    source_root: Path,
    output_csv: Path,
    image_root: Path | None = None,
    relative_to: Path | None = None,
    content_filter: str | None = "leaf",
) -> list[PlantClefMetadataRow]:
    """Convert PlantCLEF-style XML annotations into the project metadata CSV."""

    resolved_image_root = image_root or source_root
    rows = parse_plantclef_xml_directory(
        source_root=source_root,
        image_root=resolved_image_root,
        content_filter=content_filter,
    )
    write_metadata_csv(rows=rows, output_csv=output_csv, relative_to=relative_to or resolved_image_root)
    return rows


def parse_plantclef_xml_directory(
    source_root: Path,
    image_root: Path,
    content_filter: str | None = "leaf",
) -> list[PlantClefMetadataRow]:
    image_index = _build_image_index(image_root)
    rows: list[PlantClefMetadataRow] = []

    for xml_path in sorted(source_root.rglob("*.xml")):
        parsed = parse_plantclef_xml(xml_path=xml_path, image_index=image_index)
        if parsed is None:
            continue
        if content_filter and parsed.content.lower() != content_filter.lower():
            continue
        rows.append(parsed)

    if not rows:
        hint = f" under {source_root}"
        if content_filter:
            hint += f" with content={content_filter!r}"
        raise ValueError(f"No PlantCLEF metadata rows found{hint}")
    return rows


def parse_plantclef_xml(
    xml_path: Path,
    image_index: dict[str, Path],
) -> PlantClefMetadataRow | None:
    root = ET.parse(xml_path).getroot()
    values = {_clean_tag(element.tag): (element.text or "").strip() for element in root.iter()}

    family = _first(values, "family")
    genus = _first(values, "genus")
    species = _first(values, "species", "specific_epithet")
    if not genus or not species:
        return None

    image_path = _resolve_image_path(xml_path=xml_path, values=values, image_index=image_index)
    if image_path is None:
        return None

    return PlantClefMetadataRow(
        image_path=image_path,
        family=family or "",
        genus=genus,
        species=species,
        content=_first(values, "content", "organ", "view") or "",
        split=_infer_split(xml_path),
        source_xml=xml_path,
    )


def write_metadata_csv(
    rows: list[PlantClefMetadataRow],
    output_csv: Path,
    relative_to: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_path", "family", "genus", "species", "content", "split", "source_xml"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_path": _safe_relative(row.image_path, relative_to),
                    "family": row.family,
                    "genus": row.genus,
                    "species": row.species,
                    "content": row.content,
                    "split": row.split,
                    "source_xml": _safe_relative(row.source_xml, relative_to) if row.source_xml else "",
                }
            )


def _build_image_index(image_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.name.lower(), path)
            index.setdefault(path.stem.lower(), path)
    return index


def _resolve_image_path(
    xml_path: Path,
    values: dict[str, str],
    image_index: dict[str, Path],
) -> Path | None:
    candidates = [
        _first(values, "image_path", "imagepath", "filename", "file_name", "image", "media_id", "mediaid"),
        xml_path.stem,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.suffix.lower() in IMAGE_EXTENSIONS and candidate_path.exists():
            return candidate_path
        by_name = image_index.get(candidate_path.name.lower())
        if by_name:
            return by_name
        by_stem = image_index.get(candidate_path.stem.lower())
        if by_stem:
            return by_stem
    return None


def _first(values: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = values.get(key.lower(), "")
        if value:
            return value
    return ""


def _clean_tag(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1].strip().lower().replace("-", "_")


def _infer_split(xml_path: Path) -> str:
    parts = {part.lower() for part in xml_path.parts}
    if "train" in parts or "training" in parts:
        return "train"
    if "test" in parts or "testing" in parts:
        return "test"
    return ""


def _safe_relative(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)

