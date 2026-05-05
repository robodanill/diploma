from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from plant_classifier.inference.types import ImagePrediction, PredictionLabel

METADATA_FILENAMES = ("metadata.csv", "metadata.scv")
IMAGE_PATH_COLUMNS = ("image_path", "filename", "file_name", "image", "path")


@dataclass(frozen=True)
class GroundTruthLabel:
    family: str
    genus: str
    species: str

    @property
    def display_name(self) -> str:
        if not self.genus:
            return self.species
        if self.species.lower().startswith(self.genus.lower()):
            return self.species
        return f"{self.genus} {self.species}"


@dataclass(frozen=True)
class PredictionCorrectness:
    ground_truth: GroundTruthLabel
    is_correct: bool


def load_ground_truth_label(image_path: Path) -> GroundTruthLabel | None:
    """Load a label from an XML sidecar or folder metadata CSV."""

    annotation_path = _find_annotation_path(image_path)
    if annotation_path is not None:
        return parse_ground_truth_xml(annotation_path)
    return load_ground_truth_from_metadata(image_path)


def parse_ground_truth_xml(annotation_path: Path) -> GroundTruthLabel | None:
    root = ET.parse(annotation_path).getroot()
    values = {_clean_tag(element.tag): (element.text or "").strip() for element in root.iter()}

    genus = _first(values, "genus")
    species = _first(values, "species", "specific_epithet")
    if not genus or not species:
        return None

    return GroundTruthLabel(
        family=_first(values, "family"),
        genus=genus,
        species=species,
    )


def load_ground_truth_from_metadata(image_path: Path) -> GroundTruthLabel | None:
    metadata_path = _find_metadata_path(image_path.parent)
    if metadata_path is None:
        return None
    return parse_ground_truth_metadata(metadata_path, image_path)


def parse_ground_truth_metadata(
    metadata_path: Path,
    image_path: Path,
) -> GroundTruthLabel | None:
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for raw_row in reader:
            row = _clean_metadata_row(raw_row)
            if not _metadata_row_matches(row, image_path, metadata_path.parent):
                continue
            genus = _first(row, "genus")
            species = _first(row, "species", "specific_epithet")
            if not genus or not species:
                return None
            return GroundTruthLabel(
                family=_first(row, "family"),
                genus=genus,
                species=species,
            )
    return None


def evaluate_prediction(
    prediction: ImagePrediction,
    ground_truth: GroundTruthLabel | None,
) -> PredictionCorrectness | None:
    if ground_truth is None:
        return None
    return PredictionCorrectness(
        ground_truth=ground_truth,
        is_correct=_labels_match(prediction.top_label, ground_truth),
    )


def _labels_match(prediction: PredictionLabel | None, ground_truth: GroundTruthLabel) -> bool:
    if prediction is None:
        return False
    if _normalize(prediction.genus) != _normalize(ground_truth.genus):
        return False
    return _canonical_binomial(prediction.genus, prediction.species) == _canonical_binomial(
        ground_truth.genus,
        ground_truth.species,
    )


def _canonical_binomial(genus: str, species: str) -> str:
    display_name = species if species.lower().startswith(genus.lower()) else f"{genus} {species}"
    tokens = _normalize(display_name).split()
    return " ".join(tokens[:2])


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", value.lower())).strip()


def _first(values: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = values.get(key.lower(), "")
        if value:
            return value
    return ""


def _clean_tag(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1].strip().lower().replace("-", "_")


def _clean_metadata_row(row: dict[str, str]) -> dict[str, str]:
    return {
        _clean_tag(key or ""): (value or "").strip()
        for key, value in row.items()
        if key is not None
    }


def _find_annotation_path(image_path: Path) -> Path | None:
    annotation_path = image_path.with_suffix(".xml")
    if annotation_path.is_file():
        return annotation_path
    if not image_path.parent.is_dir():
        return None
    image_stem = image_path.stem.lower()
    for path in image_path.parent.iterdir():
        if path.is_file() and path.stem.lower() == image_stem and path.suffix.lower() == ".xml":
            return path
    return None


def _find_metadata_path(folder: Path) -> Path | None:
    if not folder.is_dir():
        return None
    expected_names = set(METADATA_FILENAMES)
    for path in folder.iterdir():
        if path.is_file() and path.name.lower() in expected_names:
            return path
    return None


def _metadata_row_matches(row: dict[str, str], image_path: Path, metadata_root: Path) -> bool:
    value = _first(row, *IMAGE_PATH_COLUMNS)
    if not value:
        return False

    candidate = Path(value)
    resolved_candidate = candidate if candidate.is_absolute() else metadata_root / candidate
    if _same_path(resolved_candidate, image_path):
        return True
    return not candidate.is_absolute() and len(candidate.parts) == 1 and _same_name(
        candidate,
        image_path,
    )


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _same_name(left: Path, right: Path) -> bool:
    return left.name.lower() == right.name.lower()
