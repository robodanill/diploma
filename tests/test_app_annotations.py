from pathlib import Path

from plant_classifier.app.annotations import (
    evaluate_prediction,
    load_ground_truth_label,
    parse_ground_truth_metadata,
    parse_ground_truth_xml,
)
from plant_classifier.inference.types import ImagePrediction, PredictionLabel


def test_load_ground_truth_label_reads_xml_next_to_image(tmp_path: Path) -> None:
    image_path = tmp_path / "leaf.jpg"
    image_path.write_bytes(b"")
    (tmp_path / "leaf.xml").write_text(
        """
        <Observation>
          <Family>Rosaceae</Family>
          <Genus>Prunus</Genus>
          <Species>avium</Species>
        </Observation>
        """,
        encoding="utf-8",
    )

    label = load_ground_truth_label(image_path)

    assert label is not None
    assert label.family == "Rosaceae"
    assert label.display_name == "Prunus avium"


def test_load_ground_truth_label_returns_none_without_sidecar_xml(tmp_path: Path) -> None:
    image_path = tmp_path / "leaf.jpg"
    image_path.write_bytes(b"")

    assert load_ground_truth_label(image_path) is None


def test_load_ground_truth_label_accepts_uppercase_xml_suffix(tmp_path: Path) -> None:
    image_path = tmp_path / "leaf.jpg"
    image_path.write_bytes(b"")
    (tmp_path / "leaf.XML").write_text(
        """
        <Observation>
          <Genus>Betula</Genus>
          <Species>pendula</Species>
        </Observation>
        """,
        encoding="utf-8",
    )

    label = load_ground_truth_label(image_path)

    assert label is not None
    assert label.display_name == "Betula pendula"


def test_load_ground_truth_label_reads_metadata_csv_next_to_image(tmp_path: Path) -> None:
    image_path = tmp_path / "leaf.jpg"
    image_path.write_bytes(b"")
    (tmp_path / "metadata.csv").write_text(
        "image_path,family,genus,species\nleaf.jpg,Fagaceae,Quercus,rubra\n",
        encoding="utf-8",
    )

    label = load_ground_truth_label(image_path)

    assert label is not None
    assert label.family == "Fagaceae"
    assert label.display_name == "Quercus rubra"


def test_load_ground_truth_label_accepts_metadata_scv_typo(tmp_path: Path) -> None:
    image_path = tmp_path / "leaf.jpg"
    image_path.write_bytes(b"")
    (tmp_path / "metadata.scv").write_text(
        "image_path,family,genus,species\nleaf.jpg,Betulaceae,Betula,pendula\n",
        encoding="utf-8",
    )

    label = load_ground_truth_label(image_path)

    assert label is not None
    assert label.display_name == "Betula pendula"


def test_parse_ground_truth_metadata_matches_relative_image_path(tmp_path: Path) -> None:
    image_folder = tmp_path / "images"
    image_folder.mkdir()
    image_path = image_folder / "leaf.jpg"
    image_path.write_bytes(b"")
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "image_path,family,genus,species\nimages/leaf.jpg,Rosaceae,Prunus,avium\n",
        encoding="utf-8",
    )

    label = parse_ground_truth_metadata(metadata_path, image_path)

    assert label is not None
    assert label.display_name == "Prunus avium"


def test_load_ground_truth_label_matches_project_relative_path_in_local_metadata(
    tmp_path: Path,
) -> None:
    image_folder = tmp_path / "PlantCLEF2015TestDataWithAnnotations"
    image_folder.mkdir()
    image_path = image_folder / "leaf.jpg"
    image_path.write_bytes(b"")
    (image_folder / "metadata.csv").write_text(
        "image_path,family,genus,species\n"
        "PlantCLEF2015TestDataWithAnnotations/leaf.jpg,Rosaceae,Prunus,avium\n",
        encoding="utf-8",
    )

    label = load_ground_truth_label(image_path)

    assert label is not None
    assert label.display_name == "Prunus avium"


def test_parse_ground_truth_metadata_accepts_case_insensitive_headers(tmp_path: Path) -> None:
    image_path = tmp_path / "leaf.jpg"
    image_path.write_bytes(b"")
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "File-Name,Family,Genus,Specific-Epithet\n"
        "leaf.jpg,Fagaceae,Quercus,cerris\n",
        encoding="utf-8",
    )

    label = parse_ground_truth_metadata(metadata_path, image_path)

    assert label is not None
    assert label.display_name == "Quercus cerris"


def test_load_ground_truth_label_prefers_xml_over_metadata_csv(tmp_path: Path) -> None:
    image_path = tmp_path / "leaf.jpg"
    image_path.write_bytes(b"")
    (tmp_path / "leaf.xml").write_text(
        """
        <Observation>
          <Genus>Prunus</Genus>
          <Species>avium</Species>
        </Observation>
        """,
        encoding="utf-8",
    )
    (tmp_path / "metadata.csv").write_text(
        "image_path,family,genus,species\nleaf.jpg,Fagaceae,Quercus,rubra\n",
        encoding="utf-8",
    )

    label = load_ground_truth_label(image_path)

    assert label is not None
    assert label.display_name == "Prunus avium"


def test_parse_ground_truth_xml_accepts_namespaced_specific_epithet(tmp_path: Path) -> None:
    annotation_path = tmp_path / "leaf.xml"
    annotation_path.write_text(
        """
        <ns:Observation xmlns:ns="urn:test">
          <ns:Genus>Quercus</ns:Genus>
          <ns:Specific-Epithet>rubra</ns:Specific-Epithet>
        </ns:Observation>
        """,
        encoding="utf-8",
    )

    label = parse_ground_truth_xml(annotation_path)

    assert label is not None
    assert label.display_name == "Quercus rubra"


def test_evaluate_prediction_matches_epithet_to_full_species_name(tmp_path: Path) -> None:
    prediction = ImagePrediction(
        image_path=Path("leaf.jpg"),
        labels=(PredictionLabel(family="Fagaceae", genus="Quercus", species="rubra", score=0.9),),
    )
    annotation_path = tmp_path / "leaf.xml"
    annotation_path.write_text(
        """
        <Observation>
          <Family>Fagaceae</Family>
          <Genus>Quercus</Genus>
          <Species>Quercus rubra L.</Species>
        </Observation>
        """,
        encoding="utf-8",
    )
    ground_truth = parse_ground_truth_xml(annotation_path)

    correctness = evaluate_prediction(prediction, ground_truth)

    assert correctness is not None
    assert correctness.is_correct


def test_evaluate_prediction_marks_wrong_species(tmp_path: Path) -> None:
    prediction = ImagePrediction(
        image_path=Path("leaf.jpg"),
        labels=(PredictionLabel(family="Fagaceae", genus="Quercus", species="cerris", score=0.9),),
    )
    annotation_path = tmp_path / "leaf.xml"
    annotation_path.write_text(
        """
        <Observation>
          <Family>Fagaceae</Family>
          <Genus>Quercus</Genus>
          <Species>rubra</Species>
        </Observation>
        """,
        encoding="utf-8",
    )
    ground_truth = parse_ground_truth_xml(annotation_path)

    correctness = evaluate_prediction(prediction, ground_truth)

    assert correctness is not None
    assert not correctness.is_correct
    assert correctness.ground_truth.display_name == "Quercus rubra"
