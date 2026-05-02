from pathlib import Path

from plant_classifier.data.plantclef import build_plantclef_metadata


def test_build_plantclef_metadata_from_xml(tmp_path: Path) -> None:
    source_root = tmp_path / "plantclef"
    train_dir = source_root / "train"
    image_dir = train_dir / "images"
    annotation_dir = train_dir / "xml"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    (image_dir / "123.jpg").write_bytes(b"fake-image")
    (annotation_dir / "123.xml").write_text(
        """
        <Image>
            <MediaId>123</MediaId>
            <Content>Leaf</Content>
            <Family>Rosaceae</Family>
            <Genus>Prunus</Genus>
            <Species>avium</Species>
        </Image>
        """,
        encoding="utf-8",
    )

    output_csv = source_root / "metadata.csv"
    rows = build_plantclef_metadata(
        source_root=source_root,
        image_root=source_root,
        output_csv=output_csv,
        relative_to=source_root,
    )

    assert len(rows) == 1
    assert rows[0].genus == "Prunus"
    assert rows[0].species == "avium"
    assert output_csv.read_text(encoding="utf-8").splitlines()[1].startswith("train/images/123.jpg")

