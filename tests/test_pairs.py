from pathlib import Path

from plant_classifier.data import ImageRecord, sample_pairs


def test_sample_pairs_creates_requested_positive_and_negative_pairs() -> None:
    records = [
        ImageRecord(Path("a1.jpg"), "F1", "G1", "S1"),
        ImageRecord(Path("a2.jpg"), "F1", "G1", "S1"),
        ImageRecord(Path("b1.jpg"), "F1", "G2", "S2"),
        ImageRecord(Path("b2.jpg"), "F1", "G2", "S2"),
    ]

    pairs = sample_pairs(records, "genus", positive_count=3, negative_count=4, seed=1)

    assert len(pairs) == 7
    assert sum(pair.label for pair in pairs) == 3
    assert {pair.taxonomic_level for pair in pairs} == {"genus"}


def test_sample_pairs_can_create_hard_negative_pairs_within_family() -> None:
    records = [
        ImageRecord(Path("a1.jpg"), "F1", "G1", "S1"),
        ImageRecord(Path("a2.jpg"), "F1", "G1", "S1"),
        ImageRecord(Path("b1.jpg"), "F1", "G2", "S2"),
        ImageRecord(Path("b2.jpg"), "F1", "G2", "S2"),
        ImageRecord(Path("c1.jpg"), "F2", "G3", "S3"),
        ImageRecord(Path("c2.jpg"), "F2", "G3", "S3"),
    ]

    pairs = sample_pairs(
        records,
        "genus",
        positive_count=0,
        negative_count=4,
        seed=1,
        hard_negative_ratio=1.0,
    )

    records_by_path = {record.image_path: record for record in records}
    assert len(pairs) == 4
    assert sum(pair.label for pair in pairs) == 0
    for pair in pairs:
        left = records_by_path[pair.left]
        right = records_by_path[pair.right]
        assert left.family == right.family
        assert left.genus != right.genus
