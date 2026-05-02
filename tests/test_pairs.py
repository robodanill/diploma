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

