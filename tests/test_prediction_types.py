import pytest

from plant_classifier.inference.types import PredictionLabel, normalize_label_scores


def test_display_name_does_not_duplicate_genus_for_full_species_name() -> None:
    label = PredictionLabel(
        family="Fagaceae",
        genus="Quercus",
        species="Quercus rubra L.",
        score=0.9,
    )

    assert label.display_name == "Quercus rubra L."


def test_display_name_combines_genus_and_epithet() -> None:
    label = PredictionLabel(
        family="Fagaceae",
        genus="Quercus",
        species="rubra",
        score=0.9,
    )

    assert label.display_name == "Quercus rubra"


def test_normalize_label_scores_preserves_order_and_sums_to_one() -> None:
    labels = (
        PredictionLabel(family="F", genus="A", species="A alpha", score=0.0015),
        PredictionLabel(family="F", genus="B", species="B beta", score=0.00001),
        PredictionLabel(family="F", genus="C", species="C gamma", score=0.000006),
    )

    normalized = normalize_label_scores(labels)

    assert [label.display_name for label in normalized] == [
        "A alpha",
        "B beta",
        "C gamma",
    ]
    assert sum(label.score for label in normalized) == pytest.approx(1.0)
    assert 0.0 < normalized[0].score < 1.0


def test_normalize_label_scores_uses_uniform_probabilities_for_zero_scores() -> None:
    labels = (
        PredictionLabel(family="F", genus="A", species="A alpha", score=0.0),
        PredictionLabel(family="F", genus="B", species="B beta", score=0.0),
    )

    normalized = normalize_label_scores(labels)

    assert [label.score for label in normalized] == [0.5, 0.5]
