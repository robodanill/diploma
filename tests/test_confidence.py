import pytest

from plant_classifier.inference.confidence import normalize_confidences


def test_normalize_confidences_preserves_order_and_sums_to_one() -> None:
    scores = {"low": 0.1, "middle": 0.2, "high": 0.4}

    confidences = normalize_confidences(scores)

    assert sum(confidences.values()) == pytest.approx(1.0)
    assert confidences["high"] > confidences["middle"] > confidences["low"] > 0


def test_normalize_confidences_is_deterministic() -> None:
    scores = {"a": 0.12, "b": 0.11, "c": 0.09}

    assert normalize_confidences(scores) == normalize_confidences(scores)


def test_normalize_confidences_returns_uniform_distribution_for_equal_scores() -> None:
    confidences = normalize_confidences({"a": 0.5, "b": 0.5, "c": 0.5})

    assert confidences == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})


def test_higher_temperature_flattens_confidence_distribution() -> None:
    scores = {"a": 0.1, "b": 0.2, "c": 0.4}

    sharp = normalize_confidences(scores, temperature=1.0)
    flat = normalize_confidences(scores, temperature=4.0)

    assert sharp["c"] > flat["c"]
    assert sharp["a"] < flat["a"]


def test_normalize_confidences_rejects_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        normalize_confidences({"a": 1.0}, temperature=0)
