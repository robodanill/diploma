from pathlib import Path

import pytest

from plant_classifier.data import ImageRecord
from plant_classifier.training.validation import validate_records_exist


def test_validate_records_exist_raises_helpful_error_for_missing_images() -> None:
    records = [ImageRecord(Path("missing.jpg"), "F", "G", "S")]

    with pytest.raises(FileNotFoundError, match="configs/leaf_training.yaml"):
        validate_records_exist(records)
