from __future__ import annotations


def validate_records_exist(records: list) -> None:
    missing = [record.image_path for record in records if not record.image_path.exists()]
    if not missing:
        return

    examples = "\n".join(f"  - {path}" for path in missing[:5])
    raise FileNotFoundError(
        "Some metadata image paths do not exist. "
        "Check dataset.root in the selected config. "
        "For PlantCLEF2015_leaf_only.tar.gz use configs/smoke_training.yaml "
        "or configs/leaf_training.yaml.\n"
        f"Missing examples:\n{examples}"
    )
