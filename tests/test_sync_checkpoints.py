from __future__ import annotations

from pathlib import Path

from scripts.sync_checkpoints_to_drive import copy_checkpoints, remove_old_checkpoints


def write_file(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_sync_preserves_drive_best_and_copies_local_best_as_regular(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()

    write_file(source / "scnn_genus_vgg16.pt", "regular genus")
    write_file(source / "scnn_genus_vgg16_best.pt", "best genus")
    write_file(dest / "old.pt", "old regular")
    write_file(dest / "old_best.pt", "best across trainings")

    removed = remove_old_checkpoints(dest, keep_token="_best")
    copied = copy_checkpoints(source, dest, keep_token="_best")

    assert removed == 1
    assert copied == 2
    assert not (dest / "old.pt").exists()
    assert (dest / "old_best.pt").read_text(encoding="utf-8") == "best across trainings"
    assert not (dest / "scnn_genus_vgg16_best.pt").exists()
    assert (dest / "scnn_genus_vgg16.pt").read_text(encoding="utf-8") == "best genus"
