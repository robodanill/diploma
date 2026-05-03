#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CHECKPOINT_PATTERNS = ("*.pt", "*.pth", "*.ckpt")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy checkpoints to Google Drive, replacing old non-best files. "
            "Local best checkpoints are copied as regular checkpoint names."
        )
    )
    parser.add_argument("--source", type=Path, default=Path("checkpoints"))
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--keep-token", default="_best")
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    removed = remove_old_checkpoints(args.dest, keep_token=args.keep_token)
    copied = copy_checkpoints(args.source, args.dest, keep_token=args.keep_token)

    print(f"removed old checkpoints: {removed}")
    print(f"copied checkpoints: {copied}")
    print(f"destination: {args.dest}")
    return 0


def remove_old_checkpoints(dest: Path, keep_token: str) -> int:
    removed = 0
    for pattern in CHECKPOINT_PATTERNS:
        for path in dest.glob(pattern):
            if keep_token and keep_token in path.name:
                continue
            path.unlink()
            removed += 1
    return removed


def copy_checkpoints(source: Path, dest: Path, keep_token: str) -> int:
    copied = 0
    for pattern in CHECKPOINT_PATTERNS:
        for path in sorted(source.glob(pattern), key=checkpoint_copy_priority(keep_token)):
            shutil.copy2(path, dest / drive_checkpoint_name(path, keep_token=keep_token))
            copied += 1
    return copied


def checkpoint_copy_priority(keep_token: str):
    def priority(path: Path) -> tuple[int, str]:
        is_best = bool(keep_token and keep_token in path.stem)
        return int(is_best), path.name

    return priority


def drive_checkpoint_name(path: Path, keep_token: str) -> str:
    if keep_token and keep_token in path.stem:
        return f"{path.stem.replace(keep_token, '')}{path.suffix}"
    return path.name


if __name__ == "__main__":
    raise SystemExit(main())
