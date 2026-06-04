from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ClassifierTrainResult:
    checkpoint_path: Path
    best_checkpoint_path: Path
    last_loss: float
    class_to_idx: dict[str, int]
    history_path: Path | None = None


def train_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    checkpoint_path: Path,
    class_to_idx: dict[str, int],
    backbone: str,
    image_size: int,
    crop_size: int,
    preprocessing: bool,
    epochs: int = 10,
    learning_rate: float = 0.001,
    momentum: float = 0.9,
    lr_decay_step: int = 0,
    lr_decay_gamma: float = 0.5,
    max_iterations: int | None = None,
    device: str | None = None,
    progress_every: int = 10,
) -> ClassifierTrainResult:
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"training device={resolved_device}", flush=True)
    model.to(resolved_device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = _build_scheduler(optimizer, lr_decay_step, lr_decay_gamma)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_best{checkpoint_path.suffix}"
    )

    best_loss = float("inf")
    last_loss = 0.0
    global_step = 0
    history_path = _history_path(checkpoint_path)
    _reset_history(history_path)
    for epoch in range(epochs):
        started_at = perf_counter()
        running_loss = 0.0
        correct = 0
        seen = 0
        batches_seen = 0
        total_batches = len(dataloader)
        for batch_index, (images, labels) in enumerate(dataloader, start=1):
            batches_seen = batch_index
            images = images.to(resolved_device)
            labels = labels.to(resolved_device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            running_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            seen += int(labels.numel())
            _print_batch_progress(
                epoch,
                batch_index,
                total_batches,
                running_loss,
                correct,
                seen,
                progress_every,
            )
            if max_iterations and global_step >= max_iterations:
                break

        last_loss = running_loss / max(1, batches_seen)
        train_accuracy = correct / max(1, seen)
        elapsed = perf_counter() - started_at
        if last_loss < best_loss:
            best_loss = last_loss
            _save_checkpoint(
                model,
                best_checkpoint_path,
                class_to_idx,
                backbone,
                image_size,
                crop_size,
                preprocessing,
            )
            best_marker = " best"
        else:
            best_marker = ""
        _append_history_row(
            history_path,
            {
                "epoch": epoch + 1,
                "train_loss": last_loss,
                "best_loss": best_loss,
                "train_accuracy": train_accuracy,
                "iterations": global_step,
                "elapsed_seconds": elapsed,
                "is_best": bool(best_marker),
            },
        )
        print(
            f"epoch={epoch + 1} loss={last_loss:.4f} best_loss={best_loss:.4f} "
            f"train_accuracy={train_accuracy:.3f} iterations={global_step} "
            f"time={elapsed:.1f}s{best_marker}",
            flush=True,
        )
        if max_iterations and global_step >= max_iterations:
            break

    _save_checkpoint(
        model,
        checkpoint_path,
        class_to_idx,
        backbone,
        image_size,
        crop_size,
        preprocessing,
    )
    return ClassifierTrainResult(
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        last_loss=last_loss,
        class_to_idx=class_to_idx,
        history_path=history_path,
    )


def _save_checkpoint(
    model: nn.Module,
    path: Path,
    class_to_idx: dict[str, int],
    backbone: str,
    image_size: int,
    crop_size: int,
    preprocessing: bool,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_to_idx": class_to_idx,
            "backbone": backbone,
            "image_size": image_size,
            "crop_size": crop_size,
            "preprocessing": preprocessing,
        },
        path,
    )


def _history_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_history.csv")


def _reset_history(history_path: Path) -> None:
    if history_path.exists():
        history_path.unlink()


def _append_history_row(history_path: Path, row: dict) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not history_path.exists()
    with history_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _build_scheduler(optimizer, lr_decay_step: int, lr_decay_gamma: float):
    if lr_decay_step <= 0:
        return None
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_decay_step,
        gamma=lr_decay_gamma,
    )


def _print_batch_progress(
    epoch: int,
    batch_index: int,
    total_batches: int,
    running_loss: float,
    correct: int,
    seen: int,
    progress_every: int,
) -> None:
    if progress_every <= 0:
        return
    if batch_index % progress_every != 0 and batch_index != total_batches:
        return
    average_loss = running_loss / batch_index
    accuracy = correct / max(1, seen)
    print(
        f"epoch={epoch + 1} batch={batch_index}/{total_batches} "
        f"running_loss={average_loss:.4f} running_accuracy={accuracy:.3f}",
        flush=True,
    )
