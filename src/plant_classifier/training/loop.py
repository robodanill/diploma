from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader

from plant_classifier.data import ImageRecord, sample_pairs
from plant_classifier.training.genus_eval import GenusEvalResult
from plant_classifier.training.image_pairs import PairImageDataset


@dataclass(frozen=True)
class TrainResult:
    checkpoint_path: Path
    last_loss: float
    history_path: Path | None = None


def train_siamese(
    model: nn.Module,
    dataloader: DataLoader,
    checkpoint_path: Path,
    epochs: int = 20,
    learning_rate: float = 0.001,
    momentum: float = 0.9,
    lr_decay_step: int = 0,
    lr_decay_gamma: float = 0.5,
    max_iterations: int | None = None,
    device: str | None = None,
    progress_every: int = 5,
) -> TrainResult:
    """Train a Siamese model with binary cross-entropy over pair labels."""

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"training device={resolved_device}", flush=True)
    model.to(resolved_device)
    model.train()

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = _build_scheduler(optimizer, lr_decay_step, lr_decay_gamma)
    last_loss = 0.0
    best_loss = float("inf")
    global_step = 0
    history_path = _history_path(checkpoint_path)
    _reset_history(history_path)

    for epoch in range(epochs):
        started_at = perf_counter()
        running_loss = 0.0
        batches_seen = 0
        total_batches = len(dataloader)
        for batch_index, (left, right, labels) in enumerate(dataloader, start=1):
            batches_seen = batch_index
            left = left.to(resolved_device)
            right = right.to(resolved_device)
            labels = labels.to(resolved_device).float()

            optimizer.zero_grad()
            predictions = model(left, right)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            running_loss += loss.item()
            _print_batch_progress(epoch, batch_index, total_batches, running_loss, progress_every)
            if max_iterations and global_step >= max_iterations:
                break

        last_loss = running_loss / max(1, batches_seen)
        is_best = last_loss < best_loss
        best_loss = min(best_loss, last_loss)
        elapsed = perf_counter() - started_at
        print(
            f"epoch={epoch + 1} loss={last_loss:.4f} "
            f"iterations={global_step} time={elapsed:.1f}s",
            flush=True,
        )
        _append_history_row(
            history_path,
            {
                "epoch": epoch + 1,
                "train_loss": last_loss,
                "best_loss": best_loss,
                "iterations": global_step,
                "elapsed_seconds": elapsed,
                "is_best": is_best,
            },
        )
        if max_iterations and global_step >= max_iterations:
            break

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    return TrainResult(checkpoint_path=checkpoint_path, last_loss=last_loss, history_path=history_path)


def train_siamese_with_dynamic_pairs(
    model: nn.Module,
    records: list[ImageRecord],
    taxonomic_level: str,
    view: str,
    checkpoint_path: Path,
    positive_count: int,
    negative_count: int,
    hard_negative_ratio: float = 0.0,
    targeted_negative_ratio: float = 0.0,
    targeted_negative_label_pairs: list[tuple[str, str]] | None = None,
    pair_sampling_strategy: str = "label_uniform",
    image_size: int = 224,
    crop_size: int = 32,
    preprocessing: bool = False,
    batch_size: int = 32,
    epochs: int = 20,
    learning_rate: float = 0.001,
    momentum: float = 0.9,
    lr_decay_step: int = 0,
    lr_decay_gamma: float = 0.5,
    max_iterations: int | None = None,
    num_workers: int = 2,
    seed: int = 42,
    device: str | None = None,
    eval_fn=None,
    progress_every: int = 5,
) -> TrainResult:
    """Train a Siamese model while re-sampling positive/negative pairs every epoch."""

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"training device={resolved_device}", flush=True)
    model.to(resolved_device)
    model.train()

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = _build_scheduler(optimizer, lr_decay_step, lr_decay_gamma)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_best{checkpoint_path.suffix}"
    )

    last_loss = 0.0
    best_loss = float("inf")
    best_eval = -1.0
    global_step = 0
    history_path = _history_path(checkpoint_path)
    _reset_history(history_path)

    for epoch in range(epochs):
        pairs = sample_pairs(
            records=records,
            taxonomic_level=taxonomic_level,
            positive_count=positive_count,
            negative_count=negative_count,
            hard_negative_ratio=hard_negative_ratio,
            targeted_negative_ratio=targeted_negative_ratio,
            targeted_negative_label_pairs=targeted_negative_label_pairs,
            seed=seed + epoch,
            strategy=pair_sampling_strategy,
        )
        print(
            f"epoch={epoch + 1} sampled_pairs={len(pairs)} "
            f"sampling={pair_sampling_strategy} batch_size={batch_size} "
            f"hard_negative_ratio={hard_negative_ratio:.2f} "
            f"targeted_negative_ratio={targeted_negative_ratio:.2f} "
            f"targeted_negative_pairs={len(targeted_negative_label_pairs or [])} "
            f"device={resolved_device}",
            flush=True,
        )
        dataset = PairImageDataset(
            pairs=pairs,
            view=view,
            image_size=image_size,
            crop_size=crop_size,
            preprocessing=preprocessing,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        started_at = perf_counter()
        running_loss = 0.0
        batches_seen = 0
        total_batches = len(dataloader)
        for batch_index, (left, right, labels) in enumerate(dataloader, start=1):
            batches_seen = batch_index
            left = left.to(resolved_device)
            right = right.to(resolved_device)
            labels = labels.to(resolved_device).float()

            optimizer.zero_grad()
            predictions = model(left, right)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            running_loss += loss.item()
            _print_batch_progress(epoch, batch_index, total_batches, running_loss, progress_every)
            if max_iterations and global_step >= max_iterations:
                break

        last_loss = running_loss / max(1, batches_seen)
        elapsed = perf_counter() - started_at
        if eval_fn is not None:
            print(f"epoch={epoch + 1} eval=starting", flush=True)
        eval_result = eval_fn(model, resolved_device) if eval_fn is not None else None
        is_best = _is_best(last_loss, eval_result, best_loss, best_eval)
        if is_best:
            best_loss = last_loss
            if eval_result is not None:
                best_eval = eval_result.primary_accuracy
            torch.save(model.state_dict(), best_checkpoint_path)
            best_marker = " best"
        else:
            best_marker = ""
        print(
            _format_epoch(epoch, last_loss, best_loss, elapsed, eval_result, best_eval, best_marker),
            flush=True,
        )
        _append_history_row(
            history_path,
            _history_row(
                epoch=epoch,
                loss=last_loss,
                best_loss=best_loss,
                elapsed=elapsed,
                iterations=global_step,
                eval_result=eval_result,
                best_eval=best_eval,
                is_best=is_best,
            ),
        )
        if max_iterations and global_step >= max_iterations:
            break

    torch.save(model.state_dict(), checkpoint_path)
    return TrainResult(checkpoint_path=checkpoint_path, last_loss=last_loss, history_path=history_path)


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


def _history_row(
    epoch: int,
    loss: float,
    best_loss: float,
    elapsed: float,
    iterations: int,
    eval_result: GenusEvalResult | None,
    best_eval: float,
    is_best: bool,
) -> dict:
    row = {
        "epoch": epoch + 1,
        "train_loss": loss,
        "best_loss": best_loss,
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "is_best": is_best,
    }
    if eval_result is None:
        return row

    row.update(
        {
            "val_score_mode": eval_result.score_mode,
            "val_pair_loss": "" if eval_result.pair_loss is None else eval_result.pair_loss,
            "val_primary_top_k": eval_result.primary_top_k,
            "val_primary_accuracy": eval_result.primary_accuracy,
            "best_val_primary_accuracy": best_eval,
            "val_references": eval_result.references,
            "val_queries": eval_result.queries,
            "val_primary_hits": eval_result.hits[eval_result.primary_top_k],
        }
    )
    for top_k in eval_result.top_ks:
        row[f"val_top{top_k}_genus_accuracy"] = eval_result.accuracies[top_k]
        row[f"val_top{top_k}_genus_hits"] = eval_result.hits[top_k]
    return row


def _build_scheduler(optimizer, lr_decay_step: int, lr_decay_gamma: float):
    if lr_decay_step <= 0:
        return None
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_decay_step,
        gamma=lr_decay_gamma,
    )


def _is_best(
    loss: float,
    eval_result: GenusEvalResult | None,
    best_loss: float,
    best_eval: float,
) -> bool:
    if eval_result is not None:
        return eval_result.primary_accuracy > best_eval
    return loss < best_loss


def _print_batch_progress(
    epoch: int,
    batch_index: int,
    total_batches: int,
    running_loss: float,
    progress_every: int,
) -> None:
    if progress_every <= 0:
        return
    if batch_index % progress_every != 0 and batch_index != total_batches:
        return

    average_loss = running_loss / batch_index
    print(
        f"epoch={epoch + 1} batch={batch_index}/{total_batches} "
        f"running_loss={average_loss:.4f}",
        flush=True,
    )


def _format_epoch(
    epoch: int,
    loss: float,
    best_loss: float,
    elapsed: float,
    eval_result: GenusEvalResult | None,
    best_eval: float,
    best_marker: str,
) -> str:
    parts = [
        f"epoch={epoch + 1}",
        f"loss={loss:.4f}",
        f"best_loss={best_loss:.4f}",
    ]
    if eval_result is not None:
        eval_parts = [
            " ".join(
                f"top{top_k}_genus_accuracy={eval_result.accuracies[top_k]:.3f}"
                for top_k in eval_result.top_ks
            )
        ]
        if eval_result.pair_loss is not None:
            eval_parts.append(f"val_pair_loss={eval_result.pair_loss:.4f}")
        eval_parts.extend(
            [
                f"score_mode={eval_result.score_mode}",
                f"best_top{eval_result.primary_top_k}={best_eval:.3f}",
                f"eval={eval_result.hits[eval_result.primary_top_k]}/{eval_result.queries}",
            ]
        )
        parts.extend(eval_parts)
    parts.append(f"time={elapsed:.1f}s{best_marker}")
    return " ".join(parts)
