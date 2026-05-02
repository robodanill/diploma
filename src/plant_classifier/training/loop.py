from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader

from plant_classifier.data import ImageRecord, sample_pairs
from plant_classifier.training.image_pairs import PairImageDataset


@dataclass(frozen=True)
class TrainResult:
    checkpoint_path: Path
    last_loss: float


def train_siamese(
    model: nn.Module,
    dataloader: DataLoader,
    checkpoint_path: Path,
    epochs: int = 20,
    learning_rate: float = 0.001,
    momentum: float = 0.9,
    device: str | None = None,
) -> TrainResult:
    """Train a Siamese model with binary cross-entropy over pair labels."""

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(resolved_device)
    model.train()

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    last_loss = 0.0

    for epoch in range(epochs):
        started_at = perf_counter()
        running_loss = 0.0
        for left, right, labels in dataloader:
            left = left.to(resolved_device)
            right = right.to(resolved_device)
            labels = labels.to(resolved_device).float()

            optimizer.zero_grad()
            predictions = model(left, right)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        last_loss = running_loss / max(1, len(dataloader))
        elapsed = perf_counter() - started_at
        print(f"epoch={epoch + 1} loss={last_loss:.4f} time={elapsed:.1f}s")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    return TrainResult(checkpoint_path=checkpoint_path, last_loss=last_loss)


def train_siamese_with_dynamic_pairs(
    model: nn.Module,
    records: list[ImageRecord],
    taxonomic_level: str,
    view: str,
    checkpoint_path: Path,
    positive_count: int,
    negative_count: int,
    image_size: int = 224,
    crop_size: int = 32,
    batch_size: int = 32,
    epochs: int = 20,
    learning_rate: float = 0.001,
    momentum: float = 0.9,
    num_workers: int = 2,
    seed: int = 42,
    device: str | None = None,
) -> TrainResult:
    """Train a Siamese model while re-sampling positive/negative pairs every epoch."""

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(resolved_device)
    model.train()

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_best{checkpoint_path.suffix}"
    )

    last_loss = 0.0
    best_loss = float("inf")

    for epoch in range(epochs):
        pairs = sample_pairs(
            records=records,
            taxonomic_level=taxonomic_level,
            positive_count=positive_count,
            negative_count=negative_count,
            seed=seed + epoch,
        )
        dataset = PairImageDataset(
            pairs=pairs,
            view=view,
            image_size=image_size,
            crop_size=crop_size,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        started_at = perf_counter()
        running_loss = 0.0
        for left, right, labels in dataloader:
            left = left.to(resolved_device)
            right = right.to(resolved_device)
            labels = labels.to(resolved_device).float()

            optimizer.zero_grad()
            predictions = model(left, right)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        last_loss = running_loss / max(1, len(dataloader))
        elapsed = perf_counter() - started_at
        if last_loss < best_loss:
            best_loss = last_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            best_marker = " best"
        else:
            best_marker = ""
        print(
            f"epoch={epoch + 1} loss={last_loss:.4f} "
            f"best_loss={best_loss:.4f} time={elapsed:.1f}s{best_marker}"
        )

    torch.save(model.state_dict(), checkpoint_path)
    return TrainResult(checkpoint_path=checkpoint_path, last_loss=last_loss)
