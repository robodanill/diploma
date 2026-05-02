from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


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
        print(f"epoch={epoch + 1} loss={last_loss:.4f}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    return TrainResult(checkpoint_path=checkpoint_path, last_loss=last_loss)

