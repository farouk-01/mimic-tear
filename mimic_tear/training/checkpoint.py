from pathlib import Path
from typing import Any

import torch

from mimic_tear.model.policy import LSTMPolicy


def save_checkpoint(
    path: str | Path,
    *,
    model: LSTMPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    *,
    model: LSTMPolicy,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
