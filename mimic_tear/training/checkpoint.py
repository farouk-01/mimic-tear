from pathlib import Path
from typing import Any
from collections.abc import Sequence

import torch

from mimic_tear.model.policy import LSTMPolicy
from mimic_tear.training.trainer import GamepadPredictions
from data.models.gamepad import get_inputs_names_classified


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


def save_predictions(
    path: str | Path,
    *,
    predictions: Sequence[GamepadPredictions],
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    def _concat_column(
        tensors: Sequence[torch.Tensor],
        *,
        index: int,
    ) -> list[float]:
        return torch.cat(
            [tensor.detach().reshape(-1, tensor.shape[-1]).cpu() for tensor in tensors],
            dim=0,
        )[:, index].tolist()

    analog_names, button_names = get_inputs_names_classified()

    p_analog: list[torch.Tensor] = []
    p_button: list[torch.Tensor] = []
    target_analog: list[torch.Tensor] = []
    target_button: list[torch.Tensor] = []

    for p_controller, t_analog, t_button in predictions:
        analog = p_controller.analog
        button = p_controller.button_logits

        p_analog.append(analog)
        p_button.append(button)
        target_analog.append(t_analog)
        target_button.append(t_button)

    columns: dict[str, list[float]] = {}

    for index, name in enumerate(analog_names):
        columns[f"{name}_predicted"] = _concat_column(p_analog, index=index)
        columns[f"{name}_target"] = _concat_column(target_analog, index=index)

    for index, name in enumerate(button_names):
        columns[f"{name}_logit"] = _concat_column(p_button, index=index)
        columns[f"{name}_target"] = _concat_column(target_button, index=index)

    pq.write_table(pa.table(columns), path)
