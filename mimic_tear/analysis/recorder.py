from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor

from data.models.gamepad import get_inputs_names_classified
from mimic_tear.model.components.controller import ControllerOutput


@dataclass(frozen=True, slots=True)
class _OnePredictionChunk:
    recording_id: str
    frames: Tensor  # [N]

    analog_predicted: Tensor  # [N, 6]
    analog_target: Tensor  # [N, 6]

    button_logits: Tensor  # [N, 14]
    button_target: Tensor  # [N, 14]

class PredictionRecorder:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

        self.analog_names, self.button_names = get_inputs_names_classified()

        self._chunks: list[_OnePredictionChunk] = []

        self._current_recording: str | None = None
        self._current_frame_counter: int = 0

    def begin_recording(self, recording_id: str) -> None:
        self._current_recording = str(recording_id)
        self._current_frame_counter = 0

    @torch.no_grad()
    def update(
        self,
        *,
        output: ControllerOutput,
        target_analog: Tensor,
        target_buttons: Tensor,
    ) -> None:
        if self._current_recording is None:
            raise RuntimeError("begin_recording must be called before update")

        analog_pred = self._to_cpu(output.analog)
        analog_target = self._to_cpu(target_analog)

        btn_pred = self._to_cpu(output.button_logits)
        btn_target = self._to_cpu(target_buttons)

        analog_y_count = analog_pred.shape[0]
        analog_x_count = analog_target.shape[0]
        btn_y_count = btn_pred.shape[0]
        btn_x_count = btn_target.shape[0]

        if not analog_y_count == analog_x_count == btn_y_count == btn_x_count:
            raise ValueError(
                f"Prediction and target tensors must have the same number of frames, "
                f"but got: "
                f"analog_y={analog_y_count}, "
                f"analog_x={analog_x_count}, "
                f"btn_y={btn_y_count}, "
                f"btn_x={btn_x_count}"
            )

        chunk = _OnePredictionChunk(
            recording_id=self._current_recording,
            frames=torch.arange(
                self._current_frame_counter,
                self._current_frame_counter + analog_y_count,
            ),
            analog_predicted=analog_pred,
            analog_target=analog_target,
            button_logits=btn_pred,
            button_target=btn_target,
        )

        self._chunks.append(chunk)
        self._current_frame_counter += analog_y_count

    def save(self, epoch: int | None = None) -> None:
        if not self._chunks:
            raise RuntimeError("No predictions recorded")

        if epoch is not None:
            hive_path = self.path / f"epoch={epoch}" / "predictions.parquet"
        else:
            hive_path = self.path / "predictions.parquet"

        hive_path.parent.mkdir(parents=True, exist_ok=True)

        # one recording_id per chunk -> one per row
        recording_ids = np.repeat(
            [chunk.recording_id for chunk in self._chunks],
            [len(chunk.frames) for chunk in self._chunks],
        )

        cols: dict[str, pa.Array] = {
            "recording_id": pa.array(recording_ids).dictionary_encode(),
            "frame_index": pa.array(
                torch.cat([chunk.frames for chunk in self._chunks]).numpy()
            ),
        }

        self._add_columns(
            cols,
            names=self.analog_names,
            suffix="predicted",
            tensors=[chunk.analog_predicted for chunk in self._chunks],
        )
        self._add_columns(
            cols,
            names=self.analog_names,
            suffix="target",
            tensors=[chunk.analog_target for chunk in self._chunks],
        )
        self._add_columns(
            cols,
            names=self.button_names,
            suffix="logit",
            tensors=[chunk.button_logits for chunk in self._chunks],
        )
        self._add_columns(
            cols,
            names=self.button_names,
            suffix="target",
            tensors=[chunk.button_target for chunk in self._chunks],
        )

        pq.write_table(pa.table(cols), hive_path)

        self._chunks.clear()

    @staticmethod
    def _add_columns(
        columns: dict[str, pa.Array],
        *,
        names: tuple[str, ...],
        suffix: str,
        tensors: list[Tensor],
    ) -> None:
        values = torch.cat(tensors, dim=0).numpy()

        for index, name in enumerate(names):
            columns[f"{name}_{suffix}"] = pa.array(values[:, index])

    @staticmethod
    def _to_cpu(tensor: Tensor) -> Tensor:
        # [B, T, D] -> [B*T, D]
        return tensor.detach().reshape(-1, tensor.shape[-1]).float().cpu()