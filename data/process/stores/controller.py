from pathlib import Path

import pyarrow.parquet as pq
import torch
import numpy as np

from controller import (
    ANALOG_INPUTS,
    BUTTON_INPUTS,
    AnalogState,
    ButtonState,
    ControllerState,
)
from data.datasets.controller import ControllerSample, ControllerStore


class ParquetControllerStore(ControllerStore):
    def __init__(
        self,
        *,
        path: str | Path,
    ) -> None:
        self.path = Path(path)

        if not self.path.is_file():
            raise FileNotFoundError(f"Controller parquet does not exist: {self.path}")

        table = pq.read_table(self.path)

        required_columns = (
            *ANALOG_INPUTS,
            *BUTTON_INPUTS,
        )

        missing = [
            column for column in required_columns if column not in table.column_names
        ]

        if missing:
            raise ValueError(f"Controller parquet is missing columns: {missing}")

        if table.num_rows <= 0:
            raise ValueError("Controller parquet cannot be empty")

        self._analog = torch.from_numpy(
            np.stack(
                [table[name].to_numpy(zero_copy_only=False) for name in ANALOG_INPUTS],
                axis=1,
            )
        ).to(torch.float32)

        self._buttons = torch.from_numpy(
            np.stack(
                [table[name].to_numpy(zero_copy_only=False) for name in BUTTON_INPUTS],
                axis=1,
            )
        ).to(torch.float32)

        self._length = table.num_rows

    def __len__(self) -> int:
        return self._length

    def get(self, index: int) -> ControllerState:
        if index < 0:
            index += len(self)

        if not 0 <= index < len(self):
            raise IndexError(index)

        analog = self._analog[index]
        buttons = self._buttons[index]

        state = ControllerState(
            analog=AnalogState(
                left_x=analog[0].item(),
                left_y=analog[1].item(),
                right_x=analog[2].item(),
                right_y=analog[3].item(),
                left_trigger=analog[4].item(),
                right_trigger=analog[5].item(),
            ),
            buttons=ButtonState(
                south=bool(buttons[0].item()),
                east=bool(buttons[1].item()),
                west=bool(buttons[2].item()),
                north=bool(buttons[3].item()),
                left_bumper=bool(buttons[4].item()),
                right_bumper=bool(buttons[5].item()),
                left_stick=bool(buttons[6].item()),
                right_stick=bool(buttons[7].item()),
                dpad_up=bool(buttons[8].item()),
                dpad_down=bool(buttons[9].item()),
                dpad_left=bool(buttons[10].item()),
                dpad_right=bool(buttons[11].item()),
                start=bool(buttons[12].item()),
                back=bool(buttons[13].item()),
            ),
        )

        state.validate()
        return state

    def get_range(self, start: int, end: int) -> ControllerSample:
        if start < 0:
            start += len(self)

        if end < 0:
            end += len(self)

        if not 0 <= start <= end <= len(self):
            raise IndexError(f"Invalid range [{start}:{end}]")

        return ControllerSample(
            analog=self._analog[start:end],
            buttons=self._buttons[start:end],
        )
