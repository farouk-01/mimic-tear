from pathlib import Path

import pyarrow.parquet as pq

from controller import (
    ANALOG_INPUTS,
    BUTTON_INPUTS,
    AnalogState,
    ButtonState,
    ControllerState,
)
from data.datasets.controller import ControllerStore


class ParquetControllerStore(ControllerStore):
    def __init__(
        self,
        *,
        path: str | Path,
    ) -> None:
        self.path = Path(path)

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Controller parquet does not exist: {self.path}"
            )

        table = pq.read_table(self.path)

        required_columns = (
            *ANALOG_INPUTS,
            *BUTTON_INPUTS,
        )

        missing = [
            column
            for column in required_columns
            if column not in table.column_names
        ]

        if missing:
            raise ValueError(
                f"Controller parquet is missing columns: {missing}"
            )

        if table.num_rows <= 0:
            raise ValueError(
                "Controller parquet cannot be empty"
            )

        self._analog = {
            name: table[name].to_pylist()
            for name in ANALOG_INPUTS
        }

        self._buttons = {
            name: table[name].to_pylist()
            for name in BUTTON_INPUTS
        }

        self._length = table.num_rows

    def __len__(self) -> int:
        return self._length

    def get(
        self,
        index: int,
    ) -> ControllerState:
        if index < 0:
            index += len(self)

        if not 0 <= index < len(self):
            raise IndexError(index)

        analog = AnalogState(
            left_x=float(self._analog["left_x"][index]),
            left_y=float(self._analog["left_y"][index]),
            right_x=float(self._analog["right_x"][index]),
            right_y=float(self._analog["right_y"][index]),
            left_trigger=float(
                self._analog["left_trigger"][index]
            ),
            right_trigger=float(
                self._analog["right_trigger"][index]
            ),
        )

        buttons = ButtonState(
            south=bool(self._buttons["south"][index]),
            east=bool(self._buttons["east"][index]),
            west=bool(self._buttons["west"][index]),
            north=bool(self._buttons["north"][index]),
            left_bumper=bool(
                self._buttons["left_bumper"][index]
            ),
            right_bumper=bool(
                self._buttons["right_bumper"][index]
            ),
            left_stick=bool(
                self._buttons["left_stick"][index]
            ),
            right_stick=bool(
                self._buttons["right_stick"][index]
            ),
            dpad_up=bool(
                self._buttons["dpad_up"][index]
            ),
            dpad_down=bool(
                self._buttons["dpad_down"][index]
            ),
            dpad_left=bool(
                self._buttons["dpad_left"][index]
            ),
            dpad_right=bool(
                self._buttons["dpad_right"][index]
            ),
            start=bool(self._buttons["start"][index]),
            back=bool(self._buttons["back"][index]),
        )

        state = ControllerState(
            analog=analog,
            buttons=buttons,
        )

        state.validate()

        return state