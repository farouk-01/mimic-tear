from __future__ import annotations

from collections.abc import Iterable

import pyarrow as pa

ANALOG_COLUMNS: tuple[str, ...] = (
    "left_x",
    "left_y",
    "right_x",
    "right_y",
    "left_trigger",
    "right_trigger",
)

BUTTON_COLUMNS: tuple[str, ...] = (
    "south",
    "east",
    "west",
    "north",
    "left_bumper",
    "right_bumper",
    "left_stick",
    "right_stick",
    "dpad_up",
    "dpad_down",
    "dpad_left",
    "dpad_right",
    "start",
    "back",
)

INPUT_COLUMNS: tuple[str, ...] = (
    "frame_index",
    "timestamp_ns",
    "source_timestamp_ns",
    "frame_timestamp_ns",
    "input_timestamp_ns",
    "input_offset_ns",
    *ANALOG_COLUMNS,
    *BUTTON_COLUMNS,
)

PARQUET_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("source_timestamp_ns", pa.int64(), nullable=False),
        pa.field("frame_timestamp_ns", pa.int64(), nullable=False),
        pa.field("input_timestamp_ns", pa.int64(), nullable=False),
        pa.field("input_offset_ns", pa.int64(), nullable=False),
        *(pa.field(column, pa.float32(), nullable=False) for column in ANALOG_COLUMNS),
        *(pa.field(column, pa.bool_(), nullable=False) for column in BUTTON_COLUMNS),
    ]
)

def validate_columns(fieldnames: Iterable[str]) -> None:
    """Require the one supported input column layout, including order."""

    actual = tuple(fieldnames)
    if actual == INPUT_COLUMNS:
        return

    missing = sorted(set(INPUT_COLUMNS).difference(actual))
    unexpected = sorted(set(actual).difference(INPUT_COLUMNS))
    details: list[str] = []
    if missing:
        details.append("missing: " + ", ".join(missing))
    if unexpected:
        details.append("unexpected: " + ", ".join(unexpected))
    if not details:
        details.append("columns are in the wrong order")
    raise ValueError("Recording input columns do not match (" + "; ".join(details) + ")")
