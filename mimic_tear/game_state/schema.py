from __future__ import annotations

import pyarrow as pa


GAME_STATE_VALUE_TYPES: dict[str, pa.DataType] = {
    "player_health": pa.int32(),
    "player_max_health": pa.int32(),
    "player_fp": pa.int32(),
    "player_max_fp": pa.int32(),
    "player_stamina": pa.int32(),
    "player_max_stamina": pa.int32(),
    "player_x": pa.float32(),
    "player_y": pa.float32(),
    "player_z": pa.float32(),
    "lock_on_active": pa.bool_(),
    "location_id": pa.int64(),
}

GAME_STATE_VALUE_KINDS: dict[str, str] = {
    name: (
        "bool"
        if pa.types.is_boolean(data_type)
        else "int"
        if pa.types.is_integer(data_type)
        else "float"
        if pa.types.is_floating(data_type)
        else "string"
    )
    for name, data_type in GAME_STATE_VALUE_TYPES.items()
}

GAME_STATE_COLUMNS: tuple[str, ...] = (
    "frame_index",
    "timestamp_ns",
    "frame_timestamp_ns",
    "state_timestamp_ns",
    "state_offset_ns",
    "state_valid",
    *GAME_STATE_VALUE_TYPES,
)

GAME_STATE_PARQUET_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("frame_timestamp_ns", pa.int64(), nullable=False),
        pa.field("state_timestamp_ns", pa.int64(), nullable=False),
        pa.field("state_offset_ns", pa.int64(), nullable=False),
        pa.field("state_valid", pa.bool_(), nullable=False),
        *(
            pa.field(name, data_type, nullable=True)
            for name, data_type in GAME_STATE_VALUE_TYPES.items()
        ),
    ]
)
