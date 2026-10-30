from dataclasses import dataclass


FRAME_FIELD = "frame"
MAX_HELD = 30

@dataclass(frozen=True, slots=True)
class RecordColumns:
    frame_index: str
    capture_timestamp_ns: str


DEFAULT_COLUMNS = RecordColumns(
    frame_index="frame_index",
    capture_timestamp_ns="frame_timestamp_ns",
)
