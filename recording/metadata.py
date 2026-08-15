from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class RecordingMetadata:
    format_version: int
    fps: float

    theme: str | None = None
    enemy: str | None = None
    success: bool | None = None