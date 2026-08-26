from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class RecordingMetadata:
    format_version: int
    fps: float
    width: int
    height: int
    sample_count: int

    theme: str | None = None
    enemy: str | None = None
    success: bool | None = None

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.fps