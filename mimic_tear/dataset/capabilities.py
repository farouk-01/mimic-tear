from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CapabilityAvailability:
    """Result of inspecting one optional recording capability."""

    available: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.available and self.reason is not None:
            raise ValueError("An available capability cannot have a failure reason")
        if not self.available and not self.reason:
            raise ValueError("An unavailable capability must explain why")


class RecordingCapability(Protocol):
    """A modular requirement that can inspect a recording directory."""

    @property
    def key(self) -> str: ...

    def inspect(self, session: Path) -> CapabilityAvailability: ...


@dataclass(frozen=True, slots=True)
class FileRecordingCapability:
    """An optional recording capability supplied by one relative file."""

    key: str
    relative_path: Path
    description: str

    def __post_init__(self) -> None:
        if not self.key or not self.key.isidentifier():
            raise ValueError("Capability key must be a non-empty identifier")
        if self.relative_path.is_absolute() or ".." in self.relative_path.parts:
            raise ValueError("Capability paths must remain inside the recording")
        if not self.description:
            raise ValueError("Capability description cannot be empty")

    def inspect(self, session: Path) -> CapabilityAvailability:
        path = session / self.relative_path
        if path.is_file():
            return CapabilityAvailability(available=True)
        if path.exists():
            reason = f"{self.relative_path} is not a regular file"
        else:
            reason = f"missing {self.relative_path}"
        return CapabilityAvailability(available=False, reason=reason)


GAME_STATE_CAPABILITY = FileRecordingCapability(
    key="game_state",
    relative_path=Path("game_state.parquet"),
    description="frame-aligned process-memory game state",
)


@dataclass(frozen=True, slots=True)
class MissingCapability:
    key: str
    reason: str


@dataclass(frozen=True, slots=True)
class ExcludedRecording:
    session: Path
    missing_capabilities: tuple[MissingCapability, ...]


@dataclass(frozen=True, slots=True)
class RecordingSelectionReport:
    """Structured result of applying optional capability requirements."""

    required_capabilities: tuple[str, ...]
    included: tuple[Path, ...]
    excluded: tuple[ExcludedRecording, ...]

    @property
    def discovered_count(self) -> int:
        return len(self.included) + len(self.excluded)

    @property
    def excluded_count(self) -> int:
        return len(self.excluded)

    def log_lines(self, *, label: str) -> tuple[str, ...]:
        lines = [
            f"{label.capitalize()} recording capability selection: included "
            f"{len(self.included)} of {self.discovered_count}; excluded "
            f"{self.excluded_count}."
        ]
        if not self.excluded:
            return tuple(lines)

        lines.append("Exclusion reasons:")
        reasons = Counter(
            (missing.key, missing.reason)
            for recording in self.excluded
            for missing in recording.missing_capabilities
        )
        lines.extend(
            f"  {count} recording(s): {key} ({reason})"
            for (key, reason), count in sorted(reasons.items())
        )
        return tuple(lines)


def select_recordings_by_capabilities(
    sessions: Sequence[str | Path],
    *,
    required: Sequence[RecordingCapability] = (),
) -> RecordingSelectionReport:
    """Select recordings that provide every declared optional capability."""

    requirements = tuple(required)
    keys = tuple(capability.key for capability in requirements)
    if len(set(keys)) != len(keys):
        raise ValueError("Required recording capability keys must be unique")

    included: list[Path] = []
    excluded: list[ExcludedRecording] = []
    for session_value in sessions:
        session = Path(session_value).expanduser().resolve()
        missing = tuple(
            MissingCapability(capability.key, availability.reason or "unavailable")
            for capability in requirements
            if not (availability := capability.inspect(session)).available
        )
        if missing:
            excluded.append(ExcludedRecording(session, missing))
        else:
            included.append(session)

    return RecordingSelectionReport(
        required_capabilities=keys,
        included=tuple(included),
        excluded=tuple(excluded),
    )
