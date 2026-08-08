from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


ANNOTATIONS_FILENAME = "annotations.json"


@dataclass(frozen=True, order=True, slots=True)
class FrameRange:
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if type(self.start_frame) is not int or type(self.end_frame) is not int:
            raise TypeError("Frame indexes must be integers")
        if self.start_frame < 0:
            raise ValueError("start_frame cannot be negative")
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame cannot be before start_frame")

    def contains(self, frame_index: int) -> bool:
        return self.start_frame <= frame_index <= self.end_frame

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1


def merge_frame_ranges(
    ranges: Iterable[FrameRange],
    *,
    total_frames: int | None = None,
) -> list[FrameRange]:
    if total_frames is not None and total_frames < 0:
        raise ValueError("total_frames cannot be negative")

    ordered = sorted(ranges)
    if total_frames is not None:
        for frame_range in ordered:
            if frame_range.end_frame >= total_frames:
                raise ValueError(
                    f"Excluded frame {frame_range.end_frame} is outside a "
                    f"{total_frames}-frame recording"
                )

    merged: list[FrameRange] = []
    for frame_range in ordered:
        if not merged or frame_range.start_frame > merged[-1].end_frame + 1:
            merged.append(frame_range)
            continue
        previous = merged[-1]
        merged[-1] = FrameRange(
            previous.start_frame,
            max(previous.end_frame, frame_range.end_frame),
        )
    return merged


def load_frame_ranges(
    session_directory: str | Path,
    *,
    total_frames: int | None = None,
) -> list[FrameRange]:
    path = Path(session_directory) / ANNOTATIONS_FILENAME
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")

    raw_ranges = data.get("excluded_ranges", [])
    if not isinstance(raw_ranges, list):
        raise ValueError(f"excluded_ranges must be a list in {path}")

    ranges: list[FrameRange] = []
    for index, item in enumerate(raw_ranges):
        if not isinstance(item, dict):
            raise ValueError(f"Excluded range {index} must be an object in {path}")
        try:
            start_frame = item["start_frame"]
            end_frame = item["end_frame"]
            if type(start_frame) is not int or type(end_frame) is not int:
                raise TypeError
            ranges.append(FrameRange(start_frame, end_frame))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid excluded range {index} in {path}") from error

    return merge_frame_ranges(ranges, total_frames=total_frames)


def save_frame_ranges(
    session_directory: str | Path,
    ranges: Iterable[FrameRange],
    *,
    total_frames: int | None = None,
) -> Path:
    session = Path(session_directory)
    path = session / ANNOTATIONS_FILENAME
    temporary_path = session / f".{ANNOTATIONS_FILENAME}.tmp"
    merged = merge_frame_ranges(ranges, total_frames=total_frames)
    payload = {
        "excluded_ranges": [
            {
                "start_frame": frame_range.start_frame,
                "end_frame": frame_range.end_frame,
            }
            for frame_range in merged
        ]
    }
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return path


def is_frame_excluded(frame_index: int, ranges: Iterable[FrameRange]) -> bool:
    return any(frame_range.contains(frame_index) for frame_range in ranges)
