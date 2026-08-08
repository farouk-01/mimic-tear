from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import pyarrow.parquet as pq


from ai_player.recording.schema import (  # noqa: E402
    ANALOG_COLUMNS,
    BUTTON_COLUMNS,
    PARQUET_SCHEMA,
    validate_columns,
)
from ai_player.recording.annotations import load_frame_ranges  # noqa: E402
from ai_player.game_state.schema import (  # noqa: E402
    GAME_STATE_PARQUET_SCHEMA,
    GAME_STATE_VALUE_TYPES,
)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    session_directory: str
    video_frames: int
    input_rows: int
    game_state_rows: int | None
    fps: float
    width: int
    height: int
    active_input_frames: dict[str, int]
    mean_abs_input_offset_ms: float | None
    maximum_abs_input_offset_ms: float | None
    errors: list[str]
    warnings: list[str]

    @property
    def valid(self) -> bool:
        return not self.errors


def read_metadata(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append("metadata.json is missing")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"Could not read metadata.json: {error}")
        return {}
    if not isinstance(data, dict):
        errors.append("metadata.json must contain a JSON object")
        return {}
    return data


def validate_inputs(
    path: Path,
    metadata: dict[str, Any],
    errors: list[str],
) -> tuple[int, list[int], list[int], dict[str, int], list[int]]:
    active_input_frames = {
        column: 0 for column in (*ANALOG_COLUMNS, *BUTTON_COLUMNS)
    }
    timestamps: list[int] = []
    frame_timestamps: list[int] = []
    input_offsets_ns: list[int] = []
    if not path.is_file():
        errors.append("inputs.parquet is missing")
        return 0, timestamps, frame_timestamps, active_input_frames, input_offsets_ns

    try:
        parquet_file = pq.ParquetFile(path)
    except Exception as error:
        errors.append(f"Could not open inputs.parquet: {error}")
        return 0, timestamps, frame_timestamps, active_input_frames, input_offsets_ns

    parquet_schema = parquet_file.schema_arrow
    try:
        validate_columns(parquet_schema.names)
    except ValueError as error:
        errors.append(str(error))
        return 0, timestamps, frame_timestamps, active_input_frames, input_offsets_ns
    if not parquet_schema.equals(PARQUET_SCHEMA, check_metadata=False):
        errors.append(
            f"Parquet column types do not match: expected {PARQUET_SCHEMA}, "
            f"received {parquet_schema}"
        )
        return 0, timestamps, frame_timestamps, active_input_frames, input_offsets_ns

    config = metadata.get("config", {})
    maximum_sync_offset_ms = (
        float(config.get("maximum_sync_offset_ms", 15.0))
        if isinstance(config, dict)
        else 15.0
    )
    previous_timestamp: int | None = None
    previous_frame_timestamp: int | None = None
    row_count = 0

    try:
        batches = parquet_file.iter_batches(batch_size=4096)
        for batch in batches:
            for row in batch.to_pylist():
                row_count += 1
                try:
                    frame_index = int(row["frame_index"])
                    timestamp_ns = int(row["timestamp_ns"])
                    source_timestamp_ns = int(row["source_timestamp_ns"])
                    frame_timestamp_ns = int(row["frame_timestamp_ns"])
                    input_timestamp_ns = int(row["input_timestamp_ns"])
                    input_offset_ns = int(row["input_offset_ns"])
                    analog = [float(row[column]) for column in ANALOG_COLUMNS]
                    buttons = [bool(row[column]) for column in BUTTON_COLUMNS]
                except (KeyError, TypeError, ValueError) as error:
                    errors.append(f"Invalid input value at row {row_count}: {error}")
                    continue

                expected_index = row_count - 1
                if frame_index != expected_index:
                    errors.append(
                        f"Expected frame_index {expected_index}, got {frame_index} "
                        f"at input row {row_count}"
                    )
                if previous_timestamp is not None and timestamp_ns <= previous_timestamp:
                    errors.append(
                        f"timestamp_ns does not increase at input row {row_count}"
                    )
                if (
                    previous_frame_timestamp is not None
                    and frame_timestamp_ns <= previous_frame_timestamp
                ):
                    errors.append(
                        f"frame_timestamp_ns does not increase at input row {row_count}"
                    )
                previous_timestamp = timestamp_ns
                previous_frame_timestamp = frame_timestamp_ns
                timestamps.append(timestamp_ns)
                frame_timestamps.append(frame_timestamp_ns)

                if source_timestamp_ns < 0 or input_timestamp_ns < 0:
                    errors.append(f"Negative timestamp at input row {row_count}")
                if input_offset_ns != input_timestamp_ns - frame_timestamp_ns:
                    errors.append(
                        f"input_offset_ns is inconsistent at input row {row_count}"
                    )
                input_offsets_ns.append(input_offset_ns)
                maximum_sync_offset_ns = round(maximum_sync_offset_ms * 1e6)
                if not -maximum_sync_offset_ns <= input_offset_ns <= 0:
                    errors.append(
                        f"Input synchronization offset "
                        f"{input_offset_ns / 1e6:+.2f} ms is outside the causal "
                        f"range [-{maximum_sync_offset_ms:.2f}, +0.00] ms at "
                        f"input row {row_count}"
                    )

                if any(not math.isfinite(value) for value in analog):
                    errors.append(f"Non-finite analog value at input row {row_count}")
                if any(not -1.0 <= value <= 1.0 for value in analog[:4]):
                    errors.append(f"Stick value outside [-1, 1] at input row {row_count}")
                if any(not 0.0 <= value <= 1.0 for value in analog[4:]):
                    errors.append(f"Trigger value outside [0, 1] at input row {row_count}")

                for column, value in zip(ANALOG_COLUMNS, analog):
                    if abs(value) > 1e-4:
                        active_input_frames[column] += 1
                for column, value in zip(BUTTON_COLUMNS, buttons):
                    if value:
                        active_input_frames[column] += 1
    except Exception as error:
        errors.append(f"Could not decode inputs.parquet: {error}")

    return (
        row_count,
        timestamps,
        frame_timestamps,
        active_input_frames,
        input_offsets_ns,
    )


def validate_csv_mirror(path: Path, expected_rows: int, errors: list[str]) -> None:
    try:
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                errors.append("Optional inputs.csv has no header")
                return
            try:
                validate_columns(reader.fieldnames)
            except ValueError as error:
                errors.append(f"Optional inputs.csv: {error}")
                return
            csv_rows = sum(1 for _ in reader)
    except OSError as error:
        errors.append(f"Could not read optional inputs.csv: {error}")
        return
    if csv_rows != expected_rows:
        errors.append(
            f"Optional inputs.csv has {csv_rows} rows; Parquet has {expected_rows}"
        )


def validate_game_state(
    path: Path,
    metadata: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> tuple[int, list[int], list[int]]:
    timestamps: list[int] = []
    frame_timestamps: list[int] = []
    if not path.is_file():
        errors.append("game_state.parquet is missing")
        return 0, timestamps, frame_timestamps
    try:
        parquet_file = pq.ParquetFile(path)
    except Exception as error:
        errors.append(f"Could not open game_state.parquet: {error}")
        return 0, timestamps, frame_timestamps
    parquet_schema = parquet_file.schema_arrow
    if not parquet_schema.equals(GAME_STATE_PARQUET_SCHEMA, check_metadata=False):
        errors.append(
            "Game-state Parquet schema does not match the canonical schema: "
            f"expected {GAME_STATE_PARQUET_SCHEMA}, received {parquet_schema}"
        )
        return 0, timestamps, frame_timestamps

    config = metadata.get("config", {})
    maximum_offset_ms = (
        float(config.get("maximum_game_state_sync_offset_ms", 25.0))
        if isinstance(config, dict)
        else 25.0
    )
    row_count = 0
    valid_rows = 0
    previous_frame_timestamp: int | None = None
    try:
        for batch in parquet_file.iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                row_count += 1
                frame_index = int(row["frame_index"])
                timestamp_ns = int(row["timestamp_ns"])
                frame_timestamp_ns = int(row["frame_timestamp_ns"])
                state_timestamp_ns = int(row["state_timestamp_ns"])
                state_offset_ns = int(row["state_offset_ns"])
                expected_index = row_count - 1
                if frame_index != expected_index:
                    errors.append(
                        f"Expected frame_index {expected_index}, got {frame_index} "
                        f"at game-state row {row_count}"
                    )
                if (
                    previous_frame_timestamp is not None
                    and frame_timestamp_ns <= previous_frame_timestamp
                ):
                    errors.append(
                        "frame_timestamp_ns does not increase at game-state row "
                        f"{row_count}"
                    )
                previous_frame_timestamp = frame_timestamp_ns
                timestamps.append(timestamp_ns)
                frame_timestamps.append(frame_timestamp_ns)
                if state_offset_ns != state_timestamp_ns - frame_timestamp_ns:
                    errors.append(
                        f"state_offset_ns is inconsistent at game-state row {row_count}"
                    )
                if abs(state_offset_ns) / 1e6 > maximum_offset_ms:
                    errors.append(
                        f"Game-state synchronization offset "
                        f"{state_offset_ns / 1e6:+.2f} ms exceeds "
                        f"{maximum_offset_ms:.2f} ms at row {row_count}"
                    )
                valid_rows += int(bool(row["state_valid"]))
                for field_name in GAME_STATE_VALUE_TYPES:
                    value = row[field_name]
                    if isinstance(value, float) and not math.isfinite(value):
                        errors.append(
                            f"Non-finite {field_name} at game-state row {row_count}"
                        )
    except Exception as error:
        errors.append(f"Could not decode game_state.parquet: {error}")
        return row_count, timestamps, frame_timestamps

    if row_count and valid_rows == 0:
        errors.append("All game-state rows are invalid")
    elif valid_rows < row_count:
        warnings.append(
            f"{row_count - valid_rows} of {row_count} game-state rows are invalid"
        )
    return row_count, timestamps, frame_timestamps


def validate_video(
    path: Path,
    errors: list[str],
) -> tuple[int, float, int, int]:
    if not path.is_file():
        errors.append("frames.mp4 is missing")
        return 0, 0.0, 0, 0

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        errors.append("frames.mp4 could not be opened")
        return 0, 0.0, 0, 0

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_frames = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame is None or frame.size == 0:
                errors.append(f"Decoded frame {decoded_frames} is empty")
                break
            decoded_frames += 1
    finally:
        capture.release()

    if reported_frames != decoded_frames:
        errors.append(
            f"Video reports {reported_frames} frames but only {decoded_frames} decode"
        )
    if fps <= 0 or not math.isfinite(fps):
        errors.append(f"Video reports invalid FPS: {fps}")
    if width <= 0 or height <= 0:
        errors.append(f"Video reports invalid resolution: {width}x{height}")
    return decoded_frames, fps, width, height


def validate_session(session_directory: str | Path) -> ValidationReport:
    session = Path(session_directory).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    if not session.is_dir():
        errors.append(f"Session directory does not exist: {session}")
        return ValidationReport(
            str(session), 0, 0, None, 0.0, 0, 0, {}, None, None, errors, warnings
        )

    metadata = read_metadata(session / "metadata.json", errors)
    (
        input_rows,
        timestamps,
        input_frame_timestamps,
        active_input_frames,
        input_offsets_ns,
    ) = validate_inputs(session / "inputs.parquet", metadata, errors)
    csv_path = session / "inputs.csv"
    files = metadata.get("files", {})
    csv_declared = isinstance(files, dict) and "inputs_csv" in files
    if csv_path.exists():
        validate_csv_mirror(csv_path, input_rows, errors)
    elif csv_declared:
        errors.append("metadata.json declares inputs.csv, but it is missing")

    game_state_path = session / "game_state.parquet"
    game_state_declared = isinstance(files, dict) and "game_state" in files
    game_state_rows: int | None = None
    game_state_timestamps: list[int] = []
    game_state_frame_timestamps: list[int] = []
    if game_state_declared:
        (
            game_state_rows,
            game_state_timestamps,
            game_state_frame_timestamps,
        ) = validate_game_state(
            game_state_path,
            metadata,
            errors,
            warnings,
        )
    elif game_state_path.exists():
        errors.append("game_state.parquet exists but is not declared in metadata.json")

    video_frames, fps, width, height = validate_video(session / "frames.mp4", errors)
    try:
        excluded_ranges = load_frame_ranges(session, total_frames=video_frames)
    except ValueError as error:
        errors.append(str(error))
        excluded_ranges = []
    excluded_frame_count = sum(
        frame_range.frame_count for frame_range in excluded_ranges
    )
    if excluded_frame_count:
        warnings.append(f"{excluded_frame_count} frame(s) excluded from training")
    if input_rows == 0:
        errors.append("Recording contains no input rows")
    if video_frames != input_rows:
        errors.append(
            f"Frame/label mismatch: {video_frames} video frames and "
            f"{input_rows} Parquet rows"
        )
    if game_state_rows is not None and video_frames != game_state_rows:
        errors.append(
            f"Frame/state mismatch: {video_frames} video frames and "
            f"{game_state_rows} game-state rows"
        )
    if game_state_rows is not None and game_state_timestamps != timestamps:
        errors.append("Game-state timestamp_ns values do not match input rows")
    if (
        game_state_rows is not None
        and game_state_frame_timestamps != input_frame_timestamps
    ):
        errors.append("Game-state frame_timestamp_ns values do not match input rows")

    if metadata:
        if metadata.get("status") != "complete":
            errors.append(f"Metadata status is {metadata.get('status')!r}, not 'complete'")
        config = metadata.get("config", {})
        if not isinstance(config, dict):
            errors.append("Metadata config must be an object")
        else:
            expected_theme = config.get("theme")
            expected_tag = config.get("tag")
            if not isinstance(expected_theme, str) or not isinstance(expected_tag, str):
                errors.append("Metadata must contain string theme and tag values")
            else:
                staging_name_prefix = f".{expected_tag}."
                has_expected_tag = session.name == expected_tag or (
                    session.name.startswith(staging_name_prefix)
                    and session.name.endswith(".inprogress")
                )
                theme_parts = tuple(
                    part
                    for part in expected_theme.replace("\\", "/").strip("/").split("/")
                    if part
                )
                dataset_split = config.get("split")
                split_directory = dataset_split
                expected_parent = (split_directory, *theme_parts)
                legacy_parent = theme_parts
                output_directory = config.get("output_directory")
                try:
                    actual_parent = session.parent.resolve().relative_to(
                        Path(str(output_directory)).resolve()
                    ).parts
                except (TypeError, ValueError):
                    actual_parent = tuple(session.parent.parts[-len(theme_parts):])
                if not has_expected_tag or actual_parent not in (
                    expected_parent,
                    legacy_parent,
                ):
                    errors.append(
                        "Session path must end in "
                        f"{split_directory}/{expected_theme}/{expected_tag}, got "
                        f"{session.parent}/{session.name}"
                    )
                elif actual_parent == legacy_parent:
                    warnings.append(
                        "Legacy recording layout detected; new recordings are saved "
                        f"under {split_directory}/{expected_theme}/{expected_tag}"
                    )
            dataset_split = config.get("split")
            if dataset_split not in ("train", "validation"):
                errors.append(
                    "Metadata config.split must be 'train' or 'validation'"
                )
            labels = config.get("labels")
            if not isinstance(labels, list) or not all(
                isinstance(label, str) for label in labels
            ):
                errors.append("Metadata config.labels must be an array of strings")
        recording = metadata.get("recording", {})
        declared_frames = recording.get("frame_count") if isinstance(recording, dict) else None
        if declared_frames is not None and declared_frames != input_rows:
            errors.append(
                f"Metadata declares {declared_frames} frames, but Parquet has "
                f"{input_rows} rows"
            )
        video = metadata.get("video", {})
        if isinstance(video, dict):
            expected_fps = video.get("fps")
            if expected_fps is not None and abs(float(expected_fps) - fps) > 0.1:
                errors.append(f"Expected {expected_fps} FPS, video reports {fps:.3f}")
            expected_size = (video.get("width"), video.get("height"))
            if None not in expected_size and expected_size != (width, height):
                errors.append(
                    f"Expected {expected_size[0]}x{expected_size[1]}, "
                    f"video reports {width}x{height}"
                )

    if len(timestamps) > 1 and fps > 0:
        gaps = [second - first for first, second in zip(timestamps, timestamps[1:])]
        largest_gap_ms = max(gaps) / 1e6
        warning_threshold_ns = 1.75 * 1e9 / fps
        large_gap_count = sum(gap > warning_threshold_ns for gap in gaps)
        if large_gap_count:
            warnings.append(
                f"{large_gap_count} timestamp gap(s) exceed 1.75 frame periods; "
                f"largest is {largest_gap_ms:.2f} ms"
            )

    inactive_inputs = [
        column for column, count in active_input_frames.items() if count == 0
    ]
    if input_rows and inactive_inputs:
        warnings.append(
            "Inputs never active in this session: " + ", ".join(inactive_inputs)
        )

    mean_abs_input_offset_ms = (
        sum(abs(offset) for offset in input_offsets_ns) / len(input_offsets_ns) / 1e6
        if input_offsets_ns
        else None
    )
    maximum_abs_input_offset_ms = (
        max(abs(offset) for offset in input_offsets_ns) / 1e6
        if input_offsets_ns
        else None
    )
    return ValidationReport(
        session_directory=str(session),
        video_frames=video_frames,
        input_rows=input_rows,
        game_state_rows=game_state_rows,
        fps=fps,
        width=width,
        height=height,
        active_input_frames=active_input_frames,
        mean_abs_input_offset_ms=mean_abs_input_offset_ms,
        maximum_abs_input_offset_ms=maximum_abs_input_offset_ms,
        errors=errors,
        warnings=warnings,
    )


def print_report(report: ValidationReport) -> None:
    print(f"Video frames: {report.video_frames}")
    print(f"Input rows:   {report.input_rows}")
    if report.game_state_rows is not None:
        print(f"State rows:   {report.game_state_rows}")
    print(f"FPS:          {report.fps:.2f}")
    print(f"Resolution:   {report.width}x{report.height}")
    if report.maximum_abs_input_offset_ms is not None:
        print(
            f"Input sync:   mean {report.mean_abs_input_offset_ms:.2f} ms, "
            f"max {report.maximum_abs_input_offset_ms:.2f} ms"
        )
    for warning in report.warnings:
        print(f"Warning:      {warning}")
    for error in report.errors:
        print(f"Error:        {error}")
    print(f"Validation:   {'valid' if report.valid else 'INVALID'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate a recording session.")
    parser.add_argument("session", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    report = validate_session(args.session)
    if args.as_json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print_report(report)
    if not report.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
