from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mimic_tear.recording.schema import (  # noqa: E402
    INPUT_COLUMNS,
    PARQUET_SCHEMA,
    validate_columns,
)
from mimic_tear.recording.annotations import (  # noqa: E402
    FrameRange,
    load_frame_ranges,
    merge_frame_ranges,
    save_frame_ranges,
)
from mimic_tear.dataset.dataset import (  # noqa: E402
    EldenRingDataset,
    RecordingSample,
    discover_sessions,
    ensure_frame_cache,
    load_session_samples,
    partition_sessions_by_split,
)
from mimic_tear.recording.record import (  # noqa: E402
    ControllerSampler,
    create_session_directories,
    FrameClockSynchronizer,
    InputRow,
    InputSample,
    InputStatisticsAccumulator,
    InputWriters,
    nearest_input_sample,
    open_replay,
    parse_args,
    parse_recording_label,
    parse_region,
)
from mimic_tear.recording.validation import validate_csv_mirror, validate_session  # noqa: E402
from mimic_tear.recording.replay import (  # noqa: E402
    ReplayState,
    exclude_selection,
    load_replay_session,
    render_replay_frame,
)


def write_video(path: Path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (64, 36),
    )
    if not writer.isOpened():
        raise RuntimeError("Test video writer could not be opened")
    try:
        for index in range(frame_count):
            frame = np.full((36, 64, 3), index * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def current_row(index: int) -> dict[str, int | float]:
    row: dict[str, int | float | bool] = {
        column: False if column in PARQUET_SCHEMA.names[-14:] else 0
        for column in INPUT_COLUMNS
    }
    row["frame_index"] = index
    row["timestamp_ns"] = 1_000_000_000 + index * 33_333_333
    row["source_timestamp_ns"] = 2_000_000_000 + index * 33_333_333
    row["frame_timestamp_ns"] = 1_000_000_000 + index * 33_333_333
    row["input_timestamp_ns"] = 999_000_000 + index * 33_333_333
    row["input_offset_ns"] = -1_000_000
    return row


def write_inputs(path: Path, rows: list[dict[str, int | float | bool]]) -> None:
    table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
    pq.write_table(table, path, compression="zstd")


class RecordingSchemaTests(unittest.TestCase):
    def test_only_exact_input_columns_are_accepted(self) -> None:
        validate_columns(INPUT_COLUMNS)
        self.assertEqual(tuple(field.name for field in fields(InputRow)), INPUT_COLUMNS)
        with self.assertRaises(ValueError):
            validate_columns(INPUT_COLUMNS[:-1])

    def test_parse_region(self) -> None:
        self.assertEqual(parse_region("10,20,650,380"), (10, 20, 650, 380))

    def test_recording_labels_cannot_escape_the_output_directory(self) -> None:
        self.assertEqual(parse_recording_label("basic-movement-01"), "basic-movement-01")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_recording_label("../outside")

    def test_nested_theme_creates_nested_session_directory(self) -> None:
        config = parse_args(
            [
                "--theme",
                "movement/jump",
                "--tag",
                "basic-01",
                "--split",
                "train",
            ]
        )
        self.assertEqual(config.theme, "movement/jump")

        with tempfile.TemporaryDirectory() as temporary_directory:
            _, final = create_session_directories(
                Path(temporary_directory),
                split="train",
                theme=config.theme,
                tag=config.tag,
            )
            self.assertEqual(
                final,
                Path(temporary_directory) / "train" / "movement" / "jump" / "basic-01",
            )

    def test_nested_theme_rejects_empty_or_unsafe_segments(self) -> None:
        required = ["--tag", "basic-01", "--split", "train"]
        for theme in ("movement/", "/jump", "movement//jump", "movement/../jump"):
            with self.assertRaises(SystemExit):
                parse_args(["--theme", theme, *required])

    def test_cancel_hotkey_defaults_to_global_discard_key(self) -> None:
        config = parse_args(
            [
                "--theme",
                "movement",
                "--tag",
                "basic-01",
                "--split",
                "train",
            ]
        )
        self.assertEqual(config.cancel_hotkey, "CTRL+F9")

        configured = parse_args(
            [
                "--theme",
                "movement",
                "--tag",
                "basic-02",
                "--split",
                "train",
                "--cancel-hotkey",
                "CTRL+F10",
            ]
        )
        self.assertEqual(configured.cancel_hotkey, "CTRL+F10")

    def test_open_replay_loads_and_runs_the_completed_session(self) -> None:
        session_directory = Path("recordings/train/movement/jump/basic-01")
        replay_session = object()
        with (
            patch(
                "mimic_tear.recording.replay.load_replay_session",
                return_value=replay_session,
            ) as load_session,
            patch("mimic_tear.recording.replay.run_replay") as run_session,
        ):
            open_replay(session_directory)

        load_session.assert_called_once_with(session_directory)
        run_session.assert_called_once_with(replay_session)

    def test_csv_output_is_opt_in(self) -> None:
        required = [
            "--theme",
            "exploration",
            "--tag",
            "basic-movement",
            "--split",
            "train",
        ]
        self.assertFalse(parse_args(required).write_csv)
        self.assertTrue(parse_args([*required, "--csv"]).write_csv)

    def test_reusable_labels_are_deduplicated(self) -> None:
        config = parse_args(
            [
                "--theme",
                "combat",
                "--tag",
                "dodging-01",
                "--split",
                "validation",
                "--label",
                "dodging",
                "--label",
                "lock-on",
                "--label",
                "dodging",
            ]
        )
        self.assertEqual(config.split, "validation")
        self.assertEqual(config.labels, ("dodging", "lock-on"))

    def test_input_statistics_include_presses_ratios_and_analog_summary(self) -> None:
        base = {
            column: 0.0 for column in PARQUET_SCHEMA.names[6:12]
        }
        base.update({column: False for column in PARQUET_SCHEMA.names[-14:]})
        accumulator = InputStatisticsAccumulator()
        accumulator.add(SimpleNamespace(**base))
        active = dict(base)
        active["left_x"] = 0.5
        active["south"] = True
        accumulator.add(SimpleNamespace(**active))
        accumulator.add(SimpleNamespace(**active))
        accumulator.add(SimpleNamespace(**base))

        summary = accumulator.summary(duration_seconds=30.0)
        self.assertEqual(summary["active_frames"]["south"], 2)
        self.assertEqual(summary["active_ratios"]["south"], 0.5)
        self.assertEqual(summary["button_press_counts"]["south"], 1)
        self.assertEqual(summary["button_presses_per_minute"]["south"], 2.0)
        self.assertEqual(summary["analog"]["left_x"]["minimum"], 0.0)
        self.assertEqual(summary["analog"]["left_x"]["maximum"], 0.5)
        self.assertEqual(summary["analog"]["left_x"]["mean"], 0.25)

    def test_frame_cache_preserves_frame_indices_and_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = Path(temporary_directory)
            video = session / "frames.mp4"
            write_video(video, frame_count=3)

            cache_path = ensure_frame_cache(video, width=32, height=18)
            cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
            self.assertEqual(cache.shape, (3, 18, 32, 3))
            self.assertEqual(cache.dtype, np.uint8)
            self.assertLess(float(cache[0].mean()), float(cache[2].mean()))
            del cache

            cache_mtime_ns = cache_path.stat().st_mtime_ns
            self.assertEqual(
                ensure_frame_cache(video, width=32, height=18),
                cache_path,
            )
            self.assertEqual(cache_path.stat().st_mtime_ns, cache_mtime_ns)

            manifest_path = cache_path.with_suffix(".json")
            manifest_path.write_text('{"invalid": true}\n', encoding="utf-8")
            ensure_frame_cache(video, width=32, height=18)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["frame_count"], 3)
            self.assertEqual(manifest["width"], 32)
            self.assertEqual(manifest["height"], 18)
            cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)

            sample = RecordingSample(
                session_directory=session,
                video_path=video.resolve(),
                frame_index=2,
                timestamp_ns=123,
                analog=(0.0,) * 6,
                buttons=(0.0,) * 14,
            )
            dataset = EldenRingDataset(
                [sample],
                transform=lambda frame: torch.from_numpy(frame.copy())
                .permute(2, 0, 1)
                .to(dtype=torch.float32),
                frame_cache_paths={video.resolve(): cache_path},
            )
            with patch.object(
                dataset,
                "_capture_for",
                side_effect=AssertionError("MP4 decoder should not be used"),
            ):
                loaded = dataset[0]["image"]

            expected = (
                torch.from_numpy(np.asarray(cache[2]).copy())
                .permute(2, 0, 1)
                .to(dtype=torch.float32)
            )
            self.assertTrue(torch.equal(loaded, expected))
            self.assertEqual(dataset[0]["frame_index"], 2)
            dataset.close()
            del cache

    def test_sessions_are_partitioned_without_frame_level_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            training = root / "train" / "exploration" / "movement-01"
            validation = root / "validation" / "exploration" / "movement-02"
            for session, split in (
                (training, "train"),
                (validation, "validation"),
            ):
                session.mkdir(parents=True)
                (session / "metadata.json").write_text(
                    json.dumps({"config": {"split": split}}),
                    encoding="utf-8",
                )

            self.assertEqual(
                partition_sessions_by_split([training, validation]),
                ([training.resolve()], [validation.resolve()]),
            )

    def test_classified_destination_cannot_be_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            staging, final = create_session_directories(
                root,
                split="train",
                theme="exploration",
                tag="basic-movement",
            )
            self.assertEqual(
                final,
                root / "train" / "exploration" / "basic-movement",
            )
            _, validation_final = create_session_directories(
                root,
                split="validation",
                theme="combat",
                tag="dodging-01",
            )
            self.assertEqual(
                validation_final,
                root / "validation" / "combat" / "dodging-01",
            )
            staging.rename(final)
            with self.assertRaises(FileExistsError):
                create_session_directories(
                    root,
                    split="train",
                    theme="exploration",
                    tag="basic-movement",
                )

    def test_nearest_timestamped_input_is_selected(self) -> None:
        samples = [
            InputSample(96_000_000, "before"),
            InputSample(100_000_000, "nearest"),
            InputSample(104_000_000, "after"),
        ]
        self.assertEqual(nearest_input_sample(samples, 101_000_000).state, "nearest")

    def test_duplicate_frames_use_current_output_time(self) -> None:
        clock = FrameClockSynchronizer()
        _, first_timestamp, first_duplicate = clock.align(10.0, 20_000_000_000)
        _, second_timestamp, second_duplicate = clock.align(10.0, 20_033_000_000)
        self.assertFalse(first_duplicate)
        self.assertTrue(second_duplicate)
        self.assertEqual(first_timestamp, 20_000_000_000)
        self.assertEqual(second_timestamp, 20_033_000_000)

    def test_controller_sampler_polls_independently(self) -> None:
        class FakeController:
            name = "fake"
            connected = True

            def __init__(self) -> None:
                self.poll_count = 0

            def poll(self) -> int:
                self.poll_count += 1
                return self.poll_count

        sampler = ControllerSampler(FakeController, polling_hz=250)
        sampler.start()
        try:
            time.sleep(0.025)
            sample = sampler.closest(
                time.perf_counter_ns(),
                timeout_seconds=0.02,
            )
            stats = sampler.stats()
            self.assertEqual(sampler.controller_name, "fake")
            self.assertGreaterEqual(stats.sample_count, 2)
            self.assertIsInstance(sample.state, int)
        finally:
            sampler.stop()

    def test_excluded_ranges_are_merged_and_persisted(self) -> None:
        ranges = merge_frame_ranges(
            [FrameRange(5, 8), FrameRange(1, 3), FrameRange(4, 4)]
        )
        self.assertEqual(ranges, [FrameRange(1, 8)])
        with tempfile.TemporaryDirectory() as temporary_directory:
            save_frame_ranges(temporary_directory, ranges, total_frames=10)
            self.assertEqual(
                load_frame_ranges(temporary_directory, total_frames=10),
                ranges,
            )

    def test_replay_overlay_renders_controller_state(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        row = current_row(0)
        row["left_x"] = 0.75
        row["south"] = True
        canvas = render_replay_frame(
            frame,
            row,
            ReplayState(frame_index=0),
            total_frames=1,
            fps=30.0,
        )
        self.assertEqual(canvas.shape, (360, 970, 3))
        self.assertGreater(int(canvas.sum()), 0)

    def test_exclusion_keeps_an_undo_snapshot(self) -> None:
        state = ReplayState(
            frame_index=8,
            selection_anchor=8,
            selection_end=12,
            excluded_ranges=[FrameRange(1, 10)],
        )
        exclude_selection(state, total_frames=20)
        self.assertEqual(state.excluded_ranges, [FrameRange(1, 12)])
        self.assertEqual(state.undo_stack, [[FrameRange(1, 10)]])


class RecordingValidationTests(unittest.TestCase):
    def test_completed_recording_validates_before_staging_rename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = (
                Path(temporary_directory)
                / "exploration"
                / ".basic-movement.20260805_120000_000.inprogress"
            )
            session.mkdir(parents=True)
            write_video(session / "frames.mp4", frame_count=1)
            write_inputs(session / "inputs.parquet", [current_row(0)])
            (session / "metadata.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "config": {
                            "theme": "exploration",
                            "tag": "basic-movement",
                            "split": "train",
                            "labels": [],
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = validate_session(session)
            self.assertTrue(report.valid, report.errors)

    def test_valid_current_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            recordings = Path(temporary_directory)
            session = recordings / "exploration" / "basic-movement"
            session.mkdir(parents=True)
            write_video(session / "frames.mp4")
            with InputWriters(session / "inputs.parquet") as writers:
                for index in range(3):
                    writers.write(InputRow(**current_row(index)))
            (session / "metadata.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "config": {
                            "theme": "exploration",
                            "tag": "basic-movement",
                            "split": "train",
                            "labels": ["movement"],
                            "maximum_sync_offset_ms": 10.0,
                        },
                        "recording": {"frame_count": 3},
                        "video": {"fps": 30, "width": 64, "height": 36},
                        "files": {
                            "video": "frames.mp4",
                            "inputs": "inputs.parquet",
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = validate_session(session)
            self.assertTrue(report.valid, report.errors)
            self.assertEqual(report.video_frames, 3)
            self.assertEqual(report.input_rows, 3)
            self.assertFalse((session / "inputs.csv").exists())
            self.assertEqual(len(load_session_samples(session)), 3)
            self.assertEqual(discover_sessions(recordings), [session])
            replay_session = load_replay_session(session)
            self.assertEqual(replay_session.frame_count, 3)

            save_frame_ranges(session, [FrameRange(1, 1)], total_frames=3)
            remaining = load_session_samples(session)
            self.assertEqual([sample.frame_index for sample in remaining], [0, 2])
            annotated_report = validate_session(session)
            self.assertTrue(annotated_report.valid, annotated_report.errors)
            self.assertTrue(
                any("excluded from training" in warning for warning in annotated_report.warnings)
            )

    def test_optional_csv_mirror_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = Path(temporary_directory) / "combat" / "mismatch-test"
            session.mkdir(parents=True)
            with InputWriters(
                session / "inputs.parquet",
                csv_path=session / "inputs.csv",
            ) as writers:
                writers.write(InputRow(**current_row(0)))
            self.assertTrue((session / "inputs.parquet").is_file())
            self.assertTrue((session / "inputs.csv").is_file())
            self.assertEqual(pq.read_table(session / "inputs.parquet").num_rows, 1)
            errors: list[str] = []
            validate_csv_mirror(session / "inputs.csv", 1, errors)
            self.assertEqual(errors, [])

    def test_frame_label_mismatch_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = Path(temporary_directory)
            write_video(session / "frames.mp4", frame_count=2)
            write_inputs(
                session / "inputs.parquet",
                [current_row(index) for index in range(3)],
            )
            (session / "metadata.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "config": {
                            "theme": "combat",
                            "tag": "mismatch-test",
                            "split": "train",
                            "labels": [],
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = validate_session(session)
            self.assertFalse(report.valid)
            self.assertTrue(
                any("Frame/label mismatch" in error for error in report.errors)
            )

    def test_future_input_offset_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = Path(temporary_directory) / "combat" / "sync-test"
            session.mkdir(parents=True)
            write_video(session / "frames.mp4", frame_count=1)
            row = current_row(0)
            row["input_timestamp_ns"] = int(row["frame_timestamp_ns"]) + 5_000_000
            row["input_offset_ns"] = 5_000_000
            write_inputs(session / "inputs.parquet", [row])
            (session / "metadata.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "config": {
                            "theme": "combat",
                            "tag": "sync-test",
                            "split": "validation",
                            "labels": [],
                            "maximum_sync_offset_ms": 15.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = validate_session(session)
            self.assertFalse(report.valid)
            self.assertTrue(
                any("synchronization offset" in error for error in report.errors)
            )


if __name__ == "__main__":
    unittest.main()
