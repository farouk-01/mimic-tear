from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.utils.data import Dataset

from mimic_tear.recording.annotations import is_frame_excluded, load_frame_ranges
from mimic_tear.game_state.features import (
    GAME_STATE_FEATURE_COUNT,
    encode_game_state_values,
)
from mimic_tear.game_state.schema import GAME_STATE_PARQUET_SCHEMA
from mimic_tear.recording.schema import (
    ANALOG_COLUMNS,
    BUTTON_COLUMNS,
    PARQUET_SCHEMA,
    validate_columns,
)


@dataclass(frozen=True, slots=True)
class RecordingSample:
    """Metadata needed to load one synchronized frame/action pair."""

    session_directory: Path
    video_path: Path
    frame_index: int
    timestamp_ns: int
    analog: tuple[float, ...]
    buttons: tuple[float, ...]
    game_state: tuple[float, ...] = ()


class PolicySample(TypedDict):
    image: Tensor
    analog: Tensor
    buttons: Tensor
    game_state: Tensor
    frame_index: int
    timestamp_ns: int
    session: str


FRAME_CACHE_DIRECTORY = ".frame_cache"
FRAME_CACHE_LAYOUT = "NHWC"
FRAME_CACHE_COLOR_SPACE = "BGR"
FRAME_CACHE_DTYPE = "uint8"


def frame_cache_path(video_path: str | Path, *, width: int, height: int) -> Path:
    """Return the cache path for one video and training resolution."""

    video = Path(video_path).expanduser().resolve()
    return (
        video.parent
        / FRAME_CACHE_DIRECTORY
        / f"frames-{width}x{height}-bgr-uint8.npy"
    )


def ensure_frame_cache(
    video_path: str | Path,
    *,
    width: int,
    height: int,
    rebuild: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """
    Decode a video sequentially into an atomic, memory-mapped uint8 cache.

    Cache row ``i`` is always source video frame ``i``. The accompanying
    manifest binds the cache to the source video's size, modification time,
    frame count, and requested output resolution.
    """

    if width <= 0 or height <= 0:
        raise ValueError("cache width and height must be greater than zero")

    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)

    frame_count = _read_video_frame_count(video)
    cache = frame_cache_path(video, width=width, height=height)
    manifest_path = cache.with_suffix(".json")
    source_stat = video.stat()
    manifest = {
        "source": video.name,
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "layout": FRAME_CACHE_LAYOUT,
        "color_space": FRAME_CACHE_COLOR_SPACE,
        "dtype": FRAME_CACHE_DTYPE,
    }

    if not rebuild and _frame_cache_is_current(cache, manifest_path, manifest):
        return cache

    if progress is not None:
        progress(
            f"Building frame cache: {video} -> {cache} "
            f"({frame_count} frames at {width}x{height})"
        )

    required_bytes = frame_count * height * width * 3
    free_bytes = shutil.disk_usage(video.parent).free
    if free_bytes < required_bytes + 16 * 1024**2:
        raise RuntimeError(
            f"Not enough free space to build frame cache {cache}: "
            f"requires about {required_bytes / 1024**3:.2f} GiB, "
            f"{free_bytes / 1024**3:.2f} GiB available"
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = cache.with_name(
        f".{cache.name}.{os.getpid()}.inprogress"
    )
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.inprogress"
    )
    temporary_cache.unlink(missing_ok=True)
    temporary_manifest.unlink(missing_ok=True)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video: {video}")

    cache_array: np.memmap | None = None
    try:
        cache_array = np.lib.format.open_memmap(
            temporary_cache,
            mode="w+",
            dtype=np.uint8,
            shape=(frame_count, height, width, 3),
        )
        for frame_index in range(frame_count):
            success, frame = capture.read()
            if not success or frame is None:
                raise RuntimeError(
                    f"Could not sequentially decode frame {frame_index} "
                    f"from {video}"
                )
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(
                    frame,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )
            cache_array[frame_index] = frame

        success, _ = capture.read()
        if success:
            raise RuntimeError(
                f"Video contains more frames than its reported count "
                f"({frame_count}): {video}"
            )

        cache_array.flush()
        del cache_array
        cache_array = None

        final_stat = video.stat()
        if (
            final_stat.st_size != source_stat.st_size
            or final_stat.st_mtime_ns != source_stat.st_mtime_ns
        ):
            raise RuntimeError(f"Video changed while its cache was built: {video}")

        os.replace(temporary_cache, cache)
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
    except Exception:
        if cache_array is not None:
            del cache_array
        temporary_cache.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        raise
    finally:
        capture.release()

    return cache


def prepare_frame_caches(
    samples: Sequence[RecordingSample],
    *,
    width: int,
    height: int,
    rebuild: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[Path, Path]:
    """Build or reuse one cache per unique video in a sample collection."""

    videos = sorted({sample.video_path.resolve() for sample in samples})
    return {
        video: ensure_frame_cache(
            video,
            width=width,
            height=height,
            rebuild=rebuild,
            progress=progress,
        )
        for video in videos
    }


def _frame_cache_is_current(
    cache_path: Path,
    manifest_path: Path,
    expected_manifest: dict[str, object],
) -> bool:
    if not cache_path.is_file() or not manifest_path.is_file():
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest != expected_manifest:
            return False
        array = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        valid = (
            array.dtype == np.uint8
            and array.shape
            == (
                expected_manifest["frame_count"],
                expected_manifest["height"],
                expected_manifest["width"],
                3,
            )
        )
        del array
        return bool(valid)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def discover_sessions(recordings_directory: str | Path) -> list[Path]:
    """
    Discover sessions recursively under recordings_directory.

    New sessions are stored as <split>/<theme>/<tag>, where theme can contain
    multiple path segments. Legacy <theme>/<tag> sessions remain supported.
    """

    root = Path(recordings_directory).expanduser().resolve()

    if not root.is_dir():
        raise NotADirectoryError(f"Recordings directory does not exist: {root}")

    sessions = sorted(
        frame_path.parent
        for frame_path in root.rglob("frames.mp4")
        if (frame_path.parent / "inputs.parquet").is_file()
        and all(
            not part.startswith(".")
            for part in frame_path.parent.relative_to(root).parts
        )
    )

    if not sessions:
        raise FileNotFoundError(
            f"No recording sessions containing frames.mp4 and inputs.parquet "
            f"were found under {root}/<split>/<theme>/<tag>"
        )

    return sessions


def partition_sessions_by_split(
    session_directories: Sequence[str | Path],
) -> tuple[list[Path], list[Path]]:
    """Partition complete sessions using their explicit metadata split."""

    training: list[Path] = []
    validation: list[Path] = []
    for session_directory in session_directories:
        session = Path(session_directory).expanduser().resolve()
        metadata_path = session / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Could not read session split from {metadata_path}: {error}"
            ) from error

        config = metadata.get("config")
        split = config.get("split") if isinstance(config, dict) else None
        if split == "train":
            training.append(session)
        elif split == "validation":
            validation.append(session)
        else:
            raise ValueError(
                f"Session metadata must declare config.split as 'train' or "
                f"'validation': {metadata_path}"
            )

    if not training:
        raise ValueError("No sessions are assigned to the training split")
    if not validation:
        raise ValueError("No sessions are assigned to the validation split")
    return training, validation


def load_session_samples(
    session_directory: str | Path,
    *,
    require_game_state: bool = False,
) -> list[RecordingSample]:
    """Load metadata and controller labels for one recording session."""

    session = Path(session_directory).expanduser().resolve()
    video_path = session / "frames.mp4"
    input_path = session / "inputs.parquet"
    game_state_path = session / "game_state.parquet"

    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    video_frame_count = _read_video_frame_count(video_path)
    game_state_rows = _load_game_state_rows(
        game_state_path,
        frame_count=video_frame_count,
        required=require_game_state,
    )
    excluded_ranges = load_frame_ranges(session, total_frames=video_frame_count)
    samples: list[RecordingSample] = []

    try:
        table = pq.read_table(input_path)
    except Exception as error:
        raise ValueError(f"Could not read recording inputs: {input_path}") from error

    try:
        validate_columns(table.column_names)
    except ValueError as error:
        raise ValueError(
            f"Invalid recording format in {input_path}: {error}"
        ) from error

    if not table.schema.equals(PARQUET_SCHEMA, check_metadata=False):
        raise ValueError(
            f"Invalid Parquet types in {input_path}: expected {PARQUET_SCHEMA}, "
            f"received {table.schema}"
        )

    if table.num_rows != video_frame_count:
        raise ValueError(
            f"Frame/label mismatch in {session}: {video_frame_count} video "
            f"frames and {table.num_rows} input rows"
        )

    previous_timestamp: int | None = None
    for row_number, row in enumerate(table.to_pylist(), start=1):
        try:
            frame_index = int(row["frame_index"])
            timestamp_ns = int(row["timestamp_ns"])

            analog = tuple(float(row[column]) for column in ANALOG_COLUMNS)
            buttons = tuple(float(row[column]) for column in BUTTON_COLUMNS)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid value in {input_path} at row {row_number}"
            ) from error

        if frame_index < 0:
            raise ValueError(
                f"Negative frame_index at {input_path}, row {row_number}"
            )

        expected_frame_index = row_number - 1
        if frame_index != expected_frame_index:
            raise ValueError(
                f"Expected frame_index {expected_frame_index}, got "
                f"{frame_index} at {input_path}, row {row_number}"
            )

        if frame_index >= video_frame_count:
            raise ValueError(
                f"Frame {frame_index} from {input_path} does not exist in "
                f"{video_path}, which contains {video_frame_count} frames"
            )

        if any(value not in (0.0, 1.0) for value in buttons):
            raise ValueError(
                f"Button values must be 0 or 1 at {input_path}, row {row_number}"
            )

        stick_values = analog[:4]
        trigger_values = analog[4:]
        if any(not -1.0 <= value <= 1.0 for value in stick_values):
            raise ValueError(
                f"Stick values must be in [-1, 1] at {input_path}, row {row_number}"
            )
        if any(not 0.0 <= value <= 1.0 for value in trigger_values):
            raise ValueError(
                f"Trigger values must be in [0, 1] at {input_path}, row {row_number}"
            )
        if previous_timestamp is not None and timestamp_ns <= previous_timestamp:
            raise ValueError(
                f"Timestamps must increase at {input_path}, row {row_number}"
            )
        previous_timestamp = timestamp_ns

        state_features: tuple[float, ...] = ()
        if game_state_rows is not None:
            state_row = game_state_rows[frame_index]
            if int(state_row["timestamp_ns"]) != timestamp_ns:
                raise ValueError(
                    f"Input/game-state timestamp mismatch at frame {frame_index} "
                    f"in {session}"
                )
            state_features = encode_game_state_values(
                state_row,
                valid=bool(state_row["state_valid"]),
            )

        if is_frame_excluded(frame_index, excluded_ranges):
            continue

        samples.append(
            RecordingSample(
                session_directory=session,
                video_path=video_path,
                frame_index=frame_index,
                timestamp_ns=timestamp_ns,
                analog=analog,
                buttons=buttons,
                game_state=state_features,
            )
        )

    if not samples:
        raise ValueError(f"Session contains no usable samples: {session}")

    return samples


def _load_game_state_rows(
    path: Path,
    *,
    frame_count: int,
    required: bool,
) -> list[dict[str, object]] | None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(
                f"State-aware training requires frame-aligned {path.name}: {path}"
            )
        return None
    try:
        table = pq.read_table(path)
    except Exception as error:
        raise ValueError(f"Could not read game state: {path}") from error
    if not table.schema.equals(GAME_STATE_PARQUET_SCHEMA, check_metadata=False):
        raise ValueError(
            f"Invalid game-state schema in {path}: expected "
            f"{GAME_STATE_PARQUET_SCHEMA}, received {table.schema}"
        )
    if table.num_rows != frame_count:
        raise ValueError(
            f"Frame/game-state mismatch: {frame_count} frames and "
            f"{table.num_rows} rows in {path}"
        )
    rows = table.to_pylist()
    for frame_index, row in enumerate(rows):
        if int(row["frame_index"]) != frame_index:
            raise ValueError(
                f"Expected game-state frame_index {frame_index}, got "
                f"{row['frame_index']} in {path}"
            )
        encoded = encode_game_state_values(
            row,
            valid=bool(row["state_valid"]),
        )
        if len(encoded) != GAME_STATE_FEATURE_COUNT:
            raise RuntimeError("Game-state encoder returned an unexpected width")
    return rows


def load_recording_samples(
    sessions: Sequence[str | Path],
    *,
    require_game_state: bool = False,
) -> list[RecordingSample]:
    """Load and concatenate metadata from multiple sessions."""

    if not sessions:
        raise ValueError("At least one recording session is required")

    samples: list[RecordingSample] = []

    for session in sessions:
        samples.extend(
            load_session_samples(
                session,
                require_game_state=require_game_state,
            )
        )

    return samples


class EldenRingDataset(Dataset[PolicySample]):
    """
    Dataset of synchronized game frames and controller actions.

    Video handles are opened lazily and kept per Dataset process. This avoids
    trying to pickle active OpenCV VideoCapture objects into DataLoader workers.
    """

    def __init__(
        self,
        samples: Sequence[RecordingSample],
        *,
        transform: Callable[[np.ndarray], Tensor],
        frame_cache_paths: Mapping[Path, Path] | None = None,
    ) -> None:
        if not samples:
            raise ValueError("Dataset requires at least one sample")

        self._samples = tuple(samples)
        self._transform = transform
        self._frame_cache_paths = {
            Path(video).resolve(): Path(cache).resolve()
            for video, cache in (frame_cache_paths or {}).items()
        }
        self._frame_caches: dict[Path, np.ndarray] = {}
        self._captures: dict[Path, cv2.VideoCapture] = {}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> PolicySample:
        sample = self._samples[index]
        frame = self._read_frame(sample.video_path, sample.frame_index)

        image = self._transform(frame)

        if image.dtype != torch.float32:
            raise TypeError(
                f"Transform must produce float32 tensors, got {image.dtype}"
            )

        return PolicySample(
            image=image,
            analog=torch.tensor(sample.analog, dtype=torch.float32),
            buttons=torch.tensor(sample.buttons, dtype=torch.float32),
            game_state=torch.tensor(sample.game_state, dtype=torch.float32),
            frame_index=sample.frame_index,
            timestamp_ns=sample.timestamp_ns,
            session=(
                f"{sample.session_directory.parent.name}/"
                f"{sample.session_directory.name}"
            ),
        )

    @property
    def analog_columns(self) -> tuple[str, ...]:
        return ANALOG_COLUMNS

    @property
    def button_columns(self) -> tuple[str, ...]:
        return BUTTON_COLUMNS

    @property
    def game_state_features(self) -> int:
        lengths = {len(sample.game_state) for sample in self._samples}
        if len(lengths) != 1:
            raise RuntimeError("Dataset contains inconsistent game-state vectors")
        return lengths.pop()

    @property
    def samples(self) -> tuple[RecordingSample, ...]:
        return self._samples

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()

        self._captures.clear()
        self._frame_caches.clear()

    def _read_frame(
        self,
        video_path: Path,
        frame_index: int,
    ) -> np.ndarray:
        cache_path = self._frame_cache_paths.get(video_path.resolve())
        if cache_path is not None:
            cache = self._frame_cache_for(cache_path)
            if not 0 <= frame_index < len(cache):
                raise IndexError(
                    f"Frame {frame_index} is outside cache {cache_path}, "
                    f"which contains {len(cache)} frames"
                )
            return np.asarray(cache[frame_index])

        capture = self._capture_for(video_path)

        # Explicit seeking makes __getitem__ correct even when the DataLoader
        # shuffles indices.
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        success, frame = capture.read()

        if not success or frame is None:
            # Retry once with a fresh decoder. Some codecs occasionally fail
            # after many non-sequential seeks.
            capture.release()
            self._captures.pop(video_path, None)

            capture = self._capture_for(video_path)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()

        if not success or frame is None:
            raise RuntimeError(
                f"Could not decode frame {frame_index} from {video_path}"
            )

        return frame

    def _frame_cache_for(self, cache_path: Path) -> np.ndarray:
        cache = self._frame_caches.get(cache_path)
        if cache is not None:
            return cache

        try:
            cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Could not open frame cache: {cache_path}") from error
        if cache.dtype != np.uint8 or cache.ndim != 4 or cache.shape[-1] != 3:
            raise RuntimeError(
                f"Invalid frame cache shape or dtype: {cache_path} "
                f"({cache.shape}, {cache.dtype})"
            )
        self._frame_caches[cache_path] = cache
        return cache

    def _capture_for(self, video_path: Path) -> cv2.VideoCapture:
        capture = self._captures.get(video_path)

        if capture is not None and capture.isOpened():
            return capture

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open video: {video_path}")

        self._captures[video_path] = capture
        return capture

    def __getstate__(self) -> dict:
        """
        Remove OpenCV handles when the Dataset is serialized for workers.
        """

        state = self.__dict__.copy()
        state["_captures"] = {}
        state["_frame_caches"] = {}
        return state

    def __del__(self) -> None:
        captures = getattr(self, "_captures", None)

        if captures is not None:
            self.close()


def _read_video_frame_count(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0:
            raise ValueError(
                f"Video reports an invalid frame count: {video_path}"
            )

        return frame_count
    finally:
        capture.release()
