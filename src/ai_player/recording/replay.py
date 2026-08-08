from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch


from ai_player.recording.annotations import (  # noqa: E402
    FrameRange,
    is_frame_excluded,
    load_frame_ranges,
    merge_frame_ranges,
    save_frame_ranges,
)
from ai_player.visualization.controller_layout import draw_controller_layout  # noqa: E402
from ai_player.dataset.transforms import build_eval_transform  # noqa: E402
from ai_player.game_state import game_state_tensor  # noqa: E402
from ai_player.game_state.schema import GAME_STATE_PARQUET_SCHEMA  # noqa: E402
from ai_player.recording.schema import (  # noqa: E402
    PARQUET_SCHEMA,
    validate_columns,
)

if TYPE_CHECKING:
    from ai_player.visualization import HiResCamVisualizer


WINDOW_TITLE = "AI Player Recording Replay"
LEFT_KEYS = {81, 2424832}
RIGHT_KEYS = {83, 2555904}
HOME_KEYS = {71, 2359296}
END_KEYS = {79, 2293760}
DELETE_KEYS = {127, 3014656}


@dataclass(slots=True)
class ReplaySession:
    directory: Path
    video_path: Path
    rows: list[dict[str, Any]]
    frame_count: int
    fps: float
    width: int
    height: int
    metadata: dict[str, Any]
    game_state_rows: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class ReplayState:
    frame_index: int = 0
    playing: bool = False
    selection_anchor: int | None = None
    selection_end: int | None = None
    excluded_ranges: list[FrameRange] = field(default_factory=list)
    undo_stack: list[list[FrameRange]] = field(default_factory=list)
    dirty: bool = False
    cam_enabled: bool = False
    message: str = "Space: play  [ ]: select  X/Delete: exclude"

    def selection(self) -> FrameRange | None:
        if self.selection_anchor is None:
            return None
        end = self.frame_index if self.selection_end is None else self.selection_end
        return FrameRange(min(self.selection_anchor, end), max(self.selection_anchor, end))


def load_replay_session(session_directory: str | Path) -> ReplaySession:
    directory = Path(session_directory).expanduser().resolve()
    video_path = directory / "frames.mp4"
    input_path = directory / "inputs.parquet"
    metadata_path = directory / "metadata.json"
    game_state_path = directory / "game_state.parquet"
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    table = pq.read_table(input_path)
    validate_columns(table.column_names)
    if not table.schema.equals(PARQUET_SCHEMA, check_metadata=False):
        raise ValueError(
            f"Parquet types do not match the recorder format: {table.schema}"
        )
    rows = table.to_pylist()

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("Video reports invalid dimensions, FPS, or frame count")
    if len(rows) != frame_count:
        raise ValueError(
            f"Frame/input mismatch: {frame_count} video frames and "
            f"{len(rows)} Parquet rows"
        )
    if any(int(row["frame_index"]) != index for index, row in enumerate(rows)):
        raise ValueError("Parquet frame_index values are not sequential")

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            metadata = loaded

    game_state_rows = None
    if game_state_path.is_file():
        game_state_table = pq.read_table(game_state_path)
        if not game_state_table.schema.equals(
            GAME_STATE_PARQUET_SCHEMA,
            check_metadata=False,
        ):
            raise ValueError(f"Invalid game-state schema: {game_state_path}")
        if game_state_table.num_rows != frame_count:
            raise ValueError(
                f"Frame/game-state mismatch: {frame_count} frames and "
                f"{game_state_table.num_rows} state rows"
            )
        game_state_rows = game_state_table.to_pylist()
        if any(
            int(row["frame_index"]) != index
            for index, row in enumerate(game_state_rows)
        ):
            raise ValueError("Game-state frame_index values are not sequential")

    return ReplaySession(
        directory=directory,
        video_path=video_path,
        rows=rows,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        metadata=metadata,
        game_state_rows=game_state_rows,
    )


def read_video_frame(
    capture: cv2.VideoCapture,
    frame_index: int,
) -> np.ndarray:
    next_frame = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
    if next_frame != frame_index:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = capture.read()
    if not success or frame is None:
        raise RuntimeError(f"Could not decode video frame {frame_index}")
    return frame


def render_replay_frame(
    frame: np.ndarray,
    row: dict[str, Any],
    state: ReplayState,
    *,
    total_frames: int,
    fps: float,
    cam_heatmap: np.ndarray | None = None,
    cam_opacity: float = 0.45,
    game_state_row: dict[str, Any] | None = None,
) -> np.ndarray:
    if cam_heatmap is not None:
        from ai_player.visualization import blend_heatmap

        frame = blend_heatmap(frame, cam_heatmap, opacity=cam_opacity)
    panel_width = 330
    canvas_height = max(frame.shape[0], 360)
    canvas = np.zeros((canvas_height, frame.shape[1] + panel_width, 3), dtype=np.uint8)
    canvas[: frame.shape[0], : frame.shape[1]] = frame
    panel_left = frame.shape[1]

    excluded = is_frame_excluded(state.frame_index, state.excluded_ranges)
    selection = state.selection()
    selected = selection is not None and selection.contains(state.frame_index)
    if excluded or selected:
        color = (40, 40, 210) if excluded else (0, 190, 230)
        cv2.rectangle(canvas, (0, 0), (frame.shape[1] - 1, 30), color, -1)
        label = "EXCLUDED FROM TRAINING" if excluded else "SELECTED RANGE"
        draw_text(canvas, label, 8, 21, color=(255, 255, 255), scale=0.52)

    elapsed = state.frame_index / fps
    sync_ms = float(row["input_offset_ns"]) / 1e6
    sync_color = (80, 230, 80) if abs(sync_ms) < 5 else (0, 200, 255)
    draw_text(
        canvas,
        f"Frame {state.frame_index}/{total_frames - 1}",
        panel_left + 12,
        22,
    )
    draw_text(canvas, f"Time  {elapsed:8.2f}s", panel_left + 12, 44)
    draw_text(
        canvas,
        f"Sync  {sync_ms:+6.2f} ms",
        panel_left + 12,
        66,
        color=sync_color,
    )

    draw_controller_layout(canvas, row, origin=(panel_left, 0))

    if game_state_row is not None:
        state_text = (
            f"HP {game_state_row.get('player_health')}/"
            f"{game_state_row.get('player_max_health')}  "
            f"SP {game_state_row.get('player_stamina')}/"
            f"{game_state_row.get('player_max_stamina')}  "
            f"Lock {game_state_row.get('lock_on_active')}"
        )
        draw_text(
            canvas,
            state_text,
            8,
            canvas_height - 45,
            color=(80, 230, 255),
            scale=0.45,
        )

    if cam_heatmap is not None:
        draw_text(
            canvas,
            "HiResCAM action evidence  [H: toggle]",
            8,
            22,
            color=(255, 255, 255),
            scale=0.5,
        )

    if selection is not None:
        selection_text = f"Selection: {selection.start_frame}-{selection.end_frame}"
    else:
        selection_text = f"Excluded: {sum(r.frame_count for r in state.excluded_ranges)} frames"
    draw_text(canvas, selection_text, 8, canvas_height - 25, scale=0.45)
    draw_text(canvas, state.message, 8, canvas_height - 7, scale=0.38)
    return canvas


def draw_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    color: tuple[int, int, int] = (230, 230, 230),
    scale: float = 0.46,
) -> None:
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def exclude_selection(state: ReplayState, *, total_frames: int) -> None:
    selection = state.selection() or FrameRange(state.frame_index, state.frame_index)
    state.undo_stack.append(list(state.excluded_ranges))
    state.excluded_ranges = merge_frame_ranges(
        [*state.excluded_ranges, selection],
        total_frames=total_frames,
    )
    state.selection_anchor = None
    state.selection_end = None
    state.dirty = True
    state.message = f"Excluded frames {selection.start_frame}-{selection.end_frame}"


def save_annotations(session: ReplaySession, state: ReplayState) -> None:
    path = save_frame_ranges(
        session.directory,
        state.excluded_ranges,
        total_frames=session.frame_count,
    )
    state.dirty = False
    state.message = f"Saved {path.name}"


def run_replay(
    session: ReplaySession,
    *,
    start_frame: int = 0,
    cam_visualizer: HiResCamVisualizer | None = None,
    cam_fps: float = 5.0,
    cam_opacity: float = 0.45,
) -> None:
    if cam_fps <= 0:
        raise ValueError("cam_fps must be positive")
    if (
        cam_visualizer is not None
        and cam_visualizer.model.game_state_features > 0
        and session.game_state_rows is None
    ):
        raise ValueError(
            "This state-aware checkpoint requires game_state.parquet in replay"
        )
    state = ReplayState(
        frame_index=max(0, min(start_frame, session.frame_count - 1)),
        cam_enabled=cam_visualizer is not None,
        excluded_ranges=load_frame_ranges(
            session.directory,
            total_frames=session.frame_count,
        ),
    )
    capture = cv2.VideoCapture(str(session.video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open {session.video_path}")

    requested_frame = [state.frame_index]
    suppress_trackbar = [False]
    transform = build_eval_transform() if cam_visualizer is not None else None
    cam_interval_frames = max(1, round(session.fps / cam_fps))
    cam_heatmap: np.ndarray | None = None
    cam_frame_index: int | None = None

    def on_seek(value: int) -> None:
        if suppress_trackbar[0]:
            return
        requested_frame[0] = value
        state.playing = False

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(
        "Frame",
        WINDOW_TITLE,
        state.frame_index,
        session.frame_count - 1,
        on_seek,
    )
    cv2.resizeWindow(WINDOW_TITLE, session.width + 330, max(session.height, 360))

    try:
        while True:
            if requested_frame[0] != state.frame_index:
                state.frame_index = requested_frame[0]
            frame = read_video_frame(capture, state.frame_index)
            game_state_row = (
                session.game_state_rows[state.frame_index]
                if session.game_state_rows is not None
                else None
            )
            if state.cam_enabled and cam_visualizer is not None:
                refresh_cam = (
                    cam_frame_index is None
                    or (not state.playing and cam_frame_index != state.frame_index)
                    or abs(state.frame_index - cam_frame_index) >= cam_interval_frames
                )
                if refresh_cam:
                    assert transform is not None
                    model_device = next(cam_visualizer.model.parameters()).device
                    image = transform(frame).unsqueeze(0).to(model_device)
                    model_state = (
                        game_state_tensor(
                            game_state_row,
                            valid=bool(game_state_row["state_valid"]),
                        ).unsqueeze(0).to(model_device)
                        if game_state_row is not None
                        else None
                    )
                    cam_heatmap = cam_visualizer.generate(image, model_state)
                    cam_frame_index = state.frame_index
            canvas = render_replay_frame(
                frame,
                session.rows[state.frame_index],
                state,
                total_frames=session.frame_count,
                fps=session.fps,
                cam_heatmap=cam_heatmap if state.cam_enabled else None,
                cam_opacity=cam_opacity,
                game_state_row=game_state_row,
            )
            cv2.imshow(WINDOW_TITLE, canvas)
            suppress_trackbar[0] = True
            cv2.setTrackbarPos("Frame", WINDOW_TITLE, state.frame_index)
            suppress_trackbar[0] = False

            delay_ms = max(1, round(1000 / session.fps)) if state.playing else 30
            key = cv2.waitKeyEx(delay_ms)
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (-1, 255):
                if state.playing:
                    if state.frame_index >= session.frame_count - 1:
                        state.playing = False
                    else:
                        state.frame_index += 1
                        requested_frame[0] = state.frame_index
                continue

            low_key = key & 0xFF
            if low_key in (ord("q"), 27):
                break
            if low_key == ord(" "):
                state.playing = not state.playing
            elif key in LEFT_KEYS or low_key == ord("a"):
                state.playing = False
                state.frame_index = max(0, state.frame_index - 1)
            elif key in RIGHT_KEYS or low_key == ord("d"):
                state.playing = False
                state.frame_index = min(session.frame_count - 1, state.frame_index + 1)
            elif low_key == ord("j"):
                state.playing = False
                state.frame_index = max(0, state.frame_index - round(session.fps))
            elif low_key == ord("l"):
                state.playing = False
                state.frame_index = min(
                    session.frame_count - 1,
                    state.frame_index + round(session.fps),
                )
            elif key in HOME_KEYS or low_key == ord("g"):
                state.playing = False
                state.frame_index = 0
            elif key in END_KEYS or low_key == ord("e"):
                state.playing = False
                state.frame_index = session.frame_count - 1
            elif low_key == ord("["):
                state.selection_anchor = state.frame_index
                state.selection_end = None
                state.message = f"Selection starts at frame {state.frame_index}"
            elif low_key == ord("]"):
                if state.selection_anchor is None:
                    state.selection_anchor = state.frame_index
                state.selection_end = state.frame_index
                state.message = "Press X or Delete to exclude the selected range"
            elif key in DELETE_KEYS or low_key == ord("x"):
                exclude_selection(state, total_frames=session.frame_count)
            elif low_key == ord("c"):
                state.selection_anchor = None
                state.selection_end = None
                state.message = "Selection cleared"
            elif low_key == ord("h"):
                if cam_visualizer is None:
                    state.message = "Use --cam-checkpoint to enable HiResCAM"
                else:
                    state.cam_enabled = not state.cam_enabled
                    cam_frame_index = None
                    state.message = (
                        f"HiResCAM {'enabled' if state.cam_enabled else 'disabled'}"
                    )
            elif low_key == ord("u"):
                if state.undo_stack:
                    state.excluded_ranges = state.undo_stack.pop()
                    state.dirty = True
                    state.message = "Undid the most recent exclusion"
            elif low_key == ord("s"):
                save_annotations(session, state)
            requested_frame[0] = state.frame_index
    finally:
        if state.dirty:
            save_annotations(session, state)
        capture.release()
        cv2.destroyAllWindows()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay synchronized frames and controller inputs."
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--cam-checkpoint",
        type=Path,
        help="policy checkpoint used to render HiResCAM action evidence",
    )
    parser.add_argument("--cam-fps", type=float, default=5.0)
    parser.add_argument("--cam-opacity", type=float, default=0.45)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    args = parser.parse_args(argv)
    if args.start_frame < 0:
        parser.error("--start-frame must be non-negative")
    if args.cam_fps <= 0:
        parser.error("--cam-fps must be positive")
    if not 0.0 <= args.cam_opacity <= 1.0:
        parser.error("--cam-opacity must be in [0, 1]")
    return args


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    session = load_replay_session(args.session)
    print(
        f"Loaded {session.frame_count} frames at {session.fps:.2f} FPS from "
        f"{session.directory}"
    )
    print("Space play/pause | arrows/A/D frame | J/L second | [ ] select")
    print("X/Delete exclude | U undo | C clear selection | S save | Q quit")
    visualizer = None
    try:
        if args.cam_checkpoint is not None:
            from ai_player.visualization import HiResCamVisualizer

            device = resolve_device(args.device)
            visualizer = HiResCamVisualizer.from_checkpoint(
                args.cam_checkpoint,
                device=device,
            )
            print(f"HiResCAM enabled on {device} | H toggle")
        run_replay(
            session,
            start_frame=args.start_frame,
            cam_visualizer=visualizer,
            cam_fps=args.cam_fps,
            cam_opacity=args.cam_opacity,
        )
    finally:
        if visualizer is not None:
            visualizer.close()


if __name__ == "__main__":
    main()
