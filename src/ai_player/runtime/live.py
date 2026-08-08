from __future__ import annotations

import argparse
import math
from pathlib import Path
from time import perf_counter, perf_counter_ns, sleep

import dxcam
import torch

from ai_player.controller import ControllerState, VirtualController
from ai_player.visualization.controller_overlay import ControllerOverlay
from ai_player.dataset.transforms import build_eval_transform
from ai_player.game_state import (
    EldenRingStateReader,
    GameStateSampler,
    game_state_tensor,
    load_memory_profile,
)
from ai_player.policy import EldenRingPolicy, load_policy_checkpoint
from ai_player.paths import DEFAULT_GAME_STATE_PROFILE
from ai_player.recording.schema import ANALOG_COLUMNS, BUTTON_COLUMNS
from ai_player.recording.synchronization import FrameClockSynchronizer


def load_policy(
    checkpoint_path: Path,
    device: torch.device,
) -> EldenRingPolicy:
    model, _ = load_policy_checkpoint(checkpoint_path, device=device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the current behavioral-cloning policy with live input."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--decision-hz", type=float, default=30.0)
    parser.add_argument("--button-threshold", type=float, default=0.5)
    parser.add_argument(
        "--game-state-profile",
        type=Path,
        default=DEFAULT_GAME_STATE_PROFILE,
        help="read-only memory profile used by state-aware checkpoints",
    )
    parser.add_argument(
        "--game-state-hz",
        type=int,
        default=60,
        help="process-memory polls per second (default: 60)",
    )
    parser.add_argument(
        "--max-game-state-sync-offset-ms",
        type=float,
        default=25.0,
    )
    parser.add_argument(
        "--output-index",
        type=int,
        default=0,
        help="DXGI monitor index to capture (default: primary monitor, index 0).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the captured frame size and controller output once per second.",
    )
    parser.add_argument(
        "--controller-overlay",
        action="store_true",
        help=(
            "show a click-through controller HUD that Windows excludes from "
            "the AI's screen capture"
        ),
    )
    parser.add_argument(
        "--cam-overlay",
        action="store_true",
        help=(
            "show a click-through HiResCAM overlay for the current AI action; "
            "press F8 to toggle it"
        ),
    )
    parser.add_argument(
        "--cam-fps",
        type=float,
        default=5.0,
        help="maximum HiResCAM updates per second (default: 5)",
    )
    parser.add_argument(
        "--cam-opacity",
        type=float,
        default=0.65,
        help="HiResCAM overlay opacity in (0, 1] (default: 0.65)",
    )
    parser.add_argument(
        "--cam-threshold",
        type=float,
        default=0.15,
        help="hide CAM evidence below this intensity in [0, 1) (default: 0.15)",
    )
    parser.add_argument(
        "--armed",
        action="store_true",
        help="Required acknowledgement that this process will send live input.",
    )
    args = parser.parse_args()

    if not args.armed:
        parser.error("Pass --armed to enable live controller output.")
    if args.decision_hz <= 0:
        parser.error("--decision-hz must be positive.")
    if not 0.0 <= args.button_threshold <= 1.0:
        parser.error("--button-threshold must be in [0, 1].")
    if args.output_index < 0:
        parser.error("--output-index must be non-negative.")
    if args.game_state_hz < math.ceil(args.decision_hz):
        parser.error("--game-state-hz must be at least --decision-hz.")
    if args.max_game_state_sync_offset_ms <= 0:
        parser.error("--max-game-state-sync-offset-ms must be positive.")
    if args.cam_fps <= 0:
        parser.error("--cam-fps must be positive.")
    if not 0.0 < args.cam_opacity <= 1.0:
        parser.error("--cam-opacity must be in (0, 1].")
    if not 0.0 <= args.cam_threshold < 1.0:
        parser.error("--cam-threshold must be in [0, 1).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_policy(args.checkpoint, device)
    game_state_sampler = None
    if model.game_state_features > 0:
        memory_profile = load_memory_profile(args.game_state_profile)
        game_state_sampler = GameStateSampler(
            lambda: EldenRingStateReader.open(memory_profile),
            polling_hz=args.game_state_hz,
        )
        game_state_sampler.start()
    transform = build_eval_transform()
    decision_period = 1.0 / args.decision_hz
    frame_clock = FrameClockSynchronizer()

    try:
        camera = dxcam.create(output_idx=args.output_index, output_color="BGR")
        camera.start(target_fps=max(30, round(args.decision_hz)))
    except Exception:
        if game_state_sampler is not None:
            game_state_sampler.stop()
        raise
    overlay = ControllerOverlay() if args.controller_overlay else None
    cam_overlay = None
    cam_toggle = None
    if args.cam_overlay:
        from ai_player.visualization import HiResCamOverlay, ToggleKey

        cam_overlay = HiResCamOverlay(
            args.checkpoint,
            device=device,
            cam_fps=args.cam_fps,
            opacity=args.cam_opacity,
            threshold=args.cam_threshold,
        )
        cam_toggle = ToggleKey()

    try:
        if cam_overlay is not None:
            cam_overlay.start()
            print(
                "HiResCAM overlay enabled and excluded from capture. "
                "Press F8 to toggle it.",
                flush=True,
            )
        if overlay is not None:
            overlay.start()
            print(
                "Controller overlay enabled and excluded from capture.",
                flush=True,
            )
        with VirtualController() as controller:
            print(
                f"AI controller armed on {device}. "
                f"Capturing output {args.output_index}. "
                "Press Ctrl+C to release input and stop.",
                flush=True,
            )
            next_debug_at = perf_counter()
            next_cam_at = perf_counter()
            reported_frame_shape = False
            while True:
                started = perf_counter()
                if cam_toggle is not None and cam_toggle.pressed():
                    assert cam_overlay is not None
                    visible = cam_overlay.toggle()
                    print(
                        f"HiResCAM overlay {'shown' if visible else 'hidden'}.",
                        flush=True,
                    )
                    next_cam_at = started
                frame_result = camera.get_latest_frame(with_timestamp=True)
                if frame_result is None:
                    continue
                frame, source_timestamp_seconds = frame_result
                received_timestamp_ns = perf_counter_ns()
                _, frame_timestamp_ns, _ = frame_clock.align(
                    source_timestamp_seconds,
                    received_timestamp_ns,
                )

                game_state_cpu = None
                game_state_snapshot = None
                if game_state_sampler is not None:
                    sample = game_state_sampler.closest(
                        frame_timestamp_ns,
                        timeout_seconds=max(0.025, 4 / args.game_state_hz),
                    )
                    state_offset_ns = sample.timestamp_ns - frame_timestamp_ns
                    if abs(state_offset_ns) > round(
                        args.max_game_state_sync_offset_ms * 1e6
                    ):
                        raise RuntimeError(
                            "Game state fell out of sync with live capture: "
                            f"{state_offset_ns / 1e6:+.2f} ms"
                        )
                    game_state_snapshot = sample.snapshot
                    game_state_cpu = game_state_tensor(
                        sample.snapshot.values,
                        valid=sample.snapshot.valid,
                    ).unsqueeze(0)

                cpu_image = transform(frame).unsqueeze(0)
                image = cpu_image.to(
                    device,
                    non_blocking=True,
                )
                with torch.inference_mode():
                    analog, buttons = model.predict(
                        image,
                        (
                            game_state_cpu.to(device, non_blocking=True)
                            if game_state_cpu is not None
                            else None
                        ),
                        button_threshold=args.button_threshold,
                    )

                state = ControllerState.from_predictions(
                    analog[0].tolist(),
                    buttons[0].tolist(),
                )
                controller.apply(state)
                if overlay is not None:
                    overlay.update(state)
                if (
                    cam_overlay is not None
                    and cam_overlay.visible
                    and started >= next_cam_at
                ):
                    cam_overlay.update(cpu_image, game_state_cpu)
                    next_cam_at = started + 1.0 / args.cam_fps

                if args.debug and not reported_frame_shape:
                    print(
                        f"Captured frame: {frame.shape[1]}x{frame.shape[0]} BGR",
                        flush=True,
                    )
                    reported_frame_shape = True

                now = perf_counter()
                if args.debug and now >= next_debug_at:
                    analog_values = ", ".join(
                        f"{name}={getattr(state, name):+.2f}"
                        for name in ANALOG_COLUMNS
                    )
                    active_buttons = [
                        name for name in BUTTON_COLUMNS if getattr(state, name)
                    ]
                    print(
                        f"AI output: {analog_values}; "
                        f"buttons={active_buttons or ['none']}",
                        flush=True,
                    )
                    if game_state_snapshot is not None:
                        values = game_state_snapshot.values
                        print(
                            "AI state: "
                            f"hp={values.get('player_health')}/"
                            f"{values.get('player_max_health')} "
                            f"stamina={values.get('player_stamina')}/"
                            f"{values.get('player_max_stamina')} "
                            f"lock_on={values.get('lock_on_active')}",
                            flush=True,
                        )
                    next_debug_at = now + 1.0

                remaining = decision_period - (perf_counter() - started)
                if remaining > 0:
                    sleep(remaining)
    finally:
        if overlay is not None:
            overlay.close()
        if cam_overlay is not None:
            cam_overlay.close()
        if game_state_sampler is not None:
            game_state_sampler.stop()
        camera.stop()


if __name__ == "__main__":
    main()
