from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import queue
from multiprocessing.process import BaseProcess
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor

from mimic_tear.visualization.hirescam import HiResCamVisualizer
from mimic_tear.visualization.controller_overlay import (
    GA_ROOT,
    GWL_EXSTYLE,
    GWL_STYLE,
    HWND_TOPMOST,
    SWP_FRAMECHANGED,
    SWP_NOACTIVATE,
    SWP_SHOWWINDOW,
    WDA_EXCLUDEFROMCAPTURE,
    WS_EX_LAYERED,
    WS_EX_NOACTIVATE,
    WS_EX_TOOLWINDOW,
    WS_EX_TRANSPARENT,
    WS_POPUP,
    WS_VISIBLE,
    _find_elden_ring_window_rect,
    _primary_monitor_rect,
)


OVERLAY_TITLE = "AI HiResCAM Overlay"
LWA_COLORKEY = 0x00000001
LWA_ALPHA = 0x00000002
SW_HIDE = 0
SW_SHOWNOACTIVATE = 4
VK_F8 = 0x77


class ToggleKey:
    """Edge-triggered global Windows key state used by click-through overlays."""

    def __init__(self, virtual_key: int = VK_F8) -> None:
        if os.name != "nt":
            raise RuntimeError("The global CAM toggle requires Windows")
        self._virtual_key = virtual_key
        self._was_down = False

    def pressed(self) -> bool:
        key_state = ctypes.windll.user32.GetAsyncKeyState(self._virtual_key)
        is_down = bool(key_state & 0x8000)
        pressed = is_down and not self._was_down
        self._was_down = is_down
        return pressed


class HiResCamOverlay:
    """Compute HiResCAM asynchronously and show it above the game window."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: torch.device,
        cam_fps: float = 5.0,
        opacity: float = 0.65,
        threshold: float = 0.15,
        startup_timeout_seconds: float = 20.0,
    ) -> None:
        if os.name != "nt":
            raise RuntimeError("The excluded HiResCAM overlay requires Windows")
        if cam_fps <= 0:
            raise ValueError("cam_fps must be positive")
        if not 0.0 < opacity <= 1.0:
            raise ValueError("opacity must be in (0, 1]")
        if not 0.0 <= threshold < 1.0:
            raise ValueError("threshold must be in [0, 1)")
        self._checkpoint_path = str(Path(checkpoint_path).resolve())
        self._device_name = str(device)
        self._cam_fps = cam_fps
        self._opacity = opacity
        self._threshold = threshold
        self._startup_timeout_seconds = startup_timeout_seconds
        self._context = mp.get_context("spawn")
        self._message_queue: Any = self._context.Queue(maxsize=2)
        self._status_queue: Any = self._context.Queue(maxsize=1)
        self._process: BaseProcess | None = None
        self._visible = True

    @property
    def visible(self) -> bool:
        return self._visible

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("HiResCAM overlay has already been started")
        self._process = self._context.Process(
            target=_overlay_process_main,
            args=(
                self._message_queue,
                self._status_queue,
                self._checkpoint_path,
                self._device_name,
                self._cam_fps,
                self._opacity,
                self._threshold,
            ),
            name="AIHiResCamOverlay",
            daemon=True,
        )
        self._process.start()
        try:
            status, detail = self._status_queue.get(
                timeout=self._startup_timeout_seconds
            )
        except queue.Empty as error:
            self.close()
            raise RuntimeError("Timed out starting the HiResCAM overlay") from error
        if status != "ready":
            self.close()
            raise RuntimeError(f"HiResCAM overlay could not start: {detail}")

    def update(self, image: Tensor, game_state: Tensor | None = None) -> None:
        array = image.detach().squeeze(0).to(device="cpu").numpy()
        if array.ndim != 3:
            raise ValueError(f"Expected a CHW image tensor, received {array.shape}")
        state_array = (
            game_state.detach().squeeze(0).to(device="cpu").numpy()
            if game_state is not None
            else None
        )
        self._send_latest(
            (
                "image",
                (
                    np.asarray(array, dtype=np.float32),
                    (
                        np.asarray(state_array, dtype=np.float32)
                        if state_array is not None
                        else None
                    ),
                ),
            )
        )

    def toggle(self) -> bool:
        self.set_visible(not self._visible)
        return self._visible

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        self._send_latest(("visible", visible))

    def _send_latest(self, message: tuple[str, object]) -> None:
        if self._process is None:
            raise RuntimeError("HiResCAM overlay has not been started")
        if not self._process.is_alive():
            raise RuntimeError("HiResCAM overlay process exited unexpectedly")
        try:
            self._message_queue.put_nowait(message)
        except queue.Full:
            try:
                self._message_queue.get_nowait()
            except queue.Empty:
                pass
            self._message_queue.put_nowait(message)

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.is_alive():
            self._send_latest(("close", True))
            process.join(timeout=3.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        self._process = None


def _overlay_process_main(
    message_queue: Any,
    status_queue: Any,
    checkpoint_path: str,
    device_name: str,
    cam_fps: float,
    opacity: float,
    threshold: float,
) -> None:
    reported_ready = False
    visualizer: HiResCamVisualizer | None = None
    try:
        device = torch.device(device_name)
        visualizer = HiResCamVisualizer.from_checkpoint(
            checkpoint_path,
            device=device,
        )
        try:
            left, top, right, bottom = _find_elden_ring_window_rect()
        except RuntimeError:
            left, top, right, bottom = _primary_monitor_rect()
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            raise RuntimeError("The target window has invalid dimensions")

        cv2.namedWindow(OVERLAY_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(OVERLAY_TITLE, width, height)
        cv2.imshow(OVERLAY_TITLE, np.zeros((height, width, 3), dtype=np.uint8))
        cv2.waitKey(1)
        window = _find_overlay_window()
        _configure_overlay_window(
            window,
            x=left,
            y=top,
            width=width,
            height=height,
            opacity=opacity,
        )
        status_queue.put(("ready", f"window={window}"))
        reported_ready = True

        visible = True
        pending_image: np.ndarray | None = None
        pending_state: np.ndarray | None = None
        next_cam_at = perf_counter()
        running = True
        while running:
            received = False
            try:
                while True:
                    kind, payload = message_queue.get_nowait()
                    received = True
                    if kind == "close":
                        running = False
                        break
                    if kind == "visible":
                        visible = bool(payload)
                        ctypes.windll.user32.ShowWindow(
                            window,
                            SW_SHOWNOACTIVATE if visible else SW_HIDE,
                        )
                    elif kind == "image":
                        pending_image, pending_state = payload
            except queue.Empty:
                pass
            if not running:
                break

            now = perf_counter()
            if visible and pending_image is not None and now >= next_cam_at:
                image = torch.from_numpy(pending_image).unsqueeze(0).to(device)
                state = (
                    torch.from_numpy(pending_state).unsqueeze(0).to(device)
                    if pending_state is not None
                    else None
                )
                pending_image = None
                pending_state = None
                heatmap = visualizer.generate(image, state)
                cv2.imshow(
                    OVERLAY_TITLE,
                    _render_transparent_heatmap(
                        heatmap,
                        width=width,
                        height=height,
                        threshold=threshold,
                    ),
                )
                cv2.waitKey(1)
                next_cam_at = perf_counter() + 1.0 / cam_fps
            elif not received:
                cv2.waitKey(1)
                sleep(0.005)
    except Exception as error:
        if not reported_ready:
            status_queue.put(("error", f"{type(error).__name__}: {error}"))
        raise
    finally:
        if visualizer is not None:
            visualizer.close()
        cv2.destroyWindow(OVERLAY_TITLE)


def _render_transparent_heatmap(
    heatmap: np.ndarray,
    *,
    width: int,
    height: int,
    threshold: float,
) -> np.ndarray:
    resized = cv2.resize(
        np.clip(heatmap, 0.0, 1.0),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    colored = cv2.applyColorMap(
        np.rint(resized * 255.0).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    colored[resized < threshold] = 0
    cv2.putText(
        colored,
        "HiResCAM  |  F8: toggle",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return colored


def _find_overlay_window() -> int:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
    user32.FindWindowW.restype = ctypes.c_void_p
    user32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    user32.GetAncestor.restype = ctypes.c_void_p
    window = user32.FindWindowW(None, OVERLAY_TITLE)
    if not window:
        raise RuntimeError("OpenCV HiResCAM overlay window was not found")
    return int(user32.GetAncestor(window, GA_ROOT) or window)


def _configure_overlay_window(
    window: int,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    opacity: float,
) -> None:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
    user32.GetWindowLongW.restype = ctypes.c_long
    user32.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
    user32.SetWindowLongW.restype = ctypes.c_long
    user32.SetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    user32.SetWindowDisplayAffinity.restype = ctypes.c_bool
    user32.GetWindowDisplayAffinity.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
    ]
    user32.GetWindowDisplayAffinity.restype = ctypes.c_bool
    user32.SetLayeredWindowAttributes.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_ubyte,
        ctypes.c_uint,
    ]
    user32.SetLayeredWindowAttributes.restype = ctypes.c_bool
    user32.SetWindowPos.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    user32.SetWindowPos.restype = ctypes.c_bool

    if not user32.SetWindowDisplayAffinity(window, WDA_EXCLUDEFROMCAPTURE):
        error_code = ctypes.get_last_error()
        raise RuntimeError(
            "SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) failed with "
            f"Windows error {error_code}: {ctypes.FormatError(error_code).strip()}"
        )
    affinity = ctypes.c_uint()
    if not user32.GetWindowDisplayAffinity(window, ctypes.byref(affinity)):
        error_code = ctypes.get_last_error()
        raise RuntimeError(
            f"GetWindowDisplayAffinity failed with Windows error {error_code}: "
            f"{ctypes.FormatError(error_code).strip()}"
        )
    if affinity.value != WDA_EXCLUDEFROMCAPTURE:
        raise RuntimeError(
            f"Windows reported unexpected display affinity: {affinity.value:#x}"
        )
    user32.SetWindowLongW(window, GWL_STYLE, WS_POPUP | WS_VISIBLE)
    extended_style = int(user32.GetWindowLongW(window, GWL_EXSTYLE))
    extended_style |= (
        WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
    )
    user32.SetWindowLongW(window, GWL_EXSTYLE, extended_style)
    alpha = round(opacity * 255)
    if not user32.SetLayeredWindowAttributes(
        window,
        0,
        alpha,
        LWA_COLORKEY | LWA_ALPHA,
    ):
        error_code = ctypes.get_last_error()
        raise RuntimeError(
            f"SetLayeredWindowAttributes failed with Windows error {error_code}: "
            f"{ctypes.FormatError(error_code).strip()}"
        )
    if not user32.SetWindowPos(
        window,
        HWND_TOPMOST,
        x,
        y,
        width,
        height,
        SWP_NOACTIVATE | SWP_FRAMECHANGED | SWP_SHOWWINDOW,
    ):
        error_code = ctypes.get_last_error()
        raise RuntimeError(
            f"SetWindowPos failed with Windows error {error_code}: "
            f"{ctypes.FormatError(error_code).strip()}"
        )
