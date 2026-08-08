from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import queue
from dataclasses import asdict
from multiprocessing.process import BaseProcess
from typing import Any

import cv2
import numpy as np

from ai_player.controller.state import ControllerState
from ai_player.visualization.controller_layout import (
    CONTROLLER_LAYOUT_HEIGHT,
    CONTROLLER_LAYOUT_WIDTH,
    render_controller_layout,
)
from ai_player.platform.windows.process_memory import find_process_id


WDA_EXCLUDEFROMCAPTURE = 0x00000011
GWL_STYLE = -16
GWL_EXSTYLE = -20
GA_ROOT = 2
WS_POPUP = 0x80000000
WS_VISIBLE = 0x10000000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000
HWND_TOPMOST = -1
LWA_ALPHA = 0x00000002
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
SWP_SHOWWINDOW = 0x0040
OVERLAY_WIDTH = CONTROLLER_LAYOUT_WIDTH
OVERLAY_HEIGHT = CONTROLLER_LAYOUT_HEIGHT
OVERLAY_TITLE = "AI Controller Overlay"


class ControllerOverlay:
    """Show AI controller state without including the window in capture."""

    def __init__(self, *, startup_timeout_seconds: float = 5.0) -> None:
        if os.name != "nt":
            raise RuntimeError("The excluded controller overlay requires Windows")
        if startup_timeout_seconds <= 0:
            raise ValueError("Overlay startup timeout must be greater than zero")
        self._startup_timeout_seconds = startup_timeout_seconds
        self._context = mp.get_context("spawn")
        self._state_queue: Any = self._context.Queue(maxsize=2)
        self._status_queue: Any = self._context.Queue(maxsize=1)
        self._process: BaseProcess | None = None

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("Controller overlay has already been started")
        self._process = self._context.Process(
            target=_overlay_process_main,
            args=(self._state_queue, self._status_queue),
            name="AIControllerOverlay",
            daemon=True,
        )
        self._process.start()
        try:
            status, detail = self._status_queue.get(
                timeout=self._startup_timeout_seconds
            )
        except queue.Empty as error:
            self.close()
            raise RuntimeError("Timed out starting the controller overlay") from error
        if status != "ready":
            self.close()
            raise RuntimeError(f"Controller overlay could not start: {detail}")

    def update(self, state: ControllerState) -> None:
        if self._process is None:
            raise RuntimeError("Controller overlay has not been started")
        if not self._process.is_alive():
            raise RuntimeError(
                "Controller overlay exited; stopping to prevent unverified capture"
            )
        payload = asdict(state)
        try:
            self._state_queue.put_nowait(payload)
        except queue.Full:
            try:
                self._state_queue.get_nowait()
            except queue.Empty:
                pass
            self._state_queue.put_nowait(payload)

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.is_alive():
            try:
                self._state_queue.put_nowait(None)
            except queue.Full:
                try:
                    self._state_queue.get_nowait()
                except queue.Empty:
                    pass
                self._state_queue.put_nowait(None)
            process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        self._process = None

    def __enter__(self) -> ControllerOverlay:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _overlay_process_main(state_queue: Any, status_queue: Any) -> None:
    reported_ready = False
    try:
        initial_state = asdict(ControllerState())
        cv2.namedWindow(OVERLAY_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(OVERLAY_TITLE, OVERLAY_WIDTH, OVERLAY_HEIGHT)
        cv2.imshow(OVERLAY_TITLE, render_controller_frame(initial_state))
        cv2.waitKey(1)

        try:
            left, top, right, _ = _find_elden_ring_window_rect()
        except RuntimeError:
            left, top, right, _ = _primary_monitor_rect()
        x = max(left + 20, right - OVERLAY_WIDTH - 20)
        y = top + 20
        window = _find_overlay_window()
        _configure_excluded_window(window, x=x, y=y)
        status_queue.put(("ready", f"window={window}"))
        reported_ready = True

        current_state = initial_state
        running = True
        while running:
            try:
                while True:
                    message = state_queue.get_nowait()
                    if message is None:
                        running = False
                        break
                    current_state = message
            except queue.Empty:
                pass
            if not running:
                break
            cv2.imshow(OVERLAY_TITLE, render_controller_frame(current_state))
            cv2.waitKey(16)
    except Exception as error:
        if not reported_ready:
            status_queue.put(("error", f"{type(error).__name__}: {error}"))
        raise
    finally:
        cv2.destroyWindow(OVERLAY_TITLE)


def _find_overlay_window() -> int:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
    user32.FindWindowW.restype = ctypes.c_void_p
    user32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    user32.GetAncestor.restype = ctypes.c_void_p
    window = user32.FindWindowW(None, OVERLAY_TITLE)
    if not window:
        raise RuntimeError("OpenCV controller overlay window was not found")
    return int(user32.GetAncestor(window, GA_ROOT) or window)


def _configure_excluded_window(window: int, *, x: int, y: int) -> None:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
    user32.GetWindowLongW.restype = ctypes.c_long
    user32.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
    user32.SetWindowLongW.restype = ctypes.c_long
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
    user32.SetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    user32.SetWindowDisplayAffinity.restype = ctypes.c_bool
    user32.GetWindowDisplayAffinity.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
    ]
    user32.GetWindowDisplayAffinity.restype = ctypes.c_bool

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
    if not user32.SetLayeredWindowAttributes(window, 0, 220, LWA_ALPHA):
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
        OVERLAY_WIDTH,
        OVERLAY_HEIGHT,
        SWP_NOACTIVATE | SWP_FRAMECHANGED | SWP_SHOWWINDOW,
    ):
        error_code = ctypes.get_last_error()
        raise RuntimeError(
            f"SetWindowPos failed with Windows error {error_code}: "
            f"{ctypes.FormatError(error_code).strip()}"
        )


def _find_elden_ring_window_rect() -> tuple[int, int, int, int]:
    from ctypes import wintypes

    process_id = find_process_id("eldenring.exe")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HWND,
        wintypes.LPARAM,
    )
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []

    @callback_type
    def collect(window: int, _: int) -> bool:
        if not user32.IsWindowVisible(window):
            return True
        owner_process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(window, ctypes.byref(owner_process_id))
        if int(owner_process_id.value) != process_id:
            return True
        rectangle = wintypes.RECT()
        if not user32.GetWindowRect(window, ctypes.byref(rectangle)):
            return True
        coordinates = (
            int(rectangle.left),
            int(rectangle.top),
            int(rectangle.right),
            int(rectangle.bottom),
        )
        area = max(0, coordinates[2] - coordinates[0]) * max(
            0, coordinates[3] - coordinates[1]
        )
        candidates.append((area, coordinates))
        return True

    user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    ]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    user32.EnumWindows(collect, 0)
    if not candidates:
        raise RuntimeError("No visible Elden Ring window was found for the overlay")
    return max(candidates)[1]


def _primary_monitor_rect() -> tuple[int, int, int, int]:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    user32.GetSystemMetrics.restype = ctypes.c_int
    width = int(user32.GetSystemMetrics(0))
    height = int(user32.GetSystemMetrics(1))
    if width <= 0 or height <= 0:
        raise RuntimeError("Could not determine a fallback monitor for the overlay")
    return 0, 0, width, height


def render_controller_frame(state: dict[str, object]) -> np.ndarray:
    return render_controller_layout(state)
