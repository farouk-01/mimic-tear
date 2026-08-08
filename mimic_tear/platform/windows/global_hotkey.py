from __future__ import annotations

import ctypes
import os
import threading
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable


class GlobalHotkeyError(RuntimeError):
    pass


MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
HOTKEY_ID = 0xA1F8

_KEY_CODES = {
    **{f"F{index}": 0x6F + index for index in range(1, 13)},
    "ESC": 0x1B,
    "SPACE": 0x20,
    "INSERT": 0x2D,
    "DELETE": 0x2E,
    "HOME": 0x24,
    "END": 0x23,
}
_MODIFIERS = {
    "ALT": MOD_ALT,
    "CTRL": MOD_CONTROL,
    "CONTROL": MOD_CONTROL,
    "SHIFT": MOD_SHIFT,
    "WIN": MOD_WIN,
}


@dataclass(frozen=True, slots=True)
class HotkeySpec:
    label: str
    modifiers: int
    virtual_key: int


def parse_hotkey(value: str) -> HotkeySpec | None:
    if not isinstance(value, str):
        raise ValueError("stop hotkey must be a string")
    normalized = value.strip().upper()
    if normalized in ("", "NONE", "OFF", "DISABLED"):
        return None
    tokens = [token.strip() for token in normalized.split("+")]
    if any(not token for token in tokens):
        raise ValueError("stop hotkey contains an empty key component")
    key_tokens = [token for token in tokens if token not in _MODIFIERS]
    if len(key_tokens) != 1 or key_tokens[0] not in _KEY_CODES:
        supported = ", ".join(sorted(_KEY_CODES))
        raise ValueError(
            f"stop hotkey must be a function/navigation key, optionally prefixed "
            f"with CTRL, ALT, SHIFT, or WIN (supported keys: {supported})"
        )
    modifiers = 0
    for token in tokens:
        if token in _MODIFIERS:
            modifiers |= _MODIFIERS[token]
    key = key_tokens[0]
    return HotkeySpec(
        label="+".join(tokens),
        modifiers=modifiers | MOD_NOREPEAT,
        virtual_key=_KEY_CODES[key],
    )


class GlobalHotkey:
    """Register one native Windows hotkey and invoke a callback on activation."""

    def __init__(self, spec: HotkeySpec, callback: Callable[[], None]) -> None:
        self.spec = spec
        self._callback = callback
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._ready = threading.Event()
        self._stop_requested = threading.Event()
        self._error: BaseException | None = None

    def start(self, *, timeout_seconds: float = 2.0) -> None:
        if os.name != "nt":
            raise GlobalHotkeyError("Global hotkeys are only supported on Windows")
        if self._thread is not None:
            raise GlobalHotkeyError("Global hotkey has already been started")
        self._thread = threading.Thread(
            target=self._run,
            name="RecorderGlobalHotkey",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout_seconds):
            raise GlobalHotkeyError("Timed out registering the global stop hotkey")
        if self._error is not None:
            raise GlobalHotkeyError(
                f"Could not register global stop hotkey {self.spec.label}: "
                f"{self._error}"
            ) from self._error

    def stop(self) -> None:
        self._stop_requested.set()
        if self._thread_id is not None and os.name == "nt":
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.PostThreadMessageW.argtypes = [
                wintypes.DWORD,
                wintypes.UINT,
                wintypes.WPARAM,
                wintypes.LPARAM,
            ]
            user32.PostThreadMessageW.restype = wintypes.BOOL
            user32.PostThreadMessageW(
                self._thread_id,
                WM_QUIT,
                0,
                0,
            )
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        user32.RegisterHotKey.argtypes = [
            wintypes.HWND,
            ctypes.c_int,
            wintypes.UINT,
            wintypes.UINT,
        ]
        user32.RegisterHotKey.restype = wintypes.BOOL
        user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.UnregisterHotKey.restype = wintypes.BOOL
        user32.GetMessageW.argtypes = [
            ctypes.POINTER(wintypes.MSG),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
        ]
        user32.GetMessageW.restype = ctypes.c_int

        # GetCurrentThreadId is exported by kernel32, not user32.  Looking it
        # up on user32 raises AttributeError on current Python/Windows builds
        # and leaves the recorder waiting forever for registration.
        kernel32.GetCurrentThreadId.argtypes = []
        kernel32.GetCurrentThreadId.restype = wintypes.DWORD
        self._thread_id = int(kernel32.GetCurrentThreadId())
        registered = bool(
            user32.RegisterHotKey(
                None,
                HOTKEY_ID,
                self.spec.modifiers,
                self.spec.virtual_key,
            )
        )
        if not registered:
            error_code = ctypes.get_last_error()
            self._error = GlobalHotkeyError(
                f"RegisterHotKey failed with Windows error {error_code}: "
                f"{ctypes.FormatError(error_code).strip()}"
            )
            self._ready.set()
            return

        self._ready.set()
        try:
            message = wintypes.MSG()
            while not self._stop_requested.is_set():
                result = user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if result <= 0:
                    break
                if message.message == WM_HOTKEY and message.wParam == HOTKEY_ID:
                    self._callback()
        finally:
            user32.UnregisterHotKey(None, HOTKEY_ID)

    def __enter__(self) -> "GlobalHotkey":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
