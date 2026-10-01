from __future__ import annotations

import ctypes
from collections.abc import Callable
from threading import Event, Thread
from typing import Final


VK_F8: Final = 0x77
VK_F9: Final = 0x78
VK_F10: Final = 0x79
VK_F11: Final = 0x7A

_POLL_INTERVAL_SECONDS: Final = 0.05
_KEY_DOWN_MASK: Final = 0x8000


def register_hotkey(
    virtual_key: int,
    callback: Callable[[], None],
) -> Event:
    stop_event = Event()

    thread = Thread(
        target=_watch_key,
        args=(virtual_key, callback, stop_event),
        daemon=True,
    )
    thread.start()

    return stop_event


def unregister_hotkey(stop_event: Event) -> None:
    stop_event.set()


def _watch_key(
    virtual_key: int,
    callback: Callable[[], None],
    stop_event: Event,
) -> None:
    user32 = ctypes.windll.user32

    was_pressed = False

    while not stop_event.wait(_POLL_INTERVAL_SECONDS):
        is_pressed = bool(
            user32.GetAsyncKeyState(virtual_key) & _KEY_DOWN_MASK
        )

        if is_pressed and not was_pressed:
            callback()

        was_pressed = is_pressed