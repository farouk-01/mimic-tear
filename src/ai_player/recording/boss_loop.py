from __future__ import annotations

import ctypes
import os
import signal
import time
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from ai_player.game_state import (
    EldenRingMemoryProfile,
    EldenRingStateReader,
    GameStateSnapshot,
    load_memory_profile,
)
from ai_player.platform.windows.process_memory import find_process_id
from ai_player.recording.record import RecorderConfig, RecordingCancelled


VK_CONTROL = 0x11
VK_E = 0x45
VK_O = 0x4F
VK_P = 0x50
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
SCAN_E = 0x12
SW_RESTORE = 9


class StateReader(Protocol):
    def read(self) -> GameStateSnapshot: ...

    def close(self) -> None: ...


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTUNION(ctypes.Union):
    # Windows validates cbSize against the complete INPUT structure. Even for
    # keyboard events, the union must be large enough for MOUSEINPUT.
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTUNION)]


def next_episode_tag(
    output_directory: Path,
    *,
    split: str,
    theme: str,
    base_tag: str,
) -> str:
    theme_directory = output_directory / split / Path(*theme.split("/"))
    episode = 1
    while (theme_directory / f"{base_tag}-{episode:04d}").exists():
        episode += 1
    return f"{base_tag}-{episode:04d}"


def _keyboard_input(
    virtual_key: int,
    *,
    key_up: bool,
    scan_code: int | None = None,
) -> INPUT:
    flags = KEYEVENTF_KEYUP if key_up else 0
    if scan_code is not None:
        flags |= KEYEVENTF_SCANCODE
    return INPUT(
        type=INPUT_KEYBOARD,
        ki=KEYBDINPUT(
            wVk=0 if scan_code is not None else virtual_key,
            wScan=scan_code or 0,
            dwFlags=flags,
            time=0,
            dwExtraInfo=0,
        ),
    )


def _send_keyboard_event(
    virtual_key: int,
    *,
    key_up: bool,
    scan_code: int | None = None,
) -> None:
    event = _keyboard_input(virtual_key, key_up=key_up, scan_code=scan_code)
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SendInput.argtypes = [
        wintypes.UINT,
        ctypes.POINTER(INPUT),
        ctypes.c_int,
    ]
    user32.SendInput.restype = wintypes.UINT
    sent = int(user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT)))
    if sent != 1:
        error_code = ctypes.get_last_error()
        raise RuntimeError(
            f"SendInput sent {sent}/1 keyboard events "
            f"(Windows error {error_code}: {ctypes.FormatError(error_code).strip()})"
        )


def send_key_chord(
    *virtual_keys: int,
    hold_seconds: float = 0.075,
    scan_code: int | None = None,
) -> None:
    if os.name != "nt":
        raise RuntimeError("Boss reset keyboard automation requires Windows")
    if not virtual_keys:
        raise ValueError("At least one key is required")
    if hold_seconds <= 0:
        raise ValueError("Key hold duration must be greater than zero")

    pressed: list[int] = []
    try:
        for virtual_key in virtual_keys:
            _send_keyboard_event(
                virtual_key,
                key_up=False,
                scan_code=scan_code,
            )
            pressed.append(virtual_key)
            time.sleep(hold_seconds)
    finally:
        for virtual_key in reversed(pressed):
            _send_keyboard_event(
                virtual_key,
                key_up=True,
                scan_code=scan_code,
            )
            time.sleep(hold_seconds)


def focus_process_window(process_name: str, *, timeout_seconds: float = 2.0) -> None:
    if os.name != "nt":
        raise RuntimeError("Boss reset window automation requires Windows")
    process_id = find_process_id(process_name)
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    candidates: list[tuple[int, int]] = []

    enum_callback_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HWND,
        wintypes.LPARAM,
    )
    user32.EnumWindows.argtypes = [enum_callback_type, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    ]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetClientRect.restype = wintypes.BOOL
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.ShowWindow.restype = wintypes.BOOL
    user32.BringWindowToTop.argtypes = [wintypes.HWND]
    user32.BringWindowToTop.restype = wintypes.BOOL
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.restype = wintypes.BOOL

    @enum_callback_type
    def collect_window(window: int, _: int) -> bool:
        if not user32.IsWindowVisible(window):
            return True
        owner_process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(window, ctypes.byref(owner_process_id))
        if int(owner_process_id.value) != process_id:
            return True
        rectangle = wintypes.RECT()
        if not user32.GetClientRect(window, ctypes.byref(rectangle)):
            return True
        area = max(0, rectangle.right - rectangle.left) * max(
            0, rectangle.bottom - rectangle.top
        )
        candidates.append((area, int(window)))
        return True

    user32.EnumWindows(collect_window, 0)
    if not candidates:
        raise RuntimeError(f"No visible window belongs to {process_name}")
    window = max(candidates)[1]
    if int(user32.GetForegroundWindow()) == window:
        return

    user32.ShowWindow(window, SW_RESTORE)
    user32.BringWindowToTop(window)
    user32.SetForegroundWindow(window)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if int(user32.GetForegroundWindow()) == window:
            return
        time.sleep(0.05)
    raise RuntimeError(
        "Could not focus Elden Ring. Click the game once and press the reset "
        "hotkey again."
    )


def wait_for_state(
    reader: StateReader,
    *,
    valid: bool,
    timeout_seconds: float,
    consecutive_reads: int,
    description: str,
    poll_seconds: float = 0.2,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    deadline = monotonic() + timeout_seconds
    matching_reads = 0
    last_errors: tuple[str, ...] = ()
    while monotonic() < deadline:
        snapshot = reader.read()
        last_errors = snapshot.read_errors
        if snapshot.valid is valid:
            matching_reads += 1
            if matching_reads >= consecutive_reads:
                return
        else:
            matching_reads = 0
        sleep(poll_seconds)
    details = f" Last read: {'; '.join(last_errors[:2])}" if last_errors else ""
    raise TimeoutError(f"Timed out waiting for {description}.{details}")


def continue_into_restored_save(
    reader: StateReader,
    *,
    process_name: str,
    timeout_seconds: float,
    focus_game: Callable[[str], None],
    send_chord: Callable[..., None],
    enter_interval_seconds: float = 1.5,
    poll_seconds: float = 0.2,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    if enter_interval_seconds <= 0:
        raise ValueError("Confirm-key interval must be greater than zero")

    enter_count = 3
    for index in range(enter_count):
        focus_game(process_name)
        send_chord(VK_E, scan_code=SCAN_E)
        if index < enter_count - 1:
            sleep(enter_interval_seconds)

    wait_for_state(
        reader,
        valid=True,
        timeout_seconds=timeout_seconds,
        consecutive_reads=3,
        description="the restored save to enter gameplay after three E presses",
        poll_seconds=poll_seconds,
        monotonic=monotonic,
        sleep=sleep,
    )
    return enter_count


def reset_boss_attempt(
    profile_path: Path,
    *,
    timeout_seconds: float,
    title_settle_seconds: float,
    snapshot_delay_seconds: float,
    gameplay_settle_seconds: float,
    reader_factory: Callable[[EldenRingMemoryProfile], StateReader]
    | None = None,
    focus_game: Callable[[str], None] | None = None,
    send_chord: Callable[..., None] = send_key_chord,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    profile = load_memory_profile(profile_path)
    make_reader = reader_factory or EldenRingStateReader.open
    focus = focus_game or focus_process_window
    reader = make_reader(profile)
    try:
        initial = reader.read()
        if not initial.valid:
            details = "; ".join(initial.read_errors[:2])
            raise RuntimeError(
                "Elden Ring gameplay state is not readable before reset. "
                f"Load into the game world first. {details}"
            )

        print("Reset: quitting to the title screen (P)...")
        focus(profile.process_name)
        send_chord(VK_P)
        wait_for_state(
            reader,
            valid=False,
            timeout_seconds=timeout_seconds,
            consecutive_reads=2,
            description="Elden Ring to leave gameplay",
            monotonic=monotonic,
            sleep=sleep,
        )
        sleep(title_settle_seconds)

        print("Reset: loading the selected practice-tool savefile (Ctrl+O)...")
        focus(profile.process_name)
        send_chord(VK_CONTROL, VK_O)
        sleep(snapshot_delay_seconds)

        print("Reset: continuing into the restored save (three E presses)...")
        enter_count = continue_into_restored_save(
            reader,
            process_name=profile.process_name,
            timeout_seconds=timeout_seconds,
            focus_game=focus,
            send_chord=send_chord,
            monotonic=monotonic,
            sleep=sleep,
        )
        print(f"Reset: gameplay loaded after {enter_count} E press(es).")
        sleep(gameplay_settle_seconds)
    finally:
        reader.close()


def run_boss_recording_loop(
    config: RecorderConfig,
    *,
    record_once: Callable[[RecorderConfig], Path],
) -> tuple[Path, ...]:
    if not config.boss_loop:
        raise ValueError("Boss recording loop requires --boss-loop")

    completed: list[Path] = []
    interrupt_requested = False
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(signum: int, frame: object) -> None:
        nonlocal interrupt_requested
        interrupt_requested = True
        signal.default_int_handler(signum, frame)

    print(
        f"Boss loop ready. Press {config.boss_reset_hotkey} after each attempt "
        "to save it, restore the selected practice-tool savefile, and continue."
    )
    print(f"Press {config.cancel_hotkey} to discard the current attempt and stop.")
    signal.signal(signal.SIGINT, handle_sigint)
    try:
        while config.boss_episodes is None or len(completed) < config.boss_episodes:
            episode_tag = next_episode_tag(
                config.output_directory,
                split=config.split,
                theme=config.theme,
                base_tag=config.tag,
            )
            episode_config = replace(
                config,
                tag=episode_tag,
                stop_hotkey=config.boss_reset_hotkey,
                open_replay_after_recording=False,
            )
            print(f"\nStarting boss attempt {episode_tag}")
            completed.append(record_once(episode_config))

            if interrupt_requested or (
                config.boss_episodes is not None
                and len(completed) >= config.boss_episodes
            ):
                break
            reset_boss_attempt(
                config.boss_reset_profile,
                timeout_seconds=config.boss_reset_timeout_seconds,
                title_settle_seconds=config.boss_title_settle_seconds,
                snapshot_delay_seconds=config.boss_snapshot_delay_seconds,
                gameplay_settle_seconds=config.boss_gameplay_settle_seconds,
            )
    except (KeyboardInterrupt, RecordingCancelled):
        print("Boss recording loop stopped.")
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
    return tuple(completed)
