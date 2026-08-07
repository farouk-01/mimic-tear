from __future__ import annotations

import ctypes
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from ai_player.game_state import GameStateSnapshot  # noqa: E402
from recorder.boss_loop import (  # noqa: E402
    INPUT,
    KEYEVENTF_SCANCODE,
    SCAN_E,
    VK_CONTROL,
    VK_E,
    VK_O,
    VK_P,
    _keyboard_input,
    continue_into_restored_save,
    next_episode_tag,
    reset_boss_attempt,
    send_key_chord,
)
from recorder.record import parse_args  # noqa: E402


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeReader:
    def __init__(self, validity: list[bool]) -> None:
        self.validity = iter(validity)
        self.closed = False

    def read(self) -> GameStateSnapshot:
        valid = next(self.validity)
        return GameStateSnapshot(
            values={},
            valid=valid,
            read_errors=() if valid else ("player_health: unavailable",),
        )

    def close(self) -> None:
        self.closed = True


class BossLoopTests(unittest.TestCase):
    def test_windows_input_structure_has_the_required_native_size(self) -> None:
        expected_size = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
        self.assertEqual(ctypes.sizeof(INPUT), expected_size)

    def test_key_chord_holds_modifier_while_tapping_key(self) -> None:
        with (
            patch("recorder.boss_loop._send_keyboard_event") as send_event,
            patch("recorder.boss_loop.time.sleep") as sleep,
        ):
            send_key_chord(VK_CONTROL, VK_O, hold_seconds=0.075)

        self.assertEqual(
            [call.args for call in send_event.call_args_list],
            [(VK_CONTROL,), (VK_O,), (VK_O,), (VK_CONTROL,)],
        )
        self.assertEqual(
            [call.kwargs for call in send_event.call_args_list],
            [
                {"key_up": False, "scan_code": None},
                {"key_up": False, "scan_code": None},
                {"key_up": True, "scan_code": None},
                {"key_up": True, "scan_code": None},
            ],
        )
        self.assertEqual([call.args for call in sleep.call_args_list], [(0.075,)] * 4)

    def test_scan_code_input_uses_physical_key_events(self) -> None:
        event = _keyboard_input(VK_E, key_up=False, scan_code=SCAN_E)

        self.assertEqual(event.ki.wVk, 0)
        self.assertEqual(event.ki.wScan, SCAN_E)
        self.assertEqual(event.ki.dwFlags, KEYEVENTF_SCANCODE)

    def test_next_episode_tag_skips_existing_recordings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            session_root = output / "train" / "combat" / "tutorial"
            (session_root / "soldier-0001").mkdir(parents=True)
            (session_root / "soldier-0002").mkdir()

            tag = next_episode_tag(
                output,
                split="train",
                theme="combat/tutorial",
                base_tag="soldier",
            )

        self.assertEqual(tag, "soldier-0003")

    def test_continue_presses_confirm_exactly_three_times_then_waits(self) -> None:
        reader = FakeReader([False, False, True, True, True])
        clock = FakeClock()
        focused: list[str] = []
        chords: list[tuple[int, ...]] = []

        count = continue_into_restored_save(
            reader,
            process_name="eldenring.exe",
            timeout_seconds=10.0,
            focus_game=focused.append,
            send_chord=lambda *keys, **_: chords.append(keys),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

        self.assertEqual(count, 3)
        self.assertEqual(chords, [(VK_E,)] * 3)
        self.assertEqual(focused, ["eldenring.exe"] * 3)
        self.assertEqual(clock.sleeps.count(1.5), 2)

    def test_boss_loop_options_are_opt_in(self) -> None:
        required = [
            "--theme",
            "combat",
            "--tag",
            "soldier",
            "--split",
            "train",
        ]
        normal = parse_args(required)
        self.assertFalse(normal.boss_loop)
        self.assertTrue(normal.open_replay_after_recording)

        loop = parse_args([*required, "--boss-loop", "--boss-episodes", "3"])
        self.assertTrue(loop.boss_loop)
        self.assertEqual(loop.boss_episodes, 3)
        self.assertEqual(loop.boss_reset_hotkey, "F10")
        self.assertEqual(loop.boss_snapshot_delay_seconds, 3.0)

    def test_reset_waits_for_title_then_restored_gameplay(self) -> None:
        reader = FakeReader([True, False, False, True, True, True])
        clock = FakeClock()
        focused: list[str] = []
        chords: list[tuple[int, ...]] = []
        profile = SimpleNamespace(process_name="eldenring.exe")

        with patch("recorder.boss_loop.load_memory_profile", return_value=profile):
            reset_boss_attempt(
                Path("profile.json"),
                timeout_seconds=10.0,
                title_settle_seconds=2.0,
                snapshot_delay_seconds=1.0,
                gameplay_settle_seconds=3.0,
                reader_factory=lambda _: reader,
                focus_game=focused.append,
                send_chord=lambda *keys, **_: chords.append(keys),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )

        self.assertEqual(
            chords,
            [
                (VK_P,),
                (VK_CONTROL, VK_O),
                (VK_E,),
                (VK_E,),
                (VK_E,),
            ],
        )
        self.assertEqual(focused, ["eldenring.exe"] * 5)
        self.assertTrue(reader.closed)
        self.assertIn(2.0, clock.sleeps)
        self.assertIn(1.0, clock.sleeps)
        self.assertIn(3.0, clock.sleeps)


if __name__ == "__main__":
    unittest.main()
