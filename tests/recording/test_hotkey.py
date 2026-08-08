from __future__ import annotations

import unittest

from ai_player.platform.windows.global_hotkey import parse_hotkey


class HotkeyTests(unittest.TestCase):
    def test_function_key_and_modifier_parse(self) -> None:
        f8 = parse_hotkey("F8")
        self.assertIsNotNone(f8)
        self.assertEqual(f8.label, "F8")
        self.assertEqual(f8.virtual_key, 0x77)

        ctrl_f8 = parse_hotkey("ctrl+f8")
        self.assertIsNotNone(ctrl_f8)
        self.assertEqual(ctrl_f8.label, "CTRL+F8")
        self.assertNotEqual(ctrl_f8.modifiers, f8.modifiers)

    def test_none_disables_global_registration(self) -> None:
        self.assertIsNone(parse_hotkey("NONE"))
        self.assertIsNone(parse_hotkey("off"))

    def test_unsupported_bare_letter_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_hotkey("Q")


if __name__ == "__main__":
    unittest.main()
