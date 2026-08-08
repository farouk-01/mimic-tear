from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mimic_tear.controller import (  # noqa: E402
    CONTROLLER_LAYOUT_HEIGHT,
    CONTROLLER_LAYOUT_WIDTH,
    ControllerState,
    VirtualController,
    render_controller_layout,
)
from mimic_tear.recording.schema import BUTTON_COLUMNS  # noqa: E402


class ControllerStateTests(unittest.TestCase):
    def test_predictions_follow_recording_column_order(self) -> None:
        analog = [-1.0, 0.5, 0.25, -0.25, 0.2, 0.8]
        buttons = [name in {"south", "right_bumper", "start"} for name in BUTTON_COLUMNS]

        state = ControllerState.from_predictions(analog, buttons)

        self.assertEqual(state.left_x, -1.0)
        self.assertEqual(state.right_trigger, 0.8)
        self.assertTrue(state.south)
        self.assertTrue(state.right_bumper)
        self.assertTrue(state.start)
        self.assertFalse(state.east)

    def test_analog_ranges_are_checked(self) -> None:
        with self.assertRaises(ValueError):
            ControllerState(left_x=1.1).validate()


class ControllerOverlayTests(unittest.TestCase):
    def test_layout_accepts_controller_state_and_recording_mapping(self) -> None:
        state = ControllerState(left_x=0.75, south=True)
        rendered_state = render_controller_layout(state)
        self.assertEqual(
            rendered_state.shape,
            (CONTROLLER_LAYOUT_HEIGHT, CONTROLLER_LAYOUT_WIDTH, 3),
        )
        self.assertGreater(int(rendered_state.sum()), 0)

        mapping = {
            name: getattr(state, name)
            for name in (
                "left_x",
                "left_y",
                "right_x",
                "right_y",
                "left_trigger",
                "right_trigger",
                *BUTTON_COLUMNS,
            )
        }
        rendered_mapping = render_controller_layout(mapping)
        self.assertEqual(rendered_mapping.shape, rendered_state.shape)


class BridgeClientTests(unittest.TestCase):
    def test_apply_writes_one_complete_state_message(self) -> None:
        stream = io.BytesIO()
        controller = VirtualController(_stream=stream)
        controller.apply(ControllerState(left_y=-1.0, south=True))

        message = json.loads(stream.getvalue().decode("utf-8"))
        self.assertEqual(message["type"], "state")
        self.assertEqual(message["left_y"], -1.0)
        self.assertTrue(message["south"])
        self.assertFalse(message["east"])


if __name__ == "__main__":
    unittest.main()
