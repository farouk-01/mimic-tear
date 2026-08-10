from dataclasses import fields

from controller.inputs import ANALOG_INPUTS, BUTTON_INPUTS
from controller.state import AnalogState, ButtonState


def test_analog_state_order_matches_inputs() -> None:
    field_names = tuple(
        field.name
        for field in fields(AnalogState)
    )

    assert field_names == ANALOG_INPUTS


def test_button_state_order_matches_inputs() -> None:
    field_names = tuple(
        field.name
        for field in fields(ButtonState)
    )

    assert field_names == BUTTON_INPUTS