from data.process.transforms.tensor import Clamp, TensorTransform, ConsecutiveCount
from data.models.gamepad import BUTTON_INPUTS
from data.constants import MAX_HELD


def held_buttons_transform() -> tuple[TensorTransform, ...]:
    transforms = []

    for btn in BUTTON_INPUTS:

        held = ConsecutiveCount(
            input=btn,
            output=f"{btn}_held",
            max_count=MAX_HELD,
            shift=1,
        )

        transforms.append(held)

    return tuple(transforms)


GAMEPAD_TRANSFORMS: tuple[TensorTransform, ...] = (
    Clamp(
        output="left_x",
        input="left_x",
        min=-1.0,
        max=1.0,
    ),
    Clamp(
        output="left_y",
        input="left_y",
        min=-1.0,
        max=1.0,
    ),
    Clamp(
        output="right_x",
        input="right_x",
        min=-1.0,
        max=1.0,
    ),
    Clamp(
        output="right_y",
        input="right_y",
        min=-1.0,
        max=1.0,
    ),
    Clamp(
        output="left_trigger",
        input="left_trigger",
        min=0.0,
        max=1.0,
    ),
    Clamp(
        output="right_trigger",
        input="right_trigger",
        min=0.0,
        max=1.0,
    ),
    *held_buttons_transform(),
)
