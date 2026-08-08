from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np

from mimic_tear.controller.state import ControllerState
from mimic_tear.recording.schema import BUTTON_COLUMNS


CONTROLLER_LAYOUT_WIDTH = 330
CONTROLLER_LAYOUT_HEIGHT = 360
ControllerInput = Mapping[str, object] | ControllerState


def render_controller_layout(inputs: ControllerInput) -> np.ndarray:
    """Render the controller layout for a recording row or controller state."""

    canvas = np.zeros(
        (CONTROLLER_LAYOUT_HEIGHT, CONTROLLER_LAYOUT_WIDTH, 3),
        dtype=np.uint8,
    )
    draw_controller_layout(canvas, inputs)
    return canvas


def draw_controller_layout(
    image: np.ndarray,
    inputs: ControllerInput,
    *,
    origin: tuple[int, int] = (0, 0),
) -> None:
    """Draw controller inputs onto an image using the shared replay layout.

    ``inputs`` may be a :class:`ControllerState` or any mapping containing the
    controller field names. This keeps recording rows and live AI output on the
    same rendering interface.
    """

    origin_x, origin_y = origin
    _draw_stick(
        image,
        center=(origin_x + 82, origin_y + 130),
        x=float(_value(inputs, "left_x")),
        y=float(_value(inputs, "left_y")),
        label="LEFT",
    )
    _draw_stick(
        image,
        center=(origin_x + 242, origin_y + 130),
        x=float(_value(inputs, "right_x")),
        y=float(_value(inputs, "right_y")),
        label="RIGHT",
    )
    _draw_trigger(
        image,
        x=origin_x + 18,
        y=origin_y + 195,
        width=135,
        value=float(_value(inputs, "left_trigger")),
        label="LT",
    )
    _draw_trigger(
        image,
        x=origin_x + 177,
        y=origin_y + 195,
        width=135,
        value=float(_value(inputs, "right_trigger")),
        label="RT",
    )

    for index, button in enumerate(BUTTON_COLUMNS):
        column = index % 2
        line = index // 2
        active = bool(_value(inputs, button))
        _draw_text(
            image,
            ("> " if active else "  ") + button,
            origin_x + 12 + column * 160,
            origin_y + 238 + line * 17,
            color=(80, 255, 120) if active else (140, 140, 140),
            scale=0.38,
        )


def _value(inputs: ControllerInput, name: str) -> object:
    if isinstance(inputs, Mapping):
        return inputs[name]
    return getattr(inputs, name)


def _draw_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    color: tuple[int, int, int],
    scale: float,
) -> None:
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_stick(
    image: np.ndarray,
    *,
    center: tuple[int, int],
    x: float,
    y: float,
    label: str,
) -> None:
    radius = 38
    cv2.circle(image, center, radius, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.line(
        image,
        (center[0] - radius, center[1]),
        (center[0] + radius, center[1]),
        (55, 55, 55),
        1,
    )
    cv2.line(
        image,
        (center[0], center[1] - radius),
        (center[0], center[1] + radius),
        (55, 55, 55),
        1,
    )
    dot = (
        center[0] + round(max(-1.0, min(1.0, x)) * (radius - 5)),
        center[1] + round(max(-1.0, min(1.0, y)) * (radius - 5)),
    )
    cv2.circle(image, dot, 5, (80, 230, 80), -1, cv2.LINE_AA)
    _draw_text(
        image,
        label,
        center[0] - 18,
        center[1] + radius + 16,
        color=(230, 230, 230),
        scale=0.36,
    )


def _draw_trigger(
    image: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    value: float,
    label: str,
) -> None:
    value = max(0.0, min(1.0, value))
    cv2.rectangle(image, (x, y), (x + width, y + 13), (100, 100, 100), 1)
    cv2.rectangle(
        image,
        (x + 1, y + 1),
        (x + 1 + round((width - 2) * value), y + 12),
        (80, 210, 240),
        -1,
    )
    _draw_text(
        image,
        f"{label} {value:.2f}",
        x,
        y - 5,
        color=(230, 230, 230),
        scale=0.36,
    )
