"""Print raw SDL3 input from a connected physical controller."""

from time import sleep
import ctypes
from typing import cast

import sdl3


def main() -> None:
    if not sdl3.SDL_Init(sdl3.SDL_INIT_GAMEPAD):
        raise RuntimeError(
            f"SDL initialization failed: {sdl3.SDL_GetError()}"
        )

    gamepad = None

    try:
        gamepad_count = ctypes.c_int()
        gamepad_ids = sdl3.SDL_GetGamepads(ctypes.pointer(gamepad_count))

        if not gamepad_ids or gamepad_count.value == 0:
            raise RuntimeError(
                "No controller detected. Connect it, then restart the script."
            )

        gamepad = sdl3.SDL_OpenGamepad(gamepad_ids.contents)

        if not gamepad:
            raise RuntimeError(
                f"Could not open controller: {sdl3.SDL_GetError()}"
            )

        name = sdl3.SDL_GetGamepadName(gamepad)
        print(f"Controller detected: {name}")
        print("Move the sticks or press buttons. Press Ctrl+C to stop.")

        while True:
            sdl3.SDL_PumpEvents()

            left_x = sdl3.SDL_GetGamepadAxis(
                gamepad,
                cast(sdl3.SDL_GamepadAxis, sdl3.SDL_GAMEPAD_AXIS_LEFTX),
            )
            left_y = sdl3.SDL_GetGamepadAxis(
                gamepad,
                cast(sdl3.SDL_GamepadAxis, sdl3.SDL_GAMEPAD_AXIS_LEFTY),
            )
            right_x = sdl3.SDL_GetGamepadAxis(
                gamepad,
                cast(sdl3.SDL_GamepadAxis, sdl3.SDL_GAMEPAD_AXIS_RIGHTX),
            )
            right_y = sdl3.SDL_GetGamepadAxis(
                gamepad,
                cast(sdl3.SDL_GamepadAxis, sdl3.SDL_GAMEPAD_AXIS_RIGHTY),
            )

            south = sdl3.SDL_GetGamepadButton(
                gamepad,
                cast(sdl3.SDL_GamepadButton, sdl3.SDL_GAMEPAD_BUTTON_SOUTH),
            )
            east = sdl3.SDL_GetGamepadButton(
                gamepad,
                cast(sdl3.SDL_GamepadButton, sdl3.SDL_GAMEPAD_BUTTON_EAST),
            )

            print(
                "\r"
                f"L=({left_x:6d}, {left_y:6d}) "
                f"R=({right_x:6d}, {right_y:6d}) "
                f"south={int(south)} east={int(east)}",
                end="",
                flush=True,
            )

            sleep(1 / 30)

    finally:
        if gamepad:
            sdl3.SDL_CloseGamepad(gamepad)

        sdl3.SDL_Quit()


if __name__ == "__main__":
    main()
