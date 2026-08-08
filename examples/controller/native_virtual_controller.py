"""Minimal AI-output example for the native SDL3 virtual controller."""

from time import sleep

import ai_controller


def main() -> None:
    controller = ai_controller.VirtualController()
    state = ai_controller.ControllerState()

    print(f"Created {controller.name} (SDL id {controller.instance_id})")
    try:
        # Hold the left stick forward for one second.
        state.left_y = -1.0
        controller.apply(state)
        sleep(1.0)

        # Tap the south (A/Cross) button for 100 ms.
        state.left_y = 0.0
        state.south = True
        controller.apply(state)
        sleep(0.1)
    finally:
        # Never leave an AI action held when inference exits or raises.
        controller.reset()


if __name__ == "__main__":
    main()
