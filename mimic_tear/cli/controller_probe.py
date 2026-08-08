from __future__ import annotations

import argparse
from time import perf_counter, sleep

from mimic_tear.controller import ControllerState, VirtualController


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a known forward input through the HIDMaestro bridge."
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=2.0,
        help="How long to hold the left stick forward (default: 2 seconds).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Seconds to wait so you can focus the game (default: 3).",
    )
    parser.add_argument(
        "--armed",
        action="store_true",
        help="Required acknowledgement that this process will send live input.",
    )
    args = parser.parse_args()

    if not args.armed:
        parser.error("Pass --armed to enable live controller output.")
    if not 0.1 <= args.seconds <= 10.0:
        parser.error("--seconds must be between 0.1 and 10.0.")
    if not 0.0 <= args.delay <= 10.0:
        parser.error("--delay must be between 0 and 10.0.")

    state = ControllerState(left_y=-1.0)

    with VirtualController() as controller:
        print(
            f"Focus Elden Ring now; moving in {args.delay:g} seconds...",
            flush=True,
        )
        sleep(args.delay)
        deadline = perf_counter() + args.seconds
        print(
            f"Holding the virtual left stick forward for {args.seconds:g} seconds...",
            flush=True,
        )
        while perf_counter() < deadline:
            controller.apply(state)
            sleep(0.05)
        controller.reset()
        print("Controller returned to neutral.", flush=True)


if __name__ == "__main__":
    main()
