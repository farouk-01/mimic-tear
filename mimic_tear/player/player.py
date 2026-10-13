from collections.abc import Callable, Iterator
from threading import Event

from typing import Any

from tensordict import TensorDict
import torch

from data.models.gamepad import GamepadState
from data.process import ProcessedSample
from data.write.writers.gamepad import GamepadWriter
from mimic_tear.model.components import ControllerOutput
from mimic_tear.model import Policy, PolicyInputs
from utils.logging import Logger

type ProcessStream = Callable[..., Iterator[ProcessedSample]]


class Player:
    def __init__(
        self,
        *,
        model: Policy,
        process_stream: ProcessStream,
        gamepad: GamepadWriter,
        device: torch.device | str,
        button_threshold: float = 0.5,
        analog_gain: float = 1.0,
        logger: Logger,
        log_interval: int = 30,
    ) -> None:
        self.model = model
        self.process_stream = process_stream
        self.gamepad = gamepad
        self.device = torch.device(device)
        self.button_threshold = button_threshold
        self.analog_gain = analog_gain
        self.logger = logger
        self.log_interval = log_interval

    @torch.inference_mode()
    def run(
        self,
        *,
        stop_event: Event | None = None,
        active: Event | None = None,
    ) -> None:
        self.model.eval()
        state: Any = None
        was_active = False

        stream = iter(self.process_stream(stop_event=stop_event))

        try:
            sample = next(stream, None)
            if sample is None:
                return

            self._forward(sample, None)
            self.logger.info("Mimic Tear ready")

            while True:
                if active is not None and not active.is_set():
                    if was_active:
                        self.gamepad.write(GamepadState())
                        self.logger.info("Mimic Tear paused")
                        was_active = False

                    if not self._wait_until_active(active, stop_event):
                        return

                if not was_active:
                    state = None
                    was_active = True
                    self.logger.info("Mimic Tear active")

                sample = next(stream, None)
                if sample is None:
                    return

                output, state = self._forward(sample, state)

                self._write_output(output)

        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

    @staticmethod
    def _wait_until_active(
        active: Event,
        stop_event: Event | None,
        poll_interval: float = 0.1,
    ) -> bool:
        while not active.wait(timeout=poll_interval):
            if stop_event is not None and stop_event.is_set():
                return False

        return stop_event is None or not stop_event.is_set()

    def _forward(
        self,
        sample: ProcessedSample,
        state: Any,
    ) -> tuple[ControllerOutput, Any]:
        # each dataset is [1, ...] (one frame) -> [B=1, T=1, ...]
        batch = TensorDict(sample.datasets, batch_size=[1]).unsqueeze(0)
        batch = batch.to(self.device, non_blocking=True)

        inputs = PolicyInputs.from_batch(batch, presence=sample.presence)
        return self.model(inputs, state)

    def _write_output(self, output: ControllerOutput) -> None:
        raw_analog = output.analog[0, -1]
        buttons = torch.sigmoid(output.button_logits[0, -1])

        analog = (raw_analog * self.analog_gain).clamp(-1.0, 1.0)

        gamepad_state = GamepadState.from_values(
            analog=analog.tolist(),
            buttons=(buttons >= self.button_threshold).tolist(),
        )

        self.gamepad.write(gamepad_state)

    # def _log_output(self, output: ControllerOutput) -> None:
    #     raw_analog = output.analog[0, -1]
    #     buttons = torch.sigmoid(output.button_logits[0, -1])

    #     analog = (raw_analog * self.analog_gain).clamp(-1.0, 1.0)

        