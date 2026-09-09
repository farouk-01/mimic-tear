from collections.abc import Callable, Iterator
from threading import Event

import torch

from data.models.gamepad import GamepadState
from data.process import ProcessedSample
from data.write.writers.gamepad import GamepadWriter
from mimic_tear.model.components.controller import ControllerOutput
from mimic_tear.model.components.temporal import LSTMState
from mimic_tear.model.policy import LSTMPolicy
from utils.logging import Logger

type ProcessStream = Callable[..., Iterator[ProcessedSample]]


class Player:
    def __init__(
        self,
        *,
        model: LSTMPolicy,
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
    ) -> None:
        self.model.eval()
        state: LSTMState | None = None

        for i, sample in enumerate(self.process_stream(stop_event=stop_event)):
            video = sample.datasets["video"]["frames"]
            structured_data = sample.datasets["game_state"]
            presence_mask = sample.presence["game_state"]

            output, state = self.model(
                images=video.unsqueeze(0).to(self.device),
                structured_data={
                    name: tensor.unsqueeze(0).to(self.device)
                    for name, tensor in structured_data.items()
                },
                presence_mask={
                    name: tensor.to(self.device)
                    for name, tensor in presence_mask.items()
                },
                state=state,
            )

            self._write_output(output, log=(i % self.log_interval == 0))

    def _write_output(self, output: ControllerOutput, *, log: bool = False) -> None:
        raw_analog = output.analog[0, -1]
        buttons = torch.sigmoid(output.button_logits[0, -1])

        analog = (raw_analog * self.analog_gain).clamp(-1.0, 1.0)

        gamepad_state = GamepadState.from_values(
            analog=analog.tolist(),
            buttons=(buttons >= self.button_threshold).tolist(),
        )

        self.gamepad.write(gamepad_state)

        if log:
            self.logger.debug(
                "analog=%s buttons=%s",
                [round(value, 3) for value in raw_analog.tolist()],
                [round(value, 3) for value in buttons.tolist()],
            )
