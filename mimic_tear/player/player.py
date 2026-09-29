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
    ) -> None:
        self.model.eval()
        state: Any = None

        for sample in self.process_stream(stop_event=stop_event):
            # each dataset is [1, ...] (one frame) -> [B=1, T=1, ...]
            batch = TensorDict(sample.datasets, batch_size=[1]).unsqueeze(0)
            batch = batch.to(self.device, non_blocking=True)

            inputs = PolicyInputs.from_batch(batch, presence=sample.presence)
            output, state = self.model(inputs, state)

            self._write_output(output)

    def _write_output(self, output: ControllerOutput) -> None:
        raw_analog = output.analog[0, -1]
        buttons = torch.sigmoid(output.button_logits[0, -1])

        analog = (raw_analog * self.analog_gain).clamp(-1.0, 1.0)

        gamepad_state = GamepadState.from_values(
            analog=analog.tolist(),
            buttons=(buttons >= self.button_threshold).tolist(),
        )

        self.gamepad.write(gamepad_state)