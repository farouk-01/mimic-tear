from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, overload

from pydantic import BaseModel, ConfigDict
import torch
from torch import Tensor
from tensordict import TensorDict
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from data.process import ProcessedRecording, SequenceDataset
from data.models.gamepad import get_inputs_names_classified
from mimic_tear.model import Policy, PolicyInputs
from mimic_tear.model.components.loss import GamepadLoss
from mimic_tear.model.components import ControllerOutput

from utils import profile


class DataLoaderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    batch_size: int | None = None
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: bool = False


@dataclass(slots=True)
class EpochMetrics:
    total_loss: float = 0.0
    analog_loss: float = 0.0
    button_loss: float = 0.0
    steps: int = 0

    def update(
        self,
        *,
        total: float,
        analog: float,
        buttons: float,
    ) -> None:
        self.total_loss += total
        self.analog_loss += analog
        self.button_loss += buttons
        self.steps += 1

    def average(self) -> EpochMetrics:
        if self.steps == 0:
            return EpochMetrics()

        return EpochMetrics(
            total_loss=self.total_loss / self.steps,
            analog_loss=self.analog_loss / self.steps,
            button_loss=self.button_loss / self.steps,
            steps=self.steps,
        )


class Sampler:
    @profile
    @staticmethod
    def prepare(sample: TensorDict) -> TensorDict:
        # [T, ...] -> [1, T, ...]
        return sample.unsqueeze(0)


type GamepadPredictions = tuple[ControllerOutput, torch.Tensor, torch.Tensor]


class Trainer:
    def __init__(
        self,
        *,
        policy: Policy,
        optimizer: torch.optim.Optimizer,
        loss: GamepadLoss,
        device: str | torch.device,
        gradient_clip_norm: float | None,
        use_amp: bool,
        data_loader_config: DataLoaderConfig,
    ) -> None:
        if data_loader_config.batch_size is not None:
            raise NotImplementedError(
                "Batched training is not supported yet, " "batch_size must be None"
            )

        if data_loader_config.shuffle:
            raise NotImplementedError(
                "Shuffled training is not supported yet, " "shuffle must be False"
            )

        self.policy = policy
        self.optimizer = optimizer
        self.loss = loss

        self.device = torch.device(device)

        self.gradient_clip_norm = gradient_clip_norm

        self.use_amp = use_amp and self.device.type == "cuda"

        self.scaler = torch.GradScaler(self.device.type, enabled=self.use_amp)

        self.data_loader_config = data_loader_config

        torch.backends.cudnn.benchmark = True

    @profile
    def _loader(self, recording: SequenceDataset) -> DataLoader[TensorDict]:
        return DataLoader(
            recording,
            collate_fn=lambda x: x,
            **self.data_loader_config.model_dump(),
        )

    @profile
    def train_epoch(self, recordings: Iterable[ProcessedRecording]) -> EpochMetrics:
        self.policy.train()
        metrics = EpochMetrics()

        for recording in recordings:
            state = None

            dataset = recording.dataset
            presence_mask = recording.presence

            for sample in self._loader(dataset):
                batch = Sampler.prepare(sample).to(self.device, non_blocking=True)

                analog_targets, buttons_target = self._targets(batch)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    data = PolicyInputs.from_batch(batch, presence=presence_mask)
                    output, next_state = self.policy(data, state)

                    losses = self.loss(
                        output,
                        analog_target=analog_targets,
                        button_target=buttons_target,
                    )

                self.scaler.scale(losses.total).backward()

                if self.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)

                    clip_grad_norm_(self.policy.parameters(), self.gradient_clip_norm)

                self.scaler.step(self.optimizer)

                self.scaler.update()

                state = self.policy.detach_state(next_state)

                metrics.update(
                    total=losses.total.item(),
                    analog=losses.analog.item(),
                    buttons=losses.buttons.item(),
                )

        return metrics.average()

    @overload
    def validate(
        self,
        recordings: Iterable[ProcessedRecording],
        *,
        return_predictions: Literal[False] = False,
    ) -> EpochMetrics: ...

    @overload
    def validate(
        self,
        recordings: Iterable[ProcessedRecording],
        *,
        return_predictions: Literal[True],
    ) -> tuple[EpochMetrics, list[GamepadPredictions]]: ...

    @profile
    def validate(
        self,
        recordings: Iterable[ProcessedRecording],
        *,
        return_predictions: bool = False,
    ) -> EpochMetrics | tuple[EpochMetrics, list[GamepadPredictions]]:
        self.policy.eval()
        metrics = EpochMetrics()

        if return_predictions:
            self.predictions: list[GamepadPredictions] = []

        with torch.no_grad():
            for recording in recordings:
                state = None

                dataset = recording.dataset
                presence_mask = recording.presence

                for sample in self._loader(dataset):
                    batch = Sampler.prepare(sample).to(self.device, non_blocking=True)

                    analog_targets, buttons_target = self._targets(batch)

                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.float16,
                        enabled=self.use_amp,
                    ):
                        data = PolicyInputs.from_batch(batch, presence=presence_mask)
                        output, next_state = self.policy(data, state)

                        losses = self.loss(
                            output,
                            analog_target=analog_targets,
                            button_target=buttons_target,
                        )

                    state = self.policy.detach_state(next_state)

                    metrics.update(
                        total=losses.total.item(),
                        analog=losses.analog.item(),
                        buttons=losses.buttons.item(),
                    )

                    if return_predictions:
                        self.predictions.append((output, analog_targets, buttons_target))

        if return_predictions:
            return metrics.average(), self.predictions

        return metrics.average()

    @staticmethod
    def _targets(batch: TensorDict) -> tuple[Tensor, Tensor]:
        analog_targets, buttons_target = get_inputs_names_classified()

        analog_targets = torch.stack(
            [batch["controller", name] for name in analog_targets],
            dim=-1,
        )
        buttons_target = torch.stack(
            [batch["controller", name] for name in buttons_target],
            dim=-1,
        )

        return analog_targets, buttons_target