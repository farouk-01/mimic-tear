from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict
import torch
from tensordict import TensorDict
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from data.process import SequenceDataset
from data.models.gamepad import get_inputs_names_classified
from mimic_tear.model.loss import PolicyLoss
from mimic_tear.model.policy import LSTMPolicy
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


class Trainer:
    def __init__(
        self,
        *,
        model: LSTMPolicy,
        optimizer: torch.optim.Optimizer,
        loss: PolicyLoss,
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

        self.model = model
        self.optimizer = optimizer
        self.loss = loss

        self.device = torch.device(device)

        self.gradient_clip_norm = gradient_clip_norm

        self.use_amp = use_amp and self.device.type == "cuda"

        self.scaler = torch.GradScaler(self.device.type, enabled=self.use_amp)

        self.data_loader_config = data_loader_config

    @profile
    def _loader(self, recording: SequenceDataset) -> DataLoader[TensorDict]:
        return DataLoader(
            recording,
            collate_fn=lambda x: x,
            **self.data_loader_config.model_dump(),
        )

    @profile
    def train_epoch(self, recordings: Iterable[SequenceDataset]) -> EpochMetrics:
        self.model.train()
        metrics = EpochMetrics()

        for recording in recordings:
            state = None

            for sample in self._loader(recording):
                batch = Sampler.prepare(sample).to(self.device, non_blocking=True)

                frames: TensorDict = batch.get("frames")
                controller: TensorDict = batch.get("controller")

                images = frames.get("frames")

                analogs, buttons = get_inputs_names_classified()

                analogs = torch.stack(
                    [controller.get(name) for name in analogs],
                    dim=-1,
                )
                buttons = torch.stack(
                    [controller.get(name) for name in buttons],
                    dim=-1,
                )

                game_state = batch.get("game_state") if "game_state" in batch else None

                self.optimizer.zero_grad(set_to_none=True)
                
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    output, next_state = self.model(
                        images,
                        game_state=game_state,
                        state=state,
                    )

                    losses = self.loss(
                        output,
                        analog_target=analogs,
                        button_target=buttons,
                    )

                self.scaler.scale(losses.total).backward()

                if self.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)

                    clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

                self.scaler.step(self.optimizer)

                self.scaler.update()

                state = self.model.detach_state(next_state)

                metrics.update(
                    total=losses.total.item(),
                    analog=losses.analog.item(),
                    buttons=losses.buttons.item(),
                )

        return metrics.average()

    @profile
    def validate(self, recordings: Iterable[SequenceDataset]) -> EpochMetrics:
        self.model.eval()

        metrics = EpochMetrics()

        with torch.no_grad():
            for recording in recordings:
                state = None

                for sample in self._loader(recording):
                    batch = Sampler.prepare(sample).to(self.device, non_blocking=True)

                    frames: TensorDict = batch.get("frames")
                    controller: TensorDict = batch.get("controller")

                    images = frames.get("frames")

                    analogs, buttons = get_inputs_names_classified()

                    analogs = torch.stack(
                        [controller.get(name) for name in analogs],
                        dim=-1,
                    )
                    buttons = torch.stack(
                        [controller.get(name) for name in buttons],
                        dim=-1,
                    )

                    game_state = (
                        batch.get("game_state") if "game_state" in batch else None
                    )

                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.float16,
                        enabled=self.use_amp,
                    ):
                        output, state = self.model(
                            images,
                            game_state=game_state,
                            state=state,
                        )

                        losses = self.loss(
                            output,
                            analog_target=analogs,
                            button_target=buttons,
                        )

                    state = self.model.detach_state(state)

                    metrics.update(
                        total=losses.total.item(),
                        analog=losses.analog.item(),
                        buttons=losses.buttons.item(),
                    )

        return metrics.average()
