from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from pydantic import BaseModel, ConfigDict
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from data.sequence import SequenceDataset, SequenceSample
from mimic_tear.model.loss import PolicyLoss
from mimic_tear.model.policy import PolicyConfig, EldenRingPolicy
from .hyperparameters import Hyperparameters
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


class TrainingBatch:
    def __init__(
        self,
        *,
        images: torch.Tensor,
        analog: torch.Tensor,
        buttons: torch.Tensor,
        game_state: torch.Tensor | None,
    ) -> None:
        self.images = images
        self.analog = analog
        self.buttons = buttons
        self.game_state = game_state


class Sampler:
    @classmethod
    def prepare(cls, device: torch.device, sample: SequenceSample) -> TrainingBatch:
        # images: [T, 3, H, W]
        #
        # Policy expects:
        # images: [B, T, 3, H, W]
        #
        # For now B=1 
        # B = batch
        # B1 -> B2 -> B3 
        #
        # T = sequence length
        # [T1, T2, T3] -> [T1, T2, T3] -> [T1, T2, T3]
        images = sample.images.unsqueeze(0).to(device, non_blocking=True)
        analog = sample.analog.unsqueeze(0).to(device, non_blocking=True)
        buttons = sample.buttons.unsqueeze(0).to(device, non_blocking=True)
        game_state = (
            sample.game_state.unsqueeze(0).to(device, non_blocking=True)
            if sample.game_state is not None
            else None
        )

        return TrainingBatch(
            images=images,
            analog=analog,
            buttons=buttons,
            game_state=game_state,
        )


class Trainer:
    def __init__(
        self,
        *,
        config: PolicyConfig,
        hyperparameters: Hyperparameters,
        device: torch.device | str,
        optimizer: torch.optim.Optimizer | None = None,
        data_loader_config: DataLoaderConfig,
    ) -> None:
        self.device = torch.device(device)
        self.hyperparams = hyperparameters
        self.model = EldenRingPolicy(config=config).to(self.device)

        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.AdamW(
                self.model.parameters(),
                lr=hyperparameters.learning_rate,
                weight_decay=hyperparameters.weight_decay,
            )
        )

        self.use_amp = hyperparameters.use_amp and self.device.type == "cuda"
        self.scaler = torch.GradScaler(self.device.type, enabled=self.use_amp)

        btn_weight = hyperparameters.controller_weights.button_weights
        analog_weight = hyperparameters.controller_weights.analog_weights
        self.loss = PolicyLoss(button_weight=btn_weight, analog_weight=analog_weight).to(self.device)
        self.data_loader_config = data_loader_config

    def _loader(self, recording: SequenceDataset) -> DataLoader:
        return DataLoader(
            recording,
            **self.data_loader_config.model_dump(),
        )

    @profile
    def train_epoch(self, recordings: Iterable[SequenceDataset]) -> EpochMetrics:
        self.model.train()
        metrics = EpochMetrics()

        for recording in recordings:
            state = None

            for sample in self._loader(recording):
                batch = Sampler.prepare(self.device, sample)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    output, next_state = self.model(
                        batch.images,
                        game_state=batch.game_state,
                        state=state,
                    )

                    loss_output = self.loss(
                        output,
                        analog_target=batch.analog,
                        button_target=batch.buttons,
                    )

                self.scaler.scale(loss_output.total).backward()

                if self.hyperparams.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(
                        self.model.parameters(),
                        self.hyperparams.gradient_clip_norm,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                state = self.model.detach_state(next_state)
                metrics.update(
                    total=loss_output.total.item(),
                    analog=loss_output.analog.item(),
                    buttons=loss_output.buttons.item(),
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
                    batch = Sampler.prepare(self.device, sample)

                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.float16,
                        enabled=self.use_amp,
                    ):
                        output, state = self.model(
                            batch.images,
                            game_state=batch.game_state,
                            state=state,
                        )

                        losses = self.loss(
                            output,
                            analog_target=batch.analog,
                            button_target=batch.buttons,
                        )

                    state = self.model.detach_state(state)

                    metrics.update(
                        total=losses.total.detach().item(),
                        analog=losses.analog.detach().item(),
                        buttons=losses.buttons.detach().item(),
                    )

        return metrics.average()
