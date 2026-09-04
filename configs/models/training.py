from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from mimic_tear.training.trainer import DataLoaderConfig
from mimic_tear.training import Hyperparameters


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    hyperparameters: Hyperparameters
    data_loader: DataLoaderConfig

    @classmethod
    def load(cls, raw_training: dict) -> TrainingConfig:
        return cls(
            hyperparameters=raw_training["hyperparameters"],
            data_loader=raw_training["data_loader"],
        )
