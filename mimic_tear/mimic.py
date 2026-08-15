from __future__ import annotations
import torch

from utils.logging import Logger
from configs.config import MimicTearConfig
from mimic_tear.training import Trainer
from mimic_tear.utils.timing import timed
from mimic_tear.training.checkpoint import save_checkpoint


class MimicTear:
    def __init__(
        self,
        *,
        config: MimicTearConfig,
        device: torch.device | str,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.config = config
        self.hyperparameters = config.hyperparameters
        self.trainer = Trainer(
            config=config.policy,
            hyperparameters=config.hyperparameters,
            device=device,
            optimizer=optimizer,
        )
        self.logger = Logger(**config.regular_logging.model_dump())
        self.perf_logger = Logger(**config.perf_logging.model_dump())

        self.logger.info(f"Device: %s", device)

    @timed("perf_logger")
    def train(self):
        self.logger.info("Starting training...")

        train_datasets, val_datasets = self.config.load_recordings()
        self.logger.info("Training recordings: %d", len(train_datasets))
        self.logger.info("Validation recordings: %d", len(val_datasets))

        best_val_loss = float("inf")

        for epoch in range(1, self.hyperparameters.epochs + 1):
            train_metrics = self.trainer.train_epoch(train_datasets)
            val_metrics = self.trainer.validate(val_datasets)

            metadata = {
                "validation_loss": val_metrics.total_loss,
                "network_hyperparameters": self.hyperparameters.model_dump(),
                "policy_config": self.config.policy.model_dump(),
                "sequence_length": self.hyperparameters.sequence_length,
            }

            save_checkpoint(
                self.config.artifacts_directory / "latest.pt",
                model=self.trainer.model,
                optimizer=self.trainer.optimizer,
                epoch=epoch,
                metadata=metadata,
            )

            if val_metrics.total_loss < best_val_loss:
                best_val_loss = val_metrics.total_loss
                save_checkpoint(
                    self.config.artifacts_directory / "best.pt",
                    model=self.trainer.model,
                    optimizer=self.trainer.optimizer,
                    epoch=epoch,
                    metadata=metadata,
                )

            self.logger.info(
                "Epoch %d/%d : Train Loss: %.4f, Validation Loss: %.4f",
                epoch,
                self.hyperparameters.epochs,
                train_metrics.total_loss,
                val_metrics.total_loss,
            )
