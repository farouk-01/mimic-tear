from __future__ import annotations
from time import sleep
import torch
from pathlib import Path
from typing import TYPE_CHECKING
from threading import Event

from game_state.elden_ring.reader import EldenRingGameStateReader
from mimic_tear.model.policy import EldenRingPolicy
from mimic_tear.player.player import Player
from utils.logging.logger import Logger
from mimic_tear.training import Trainer
from mimic_tear.training.checkpoint import load_checkpoint, save_checkpoint
from utils.logging.profiling import Profiler
from recording.session import RecordingSession
from capture.screen.reader import ScreenReader
from recording.writers.gamepad import GamepadWriter
from data.transforms.frames import FrameTransform
from data.transforms.game_state import GameStateTransform, GameStateTransformConfig

if TYPE_CHECKING:
    from configs.config import MimicTearConfig


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
        self.device = device
        self._optimizer = optimizer
        self._trainer: Trainer | None = None
        self.logger = Logger(**config.regular_logging.model_dump())
        self.perf_logger = Logger(**config.perf_logging.model_dump())

        self.profiler = Profiler(
            config.profiling,
            logger=self.perf_logger,
            device=self.device,
        )

        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(device)
            self.logger.info("Device: %s (%s)", device, gpu_name)
        else:
            self.logger.info("Device: %s", device)

    @property
    def trainer(self) -> Trainer:  # lazy trainer
        if self._trainer is None:
            self.logger.info("Initializing trainer on %s...", self.device)
            self._trainer = Trainer(
                config=self.config.policy,
                hyperparameters=self.hyperparameters,
                device=self.device,
                optimizer=self._optimizer,
                data_loader_config=self.config.data_loader,
            )
        return self._trainer

    def mimic(self) -> None:
        self.logger.info("Starting training...")

        train_datasets, val_datasets, game_state_transform = self.config.load_recordings()
        self.logger.info("Training recordings: %d", len(train_datasets))
        self.logger.info("Validation recordings: %d", len(val_datasets))

        best_val_loss = float("inf")

        trainer = self.trainer  # initialize trainer

        best_val_loss = float("inf")
        early_stopping_best = float("inf")
        epochs_without_improvement = 0
        early_stopping = self.hyperparameters.early_stopping

        for epoch in range(1, self.hyperparameters.epochs + 1):
            self.logger.info(
                "Training epoch %d/%d...", epoch, self.hyperparameters.epochs
            )
            train_metrics = self.trainer.train_epoch(train_datasets)

            self.logger.info(
                "Validating epoch %d/%d...", epoch, self.hyperparameters.epochs
            )
            val_metrics = self.trainer.validate(val_datasets)

            metadata = {
                "validation_loss": val_metrics.total_loss,
                "network_hyperparameters": self.hyperparameters.model_dump(),
                "policy_config": self.config.policy.model_dump(),
                "sequence_length": self.hyperparameters.sequence_length,
                "game_state_transform": game_state_transform.model_dump(),
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

            if not early_stopping.enabled:
                continue

            improvement = early_stopping_best - val_metrics.total_loss

            if improvement > early_stopping.min_delta:
                early_stopping_best = val_metrics.total_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping.patience:
                self.logger.info(
                    "Early stopping at epoch %d after %d epochs without improvement.",
                    epoch,
                    epochs_without_improvement,
                )
                self.logger.info(
                    "Best validation loss: %.4f at epoch %d",
                    early_stopping_best,
                    epoch - epochs_without_improvement,
                )
                break

    def train(self) -> None:
        self.mimic()

    def record(
        self,
        *,
        name: str | None = None,
        seconds: float | None = None,
    ) -> Path:
        self.logger.info("Starting gameplay recording...")

        output = RecordingSession(config=self.config).run(name=name, seconds=seconds)

        self.logger.info("Recording saved to %s", output)

        return output

    def summon(
        self,
        *,
        stop_event: Event | None = None,
    ) -> None:
        input("Press Enter to summon (Ctrl+C to dismiss)")
        self.logger.info("Summoning Mimic Tear...")
        sleep(3.0)

        model = EldenRingPolicy(config=self.config.policy).to(self.device)

        checkpoint = load_checkpoint(
            self.config.artifacts_directory / "best.pt",
            model=model,
        )

        metadata = checkpoint["metadata"]

        game_state_transform_config = GameStateTransformConfig.model_validate(
            metadata["game_state_transform"]
        )

        frame_transform = FrameTransform(**self.config.transform_frames.model_dump())
        game_state_transform = GameStateTransform(
            **game_state_transform_config.model_dump()
        )

        screen = ScreenReader(**self.config.capture_screen.model_dump())
        game_state_reader = EldenRingGameStateReader.open(self.config.game_state)
        gamepad = GamepadWriter()

        grace_logger = Logger(**self.config.grace_logging.model_dump())

        player = Player(
            model=model,
            screen=screen,
            gamepad=gamepad,
            frame_transform=frame_transform,
            game_state_reader=game_state_reader,
            game_state_transform=game_state_transform,
            game_state_features=self.config.game_state.schema_.features,
            device=self.device,
            fps=self.config.video_config.fps,
            logger=grace_logger,
        )

        try:
            player.run()
        except KeyboardInterrupt:
            self.logger.info("Mimic Tear dismissed.")
        finally:
            game_state_reader.close()
            screen.close()
            gamepad.close()

    def eval(
        self,
        *,
        stop_event: Event | None = None,
    ) -> None:
        self.summon(stop_event=stop_event)
