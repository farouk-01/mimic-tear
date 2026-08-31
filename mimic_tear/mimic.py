from __future__ import annotations
from time import sleep
import torch
from pathlib import Path
from typing import TYPE_CHECKING
from threading import Event

from data import DataPipeline
from mimic_tear.model.loss import PolicyLoss
from mimic_tear.model.policy import LSTMPolicy

# from mimic_tear.player.player import Player
from utils.logging.logger import Logger
from mimic_tear.training import Trainer
from mimic_tear.training.checkpoint import load_checkpoint, save_checkpoint
from utils.logging.profiling import Profiler

if TYPE_CHECKING:
    from configs.config import MimicTearConfig


class MimicTear:
    def __init__(
        self,
        *,
        config: MimicTearConfig,
        device: torch.device | str,
    ) -> None:
        self.config = config
        self.device = torch.device(device)

        self.logger = Logger(**config.logging.regular.model_dump())
        self.perf_logger = Logger(**config.logging.performance.model_dump())
        self.profiler = Profiler(
            config.logging.profiling,
            logger=self.perf_logger,
            device=self.device,
        )

        self.hyperparams = config.training.hyperparameters

        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(self.device)
            self.logger.info("Device: %s (%s)", self.device, gpu_name)
        else:
            self.logger.info("Device: %s", self.device)

        self.data_pipeline = DataPipeline(
            capture_config=config.data.capture,
            process_config=config.data.process,
            writer_config=config.data.writer,
        )

    def mimic(self, *, discover_encodings: bool = False) -> None:
        self.logger.info("Starting training...")

        if discover_encodings:
            self.logger.info("Discovering encodings...")
            # TODO : timer
            self.data_pipeline.discover_encodings(
                root=self.config.paths.training_recordings
            )

        train_path = self.config.paths.training_recordings
        val_path = self.config.paths.validation_recordings

        cardinalities = self.data_pipeline.encoding_cardinalities

        model_cfg = self.config.load_model_config(encoding_cardinalities=cardinalities)
        model = LSTMPolicy(config=model_cfg.policy).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.hyperparams.learning_rate,
            weight_decay=self.hyperparams.weight_decay,
        )

        # self.logger.info("Training recordings: %d", len(train_dataset))
        # self.logger.info("Validation recordings: %d", len(val_dataset))

        loss = PolicyLoss(
            button_weight=self.hyperparams.controller_weights.button_weights,
            analog_weight=self.hyperparams.controller_weights.analog_weights,
        ).to(self.device)

        self.logger.info("Initializing trainer on %s...", self.device)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss=loss,
            device=self.device,
            gradient_clip_norm=self.hyperparams.gradient_clip_norm,
            use_amp=self.hyperparams.use_amp,
            data_loader_config=self.config.training.data_loader,
        )

        best_val_loss = float("inf")
        early_stopping_best = float("inf")
        epochs_without_improvement = 0
        early_stopping = self.hyperparams.early_stopping

        for epoch in range(1, self.hyperparams.epochs + 1):
            train_datasets = self.data_pipeline.prepare_recordings(root=train_path)
            val_datasets = self.data_pipeline.prepare_recordings(root=val_path)

            self.logger.info("Training epoch %d/%d...", epoch, self.hyperparams.epochs)
            train_metrics = trainer.train_epoch(train_datasets)

            self.logger.info(
                "Validating epoch %d/%d...", epoch, self.hyperparams.epochs
            )
            val_metrics = trainer.validate(val_datasets)

            metadata = {
                "validation_loss": val_metrics.total_loss,
                "network_hyperparameters": self.hyperparams.model_dump(),
                "policy_config": model.config.model_dump(),
                "sequence_length": self.hyperparams.sequence_length,
                "encoders": [
                    encoders.model_dump() for encoders in self.config.gstate.encoders
                ],
                "encoding_cardinalities": dict(cardinalities),
                "game_state_schema": self.config.gstate.processed_schema.model_dump(),
            }

            save_checkpoint(
                self.config.paths.artifacts / "latest.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metadata=metadata,
            )

            if val_metrics.total_loss < best_val_loss:
                best_val_loss = val_metrics.total_loss
                save_checkpoint(
                    self.config.paths.artifacts / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metadata=metadata,
                )

            self.logger.info(
                "Epoch %d/%d : Train Loss: %.4f, Validation Loss: %.4f",
                epoch,
                self.hyperparams.epochs,
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

    def train(self, *, discover_encodings: bool = False) -> None:
        self.mimic(discover_encodings=discover_encodings)

    # def record(
    #     self,
    #     *,
    #     name: str | None = None,
    #     seconds: float | None = None,
    # ) -> Path:
    #     self.logger.info("Starting gameplay recording...")

    #     output = RecordingSession(config=self.config).run(name=name, seconds=seconds)

    #     self.logger.info("Recording saved to %s", output)

    #     return output

    # def summon(
    #     self,
    #     *,
    #     stop_event: Event | None = None,
    # ) -> None:
    #     input("Press Enter to summon (Ctrl+C to dismiss)")
    #     self.logger.info("Summoning Mimic Tear...")
    #     sleep(3.0)

    #     model = EldenRingPolicy(config=self.config.policy).to(self.device)

    #     checkpoint = load_checkpoint(
    #         self.config.artifacts_directory / "best.pt",
    #         model=model,
    #     )

    #     metadata = checkpoint["metadata"]

    #     game_state_transform_config = GameStateTransformConfig.model_validate(
    #         metadata["game_state_transform"]
    #     )

    #     frame_transform = FrameTransform(**self.config.transform_frames.model_dump())
    #     game_state_transform = GameStateTransform(
    #         **game_state_transform_config.model_dump()
    #     )

    #     screen = ScreenReader(**self.config.capture_screen.model_dump())
    #     game_state_reader = EldenRingReader.open(self.config.game_state)
    #     gamepad = GamepadWriter()

    #     grace_logger = Logger(**self.config.grace_logging.model_dump())

    #     player = Player(
    #         model=model,
    #         screen=screen,
    #         gamepad=gamepad,
    #         frame_transform=frame_transform,
    #         game_state_reader=game_state_reader,
    #         game_state_transform=game_state_transform,
    #         game_state_features=self.config.game_state.schema_.features,
    #         device=self.device,
    #         fps=self.config.video_config.fps,
    #         logger=grace_logger,
    #     )

    #     try:
    #         player.run()
    #     except KeyboardInterrupt:
    #         self.logger.info("Mimic Tear dismissed.")
    #     finally:
    #         game_state_reader.close()
    #         screen.close()
    #         gamepad.close()

    # def eval(
    #     self,
    #     *,
    #     stop_event: Event | None = None,
    # ) -> None:
    #     self.summon(stop_event=stop_event)
