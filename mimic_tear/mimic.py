from __future__ import annotations
import torch
from pathlib import Path
from typing import TYPE_CHECKING, Any
from threading import Event
import datetime

from configs.models.policy import PolicyConfig
from data import DataPipeline
from mimic_tear.model import get_policy, Policy
from mimic_tear.model.components import GamepadLoss

from mimic_tear.player.player import Player
from utils.logging.logger import Logger
from mimic_tear.training import Trainer
from mimic_tear.training.checkpoint import load_checkpoint, save_checkpoint
from mimic_tear.analysis.recorder import PredictionRecorder
from utils.logging.profiling import Profiler
from utils.hotkeys import VK_F10, register_hotkey, unregister_hotkey
from utils.files import dump_json

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
        self.grace_logger: Logger | None = None

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

        self.recorder: PredictionRecorder | None = None

    def mimic(self, *, discover_encodings: bool = False) -> None:
        self.logger.info("Starting training...")

        artififact_path = (
            self.config.paths.artifacts
            / f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        )

        if self.config.training.save_predictions:
            self.recorder = PredictionRecorder(artififact_path / "predictions")

        if discover_encodings:
            self.logger.info("Discovering encodings...")
            # TODO : timer
            self.data_pipeline.discover_encodings(
                root=self.config.paths.training_recordings
            )

        train_path = self.config.paths.training_recordings
        val_path = self.config.paths.validation_recordings

        policy, policy_cfg = self._build_policy()

        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=self.hyperparams.learning_rate,
            weight_decay=self.hyperparams.weight_decay,
        )

        loss = GamepadLoss(
            button_weight=self.hyperparams.controller_weights.button_weights,
            analog_weight=self.hyperparams.controller_weights.analog_weights,
            button_transition_weight=self.hyperparams.button_transition_weights.weights,
        ).to(self.device)

        self.logger.info("Initializing trainer on %s...", self.device)
        trainer = Trainer(
            policy=policy,
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

        self._write_config_metadata(artififact_path, policy_cfg)

        for epoch in range(1, self.hyperparams.epochs + 1):
            train_datasets = self.data_pipeline.prepare_recordings(root=train_path)
            val_datasets = self.data_pipeline.prepare_recordings(root=val_path)

            self.logger.info("Training epoch %d/%d...", epoch, self.hyperparams.epochs)
            train_metrics = trainer.train_epoch(train_datasets)

            self.logger.info(
                "Validating epoch %d/%d...", epoch, self.hyperparams.epochs
            )

            val_metrics = trainer.validate(val_datasets, observer=self.recorder)

            save_checkpoint(
                artififact_path / "latest.pt",
                model=policy,
                optimizer=optimizer,
                epoch=epoch,
            )

            if self.recorder is not None:
                self.recorder.save(epoch=epoch)

            if val_metrics.total_loss < best_val_loss:
                best_val_loss = val_metrics.total_loss
                save_checkpoint(
                    artififact_path / "best.pt",
                    model=policy,
                    optimizer=optimizer,
                    epoch=epoch,
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
                    epoch - epochs_without_improvement,  # TODO i think off by one
                )
                break

    def summon(self, *, stop_event: Event | None = None) -> None:
        self.logger.info("Summoning Mimic Tear...")

        policy, _ = self._build_policy()

        checkpoint = self._latest_checkpoint()
        self.logger.info("Loading checkpoint %s", checkpoint)
        load_checkpoint(checkpoint, model=policy)

        if self.grace_logger is None:
            self.grace_logger = Logger(**self.config.logging.grace.model_dump())

        active = Event()

        def toggle_active() -> None:
            if active.is_set():
                active.clear()
            else:
                active.set()

        hotkey = register_hotkey(VK_F10, toggle_active)
        self.logger.info("Press F10 to start/pause the Mimic Tear (Ctrl+C to dismiss)")

        try:
            with self.data_pipeline.writer.gamepad() as gamepad:
                player = Player(
                    model=policy,
                    process_stream=self.data_pipeline.process_stream,
                    gamepad=gamepad,
                    device=self.device,
                    button_thresholds=self.hyperparams.button_thresholds.thresholds,
                    analog_gain=self.hyperparams.analog_gain,
                    logger=self.grace_logger,
                )

                player.run(stop_event=stop_event, active=active)

        except KeyboardInterrupt:
            self.logger.info("Mimic Tear dismissed.")

        finally:
            unregister_hotkey(hotkey)

    def start_virtual_gamepad(self) -> bool:
        self.logger.info("Connecting gamepad bridge...")

        started = self.data_pipeline.writer.start_gamepad_bridge()

        return started

    def stop_virtual_gamepad(self) -> bool:
        self.logger.info("Disconnecting gamepad bridge...")

        stopped = self.data_pipeline.writer.stop_gamepad_bridge()

        return stopped

    def train(self, *, discover_encodings: bool = False) -> None:
        self.mimic(discover_encodings=discover_encodings)

    def eval(self, *, stop_event: Event | None = None) -> None:
        self.summon(stop_event=stop_event)

    def _latest_checkpoint(self) -> Path:
        checkpoint = self.config.paths.artifacts / "best.pt"

        if checkpoint.exists():
            return checkpoint

        checkpoints = list(self.config.paths.artifacts.glob("*/best.pt"))

        if not checkpoints:
            raise FileNotFoundError(
                f"No best.pt checkpoint found in {self.config.paths.artifacts}"
            )

        return max(checkpoints, key=lambda path: path.stat().st_mtime)

    def _build_policy(self) -> tuple[Policy, PolicyConfig]:
        policy_cfg = self.config.policy
        cardinalities = self.data_pipeline.encoding_cardinalities

        policy = get_policy(
            policy_cfg,
            schemas=self.config.data.process.schemas,  # type: ignore
            cardinalities=cardinalities,
        )

        return policy.to(self.device), policy_cfg

    def _write_config_metadata(self, path: Path, policy_cfg: PolicyConfig) -> None:
        metadata = {
            "hyperparameters": self.hyperparams.model_dump(),
            "encoders": [
                encoders.model_dump() for encoders in self.config.gstate.encoders
            ],
            "encoding_cardinalities": self.data_pipeline.encoding_cardinalities,
            "game_state_schema": self.config.gstate.tensor_schema.model_dump(),
        }

        path.mkdir(parents=True, exist_ok=True)

        dump_json(metadata, path=path / "metadata.json", indent=4, sort_keys=True)

    def record(
        self,
        *,
        name: str,
        theme: str | None = None,
        sub_theme: str | None = None,
        seconds: float | None = None,
        validation: bool = False,
    ) -> Path:
        self.logger.info("Starting gameplay recording...")

        root = (
            self.config.paths.validation_recordings
            if validation
            else self.config.paths.training_recordings
        )

        output = self.data_pipeline.record_session(
            root=root,
            theme=theme,
            sub_theme=sub_theme,
            name=name,
            seconds=seconds,
        )

        self.logger.info("Recording saved to %s", output)

        return output
