from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from mimic_tear.dataset import (
    BUTTON_COLUMNS,
    DEFAULT_NUM_WORKERS,
    DataModuleConfig,
    EldenRingDataModule,
)
from mimic_tear.game_state import GAME_STATE_FEATURE_COUNT
from mimic_tear.policy import EldenRingPolicy, PolicyLoss, policy_model_config


@dataclass(frozen=True, slots=True)
class TrainConfig:
    recordings_directory: Path
    output_directory: Path
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = DEFAULT_NUM_WORKERS
    prefetch_factor: int = 2
    seed: int = 42
    gradient_clip_norm: float = 1.0
    analog_loss_weight: float = 1.0
    button_loss_weight: float = 1.0
    use_amp: bool = True
    frame_cache: bool = True
    rebuild_frame_cache: bool = False
    early_stopping_patience: int = 5
    button_positive_weight_cap: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be greater than zero")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than zero")

        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")

        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")

        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be greater than zero")

        if self.rebuild_frame_cache and not self.frame_cache:
            raise ValueError("cannot rebuild the frame cache when it is disabled")

        if self.gradient_clip_norm <= 0.0:
            raise ValueError(
                "gradient_clip_norm must be greater than zero"
            )

        if self.early_stopping_patience <= 0:
            raise ValueError(
                "early_stopping_patience must be greater than zero"
            )

        if self.button_positive_weight_cap < 1.0:
            raise ValueError(
                "button_positive_weight_cap must be at least 1"
            )


@dataclass(slots=True)
class EpochMetrics:
    total_loss: float = 0.0
    analog_loss: float = 0.0
    button_loss: float = 0.0
    analog_mae: float = 0.0
    button_accuracy: float = 0.0
    button_true_positives: float = 0.0
    button_false_positives: float = 0.0
    button_false_negatives: float = 0.0
    samples: int = 0

    def update(
        self,
        *,
        batch_size: int,
        total_loss: Tensor,
        analog_loss: Tensor,
        button_loss: Tensor,
        analog_prediction: Tensor,
        analog_target: Tensor,
        button_logits: Tensor,
        button_target: Tensor,
    ) -> None:
        self.total_loss += total_loss.item() * batch_size
        self.analog_loss += analog_loss.item() * batch_size
        self.button_loss += button_loss.item() * batch_size

        self.analog_mae += (
            torch.abs(analog_prediction - analog_target)
            .mean()
            .item()
            * batch_size
        )

        button_prediction = torch.sigmoid(button_logits) >= 0.5
        button_truth = button_target >= 0.5

        self.button_true_positives += (
            button_prediction & button_truth
        ).sum().item()

        self.button_false_positives += (
            button_prediction & ~button_truth
        ).sum().item()

        self.button_false_negatives += (
            ~button_prediction & button_truth
        ).sum().item()

        self.button_accuracy += (
            (button_prediction == button_truth)
            .float()
            .mean()
            .item()
            * batch_size
        )

        self.samples += batch_size

    def averages(self) -> dict[str, float]:
        if self.samples == 0:
            raise RuntimeError("Cannot average empty epoch metrics")
        
        precision_denominator = (
            self.button_true_positives
            + self.button_false_positives
        )

        recall_denominator = (
            self.button_true_positives
            + self.button_false_negatives
        )

        precision = (
            self.button_true_positives / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall = (
            self.button_true_positives / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        return {
            "total_loss": self.total_loss / self.samples,
            "analog_loss": self.analog_loss / self.samples,
            "button_loss": self.button_loss / self.samples,
            "analog_mae": self.analog_mae / self.samples,
            "button_accuracy": self.button_accuracy / self.samples,
            "button_precision": precision,
            "button_recall": recall,
            "button_f1": f1,
        }


def train_one_epoch(
    *,
    model: EldenRingPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: PolicyLoss,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    metrics = EpochMetrics()

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        analog_target = batch["analog"].to(
            device,
            non_blocking=True,
        )
        button_target = batch["buttons"].to(
            device,
            non_blocking=True,
        )
        game_state = batch["game_state"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            prediction = model(images, game_state)
            losses = criterion(
                prediction,
                analog_target,
                button_target,
            )

        scaler.scale(losses.total).backward()
        scaler.unscale_(optimizer)

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip_norm,
        )

        scaler.step(optimizer)
        scaler.update()

        metrics.update(
            batch_size=images.shape[0],
            total_loss=losses.total.detach(),
            analog_loss=losses.analog,
            button_loss=losses.buttons,
            analog_prediction=prediction.analog.detach(),
            analog_target=analog_target,
            button_logits=prediction.button_logits.detach(),
            button_target=button_target,
        )

    return metrics.averages()


@torch.inference_mode()
def validate_one_epoch(
    *,
    model: EldenRingPolicy,
    loader: DataLoader,
    criterion: PolicyLoss,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    metrics = EpochMetrics()

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        analog_target = batch["analog"].to(
            device,
            non_blocking=True,
        )
        button_target = batch["buttons"].to(
            device,
            non_blocking=True,
        )
        game_state = batch["game_state"].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            prediction = model(images, game_state)
            losses = criterion(
                prediction,
                analog_target,
                button_target,
            )

        metrics.update(
            batch_size=images.shape[0],
            total_loss=losses.total,
            analog_loss=losses.analog,
            button_loss=losses.buttons,
            analog_prediction=prediction.analog,
            analog_target=analog_target,
            button_logits=prediction.button_logits,
            button_target=button_target,
        )

    return metrics.averages()


def save_checkpoint(
    *,
    path: Path,
    model: EldenRingPolicy,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    validation_loss: float,
    config: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "validation_loss": validation_loss,
            "model_config": policy_model_config(model.game_state_features),
            "config": {
                **asdict(config),
                "recordings_directory": str(
                    config.recordings_directory
                ),
                "output_directory": str(config.output_directory),
            },
        },
        path,
    )


def train(config: TrainConfig) -> None:
    set_seed(config.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = config.use_amp and device.type == "cuda"

    print(f"Device: {device}")
    print(f"Mixed precision: {use_amp}")

    data_module = EldenRingDataModule(
        DataModuleConfig(
            recordings_directory=config.recordings_directory,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            seed=config.seed,
            frame_cache=config.frame_cache,
            rebuild_frame_cache=config.rebuild_frame_cache,
        )
    )

    data_module.setup()

    train_loader = data_module.train_dataloader()
    validation_loader = data_module.validation_dataloader()

    print(f"Train samples: {len(data_module.train_dataset)}")
    print(
        "Validation samples: "
        f"{len(data_module.validation_dataset)}"
    )

    button_positive_weights = calculate_button_positive_weights(
        data_module,
        cap=config.button_positive_weight_cap,
    )
    print(
        "Button positive weights: "
        + ", ".join(
            f"{name}={weight:.2f}"
            for name, weight in zip(
                BUTTON_COLUMNS,
                button_positive_weights.tolist(),
                strict=True,
            )
        )
    )

    model = EldenRingPolicy(
        game_state_features=GAME_STATE_FEATURE_COUNT,
    ).to(device)

    criterion = PolicyLoss(
        analog_weight=config.analog_loss_weight,
        button_weight=config.button_loss_weight,
        button_positive_weights=button_positive_weights,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=use_amp,
    )

    config.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    try:
        for epoch in range(1, config.epochs + 1):
            started_at = time.perf_counter()

            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
                gradient_clip_norm=config.gradient_clip_norm,
            )

            validation_metrics = validate_one_epoch(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
            )

            scheduler.step(validation_metrics["total_loss"])

            elapsed_seconds = time.perf_counter() - started_at
            learning_rate = optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "elapsed_seconds": elapsed_seconds,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            history.append(row)

            print(
                f"Epoch {epoch:03d}/{config.epochs:03d} | "
                f"train={train_metrics['total_loss']:.4f} | "
                f"val={validation_metrics['total_loss']:.4f} | "
                f"analog_mae="
                f"{validation_metrics['analog_mae']:.4f} | "
                f"button_acc="
                f"{validation_metrics['button_accuracy']:.4f} | "
                f"lr={learning_rate:.2e} | "
                f"{elapsed_seconds:.1f}s"
            )

            save_checkpoint(
                path=config.output_directory / "latest.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                validation_loss=validation_metrics["total_loss"],
                config=config,
            )

            if (
                validation_metrics["total_loss"]
                < best_validation_loss
            ):
                best_validation_loss = validation_metrics[
                    "total_loss"
                ]
                epochs_without_improvement = 0

                save_checkpoint(
                    path=config.output_directory / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    validation_loss=best_validation_loss,
                    config=config,
                )

                print("  Saved new best checkpoint.")
            else:
                epochs_without_improvement += 1

            write_history(
                config.output_directory / "history.json",
                history,
            )

            if epochs_without_improvement >= config.early_stopping_patience:
                print(
                    "Early stopping: validation loss did not improve for "
                    f"{config.early_stopping_patience} consecutive epoch(s)."
                )
                break

    finally:
        data_module.close()

    print(
        "Training complete. Best validation loss: "
        f"{best_validation_loss:.4f}"
    )


def calculate_button_positive_weights(
    data_module: EldenRingDataModule,
    *,
    cap: float,
) -> Tensor:
    """Return capped inverse-frequency weights for observed button presses."""

    if cap < 1.0:
        raise ValueError("cap must be at least 1")

    dataset = data_module.train_dataset
    if dataset is None:
        raise RuntimeError("Data module must be set up before calculating weights")

    positive_counts = torch.zeros(len(BUTTON_COLUMNS), dtype=torch.float32)
    for sample in dataset.samples:
        positive_counts += torch.tensor(sample.buttons, dtype=torch.float32)

    total_samples = len(dataset)
    negative_counts = total_samples - positive_counts
    weights = torch.ones_like(positive_counts)
    observed = positive_counts > 0
    weights[observed] = negative_counts[observed] / positive_counts[observed]
    return weights.clamp_(min=1.0, max=cap)


def write_history(
    path: Path,
    history: list[dict[str, Any]],
) -> None:
    path.write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train the behavioral-cloning game policy."
    )

    parser.add_argument(
        "--recordings",
        type=Path,
        default=Path("recordings"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "policy",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help=(
            "stop after this many consecutive validation epochs without an "
            "improvement"
        ),
    )
    parser.add_argument(
        "--button-positive-weight-cap",
        type=float,
        default=1.0,
        help=(
            "maximum inverse-frequency weight applied to positive button "
            "examples; 1 disables extra positive weighting (default: 1)"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable-amp",
        action="store_true",
    )
    parser.add_argument(
        "--no-frame-cache",
        action="store_true",
        help="decode frames directly from MP4 instead of using the local cache",
    )
    parser.add_argument(
        "--rebuild-frame-cache",
        action="store_true",
        help="rebuild all resolution-specific frame caches before training",
    )

    args = parser.parse_args()

    return TrainConfig(
        recordings_directory=args.recordings,
        output_directory=args.output,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        button_positive_weight_cap=args.button_positive_weight_cap,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
        use_amp=not args.disable_amp,
        frame_cache=not args.no_frame_cache,
        rebuild_frame_cache=args.rebuild_frame_cache,
    )


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
