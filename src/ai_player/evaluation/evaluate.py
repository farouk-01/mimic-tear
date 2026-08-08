from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ai_player.dataset import (
    ANALOG_COLUMNS,
    BUTTON_COLUMNS,
    DEFAULT_NUM_WORKERS,
    DataModuleConfig,
    EldenRingDataModule,
)
from ai_player.policy import EldenRingPolicy, load_policy_checkpoint


@dataclass(slots=True)
class EvaluationMetrics:
    analog_absolute_error: Tensor
    button_true_positives: Tensor
    button_false_positives: Tensor
    button_false_negatives: Tensor
    button_true_negatives: Tensor
    sample_count: int = 0

    @classmethod
    def create(
        cls,
        *,
        analog_count: int,
        button_count: int,
        device: torch.device,
    ) -> "EvaluationMetrics":
        return cls(
            analog_absolute_error=torch.zeros(
                analog_count,
                dtype=torch.float64,
                device=device,
            ),
            button_true_positives=torch.zeros(
                button_count,
                dtype=torch.float64,
                device=device,
            ),
            button_false_positives=torch.zeros(
                button_count,
                dtype=torch.float64,
                device=device,
            ),
            button_false_negatives=torch.zeros(
                button_count,
                dtype=torch.float64,
                device=device,
            ),
            button_true_negatives=torch.zeros(
                button_count,
                dtype=torch.float64,
                device=device,
            ),
        )

    def update(
        self,
        *,
        analog_prediction: Tensor,
        analog_target: Tensor,
        button_probabilities: Tensor,
        button_target: Tensor,
        button_threshold: float,
    ) -> None:
        batch_size = analog_target.shape[0]

        self.analog_absolute_error += (
            analog_prediction - analog_target
        ).abs().sum(dim=0, dtype=torch.float64)

        button_prediction = button_probabilities >= button_threshold
        button_truth = button_target >= 0.5

        self.button_true_positives += (
            button_prediction & button_truth
        ).sum(dim=0, dtype=torch.float64)

        self.button_false_positives += (
            button_prediction & ~button_truth
        ).sum(dim=0, dtype=torch.float64)

        self.button_false_negatives += (
            ~button_prediction & button_truth
        ).sum(dim=0, dtype=torch.float64)

        self.button_true_negatives += (
            ~button_prediction & ~button_truth
        ).sum(dim=0, dtype=torch.float64)

        self.sample_count += batch_size

    def analog_mae(self) -> Tensor:
        if self.sample_count == 0:
            raise RuntimeError("No samples were evaluated")

        return self.analog_absolute_error / self.sample_count

    def button_precision(self) -> Tensor:
        denominator = (
            self.button_true_positives
            + self.button_false_positives
        )

        return torch.where(
            denominator > 0,
            self.button_true_positives / denominator,
            torch.zeros_like(denominator),
        )

    def button_recall(self) -> Tensor:
        denominator = (
            self.button_true_positives
            + self.button_false_negatives
        )

        return torch.where(
            denominator > 0,
            self.button_true_positives / denominator,
            torch.zeros_like(denominator),
        )

    def button_f1(self) -> Tensor:
        precision = self.button_precision()
        recall = self.button_recall()
        denominator = precision + recall

        return torch.where(
            denominator > 0,
            2.0 * precision * recall / denominator,
            torch.zeros_like(denominator),
        )

    def button_positive_rate(self) -> Tensor:
        if self.sample_count == 0:
            raise RuntimeError("No samples were evaluated")

        return (
            self.button_true_positives
            + self.button_false_negatives
        ) / self.sample_count


def load_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[EldenRingPolicy, dict]:
    return load_policy_checkpoint(checkpoint_path, device=device)


@torch.inference_mode()
def evaluate(
    *,
    model: EldenRingPolicy,
    loader: DataLoader,
    device: torch.device,
    button_threshold: float,
    sample_examples: int,
) -> EvaluationMetrics:
    metrics = EvaluationMetrics.create(
        analog_count=len(ANALOG_COLUMNS),
        button_count=len(BUTTON_COLUMNS),
        device=device,
    )

    examples_remaining = sample_examples

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

        output = model(images, game_state)
        button_probabilities = torch.sigmoid(
            output.button_logits
        )

        metrics.update(
            analog_prediction=output.analog,
            analog_target=analog_target,
            button_probabilities=button_probabilities,
            button_target=button_target,
            button_threshold=button_threshold,
        )

        if examples_remaining > 0:
            example_count = min(
                examples_remaining,
                images.shape[0],
            )

            print_examples(
                analog_prediction=output.analog[:example_count],
                analog_target=analog_target[:example_count],
                button_probabilities=button_probabilities[:example_count],
                button_target=button_target[:example_count],
                frame_indices=batch["frame_index"][:example_count],
                sessions=batch["session"][:example_count],
                button_threshold=button_threshold,
            )

            examples_remaining -= example_count

    return metrics


def print_examples(
    *,
    analog_prediction: Tensor,
    analog_target: Tensor,
    button_probabilities: Tensor,
    button_target: Tensor,
    frame_indices: Tensor,
    sessions: list[str] | tuple[str, ...],
    button_threshold: float,
) -> None:
    analog_prediction = analog_prediction.detach().cpu()
    analog_target = analog_target.detach().cpu()
    button_probabilities = button_probabilities.detach().cpu()
    button_target = button_target.detach().cpu()

    for index in range(analog_prediction.shape[0]):
        print()
        print(
            f"Example: session={sessions[index]} "
            f"frame={int(frame_indices[index])}"
        )

        print("  Analog:")

        for column_index, name in enumerate(ANALOG_COLUMNS):
            actual = analog_target[index, column_index].item()
            predicted = analog_prediction[index, column_index].item()

            print(
                f"    {name:16} "
                f"actual={actual:+.3f} "
                f"predicted={predicted:+.3f} "
                f"error={abs(actual - predicted):.3f}"
            )

        print("  Buttons:")

        for column_index, name in enumerate(BUTTON_COLUMNS):
            actual = bool(
                button_target[index, column_index].item() >= 0.5
            )
            probability = button_probabilities[
                index,
                column_index,
            ].item()
            predicted = probability >= button_threshold

            print(
                f"    {name:16} "
                f"actual={int(actual)} "
                f"predicted={int(predicted)} "
                f"probability={probability:.3f}"
            )


def print_summary(
    metrics: EvaluationMetrics,
    *,
    checkpoint: dict,
) -> None:
    analog_mae = metrics.analog_mae().cpu()
    button_precision = metrics.button_precision().cpu()
    button_recall = metrics.button_recall().cpu()
    button_f1 = metrics.button_f1().cpu()
    positive_rate = metrics.button_positive_rate().cpu()

    print()
    print("=" * 72)
    print("Evaluation summary")
    print("=" * 72)
    print(f"Samples: {metrics.sample_count}")

    checkpoint_epoch = checkpoint.get("epoch")
    checkpoint_loss = checkpoint.get("validation_loss")

    if checkpoint_epoch is not None:
        print(f"Checkpoint epoch: {checkpoint_epoch}")

    if checkpoint_loss is not None:
        print(
            "Checkpoint validation loss: "
            f"{float(checkpoint_loss):.6f}"
        )

    print()
    print("Analog MAE:")

    for index, name in enumerate(ANALOG_COLUMNS):
        print(f"  {name:16} {analog_mae[index].item():.6f}")

    print(
        f"  {'overall':16} "
        f"{analog_mae.mean().item():.6f}"
    )

    print()
    print(
        f"{'Button':16} "
        f"{'Rate':>9} "
        f"{'Precision':>10} "
        f"{'Recall':>10} "
        f"{'F1':>10}"
    )

    for index, name in enumerate(BUTTON_COLUMNS):
        print(
            f"{name:16} "
            f"{positive_rate[index].item():9.4f} "
            f"{button_precision[index].item():10.4f} "
            f"{button_recall[index].item():10.4f} "
            f"{button_f1[index].item():10.4f}"
        )

    macro_f1 = button_f1.mean().item()

    true_positives = metrics.button_true_positives.sum()
    false_positives = metrics.button_false_positives.sum()
    false_negatives = metrics.button_false_negatives.sum()

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives

    micro_precision = (
        (true_positives / precision_denominator).item()
        if precision_denominator.item() > 0
        else 0.0
    )
    micro_recall = (
        (true_positives / recall_denominator).item()
        if recall_denominator.item() > 0
        else 0.0
    )
    micro_f1 = (
        2.0
        * micro_precision
        * micro_recall
        / (micro_precision + micro_recall)
        if micro_precision + micro_recall > 0
        else 0.0
    )

    print()
    print(f"Button macro F1: {macro_f1:.4f}")
    print(f"Button micro F1: {micro_f1:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a behavioral-cloning policy checkpoint."
    )

    parser.add_argument(
        "--recordings",
        type=Path,
        default=Path("recordings"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts") / "policy" / "best.pt",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-frame-cache", action="store_true")
    parser.add_argument("--rebuild-frame-cache", action="store_true")
    parser.add_argument(
        "--button-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")

    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")

    if args.prefetch_factor <= 0:
        parser.error("--prefetch-factor must be greater than zero")

    if args.no_frame_cache and args.rebuild_frame_cache:
        parser.error("--rebuild-frame-cache requires the frame cache")

    if not 0.0 <= args.button_threshold <= 1.0:
        parser.error("--button-threshold must be in [0, 1]")

    if args.examples < 0:
        parser.error("--examples cannot be negative")

    return args


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available"
        )

    return torch.device(name)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    data_module = EldenRingDataModule(
        DataModuleConfig(
            recordings_directory=args.recordings,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            augment_training=False,
            frame_cache=not args.no_frame_cache,
            rebuild_frame_cache=args.rebuild_frame_cache,
        )
    )

    try:
        data_module.setup()

        model, checkpoint = load_model(
            args.checkpoint,
            device=device,
        )

        metrics = evaluate(
            model=model,
            loader=data_module.validation_dataloader(),
            device=device,
            button_threshold=args.button_threshold,
            sample_examples=args.examples,
        )

        print_summary(
            metrics,
            checkpoint=checkpoint,
        )
    finally:
        data_module.close()


if __name__ == "__main__":
    main()
