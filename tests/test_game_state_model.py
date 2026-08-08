from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mimic_tear.dataset.dataset import load_session_samples  # noqa: E402
from mimic_tear.game_state import (  # noqa: E402
    GAME_STATE_FEATURE_COUNT,
    GAME_STATE_FEATURE_NAMES,
    GameStateSample,
    GameStateSnapshot,
    encode_game_state_values,
)
from mimic_tear.policy import (  # noqa: E402
    EldenRingPolicy,
    load_policy_checkpoint,
    policy_model_config,
)
from mimic_tear.recording.schema import BUTTON_COLUMNS  # noqa: E402
from mimic_tear.policy.loss import PolicyLoss  # noqa: E402
from mimic_tear.training.train import train_one_epoch  # noqa: E402
from mimic_tear.recording.game_state_capture import GameStateWriter  # noqa: E402
from tests.recording.test_recording import (  # noqa: E402
    current_row,
    write_inputs,
    write_video,
)


class GameStateModelTests(unittest.TestCase):
    def test_feature_encoder_normalizes_values_and_marks_missing_fields(self) -> None:
        features = encode_game_state_values(
            {
                "player_health": 250,
                "player_max_health": 500,
                "player_fp": None,
                "player_max_fp": 100,
                "player_stamina": 60,
                "player_max_stamina": 120,
                "lock_on_active": True,
                "player_x": 100.0,
                "player_y": None,
                "player_z": -500.0,
            },
            valid=True,
        )
        values = dict(zip(GAME_STATE_FEATURE_NAMES, features, strict=True))
        self.assertEqual(len(features), GAME_STATE_FEATURE_COUNT)
        self.assertEqual(values["player_health_ratio"], 0.5)
        self.assertEqual(values["player_fp_available"], 0.0)
        self.assertEqual(values["player_stamina_ratio"], 0.5)
        self.assertEqual(values["lock_on_active"], 1.0)
        self.assertEqual(values["player_x_scaled"], 0.1)
        self.assertEqual(values["player_y_available"], 0.0)
        self.assertEqual(values["player_z_scaled"], -0.5)

    def test_state_aware_policy_requires_aligned_state_tensor(self) -> None:
        model = EldenRingPolicy(game_state_features=GAME_STATE_FEATURE_COUNT)
        images = torch.zeros((2, 3, 64, 64), dtype=torch.float32)
        state = torch.zeros(
            (2, GAME_STATE_FEATURE_COUNT),
            dtype=torch.float32,
        )
        output = model(images, state)
        self.assertEqual(output.analog.shape, (2, 6))
        with self.assertRaisesRegex(ValueError, "requires game-state"):
            model(images)

    def test_checkpoint_round_trip_preserves_state_schema(self) -> None:
        model = EldenRingPolicy(game_state_features=GAME_STATE_FEATURE_COUNT)
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "policy.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": policy_model_config(
                        model.game_state_features
                    ),
                },
                checkpoint,
            )
            loaded, _ = load_policy_checkpoint(
                checkpoint,
                device=torch.device("cpu"),
            )
        self.assertEqual(loaded.game_state_features, GAME_STATE_FEATURE_COUNT)

    def test_training_step_consumes_game_state_batch(self) -> None:
        model = EldenRingPolicy(game_state_features=GAME_STATE_FEATURE_COUNT)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        metrics = train_one_epoch(
            model=model,
            loader=[
                {
                    "image": torch.zeros((2, 3, 64, 64), dtype=torch.float32),
                    "game_state": torch.zeros(
                        (2, GAME_STATE_FEATURE_COUNT),
                        dtype=torch.float32,
                    ),
                    "analog": torch.zeros((2, 6), dtype=torch.float32),
                    "buttons": torch.zeros(
                        (2, len(BUTTON_COLUMNS)),
                        dtype=torch.float32,
                    ),
                }
            ],
            optimizer=optimizer,
            criterion=PolicyLoss(),
            device=torch.device("cpu"),
            scaler=torch.amp.GradScaler(device="cpu", enabled=False),
            use_amp=False,
            gradient_clip_norm=1.0,
        )
        self.assertGreater(metrics["total_loss"], 0.0)

    def test_dataset_requires_and_loads_frame_aligned_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = Path(temporary_directory) / "train" / "combat" / "one"
            session.mkdir(parents=True)
            write_video(session / "frames.mp4", frame_count=1)
            row = current_row(0)
            write_inputs(session / "inputs.parquet", [row])

            with self.assertRaises(FileNotFoundError):
                load_session_samples(session, require_game_state=True)

            frame_timestamp_ns = int(row["frame_timestamp_ns"])
            sample = GameStateSample(
                timestamp_ns=frame_timestamp_ns,
                snapshot=GameStateSnapshot(
                    values={
                        "player_health": 400,
                        "player_max_health": 500,
                        "player_stamina": 80,
                        "player_max_stamina": 100,
                        "lock_on_active": False,
                    },
                    valid=True,
                    read_errors=(),
                ),
            )
            with GameStateWriter(session / "game_state.parquet") as writer:
                writer.write(
                    frame_index=0,
                    timestamp_ns=int(row["timestamp_ns"]),
                    frame_timestamp_ns=frame_timestamp_ns,
                    sample=sample,
                )
            samples = load_session_samples(session, require_game_state=True)
            self.assertEqual(len(samples[0].game_state), GAME_STATE_FEATURE_COUNT)


if __name__ == "__main__":
    unittest.main()
