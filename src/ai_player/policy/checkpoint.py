from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ai_player.game_state.features import (
    GAME_STATE_FEATURE_COUNT,
    GAME_STATE_FEATURE_NAMES,
)
from ai_player.policy.model import EldenRingPolicy


def policy_model_config(game_state_features: int) -> dict[str, object]:
    if game_state_features not in (0, GAME_STATE_FEATURE_COUNT):
        raise ValueError(
            "game_state_features must be zero or match the current feature schema"
        )
    return {
        "game_state_features": game_state_features,
        "game_state_feature_names": (
            list(GAME_STATE_FEATURE_NAMES) if game_state_features else []
        ),
    }


def load_policy_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[EldenRingPolicy, dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain model_state_dict: {path}")

    raw_config = checkpoint.get("model_config", {})
    if not isinstance(raw_config, dict):
        raise ValueError(f"Checkpoint model_config must be an object: {path}")
    game_state_features = int(raw_config.get("game_state_features", 0))
    if game_state_features < 0:
        raise ValueError("Checkpoint game_state_features cannot be negative")
    feature_names = tuple(raw_config.get("game_state_feature_names", ()))
    if game_state_features and feature_names != GAME_STATE_FEATURE_NAMES:
        raise ValueError(
            "Checkpoint game-state feature schema does not match this build: "
            f"{feature_names!r}"
        )
    if game_state_features != len(feature_names):
        raise ValueError(
            "Checkpoint game-state feature count does not match its feature names"
        )

    model = EldenRingPolicy(game_state_features=game_state_features)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
