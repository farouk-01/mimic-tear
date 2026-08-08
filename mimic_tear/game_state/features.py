from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

import torch
from torch import Tensor

from mimic_tear.game_state.reader import GameStateSnapshot


GAME_STATE_FEATURE_NAMES: tuple[str, ...] = (
    "state_valid",
    "player_health_ratio",
    "player_health_available",
    "player_fp_ratio",
    "player_fp_available",
    "player_stamina_ratio",
    "player_stamina_available",
    "lock_on_active",
    "lock_on_available",
    "player_x_scaled",
    "player_x_available",
    "player_y_scaled",
    "player_y_available",
    "player_z_scaled",
    "player_z_available",
)
GAME_STATE_FEATURE_COUNT = len(GAME_STATE_FEATURE_NAMES)
POSITION_SCALE = 1_000.0


def encode_game_state_values(
    values: Mapping[str, object | None],
    *,
    valid: bool,
) -> tuple[float, ...]:
    features: list[float] = [float(valid)]
    for current_name, maximum_name in (
        ("player_health", "player_max_health"),
        ("player_fp", "player_max_fp"),
        ("player_stamina", "player_max_stamina"),
    ):
        current = _finite_float(values.get(current_name))
        maximum = _finite_float(values.get(maximum_name))
        available = current is not None and maximum is not None and maximum > 0.0
        ratio = min(1.0, max(0.0, current / maximum)) if available else 0.0
        features.extend((ratio, float(available)))

    lock_on = values.get("lock_on_active")
    lock_on_available = isinstance(lock_on, bool)
    features.extend((float(lock_on) if lock_on_available else 0.0, float(lock_on_available)))

    for name in ("player_x", "player_y", "player_z"):
        coordinate = _finite_float(values.get(name))
        available = coordinate is not None
        scaled = (
            min(1.0, max(-1.0, coordinate / POSITION_SCALE))
            if available
            else 0.0
        )
        features.extend((scaled, float(available)))

    return tuple(features)


def encode_game_state_snapshot(snapshot: GameStateSnapshot) -> tuple[float, ...]:
    return encode_game_state_values(snapshot.values, valid=snapshot.valid)


def game_state_tensor(
    values: Mapping[str, object | None],
    *,
    valid: bool,
) -> Tensor:
    return torch.tensor(
        encode_game_state_values(values, valid=valid),
        dtype=torch.float32,
    )


def _finite_float(value: object | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if isfinite(converted) else None
