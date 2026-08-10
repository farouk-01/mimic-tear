from dataclasses import dataclass

from pathlib import Path
from dataclasses import dataclass, fields
from typing import Any

import yaml

from mimic_tear.model.components import (
    ControllerConfig,
    FusionConfig,
    GameStateConfig,
    TemporalConfig,
    VisionConfig,
)
from mimic_tear.model.policy import PolicyConfig


@dataclass(frozen=True, slots=True)
class Config:

    def unpack(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def load_config(path: str | Path) -> PolicyConfig:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Expected config to be a dictionary, got {type(cfg)}")

    model = cfg.get("model")

    if not isinstance(model, dict):
        raise ValueError("Config must contain a 'model' section")

    vision_cfg = VisionConfig(**model["vision"])

    temporal_cfg = TemporalConfig(
        input_features=vision_cfg.output_features,
        **model["temporal"],
    )

    game_state_cfg = GameStateConfig(**model["game_state"])

    fusion_cfg = FusionConfig(
        input_features=(
            temporal_cfg.hidden_features,
            game_state_cfg.output_features,
        ),
        **model["fusion"],
    )

    controller_cfg = ControllerConfig(
        input_features=fusion_cfg.output_features,
        **model["controller"],
    )

    return PolicyConfig(
        vision_cfg=vision_cfg,
        temporal_cfg=temporal_cfg,
        game_state_cfg=game_state_cfg,
        fusion_cfg=fusion_cfg,
        controller_cfg=controller_cfg,
    )
