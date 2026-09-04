from typing import Any, Self
from pathlib import Path
import copy
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from utils.files import load_toml

from .models.version import VersionConfig
from .models.logging import LoggingSettings
from .models.model import ModelConfig
from .models.paths import PathsConfig
from .models.schema import Schemas
from .models.pipeline import DataPipelineConfig
from .models.training import TrainingConfig
from .models.game_state import GameStateConfig
from .models.frame import FrameConfig
from .models.controller import ControllerConfig

DEFAULT_CONFIG_PATH = Path("configs/config.toml")
DEFAULT_OVERRIDE_PATH = Path("configs/config.override.toml")


class MimicTearConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    raw_cfg: Mapping[str, Any]

    logging: LoggingSettings
    paths: PathsConfig
    data: DataPipelineConfig
    training: TrainingConfig
    gstate: GameStateConfig

    @classmethod
    def load(cls) -> Self:
        cfg = _load_config()

        version = VersionConfig.load(cfg["version"])
        paths = PathsConfig.load(cfg["paths"])
        logging = LoggingSettings.load(cfg["logging"])

        schemas = Schemas()

        tensor_gstate_schema = schemas.tensor.game_state(version.gstate_tensor_schema)
        memory_gstate_profile = schemas.memory.game_state(version.gstate_memory_schema)
        enc_gstate_path = paths.encodings.game_state

        gstate_cfg = GameStateConfig.load(
            memory_profile=memory_gstate_profile,
            tensor_gstate_schema=tensor_gstate_schema,
            encodings_path=enc_gstate_path,
        )

        training = TrainingConfig.load(cfg["training"])

        frame = FrameConfig.load(
            schema=schemas.tensor.frame(),
            video_store_cfg=cfg["data"]["stores"]["frames"],
            transform_cfg=cfg["data"]["transforms"]["frames"],
            weights_name=cfg["model"]["vision"]["weights_name"],
        )

        controller_version = version.controller_tensor_schema
        controller = ControllerConfig.load(schema=schemas.tensor.controller("gamepad", controller_version))

        data = DataPipelineConfig.load(
            cfg,
            gstate=gstate_cfg,
            video_store_cfg=frame.video_store_cfg,
            frame_schema=frame.tensor_frame_schema,
            frame_plan=frame.plan,
            controller_schema=controller.tensor_controller_schema,
            controller_plan=controller.plan,
            training=training,
        )

        return cls(
            raw_cfg=cfg,
            logging=logging,
            paths=paths,
            data=data,
            training=training,
            gstate=gstate_cfg,
        )

    def load_model_config(
        self,
        *,
        encoding_cardinalities: Mapping[str, int],
    ) -> ModelConfig:
        return ModelConfig.load(
            self.raw_cfg["model"],
            gstate_schema=self.gstate.tensor_gstate_schema,
            encoding_cardinalities=encoding_cardinalities,
        )


def _load_config(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    override_path: str | Path | None = DEFAULT_OVERRIDE_PATH,
) -> dict[str, Any]:
    cfg = load_toml(config_path)

    def _merge(base, override):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                _merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)

    if override_path is not None:
        override_path = Path(override_path)

        if override_path.exists():
            override = load_toml(override_path)
            _merge(cfg, override)

    return cfg
