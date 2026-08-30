from typing import Any, Mapping, Self
from pathlib import Path
import copy

from pydantic import BaseModel, ConfigDict

from utils.files import load_toml

from .models.version import VersionConfig
from .models.logging import LoggingSettings
from .models.model import ModelConfig
from .models.paths import PathsConfig, ConfigDomain as cd
from .models.pipeline import DataPipelineConfig
from .models.training import TrainingConfig
from .models.game_state import GameStateConfig

DEFAULT_CONFIG_PATH = Path("configs/config.toml")
DEFAULT_OVERRIDE_PATH = Path("configs/config.override.toml")


class MimicTearConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    raw_cfg: Mapping[str, Any]

    logging: LoggingSettings
    data: DataPipelineConfig
    training: TrainingConfig
    gstate: GameStateConfig

    @classmethod
    def load(cls) -> Self:
        cfg = _load_config()

        version = VersionConfig.load(cfg["version"])
        paths = PathsConfig.load(cfg["paths"])
        logging = LoggingSettings.load(cfg["logging"])

        mem_path = paths.memory_schema(cd.GSTATE, version.gstate_memory_schema)
        proc_path = paths.processed_schema(cd.GSTATE, version.gstate_processed_schema)
        enc_path = paths.encodings_for(cd.GSTATE)
        gstate_cfg = GameStateConfig.load(
            memory_schema=mem_path,
            processed_schema=proc_path,
            encodings_path=enc_path,
        )

        training = TrainingConfig.load(cfg["training"])

        weights_name = cfg["model"]["vision"]["weights_name"]

        data = DataPipelineConfig.load(
            cfg, resnet_weights_name=weights_name, gstate=gstate_cfg, training=training
        )

        return cls(
            raw_cfg=cfg,
            logging=logging,
            data=data,
            training=training,
            gstate=gstate_cfg,
        )

    def load_model_config(self, *, encoding_cardinalities: dict[str, int]) -> ModelConfig:
        return ModelConfig.load(
            self.raw_cfg["model"],
            gstate_schema=self.gstate.processed_schema,
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
