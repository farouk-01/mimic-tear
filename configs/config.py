from typing import Self

from pydantic import BaseModel, ConfigDict

from data.capture import EldenRingMemoryProfile

from .loader import load_raw_config

from .models.logging import LoggingSettings
from .models.model import ModelConfig
from .models.paths import PathsConfig
from .models.pipeline import DataPipelineConfig
from .models.training import TrainingConfig


class MimicTearConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    logging: LoggingSettings
    paths: PathsConfig
    data: DataPipelineConfig
    model: ModelConfig
    training: TrainingConfig
    game_state: EldenRingMemoryProfile

    @classmethod
    def load(cls) -> Self:
        raw = load_raw_config()
        settings = raw.settings

        game_state = EldenRingMemoryProfile.model_validate(raw.game_state)
        logging = LoggingSettings.load(settings["logging"])
        training = TrainingConfig.load(settings["training"])
        paths = PathsConfig.load(settings["paths"])

        model = ModelConfig.load(
            settings["model"],
            game_state_features=len(game_state.fields),
        )

        data = DataPipelineConfig.load(
            settings,
            game_state=game_state,
            model=model,
            training=training,
        )

        return cls(
            logging=logging,
            data=data,
            model=model,
            training=training,
            game_state=game_state,
            paths=paths,
        )