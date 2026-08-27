from typing import Self

from pydantic import BaseModel, ConfigDict

from data.capture import EldenRingMemoryProfile

from .loader import load_raw_config

from .models.logging import LoggingSettings
from .models.model import ModelConfig
from .models.paths import PathsConfig
from .models.pipeline import DataPipelineConfig
from .models.training import TrainingConfig
from .models.game_state import ProcessedGameStateSchema


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
        processed_game_state_schema = ProcessedGameStateSchema.from_json()

        model = ModelConfig.load(
            settings["model"],
            game_state_features=processed_game_state_schema.required_feature_count,
        )

        data = DataPipelineConfig.load(
            settings,
            game_state=game_state,
            model=model,
            training=training,
            processed_game_state_schema=processed_game_state_schema,
        )

        return cls(
            logging=logging,
            data=data,
            model=model,
            training=training,
            game_state=game_state,
            paths=paths,
        )
