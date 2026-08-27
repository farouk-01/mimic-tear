from typing import Self

from pydantic import BaseModel, ConfigDict

from utils.logging.logger import LoggingConfig
from utils.logging.profiling import ProfilerConfig


class LoggingSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    regular: LoggingConfig
    performance: LoggingConfig
    grace: LoggingConfig
    profiling: ProfilerConfig

    @classmethod
    def load(cls, raw_logging: dict) -> Self:
        return cls(
            regular=raw_logging["regular"],
            performance=raw_logging["perf_logger"],
            grace=raw_logging["grace"],
            profiling=raw_logging["profiling"],
        )
