from __future__ import annotations
from typing import Literal
from time import perf_counter
from pydantic import BaseModel, ConfigDict

from .base import ProfileMetric


class TimerMetricConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    unit: Literal["ms", "s"] = "ms"
    precision: int = 2


class TimerMetric(ProfileMetric):
    def __init__(self, unit: Literal["ms", "s"], precision: int = 2) -> None:
        self._started_at = 0.0
        self.unit = unit
        self.precision = precision

    def start(self) -> None:
        self._started_at = perf_counter()

    def stop(self) -> dict[str, float]:
        elapsed_seconds = perf_counter() - self._started_at

        return {"time_ms": elapsed_seconds * 1000}
