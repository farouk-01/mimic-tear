from __future__ import annotations

import psutil
from pydantic import BaseModel, ConfigDict

from .base import ProfileMetric


class RAMMetricConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


def _megabytes(value: int) -> float:
    return value / (1024 * 1024)


class RAMMetric(ProfileMetric):
    def __init__(self) -> None:
        self._process = psutil.Process()
        self._ram_start = 0

    def start(self) -> None:
        self._ram_start = self._process.memory_info().rss

    def stop(self) -> dict[str, float]:
        ram_end = self._process.memory_info().rss

        return {
            "ram_start_mb": _megabytes(self._ram_start),
            "ram_end_mb": _megabytes(ram_end),
            "ram_delta_mb": _megabytes(ram_end - self._ram_start),
        }
