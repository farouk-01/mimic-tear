# cpu.py

from __future__ import annotations

from time import perf_counter

import psutil
from pydantic import BaseModel, ConfigDict

from .base import ProfileMetric


class CPUMetricConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class CPUMetric(ProfileMetric):
    def __init__(self) -> None:
        self._process = psutil.Process()
        self._started_at = 0.0
        self._cpu_start = None

    def start(self) -> None:
        self._started_at = perf_counter()
        self._cpu_start = self._process.cpu_times()

    def stop(self) -> dict[str, float]:
        if self._cpu_start is None:
            raise RuntimeError("CPUMetric was not started")

        elapsed_seconds = perf_counter() - self._started_at
        cpu_end = self._process.cpu_times()

        cpu_seconds = (
            cpu_end.user
            + cpu_end.system
            - self._cpu_start.user
            - self._cpu_start.system
        )

        cpu_percent = (
            cpu_seconds / elapsed_seconds * 100 if elapsed_seconds > 0 else 0.0
        )

        return {
            "cpu_percent": cpu_percent,
        }
