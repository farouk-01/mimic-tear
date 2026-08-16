from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from time import perf_counter
from typing import (
    Any,
    Concatenate,
    ParamSpec,
    TypeVar,
)

import psutil
import torch

Instance = TypeVar("Instance")
Parameters = ParamSpec("Parameters")
Result = TypeVar("Result")


def _megabytes(value: int) -> float:
    return value / (1024 * 1024)


def timed(
    logger_attribute: str,
) -> Callable[
    [Callable[Concatenate[Instance, Parameters], Result]],
    Callable[Concatenate[Instance, Parameters], Result],
]:
    def decorator(
        function: Callable[Concatenate[Instance, Parameters], Result],
    ) -> Callable[
        Concatenate[Instance, Parameters],
        Result,
    ]:
        @wraps(function)
        def wrapper(
            instance: Instance,
            /,
            *args: Parameters.args,
            **kwargs: Parameters.kwargs,
        ) -> Result:
            start = perf_counter()

            try:
                return function(instance, *args, **kwargs)
            finally:
                elapsed_ms = (perf_counter() - start) * 1000

                logger: Any = getattr(instance, logger_attribute)

                logger.debug("%s took %.2f ms", function.__qualname__, elapsed_ms)

        return wrapper

    return decorator


def profile(
    logger_attribute: str,
    device_attribute: str = "device",
) -> Callable[
    [Callable[Concatenate[Instance, Parameters], Result]],
    Callable[Concatenate[Instance, Parameters], Result],
]:
    def decorator(
        function: Callable[Concatenate[Instance, Parameters], Result],
    ) -> Callable[
        Concatenate[Instance, Parameters],
        Result,
    ]:
        @wraps(function)
        def wrapper(
            instance: Instance,
            /,
            *args: Parameters.args,
            **kwargs: Parameters.kwargs,
        ) -> Result:
            logger: Any = getattr(instance, logger_attribute)
            device = torch.device(getattr(instance, device_attribute))
            process = psutil.Process()

            uses_cuda = device.type == "cuda" and torch.cuda.is_available()

            if uses_cuda:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                vram_start = torch.cuda.memory_allocated(device)
                reserved_start = torch.cuda.memory_reserved(device)
            else:
                vram_start = 0
                reserved_start = 0

            cpu_start = process.cpu_times()
            ram_start = process.memory_info().rss
            started_at = perf_counter()

            try:
                return function(instance, *args, **kwargs)
            finally:
                if uses_cuda:
                    torch.cuda.synchronize(device)

                elapsed_seconds = perf_counter() - started_at
                cpu_end = process.cpu_times()
                ram_end = process.memory_info().rss

                cpu_seconds = (
                    cpu_end.user
                    + cpu_end.system
                    - cpu_start.user
                    - cpu_start.system
                )
                cpu_percent = (
                    cpu_seconds / elapsed_seconds * 100
                    if elapsed_seconds > 0
                    else 0.0
                )

                if uses_cuda:
                    vram_end = torch.cuda.memory_allocated(device)
                    vram_peak = torch.cuda.max_memory_allocated(device)
                    reserved_end = torch.cuda.memory_reserved(device)
                    reserved_peak = torch.cuda.max_memory_reserved(device)
                else:
                    vram_end = 0
                    vram_peak = 0
                    reserved_end = 0
                    reserved_peak = 0

                logger.debug(
                    "%s | time=%.2f ms | cpu=%.1f%% | "
                    "ram=%.1f->%.1f MB (%+.1f MB) | "
                    "vram=%.1f->%.1f MB (peak %.1f MB) | "
                    "reserved=%.1f->%.1f MB (peak %.1f MB)",
                    function.__qualname__,
                    elapsed_seconds * 1000,
                    cpu_percent,
                    _megabytes(ram_start),
                    _megabytes(ram_end),
                    _megabytes(ram_end - ram_start),
                    _megabytes(vram_start),
                    _megabytes(vram_end),
                    _megabytes(vram_peak),
                    _megabytes(reserved_start),
                    _megabytes(reserved_end),
                    _megabytes(reserved_peak),
                )

        return wrapper

    return decorator
