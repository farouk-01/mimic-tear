from functools import wraps
from time import perf_counter
from typing import Any, Callable


def timed(logger_attribute: str):
    def decorator(func: Callable[..., Any]):
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any):
            start = perf_counter()

            try:
                return func(self, *args, **kwargs)
            finally:
                elapsed_ms = (perf_counter() - start) * 1000
                logger = getattr(self, logger_attribute)
                logger.debug(
                    "%s took %.2f ms",
                    func.__qualname__,
                    elapsed_ms,
                )

        return wrapper

    return decorator