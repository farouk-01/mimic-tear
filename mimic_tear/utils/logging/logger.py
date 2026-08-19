import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from rich.logging import RichHandler


class LoggingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str
    level: str
    log_file_enable: bool
    enabled: bool
    log_file: str | None = None


class Logger(logging.Logger):
    def __init__(
        self,
        name: str,
        level: int = logging.DEBUG,
        log_file: str | None = None,
        log_file_enable: bool = False,
        enabled: bool = False,
    ) -> None:
        super().__init__(name, level)

        self.disabled = not enabled
        self.propagate = False

        if not self.handlers:
            self._console_handler()

            if log_file_enable and log_file:
                self._add_file_handler(log_file)

    def _console_handler(self) -> None:
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=False,
            log_time_format="%Y-%m-%d %H:%M:%S",
        )

        console_handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))

        self.addHandler(console_handler)

    def _add_file_handler(self, log_file: str) -> None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")

        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s "
                "(%(filename)s:%(lineno)d) - %(message)s"
            )
        )

        self.addHandler(file_handler)

    def print(self, renderable: object) -> None:
        for handler in self.handlers:
            if isinstance(handler, RichHandler):
                handler.console.print(renderable)
                return


logging.setLoggerClass(Logger)
