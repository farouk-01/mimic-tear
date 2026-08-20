from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from pydantic import BaseModel, ConfigDict

class LoggingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str
    level: str
    log_file_enable: bool
    enabled: bool
    log_file: str | None = None

class Logger(logging.Logger):
    _COLORS = {
        logging.DEBUG: "\033[35m",  # purple
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[41m",  # red background
    }
    _RESET = "\033[0m"

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

        if not self.handlers:
            self.propagate = False
            self._console_handler()
            if log_file_enable:
                self._add_file_handler(log_file) if log_file else None

    def _console_handler(self) -> None:
        console_handler = logging.StreamHandler(sys.stdout)

        class ColorFormatter(logging.Formatter):
            def __init__(self, colors, reset):
                super().__init__(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    "%Y-%m-%d %H:%M:%S",
                )
                self.colors = colors
                self.reset = reset

            def format(self, record):
                color = self.colors.get(record.levelno, "")
                record.levelname = f"{color}{record.levelname}{self.reset}"
                return super().format(record)

        console_handler.setFormatter(ColorFormatter(self._COLORS, self._RESET))
        self.addHandler(console_handler)

    def _add_file_handler(self, log_file: str):
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.addHandler(file_handler)

logging.setLoggerClass(Logger)