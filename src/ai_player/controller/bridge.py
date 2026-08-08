from __future__ import annotations

import ctypes
import json
from dataclasses import asdict
from typing import BinaryIO

from ai_player.controller.state import ControllerState


PIPE_NAME = r"\\.\pipe\ai-player-controller"


class VirtualController:
    """Client for the elevated HIDMaestro system-controller bridge."""

    def __init__(
        self,
        pipe_name: str = PIPE_NAME,
        connect_timeout_ms: int = 5000,
        *,
        _stream: BinaryIO | None = None,
    ) -> None:
        if connect_timeout_ms <= 0:
            raise ValueError("connect_timeout_ms must be positive")

        self._pipe_name = pipe_name
        self._stream = _stream or self._connect(connect_timeout_ms)
        self._closed = False

        if _stream is None:
            response = self._stream.readline()
            if not response:
                self.close()
                raise RuntimeError("Controller bridge closed during startup.")
            message = json.loads(response.decode("utf-8"))
            if message.get("type") != "ready":
                self.close()
                raise RuntimeError(f"Controller bridge rejected connection: {message}")

    @property
    def connected(self) -> bool:
        return not self._closed

    def apply(self, state: ControllerState) -> None:
        state.validate()
        self._send({"type": "state", **asdict(state)})

    def reset(self) -> None:
        if not self._closed:
            self._send({"type": "reset"})

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._send({"type": "reset"})
            self._send({"type": "disconnect"})
        except OSError:
            pass
        finally:
            self._closed = True
            self._stream.close()

    def __enter__(self) -> VirtualController:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _send(self, message: dict[str, object]) -> None:
        if self._closed:
            raise RuntimeError("Controller is closed.")
        payload = (json.dumps(message, separators=(",", ":")) + "\n").encode()
        self._stream.write(payload)
        self._stream.flush()

    def _connect(self, timeout_ms: int) -> BinaryIO:
        if not hasattr(ctypes, "windll"):
            raise RuntimeError("The HIDMaestro controller bridge requires Windows.")

        kernel32 = ctypes.windll.kernel32
        if not kernel32.WaitNamedPipeW(self._pipe_name, timeout_ms):
            error = kernel32.GetLastError()
            raise ConnectionError(
                "HIDMaestro controller bridge is not available. Run "
                "bridges\\hidmaestro\\start.ps1 and approve its administrator "
                f"prompt. Windows error: {error}"
            )
        return open(self._pipe_name, "r+b", buffering=0)
