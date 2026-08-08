from __future__ import annotations


class FrameClockSynchronizer:
    """Translate DXGI presentation ticks into perf_counter's clock domain."""

    def __init__(self) -> None:
        self._minimum_clock_offset_ns: int | None = None
        self._previous_source_ns: int | None = None
        self._previous_frame_ns: int | None = None

    def align(
        self,
        source_timestamp_seconds: float,
        received_timestamp_ns: int,
    ) -> tuple[int, int, bool]:
        source_timestamp_ns = round(source_timestamp_seconds * 1e9)
        clock_offset_ns = received_timestamp_ns - source_timestamp_ns
        if (
            self._minimum_clock_offset_ns is None
            or clock_offset_ns < self._minimum_clock_offset_ns
        ):
            self._minimum_clock_offset_ns = clock_offset_ns

        duplicate = source_timestamp_ns == self._previous_source_ns
        if duplicate:
            frame_timestamp_ns = received_timestamp_ns
        else:
            frame_timestamp_ns = source_timestamp_ns + self._minimum_clock_offset_ns
            frame_timestamp_ns = min(frame_timestamp_ns, received_timestamp_ns)

        if self._previous_frame_ns is not None:
            frame_timestamp_ns = max(frame_timestamp_ns, self._previous_frame_ns + 1)
        self._previous_source_ns = source_timestamp_ns
        self._previous_frame_ns = frame_timestamp_ns
        return source_timestamp_ns, frame_timestamp_ns, duplicate
