from abc import ABC, abstractmethod
from collections.abc import Mapping

from rich.table import Table

MetricResult = dict[str, float]
FormattedMetricLines = tuple[str, ...]


class ProfileMetric(ABC):
    name: str

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> MetricResult: ...
