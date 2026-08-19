

from abc import ABC, abstractmethod

class ProfileMetric(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> dict[str, float]: ...