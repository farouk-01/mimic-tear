from collections.abc import Mapping
from types import MappingProxyType

from tensordict import TensorDict
from torch.utils.data import Dataset

from .datasets.tensor import TensorDataset
from utils import profile

class SequenceDataset(Dataset[TensorDict]):
    def __init__(
        self,
        *,
        datasets: Mapping[str, TensorDataset],
        sequence_length: int,
        drop_incomplete: bool = True,
    ) -> None:
        if not datasets:
            raise ValueError("Datasets cannot be empty")

        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero")

        lengths = {len(dataset) for dataset in datasets.values()}

        if len(lengths) != 1:
            raise ValueError("All datasets must have the same length")

        self.datasets = dict(datasets)
        self.sequence_length = sequence_length
        self.drop_incomplete = drop_incomplete

        self._sample_count = next(iter(lengths))

    def __len__(self) -> int:
        if self.drop_incomplete:
            return self._sample_count // self.sequence_length

        return (self._sample_count + self.sequence_length - 1) // self.sequence_length

    @profile
    def __getitem__(self, index: int) -> TensorDict:
        if index < 0:
            index += len(self)

        if not 0 <= index < len(self):
            raise IndexError(index)

        start = index * self.sequence_length
        end = min(start + self.sequence_length, self._sample_count)

        return TensorDict(
            {
                name: dataset.get_range(start, end)
                for name, dataset in self.datasets.items()
            },
            batch_size=[end - start],
            # device="cuda",
        ).lock_()
