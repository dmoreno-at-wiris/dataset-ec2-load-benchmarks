from dataclasses import (
    dataclass,
    # field
)

# from typing import Iterable, ClassVar, Optional, Callable
from enum import Enum

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.timer import Timer


class DatasetSource(Enum):
    DISK = 0
    S3 = 1
    OTHER = -1


class DatasetMode(Enum):
    FILES = 0
    DF = 1
    COMPRESSED = 2
    OTHER = -1


@dataclass
class DatasetLoadBenchmark:
    load_from: DatasetSource
    load_as: DatasetMode
    # dataset: Dataset
    dataloader: DataLoader
    iterations_number: int = 4
    train_epochs: int = 1

    # train_features: ClassVar[Iterable]
    # train_labels: ClassVar[Iterable]

    def __post_init__(self) -> None:
        train_features, train_labels = next(iter(self.dataloader))

        # print(f"First sample content: {train_features[0]}")
        # print(f"First lable: {train_labels[0]}")

    def train_mock(self) -> float:
        t = Timer(name=f"{self.load_from}-{self.load_as}-train-mock")
        sum_t = 0
        for epoch in range(self.train_epochs):
            t.start()
            # for train_features, train_labels in tqdm(self.dataloader):
            for train_features, train_labels in tqdm(self.dataloader):
                pass
            sum_t += t.stop()
        mean_t = sum_t / self.iterations_number
        return mean_t

    def measure(self) -> float:
        sum_t = 0
        for i in range(self.iterations_number):
            sum_t += self.train_mock()

        mean_t = sum_t / self.iterations_number
        print(
            "\n\nMean time for getting all taining samples"
            "\n---"
            f"\n\t- loaded from {self.load_from}"
            f"\n\t- as {self.load_as}"
            f"\n\n Mean time: {mean_t}"
        )
        return mean_t
