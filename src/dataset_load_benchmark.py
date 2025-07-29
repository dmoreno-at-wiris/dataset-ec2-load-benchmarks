from dataclasses import (
    dataclass,
    # field
)

# from typing import Iterable, ClassVar, Optional, Callable
from enum import Enum

from tqdm import tqdm
from torch.utils.data import DataLoader

# TODO: Import rich print
from rich import print, markdown, table, align

from src.timer import Timer


class DatasetSource(Enum):
    DISK = 0
    S3 = 1
    OTHER = -1


class DatasetMode(
    Enum
):  # TODO: Consider moving towards a more descriptive enums. E.g.: PARQUET, PARQUET_COMPRESSED, ...
    FILES = 0
    DF = 1
    COMPRESSED = 2
    WDS = 3
    OTHER = -1


@dataclass
class DatasetLoadBenchmark:
    load_from: DatasetSource
    load_as: DatasetMode
    # dataset: Dataset
    dataloader: DataLoader
    log_table: table.Table
    benchmark_name: str = ""
    iterations_number: int = 4
    train_epochs: int = 1
    # train_features: ClassVar[Iterable]
    # train_labels: ClassVar[Iterable]

    def __post_init__(self) -> None:
        train_features, train_labels = next(iter(self.dataloader))

        # print(f"First sample content: {train_features[0]}")
        # print(f"First lable: {train_labels[0]}")

    def train_mock(self) -> float:
        t = Timer(
            name=f"{self.load_from}-{self.load_as}-train-mock",
            text="**One epoch training time**:\n{:0.4f} seconds\n",
            logger=(lambda text: print(align.Align.center(markdown.Markdown(text)))),
        )
        sum_t = 0
        for epoch in range(self.train_epochs):
            t.start()
            # for train_features, train_labels in tqdm(self.dataloader):
            for train_features, train_labels in tqdm(self.dataloader):
                pass
            sum_t += t.stop()
        mean_t = sum_t / self.train_epochs
        return mean_t

    def measure(self) -> float:
        sum_t = 0
        for i in range(self.iterations_number):
            sum_t += self.train_mock()

        mean_t = sum_t / self.iterations_number

        self.log_table.add_row(
            self.benchmark_name,
            f"{self.load_from}",
            f"{self.load_as}",
            f"{mean_t}",
        )
        print(align.Align.center(self.log_table))
        del self.log_table
        return mean_t
