from typing import List
from abc import abstractmethod
from pathlib import Path

from torch.utils.data import Dataset
import polars as pl


class UbiquitousDataset(Dataset):
    @abstractmethod
    def load_dataset(
        self, file_path: Path, *args, **kwargs
    ) -> List[List[str]] | pl.LazyFrame:
        pass
