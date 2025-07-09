# from abc import ABC, abstractmethod ?
# from typing import protocol?
from typing import (
    # List,
    Optional,
)
# from dataclasses import dataclass

import polars as pl

# TODO: Consider adding GPU
# https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.lazyframe.engine_config.GPUEngine.html
# pl.lazyframe.engine_config.GPUEngine(
#     device: int | None = None,
#     memory_resource: Any | None = None,
#     raise_on_fail: bool = False,
# )

import logging
from pathlib import Path

# TODO: Consider parallelism in this case collecting df with collect_async() instead
# import asyncio

from src.file_loader import LocalParquetLoader
from src.ubiquitous_datasets import UbiquitousDataset


class LocalParquetDataset(UbiquitousDataset):
    def __init__(
        self,
        dataset_file_path: Path,
        _samples_dir_path: Optional[Path] = None,
        # TODO: Consider slicing the LazyFrame to only load the required buffer of data
        # _collect_buffer_size: int = 1000
    ) -> None:
        self.file_loader: LocalParquetLoader = LocalParquetLoader()
        self.dataset_file_path = dataset_file_path
        self._samples_dir_path = _samples_dir_path

        self.items: pl.DataFrame = self.load_parquet_dataset().collect(
            engine="streaming"
        )
        self.df_length: int = self.items.select(pl.len()).item()

        # self.df_length: int = 0
        # self.items: pl.LazyFrame = self.load_parquet_dataset()

        # TODO: Consider slicing the LazyFrame to only load the required buffer of data
        # self._collect_buffer_size = _collect_buffer_size

    @property
    def samples_dir_path(self) -> Path:
        if self._samples_dir_path is None:
            return self.dataset_file_path.parent

        return self._samples_dir_path

    # TODO: Consider slicing the LazyFrame to only load the required buffer of data
    # @property
    # def collect_buffer_size(self) -> int:
    #     return self._collect_buffer_size

    # TODO: Consider slicing the LazyFrame to only load the required buffer of data
    # @collect_buffer_size.setter
    # def collect_buffer_size(self, value) -> None:
    #     raise Exception("collect_buffer_size cannot be modified on the fly")

    def __len__(self) -> int:
        return self.df_length

    # TODO: The following code returns an error. Improve Dataloader collate to
    # be able to work with the specific data as in the train repo
    # -
    # def __getitem__(self, idx: int) -> tuple[List[List[List[float]]], str]:
    #     return self.items["sample"][idx].to_list(), self.items["label"][idx]

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return str(self.items["sample"][idx].to_list()), self.items["label"][idx]

    # TODO: Consider returning directly a Tensor or Numpy array of the sample content
    # def __getitem__(self, idx: int) -> tuple[List[List[List[float]]], str]:
    #     return self.items["sample"][idx].to_torch(), self.items["label"][idx]

    # def __getitem__(self, idx: int) -> tuple[List[List[List[float]]], str]:
    #     m =(idx - idx % self.collect_buffer_size) / self.collect_buffer_size
    #     buffer_df = self.items[m*self.collect_buffer_size : (m+1)*self.collect_buffer_size]
    #     return buffer_df["sample"][idx], buffer_df["label"][idx]

    def load_parquet_dataset(self) -> pl.LazyFrame:
        try:
            df = self.file_loader.load(
                self.dataset_file_path,
                schema={
                    "sample_path": pl.String,
                    "label": pl.String,
                    "sample": pl.List(pl.List(pl.List(pl.Float64))),
                },
            ).select(pl.col("sample"), pl.col("label"))

            # self.df_length = df.select(pl.len()).collect().item()

            return df
        except Exception as e:
            logging.error(e)
            raise e

    def load_dataset(self, *args, **kwargs) -> pl.LazyFrame:
        return self.load_parquet_dataset()
