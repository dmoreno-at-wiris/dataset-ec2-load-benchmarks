from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Dict, IO
from dataclasses import dataclass
from pathlib import Path
from io import IOBase

from s3fs import S3FileSystem
import polars as pl


class fileLoader(ABC):
    @abstractmethod
    def load(self, file_path: Path, *args, **kwargs) -> IO | IOBase | pl.LazyFrame:
        pass

    # TODO: Consider if the following is really needed
    # @abstractmethod
    # def read(self) -> str | bytes | pl.LazyFrame:
    #     pass

    # @abstractmethod
    # def write(self) -> None:
    #     pass

    # TODO: Consider if the following is needed
    # def __enter__(self):
    #     return self

    # def __exit__(self, *exc_info):
    #     pass


@dataclass
class FSFileLoader(fileLoader):
    def load(
        self, file_path: Path, mode: str = "r", encoding: Optional[str] = None
    ) -> IO:
        return file_path.open(
            mode=mode,
            encoding=encoding,
        )


@dataclass
class S3FileLoader(fileLoader):
    fs: ClassVar[S3FileSystem] = S3FileSystem()
    s3_bucket_name: str

    def load(
        self, file_path: Path, mode: str = "r", encoding: Optional[str] = None
    ) -> IOBase:
        return self.fs.open(
            f"s3://{self.s3_bucket_name}/{file_path}",
            mode=mode,
            encoding=encoding,
        )


@dataclass
class S3CSVLoader(fileLoader):
    s3_bucket_name: str

    def load(self, file_path: Path, schema: Optional[Dict] = None) -> pl.LazyFrame:
        return pl.scan_csv(
            f"s3://{self.s3_bucket_name}/{file_path}",
            schema=schema,
            # ignore_errors = True,
            # storage_options = {"aws_region": AWS_FIREHOSE_EVENTS_S3_REGION},
        )


@dataclass
class S3ParquetLoader(fileLoader):
    s3_bucket_name: str

    def load(
        self,
        file_path: Path,
        schema: Optional[Dict] = None,
        aws_region: str = "eu-central-1",
    ) -> pl.LazyFrame:
        return pl.scan_parquet(
            f"s3://{self.s3_bucket_name}/{file_path}",
            schema=schema,
            # ignore_errors = True,
            storage_options={"aws_region": aws_region},
        )
