from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Dict, IO
from dataclasses import dataclass
from pathlib import Path
from io import IOBase
import logging

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
        logging.info(f"Loading {file_path}...")
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
        logging.info(f"Loading {file_path}...")
        return self.fs.open(
            f"s3://{self.s3_bucket_name}/{file_path}",
            mode=mode,
            encoding=encoding,
        )


@dataclass
class S3CSVLoader(fileLoader):
    s3_bucket_name: str

    def load(self, file_path: Path, schema: Optional[Dict] = None) -> pl.LazyFrame:
        logging.info(f"Loading {file_path}...")
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
        logging.info(f"Loading {file_path}...")
        return pl.scan_parquet(
            f"s3://{self.s3_bucket_name}/{file_path}",
            schema=schema,
            # ignore_errors = True,
            # TODO: Consider proper sharding config
            # glob=True,
            # rechunk=True,
            storage_options={"aws_region": aws_region},
        )


@dataclass
class S3ParquetPLArrowLoader(fileLoader):
    s3_bucket_name: str

    def load(
        self,
        file_path: Path,
        schema: Optional[Dict] = None,
        aws_region: str = "eu-central-1",  # TODO: Consider moving the region as part of the Object definition instead of being a class parameter
    ) -> pl.LazyFrame:
        logging.debug(f"Loading {file_path}...")
        return pl.scan_pyarrow_dataset(
            f"s3://{self.s3_bucket_name}/{file_path}",
            schema=schema,
            # ignore_errors = True,
            # TODO: Consider proper sharding config
            # glob=True,
            # rechunk=True,
            storage_options={"aws_region": aws_region},
        )
