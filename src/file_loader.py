from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Dict, IO, ContextManager
from dataclasses import dataclass
from pathlib import Path
from io import IOBase
import logging

from s3fs import S3FileSystem
import polars as pl

import dvc.api


class fileLoader(ABC):
    @abstractmethod
    def load(
        self, file_path: Path, *args, **kwargs
    ) -> IO | IOBase | ContextManager | pl.LazyFrame:
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
        logging.debug(f"Loading {file_path}...")
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
        logging.debug(f"Loading {file_path}...")
        return self.fs.open(
            f"s3://{self.s3_bucket_name}/{file_path}",
            mode=mode,
            encoding=encoding,
        )

    def cp_to_s3(
        self, file_path: Path, copy_path: Optional[Path] = None, recursive: bool = False
    ) -> None:
        logging.debug(f"Copying {file_path}...")
        if copy_path:
            return self.fs.cp(
                f"{file_path}",
                f"s3://{self.s3_bucket_name}/{copy_path}",
                recursive=recursive,
            )
        return self.fs.cp(
            f"{file_path}",
            f"s3://{self.s3_bucket_name}/{file_path}",
            recursive=recursive,
        )


@dataclass
class S3CSVLoader(fileLoader):
    s3_bucket_name: str

    def load(self, file_path: Path, schema: Optional[Dict] = None) -> pl.LazyFrame:
        logging.debug(f"Loading {file_path}...")
        return pl.scan_csv(
            f"s3://{self.s3_bucket_name}/{file_path}",
            schema=schema,
            # ignore_errors = True,
            # storage_options = {"aws_region": AWS_FIREHOSE_EVENTS_S3_REGION},
        )


@dataclass
class LocalParquetLoader(fileLoader):
    def load(
        self,
        file_path: Path,
        schema: Optional[Dict] = None,
    ) -> pl.LazyFrame:
        logging.debug(f"Loading {file_path}...")
        return pl.scan_parquet(
            file_path,
            schema=schema,
            # ignore_errors = True,
        )


@dataclass
class S3ParquetLoader(fileLoader):
    s3_bucket_name: str

    def load(
        self,
        file_path: Path,
        schema: Optional[Dict] = None,
        aws_region: str = "eu-central-1",  # TODO: Consider moving the region as part of the Object definition instead of being a class parameter
    ) -> pl.LazyFrame:
        logging.debug(f"Loading {file_path}...")
        return pl.scan_parquet(
            f"s3://{self.s3_bucket_name}/{file_path}",
            schema=schema,
            # ignore_errors = True,
            storage_options={"aws_region": aws_region},
        )


# TODO: PyArrow Loader if needed?

# TODO: HFLoader if needed?


@dataclass
class DVCLoader(fileLoader):
    def load(
        self,
        file_path: Path,
        repo: Optional[str] = None,
        rev: Optional[str] = None,
        remote: str = "",
        # remote_config: Optional[dict] = None,
        # config: Optional[dict] = None,
        mode: str = "r",
        encoding: Optional[str] = None,
    ) -> ContextManager:
        logging.debug(f"Loading {file_path}...")
        return dvc.api.open(
            path=f"{file_path}",
            repo=repo,
            rev=rev,
            remote=remote,
            mode=mode,
            encoding=encoding,
        )
