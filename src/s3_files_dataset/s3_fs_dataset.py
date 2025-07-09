# from abc import ABC, abstractmethod ?
# from typing import protocol?
from typing import List, Optional
# from dataclasses import dataclass

# import polars as pl
# from s3fs.core import S3FileSystem
# from s3fs import S3FileSystem
import logging
from pathlib import Path

from src.file_loader import S3FileLoader
from src.ubiquitous_datasets import UbiquitousDataset

# TODO: Consider the possibility of adapting to specific Abstract Factories for our case

# class S3FSDataset(ABC?): OR class S3FSDataset(Protocol?):

#     @abstractmethod
#     def load_data_annotation(self, separator: str = " ") -> List? | pl.LazyFrame?:
#         pass

#     @abstractmethod
#     def load_data_samples(self) -> List? | pl.LazyFrame?:
#         pass

#     @abstractmethod
#     def get_data_samples(self, idx: int) -> List[str, str]:
#         pass

#     @abstractmethod
#     def f(self) -> ?:
#         pass

# TODO: Consider the possibility of adding an abstract factory of S3Loader that
# could be used to create the needed classes to load S3 data

# class S3Loader(ABC?): OR class S3Loader(Protocol?):

#     @abstractmethod
#     def load_data_annotation(self, separator: str = " ") -> List? | pl.LazyFrame?:
#         pass

#     @abstractmethod
#     def load_data_samples(self) -> List? | pl.LazyFrame?:
#         pass


class S3FSDataset(UbiquitousDataset):
    def __init__(
        self,
        s3_bucket_name: str,
        data_annotation_file_path: Path,
        _samples_dir_path: Optional[Path] = None,
        data_annotation_separator: str = " ",
    ) -> None:
        self.s3_bucket_name = s3_bucket_name
        self.file_loader: S3FileLoader = S3FileLoader(s3_bucket_name)
        self.data_annotation_file_path = data_annotation_file_path
        self._samples_dir_path = _samples_dir_path
        self.data_annotation_separator = data_annotation_separator

        self.items: List[List[str]] = self.load_s3_data_annotation()

    @property
    def samples_dir_path(self) -> Path:
        if self._samples_dir_path is None:
            return self.data_annotation_file_path.parent

        return self._samples_dir_path

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[str | bytes, str]:
        path, label_str = self.items[idx]
        # sample_path = self.samples_dir_path / path
        sample_path = Path(f"{self.samples_dir_path}/{path}")
        logging.debug(f"self.samples_dir_path: {self.samples_dir_path}")
        logging.debug(f"sample_path: {sample_path}")
        # return sample_path, label_str
        sample = self.load_s3_sample(sample_path)
        return sample, label_str

    def load_s3_data_annotation(self) -> List[List[str]]:
        items = []
        try:
            with self.file_loader.load(
                self.data_annotation_file_path,
                mode="r",
                encoding="utf-8",
            ) as f:
                for line in f:
                    path, label = f"{line}".split(self.data_annotation_separator, 1)

                    # TODO: We'll need to deal with unexisting samples' path.
                    # However I would deal with it in a later step, during
                    # sample load

                    # if not os.path.exists(path):
                    #     logging.warning(
                    #         f"Image path specified in {data_file} not found: {path}. Skipping item."
                    #     )
                    #     continue

                    items.append((path, label))
            return items
        except Exception as e:
            logging.error(e)
            raise e

    def load_s3_sample(self, sample_path: Path) -> str | bytes:
        # logging.debug(f"Sample path to load: {self.s3_bucket_name}{sample_path}")
        try:
            with self.file_loader.load(
                sample_path,
                mode="rb",
            ) as f:
                sample = f.read()
            return sample
        except Exception as e:
            logging.error(f"Error loading s3://{self.s3_bucket_name}/{sample_path}")
            logging.error(e)
            raise e

    def load_dataset(self, file_path: Path, *args, **kwargs) -> List[List[str]]:
        return self.load_s3_data_annotation()
