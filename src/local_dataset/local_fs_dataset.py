from typing import List, Optional

import logging
from pathlib import Path
# TODO: Improve Dataloader collate to be able to work with the specific
# data as in the train repo
# -
# import json

from src.file_loader import FSFileLoader
from src.ubiquitous_datasets import UbiquitousDataset


class LocalFSDataset(UbiquitousDataset):
    def __init__(
        self,
        datasets_path: Path,
        data_annotation_file_path: Path,
        _samples_dir_path: Optional[Path] = None,
        data_annotation_separator: str = " ",
    ) -> None:
        self.file_loader: FSFileLoader = FSFileLoader()
        self.datasets_path = datasets_path
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

    # TODO: Improve Dataloader collate to be able to work with the specific
    # data as in the train repo
    # -
    # def __getitem__(self, idx: int) -> tuple[List[List[List[float]]], str]:
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
                self.datasets_path / self.data_annotation_file_path,
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
                self.datasets_path / sample_path,
                mode="rb",
            ) as f:
                sample = f.read()
            return sample
        except Exception as e:
            logging.error(f"Error loading {self.datasets_path}/{sample_path}")
            logging.error(e)
            raise e

    # TODO: Improve Dataloader collate to be able to work with the specific
    # data as in the train repo
    # -
    # def load_s3_sample(self, sample_path: Path) -> List[List[List[float]]]:
    #     # logging.debug(f"Sample path to load: {self.s3_bucket_name}{sample_path}")
    #     try:
    #         with self.file_loader.load(
    #             self.datasets_path / sample_path,
    #             mode="rb",
    #         ) as f:
    #             sample = f.read()
    #             if sample:
    #                 return json.loads(sample)
    #             return []
    #     except Exception as e:
    #         logging.error(f"Error loading {self.datasets_path}/{sample_path}")
    #         logging.error(e)
    #         raise e

    def load_dataset(self, file_path: Path, *args, **kwargs) -> List[List[str]]:
        return self.load_s3_data_annotation()
