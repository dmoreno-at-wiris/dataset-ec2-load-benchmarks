# NOTE:
# Reference: https://www.youtube.com/watch?v=v_PacO-3OGQ

from typing import Dict, Generator  # , Optional
import logging
from pathlib import Path
# from itertools import islice

# from s3fs import S3FileSystem
from tqdm import tqdm
import webdataset as wds

from src.file_loader import FSFileLoader  # , S3FileLoader

# logging.basicConfig(level=logging.DEBUG)


def load_sample(sample_path: Path, file_loader: FSFileLoader = FSFileLoader()) -> str:
    # logging.debug(f"Sample path to load: {self.s3_bucket_name}{sample_path}")

    try:
        with file_loader.load(
            sample_path,
            mode="r",
        ) as f:
            return f.read()

    except Exception as e:
        logging.error(f"Error loading {sample_path}")
        logging.error(e)
        raise e


def webdataset_sample_generator(
    data_annotation_file_path: Path,
    file_loader: FSFileLoader = FSFileLoader(),
    data_annotation_separator: str = " ",
) -> Generator[Dict[str, str]]:
    try:
        with file_loader.load(
            data_annotation_file_path,
            mode="r",
            encoding="utf-8",
        ) as f:
            for line in f:
                _path, label = f"{line}".split(data_annotation_separator, 1)
                path = Path(_path)

                sample = {
                    "__key__": f"{data_annotation_file_path.stem}/{path.stem}",
                    "stroke.txt": load_sample(data_annotation_file_path.parent / path),
                    "label.txt": label,
                }
                logging.debug(f"Writing {sample}")
                yield sample
    except Exception as e:
        logging.error(e)
        raise e


def strokes_dataset_to_webdataset(
    dataset_file_path: Path,
    webdataset_path: Path,
    # TODO: Upload tar file to S3
    # s3_bucket_name: Optional[str] = None,
    # s3_folder_path: Optional[Path] = None,
):
    logging.info(f"{webdataset_path}")

    with wds.TarWriter(str(webdataset_path)) as sink:
        logging.info(f"Writing {webdataset_path}")
        for sample in tqdm(webdataset_sample_generator(dataset_file_path)):
            sink.write(sample)

        # NOTE:
        # if s3_bucket_name:
        #     logging.info(
        #         f"Writing s3://{s3_bucket_name}/{s3_folder_path / sink.fileobj}"
        #     )
        #     S3FileLoader(s3_bucket_name).cp_to_s3(
        #         sink.fileobj, copy_path=s3_folder_path / sink.fileobj
        #     )


def strokes_dataset_to_sharded_webdataset(
    dataset_file_path: Path,
    webdataset_path: str,
    shard_size: int,
    # TODO: Upload tar file to S3
    # s3_bucket_name: Optional[str] = None,
    # s3_folder_path: Optional[Path] = None,
):
    logging.info(f"{webdataset_path}")

    with wds.ShardWriter(webdataset_path, maxcount=shard_size) as sink:
        logging.info(f"Writing {webdataset_path}")
        for sample in tqdm(webdataset_sample_generator(dataset_file_path)):
            sink.write(sample)

        # NOTE: EITHER THIS:
        # if s3_bucket_name:
        #     logging.info(
        #         f"Writing s3://{s3_bucket_name}/{s3_folder_path / sink.pattern}"
        #     )
        #     S3FileLoader(s3_bucket_name).cp_to_s3(
        #         sink.pattern, copy_path=s3_folder_path / sink.pattern, recursive=True
        #     )
    # NOTE: OR THIS:
    # if s3_bucket_name:
    #     logging.info(f"Writing s3://{s3_bucket_name}/{webdataset_path}")
    #     S3FileLoader(s3_bucket_name).cp_to_s3(
    #         Path(webdataset_path).parent, copy_path=s3_folder_path, recursive=True
    #     )


# strokes_dataset_to_webdataset(
#     dataset_file_path=Path(
#         "data/strokes/wiris-math-online-incomplete/train_21082019.txt"
#     ),
#     webdataset_path=Path(
#         "data/wds/strokes/wiris-math-online-incomplete/train_21082019.tar"
#     ),
#     # s3_bucket_name="wiris-ml-datasets",
#     # s3_folder_path=Path("wds/strokes/wiris-math-online-incomplete/"),
# )

strokes_dataset_to_sharded_webdataset(
    dataset_file_path=Path(
        "data/strokes/wiris-math-online-incomplete/train_21082019.txt"
    ),
    webdataset_path="data/wds/strokes/wiris-math-online-incomplete/train_21082019_part%06d.tar",
    shard_size=30000,
    # s3_bucket_name="wiris-ml-datasets",
    # s3_folder_path=Path("wds/strokes/wiris-math-online-incomplete/"),
)
