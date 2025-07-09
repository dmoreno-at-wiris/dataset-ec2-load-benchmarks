from typing import List
import logging
from pathlib import Path
import json

# from s3fs import S3FileSystem
import polars as pl
from tqdm import tqdm

from src.file_loader import FSFileLoader
from src.timer import Timer

# logging.basicConfig(level=logging.DEBUG)


def load_data_annotation(
    data_annotation_file_path: Path,
    file_loader: FSFileLoader = FSFileLoader(),
    data_annotation_separator: str = " ",
) -> List[List[str]]:
    items = []
    paths = []
    labels = []
    try:
        with file_loader.load(
            data_annotation_file_path,
            mode="r",
            encoding="utf-8",
        ) as f:
            for line in tqdm(f):
                path, label = f"{line}".split(data_annotation_separator, 1)

                paths.append(path)
                labels.append(label)
            items = [paths, labels]
        return items
    except Exception as e:
        logging.error(e)
        raise e


def load_sample(
    sample_path: str, file_loader: FSFileLoader = FSFileLoader()
) -> List[List[List[float]]]:
    # logging.debug(f"Sample path to load: {self.s3_bucket_name}{sample_path}")

    try:
        with file_loader.load(
            Path(sample_path),
            mode="r",
        ) as f:
            sample = f.read()
            if sample:
                return json.loads(sample)
            return []
    except Exception as e:
        logging.error(f"Error loading {sample_path}")
        logging.error(e)
        raise e


@Timer(name="Dataset transformation to polars df as parquet file")
def strokes_dataset_to_df(
    dataset_file_path: Path, dataset_parquet_path: Path, s3_bucket_name: str
):
    # TODO: Get dataset name from the file name and the folder outside?
    # dataset_file_path...

    # train_df = pl.scan_csv(
    #         dataset_file_path,
    #         # schema=schema,
    #         separator = " ",
    #     )

    # logging.debug(f"Daily transformation schema: {train_df.collect_schema().names()}")
    # logging.debug(f"Daily transformation head: {train_df.head()}")

    items = load_data_annotation(dataset_file_path, FSFileLoader())

    train_df = (
        pl.DataFrame(items)
        .rename(
            {
                "column_0": "sample_path",
                "column_1": "label",
            }
        )
        .with_columns(
            pl.col("sample_path")
            .str.replace("^./", f"{dataset_file_path.parent}/")
            .map_elements(
                load_sample, return_dtype=pl.List(pl.List(pl.List(pl.Float64)))
            )
            # .map_elements(load_sample, return_dtype=pl.List)
            # .map_elements(load_sample, return_dtype=pl.List(pl.Float64))
            .alias("sample"),
        )
    )
    print(train_df.head())
    print(train_df.collect_schema().names())

    logging.info(f"data/{dataset_parquet_path}")
    train_df.write_parquet(
        Path(f"data/{dataset_parquet_path.name}"),
        compression="zstd",
        # compression_level=22,
        compression_level=22,
        partition_chunk_size_bytes=500000,
    )
    logging.info(f"Writing s3://{s3_bucket_name}/{dataset_parquet_path}")
    train_df.write_parquet(
        f"s3://{s3_bucket_name}/{dataset_parquet_path}",
        compression="zstd",
        compression_level=22,
        partition_chunk_size_bytes=500000,
        storage_options={"aws_region": "eu-central-1"},
    )


strokes_dataset_to_df(
    dataset_file_path=Path(
        "data/strokes/wiris-math-online-incomplete/train_21082019.txt"
    ),
    dataset_parquet_path=Path(
        "df/strokes/wiris-math-online-incomplete/train_21082019.sharded.parquet"
    ),
    s3_bucket_name="wiris-ml-datasets",
)
