from typing import List, Optional
import logging
from datetime import datetime
from pathlib import Path
import multiprocessing

# from s3fs import S3FileSystem
import polars as pl
from tqdm import tqdm

from src.file_loader import FSFileLoader
from src.timer import Timer

logging.basicConfig(level=logging.INFO)

NOW = datetime.today().strftime("%d%m%y%H%M")


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
            for i, line in enumerate(tqdm(f)):
                # TODO: Remove. Temporal exit for testing purposes
                # if i > 5000:
                #     return [paths, labels]

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
) -> bytes | None:
    # logging.debug(f"Sample path to load: {self.s3_bucket_name}{sample_path}")

    try:
        with file_loader.load(
            Path(sample_path),
            mode="rb",
        ) as f:
            sample = f.read()
            logging.debug(f"Loading sample content: {sample}")
        return sample
    except FileNotFoundError as e:
        logging.debug(e)
        logfolder_path = Path(f"logs/{Path(__file__).stem}")
        logfolder_path.mkdir(parents=True, exist_ok=True)
        logfile_path = logfolder_path / f"missing_image_samples-{NOW}.log"
        with logfile_path.open(mode="a") as f:
            f.write(f"{sample_path}\n")
    except Exception as e:
        logging.error(f"Error loading {sample_path}")
        logging.error(e)
        raise e


@Timer(name="Dataset transformation to polars df as parquet file")
def image_data_annotation_to_df(
    items: List[List[str]],
    shard_i: int,
    dataset_file_path: Path,
    dataset_parquet_path: Path,
    s3_bucket_name: str,
    compression: pl._typing.ParquetCompression = "zstd",
    compression_level: Optional[int] = None,
    use_pyarrow: bool = False,
):
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
                load_sample,
                return_dtype=pl.Binary,
            )
            .alias("sample"),
        )
        .drop_nulls()  # NOTE: Clean missing images. List logged in the missing_images_...log file generated during runtime.
        .with_columns(pl.col("label").str.split(by=" "))
        .select(["sample", "label"])
    )

    logging.info(f"sharded_train_df.head(): {train_df.head()}")
    logging.debug(f"sharded_train_df.head(): {train_df.head()}")
    logging.debug(f"sharded_train_df.tail(): {train_df.tail()}")
    logging.debug(train_df.collect_schema().names())
    logging.info(train_df.describe())

    # NOTE: Data preparation steps could be done here

    logging.info(f"data/{dataset_parquet_path}")
    train_df.write_parquet(
        Path(
            f"data/{dataset_parquet_path.stem}.{shard_i:04d}{dataset_parquet_path.suffix}"
        ),
        use_pyarrow=use_pyarrow,
        compression=compression,
        compression_level=compression_level,
    )
    logging.info(f"Writing s3://{s3_bucket_name}/{dataset_parquet_path}")
    train_df.write_parquet(
        f"s3://{s3_bucket_name}/{dataset_parquet_path.parent}/{dataset_parquet_path.stem}.{shard_i:04d}{dataset_parquet_path.suffix}",
        use_pyarrow=use_pyarrow,
        compression=compression,
        compression_level=compression_level,
        storage_options={"aws_region": "eu-central-1"},
    )
    logging.info(f"s3://{s3_bucket_name}/{dataset_parquet_path} file written")


@Timer(name="Dataset transformation to polars df as a set of sharded parquet files")
def mproc_image_dataset_to_sharded_df(
    dataset_file_path: Path,
    dataset_parquet_path: Path,
    s3_bucket_name: str,
    num_of_shards: int = multiprocessing.cpu_count(),
    compression: pl._typing.ParquetCompression = "zstd",
    compression_level: Optional[int] = None,
    use_polars_pyarrow: bool = False,
):
    items = load_data_annotation(dataset_file_path, FSFileLoader())
    num_of_items = len(items[0])

    logging.debug(f"num_of_items: {num_of_items}")

    slice_length = -(num_of_items // -num_of_shards)  # NOTE: Ceil division
    logging.debug(f"slice_length: {slice_length}")

    for shard_i in range(num_of_shards):
        offset = shard_i * slice_length
        logging.debug(f"offset: {offset}")
        logging.debug(f"offset + slice_length: {offset + slice_length}")

        if offset + slice_length < num_of_items:
            sharded_items = [
                items[0][offset : offset + slice_length],
                items[1][offset : offset + slice_length],
            ]
        else:
            sharded_items = [items[0][offset::], items[1][offset::]]

        proc = multiprocessing.get_context("spawn").Process(
            target=image_data_annotation_to_df,
            args=(
                sharded_items,
                shard_i,
                dataset_file_path,
                dataset_parquet_path,
                s3_bucket_name,
                compression,
                compression_level,
                use_polars_pyarrow,
            ),
        )
        proc.start()

        logging.info(f"Executed sub process to create shard_{shard_i:04d}")


if __name__ == "__main__":
    mproc_image_dataset_to_sharded_df(
        dataset_file_path=Path(
            "data/images/wiris-math-online-incomplete/train_21082019.txt"
        ),
        dataset_parquet_path=Path(
            "df/images/wiris-math-online-incomplete/zstd-none-15-pyarrow/train_21082019.parquet"
        ),
        s3_bucket_name="wiris-ml-datasets",
        num_of_shards=15,
        compression="zstd",
        compression_level=None,  # NOTE: 22 is maximal compression
        use_polars_pyarrow=True,
    )
