# from typing import List, cast
import logging
from pathlib import Path
import gc

from torch.utils.data import DataLoader
from torch import cuda

# import webdataset as wds
# from datasets import load_dataset
from memory_profiler import profile
from rich import print, table, markdown

from src.ubiquitous_datasets import UbiquitousDataset

# from src.s3_files_dataset.s3_fs_dataset import S3FSDataset
from src.local_dataset.local_fs_dataset import LocalFSDataset
from src.local_dataset.local_df_dataset import LocalParquetDataset

from src.s3_parquet_dataset.s3_df_dataset import S3ParquetDataset
from src.dataset_load_benchmark import DatasetLoadBenchmark, DatasetSource, DatasetMode
from src.timer import Timer

logging.basicConfig(level=logging.INFO)


def benchmark(
    benchmark_name: str,
    benchmarks_log_table: table.Table,
    dataset: UbiquitousDataset,
    load_from: DatasetSource,
    load_as: DatasetMode,
):
    # NOTE: Time of Dataloader instantiation is negligible
    train_local_dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=8,
        prefetch_factor=4,
        pin_memory=cuda.is_available(),
    )

    local_fs_benchmark = DatasetLoadBenchmark(
        load_from=load_from,
        load_as=load_as,
        dataloader=train_local_dataloader,
        log_table=benchmarks_log_table,
        benchmark_name=benchmark_name,
        iterations_number=1,
    )

    local_fs_benchmark.measure()

    del local_fs_benchmark
    del train_local_dataloader
    del dataset
    gc.collect()


@profile
def main():
    benchmarks_log_table = table.Table(title="Mean (mock) training time")
    benchmarks_log_table.add_column("Benchmark", justify="left")
    benchmarks_log_table.add_column("Dataset source", justify="center")
    benchmarks_log_table.add_column("Dataset format", justify="center")
    benchmarks_log_table.add_column("Mean time (s)", justify="right")

    # NOTE: Local FS
    benchmark_name = "Local FS"
    # print(align.Align.center("\n[yellow underline bold]Local FS"))
    print(markdown.Markdown(f"# {benchmark_name}"))
    train_dataset = LocalFSDataset(
        datasets_path=Path("/home/daniel/ML/dataset-ec2-load-benchmarks/data"),
        data_annotation_file_path=Path(
            "strokes/wiris-math-online-incomplete/train_21082019.txt"
        ),
    )

    benchmark(
        benchmark_name=benchmark_name,
        benchmarks_log_table=benchmarks_log_table,
        dataset=train_dataset,
        load_from=DatasetSource.DISK,
        load_as=DatasetMode.FILES,
    )

    # NOTE: Local Parquet
    benchmark_name = "Local Parquet (data/train_21082019.parquet)"
    print(markdown.Markdown(f"# {benchmark_name}"))
    print(markdown.Markdown("**Dataset instantiation time**:"))
    t_local_df = Timer(
        name="Loading Parquet data time", text="Elapsed time: {:0.4f} seconds\n"
    )
    t_local_df.start()
    train_dataset = LocalParquetDataset(
        dataset_file_path=Path("data/train_21082019.parquet"),
    )
    t_local_df.stop()

    benchmark(
        benchmark_name=benchmark_name,
        benchmarks_log_table=benchmarks_log_table,
        dataset=train_dataset,
        load_from=DatasetSource.DISK,
        load_as=DatasetMode.DF,
    )

    gc.collect()

    # # NOTE: Local HF Parquet
    # t_s3_df = Timer(name="Loading Parquet data time", text="Elapsed time: {:0.4f} seconds\n")
    # t_s3_df.start()
    # train_s3_hf_dataset = load_dataset("parquet", data_files={"train": "data/train_21082019.parquet"}, split="train")

    # t_s3_df.stop()

    # print(markdown.Markdown("**Dataload instantiation time**:"))
    # t_s3_df.start()
    # train_s3_hf_dataloader = DataLoader(
    #     train_s3_hf_dataset,
    #     batch_size=32,
    #     num_workers=8,
    #     prefetch_factor=4,
    #     pin_memory=cuda.is_available(),
    #     # shuffle=True,
    # )
    # t_s3_df.stop()

    # s3_hf_benchmark = DatasetLoadBenchmark(
    #     load_from=DatasetSource.DISK,
    #     load_as=DatasetMode.DF,
    #     dataloader=train_s3_hf_dataloader,
    #     log_table=benchmarks_log_table,
    #     benchmark_name=benchmark_name,
    #     iterations_number=1,
    #     train_epochs=2,
    # )

    # s3_hf_benchmark.measure()

    # TODO: Remove variable with del to avoid memory usage

    # NOTE: S3 Parquet
    benchmark_name = "S3 Parquet"
    print(markdown.Markdown(f"# {benchmark_name}"))
    print(markdown.Markdown("**Dataset instantiation time**:"))
    t_s3_df = Timer(
        name="Loading Parquet data time", text="Elapsed time: {:0.4f} seconds\n"
    )
    t_s3_df.start()
    train_dataset = S3ParquetDataset(
        s3_bucket_name="wiris-ml-datasets",
        dataset_file_path=Path(
            "df/strokes/wiris-math-online-incomplete/train_21082019.parquet"
        ),
    )
    t_s3_df.stop()

    benchmark(
        benchmark_name=benchmark_name,
        benchmarks_log_table=benchmarks_log_table,
        dataset=train_dataset,
        load_from=DatasetSource.S3,
        load_as=DatasetMode.DF,
    )

    gc.collect()

    # # NOTE: S3 HF Parquet
    # t_s3_df = Timer(name="Loading Parquet data time", text="Elapsed time: {:0.4f} seconds\n")
    # t_s3_df.start()
    # train_s3_hf_dataset = load_dataset("parquet", data_files={"train": "s3://wiris-ml-datasets/df/strokes/wiris-math-online-incomplete/train_21082019.parquet"}, split="train")

    # t_s3_df.stop()

    # print(markdown.Markdown("**Dataload instantiation time**:"))
    # t_s3_df.start()
    # train_s3_hf_dataloader = DataLoader(
    #     train_s3_hf_dataset,
    #     batch_size=32,
    #     num_workers=8,
    #     prefetch_factor=4,
    #     pin_memory=cuda.is_available(),
    #     # shuffle=True,
    # )
    # t_s3_df.stop()

    # s3_hf_benchmark = DatasetLoadBenchmark(
    #     load_from=DatasetSource.S3,
    #     load_as=DatasetMode.DF,
    #     dataloader=train_s3_hf_dataloader,
    #     log_table=benchmarks_log_table,
    #     benchmark_name=benchmark_name,
    #     iterations_number=1,
    #     train_epochs=2,
    # )

    # s3_hf_benchmark.measure()

    # TODO: Remove variable with del to avoid memory usage

    # NOTE: S3 Parquet sharded zstd22
    benchmark_name = "S3 Parquet sharded zstd22"
    print(markdown.Markdown(f"# {benchmark_name}"))
    print(markdown.Markdown("**Dataset instantiation time**:"))
    t_s3_sharded_zstd22_df = Timer(
        name="Loading Parquet sharded data time", text="Elapsed time: {:0.4f} seconds\n"
    )
    t_s3_sharded_zstd22_df.start()
    train_dataset = S3ParquetDataset(
        s3_bucket_name="wiris-ml-datasets",
        dataset_file_path=Path(
            "df/strokes/wiris-math-online-incomplete/train_21082019.*.parquet"
        ),
    )
    t_s3_sharded_zstd22_df.stop()

    benchmark(
        benchmark_name=benchmark_name,
        benchmarks_log_table=benchmarks_log_table,
        dataset=train_dataset,
        load_from=DatasetSource.S3,
        load_as=DatasetMode.DF,
    )

    gc.collect()

    # NOTE: S3 Parquet sharded zstd10
    benchmark_name = "S3 Parquet sharded zstd10"
    print(markdown.Markdown(f"# {benchmark_name}"))
    print(markdown.Markdown("**Dataset instantiation time**:"))
    t_s3_sharded_zstd10_df = Timer(
        name="Loading Parquet sharded data time", text="Elapsed time: {:0.4f} seconds\n"
    )
    t_s3_sharded_zstd10_df.start()
    train_dataset = S3ParquetDataset(
        s3_bucket_name="wiris-ml-datasets",
        dataset_file_path=Path(
            "df/strokes/wiris-math-online-incomplete/zstd-10/train_21082019.*.parquet"
        ),
    )
    t_s3_sharded_zstd10_df.stop()

    benchmark(
        benchmark_name=benchmark_name,
        benchmarks_log_table=benchmarks_log_table,
        dataset=train_dataset,
        load_from=DatasetSource.S3,
        load_as=DatasetMode.DF,
    )

    gc.collect()

    # NOTE: S3 Parquet sharded zstdNone
    benchmark_name = "S3 Parquet sharded zstdNone"
    print(markdown.Markdown(f"# {benchmark_name}"))
    print(markdown.Markdown("**Dataset instantiation time**:"))
    t_s3_sharded_zstdNone_df = Timer(
        name="Loading Parquet sharded data time", text="Elapsed time: {:0.4f} seconds\n"
    )
    t_s3_sharded_zstdNone_df.start()
    train_dataset = S3ParquetDataset(
        s3_bucket_name="wiris-ml-datasets",
        dataset_file_path=Path(
            "df/strokes/wiris-math-online-incomplete/zstd-none/train_21082019.*.parquet"
        ),
    )
    t_s3_sharded_zstdNone_df.stop()

    benchmark(
        benchmark_name=benchmark_name,
        benchmarks_log_table=benchmarks_log_table,
        dataset=train_dataset,
        load_from=DatasetSource.S3,
        load_as=DatasetMode.DF,
    )

    gc.collect()

    # TODO: WebDataset NOTE: Pytorch does not support WebDataset anymore
    # Ref:
    # https://docs.pytorch.org/data/0.7/generated/torchdata.datapipes.iter.WebDataset.html
    # t_s3_wds = Timer(name="Loading WebDataset time", text="Elapsed time: {:0.4f} seconds\n")
    # t_s3_wds.start()
    # # NOTE: The following code is not directly supported
    # # train_s3_wds_dataset = wds.WebDataset(
    # #     "s3://wiris-ml-datasets/wds/strokes/wiris-math-online-incomplete/train_21082019_part{000000..000039}.tar"
    # # )
    # # train_s3_wds_dataset = wds.WebDataset(
    # #     "pipe:aws s3 sync s3://wiris-ml-datasets/wds/strokes/wiris-math-online-incomplete/train_21082019_part{000000..000039}.tar ."
    # # )
    # train_s3_wds_dataset = wds.WebDataset(
    #     "pipe:aws s3api get-object --bucket wiris-ml-datasets --key wds/strokes/wiris-math-online-incomplete/train_21082019_part{000000..000039}.tar out | cat out"
    # )
    # t_s3_wds.stop()

    # print(markdown.Markdown("**Dataload instantiation time**:"))
    # t_s3_wds.start()
    # train_s3_wds_dataloader = DataLoader(
    #     train_s3_wds_dataset,
    #     batch_size=32,
    #     num_workers=8,
    #     prefetch_factor=4,
    #     pin_memory=cuda.is_available(),
    # )
    # t_s3_wds.stop()

    # s3_wds_benchmark = DatasetLoadBenchmark(
    #     load_from=DatasetSource.S3,
    #     load_as=DatasetMode.DF,
    #     dataloader=train_s3_wds_dataloader,
    #     log_table=benchmarks_log_table,
    #     benchmark_name=benchmark_name,
    #     iterations_number=1,
    #     train_epochs=2,
    # )

    # s3_wds_benchmark.measure()

    # TODO: Remove variable with del to avoid memory usage


if __name__ == "__main__":
    main()
