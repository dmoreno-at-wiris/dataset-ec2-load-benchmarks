import logging

from memory_profiler import profile
from rich import print, table, markdown, inspect
from torch.utils.data import DataLoader
from torch import cuda
from datasets import load_dataset

from src.dataset_load_benchmark import DatasetLoadBenchmark, DatasetSource, DatasetMode
from src.timer import Timer

logging.basicConfig(level=logging.INFO)


def local_full_hf_dataset_benchmark(benchmarks_log_table):
    benchmark_name = "Local HF Parquet"
    print(markdown.Markdown(f"# {benchmark_name}"))
    t_s3_df = Timer(
        name=f"Loading {benchmark_name} data time",
        text="Elapsed time: {:0.4f} seconds\n",
    )
    print(markdown.Markdown("**Dataset instantiation time**:"))
    t_s3_df.start()
    train_s3_hf_dataset = load_dataset(
        "parquet",
        data_files={"train": "data/train_21082019.parquet"},
        split="train",
        streaming=True,
    ).remove_columns("sample_path")

    t_s3_df.stop()

    print(markdown.Markdown("**Dataload instantiation time**:"))
    t_s3_df.start()
    train_s3_hf_dataloader = DataLoader(
        # NOTE: Type ignore due to https://github.com/microsoft/pyright/issues/698
        train_s3_hf_dataset.with_format("torch"),  # type: ignore
        batch_size=32,
        num_workers=8,
        prefetch_factor=4,
        pin_memory=cuda.is_available(),
        # shuffle=True,
    )
    t_s3_df.stop()

    s3_hf_benchmark = DatasetLoadBenchmark(
        load_from=DatasetSource.DISK,
        load_as=DatasetMode.DF,
        dataloader=train_s3_hf_dataloader,
        log_table=benchmarks_log_table,
        benchmark_name=benchmark_name,
        iterations_number=1,
        train_epochs=2,
    )

    s3_hf_benchmark.measure()


def local_sharded_hf_dataset_benchmark(benchmarks_log_table):
    benchmark_name = "Local HF Sharded Streaming Parquet"
    print(markdown.Markdown(f"# {benchmark_name}"))
    t_s3_df = Timer(
        name=f"Loading {benchmark_name} data time",
        text="Elapsed time: {:0.4f} seconds\n",
    )
    print(markdown.Markdown("**Dataset instantiation time**:"))
    t_s3_df.start()
    train_s3_hf_dataset = load_dataset(
        "parquet",
        data_files={"train": "data/train_21082019.*.parquet"},
        split="train",
        streaming=True,
    ).select_columns(["sample", "label"])
    t_s3_df.stop()

    inspect(train_s3_hf_dataset)

    print(markdown.Markdown("**Dataload instantiation time**:"))
    t_s3_df.start()
    train_s3_hf_dataloader = DataLoader(
        # NOTE: Type ignore due to https://github.com/microsoft/pyright/issues/698
        train_s3_hf_dataset,  # type: ignore
        # train_s3_hf_dataset.with_format("torch"),
        # batch_size=32,
        # num_workers=8,
        # prefetch_factor=4,
        pin_memory=cuda.is_available(),
        # shuffle=True,
    )
    t_s3_df.stop()

    s3_hf_benchmark = DatasetLoadBenchmark(
        load_from=DatasetSource.DISK,
        load_as=DatasetMode.DF,
        dataloader=train_s3_hf_dataloader,
        log_table=benchmarks_log_table,
        benchmark_name=benchmark_name,
        iterations_number=1,
        train_epochs=2,
    )

    s3_hf_benchmark.measure()


@profile
def main():
    benchmarks_log_table = table.Table(title="Mean (mock) training time")
    benchmarks_log_table.add_column("Benchmark", justify="left")
    benchmarks_log_table.add_column("Dataset source", justify="center")
    benchmarks_log_table.add_column("Dataset format", justify="center")
    benchmarks_log_table.add_column("Mean time (s)", justify="right")

    # NOTE: Local HF Parquet
    # local_full_hf_dataset_benchmark(benchmarks_log_table)

    # NOTE: Local HF Sharded Parquet
    local_sharded_hf_dataset_benchmark(benchmarks_log_table)


if __name__ == "__main__":
    main()
