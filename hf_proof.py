import logging

from memory_profiler import profile
from rich import print, table, markdown
from torch.utils.data import DataLoader
from torch import cuda
from datasets import load_dataset

from src.dataset_load_benchmark import DatasetLoadBenchmark, DatasetSource, DatasetMode
from src.timer import Timer

logging.basicConfig(level=logging.INFO)


@profile
def main():
    benchmarks_log_table = table.Table(title="Mean (mock) training time")
    benchmarks_log_table.add_column("Benchmark", justify="left")
    benchmarks_log_table.add_column("Dataset source", justify="center")
    benchmarks_log_table.add_column("Dataset format", justify="center")
    benchmarks_log_table.add_column("Mean time (s)", justify="right")

    # NOTE: Local HF Parquet
    benchmark_name = "Local HF Parquet"
    print(markdown.Markdown(f"# {benchmark_name}"))
    t_s3_df = Timer(
        name=f"Loading {benchmark_name} data time",
        text="Elapsed time: {:0.4f} seconds\n",
    )
    t_s3_df.start()
    train_s3_hf_dataset = load_dataset(
        "parquet", data_files={"train": "data/train_21082019.parquet"}, split="train"
    )

    t_s3_df.stop()

    print(markdown.Markdown("**Dataload instantiation time**:"))
    t_s3_df.start()
    train_s3_hf_dataloader = DataLoader(
        train_s3_hf_dataset,
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


if __name__ == "__main__":
    main()
