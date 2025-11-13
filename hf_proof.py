import logging
from pathlib import Path

from memory_profiler import profile
from rich import print, table, markdown, inspect
from torch.utils.data import DataLoader
from torch import cuda
from datasets import (
    load_dataset,
    Features,
    LargeList,
    # List,
    Array2D,
    Value,
    # ClassLabel,
)

from src.dataset_load_benchmark import DatasetLoadBenchmark, DatasetSource, DatasetMode
from src.timer import Timer

# NOTE: Below added due to https://github.com/huggingface/datasets/issues/6396
# import pyarrow
# pyarrow.PyExtensionType.set_auto_load(True)

logging.basicConfig(level=logging.INFO)

# SYMBOL_FILE = Path("data/symbols.txt")
SYMBOL_FILE = Path(
    "/fsx/datasets/files/strokes/wiris-math-online-incomplete/symbols_21082019.txt"
)
with open(SYMBOL_FILE, "r", encoding="utf-8") as f:
    symbols = [s_line.rstrip() for s_line in f]
token_to_index = {tok: i for i, tok in enumerate(symbols)}

# inspect(symbols)
# inspect(token_to_index)

FEATURES = Features(
    {
        # "sample_path:": Value("string"),
        # "label": Value("string"),
        # "sample": LargeList(LargeList(LargeList(Value("float64")))),
        "sample": LargeList(LargeList(Array2D(shape=(1, 2), dtype="float64"))),
        "label": LargeList(Value("string")),
    }
)


def tokenize(row):
    valid_tokens = [t for t in row["label"] if t in symbols]
    label_tokenized = [token_to_index[t] for t in valid_tokens]

    return row["sample"], label_tokenized


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
        # data_files={"train": "data/train_21082019.parquet"},
        data_files={
            "train": "/fsx/datasets/df/strokes/wiris-math-online-incomplete/train_21082019.parquet"
        },
        split="train",
        streaming=True,
    ).select_columns(["sample", "label"])

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
        # data_files={"train": "data/train_21082019.*.parquet"},
        data_files={
            "train": "/fsx/datasets/df/strokes/wiris-math-online-incomplete/train_21082019.*.parquet"
        },
        split="train",
        streaming=True,
        # column_names=["sample", "label"],
        # features=FEATURES,
    )  # .with_format("torch")#.map(tokenize, batched=True)
    t_s3_df.stop()

    # inspect(train_s3_hf_dataset)
    # inspect(train_s3_hf_dataset[0])
    # inspect(train_s3_hf_dataset[0]["sample"])
    # inspect(train_s3_hf_dataset[0]["label"])
    # inspect(train_s3_hf_dataset.with_format("torch"))

    print(markdown.Markdown("**Dataload instantiation time**:"))
    t_s3_df.start()
    train_s3_hf_dataloader = DataLoader(
        # NOTE: Type ignore due to https://github.com/microsoft/pyright/issues/698
        # train_s3_hf_dataset,  # type: ignore
        train_s3_hf_dataset.with_format("torch"),  # type: ignore
        batch_size=32,
        num_workers=8,
        prefetch_factor=4,
        pin_memory=cuda.is_available(),
        # shuffle=True,
    )
    t_s3_df.stop()

    inspect(train_s3_hf_dataloader)
    inspect(next(iter(train_s3_hf_dataloader)))

    s3_hf_benchmark = DatasetLoadBenchmark(
        load_from=DatasetSource.DISK,
        load_as=DatasetMode.DF,
        dataloader=train_s3_hf_dataloader,
        log_table=benchmarks_log_table,
        benchmark_name=benchmark_name,
        iterations_number=1,
        train_epochs=1,
    )

    s3_hf_benchmark.measure()


def s3_sharded_hf_dataset_benchmark(benchmarks_log_table):
    benchmark_name = "S3 HF Sharded Streaming Parquet"
    print(markdown.Markdown(f"# {benchmark_name}"))
    t_s3_df = Timer(
        name=f"Loading {benchmark_name} data time",
        text="Elapsed time: {:0.4f} seconds\n",
    )
    print(markdown.Markdown("**Dataset instantiation time**:"))
    t_s3_df.start()
    train_s3_hf_dataset = load_dataset(
        "parquet",
        data_files={
            "train": "s3://wiris-ml-datasets/df/strokes/wiris-math-online-incomplete/train_21082019.*.parquet"
        },
        split="train",
        streaming=True,
    ).select_columns(["sample", "label"])
    t_s3_df.stop()

    logging.debug(inspect(train_s3_hf_dataset))

    print(markdown.Markdown("**Dataload instantiation time**:"))
    t_s3_df.start()
    train_s3_hf_dataloader = DataLoader(
        # NOTE: Type ignore due to https://github.com/microsoft/pyright/issues/698
        # train_s3_hf_dataset,  # type: ignore
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
        train_epochs=1,
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

    # NOTE: S3 HF Sharded Parquet
    s3_sharded_hf_dataset_benchmark(benchmarks_log_table)


if __name__ == "__main__":
    main()
