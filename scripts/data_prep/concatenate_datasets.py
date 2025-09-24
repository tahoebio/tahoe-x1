# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
import argparse
from pathlib import Path
from typing import Iterable

import datasets


def get_files(path: str) -> Iterable[str]:
    files = [str(f.resolve()) for f in Path(path).glob("chunk*.dataset")]
    return files


def get_datasets(files: Iterable[str]) -> Iterable[datasets.Dataset]:
    return [datasets.load_from_disk(file) for file in files]


def process_datasets(path: str, dataset_name: str):
    merged_dataset = datasets.concatenate_datasets(get_datasets(get_files(path)))
    merged_dataset = merged_dataset.train_test_split(
        test_size=0.01,
        shuffle=True,
        seed=44,
    )
    train_dataset = merged_dataset["train"]
    valid_dataset = merged_dataset["test"]

    print(f"train set number of samples: {len(train_dataset)}")
    print(f"valid set number of samples: {len(valid_dataset)}")

    train_dataset.save_to_disk(f"{path}/{dataset_name}_train.dataset")
    valid_dataset.save_to_disk(f"{path}/{dataset_name}_valid.dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and split dataset files.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="The directory containing dataset chunks.",
    )  
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The base name for the dataset.",
    )  
    args = parser.parse_args()
    process_datasets(args.path, args.dataset_name)
