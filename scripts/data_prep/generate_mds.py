# Copyright (C) Vevo Therapeutics 2024. All rights reserved.
import argparse
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Tuple

import datasets
import torch
from streaming import MDSWriter
from streaming.base.util import merge_index

# Constants
COLUMNS = {"genes": "pkl", "id": "int64", "expressions": "pkl"}
COMPRESSION = "zstd"
HASHES = ("sha1", "xxh64")


def init_worker():
    pid = os.getpid()
    print(f"\nInitialize Worker PID: {pid}", flush=True, end="")


def get_files(path: str) -> Iterable[str]:
    return [str(f.resolve()) for f in Path(path).glob("data*.arrow")]


def process_data(out_root: str, dataset_path: str):
    num_files = len(get_files(dataset_path))
    arg_tuples = each_task(out_root, dataset_path)

    with Pool(initializer=init_worker, processes=num_files) as pool:
        for _ in pool.imap(convert_to_mds, arg_tuples):
            pass

    merge_index(out_root, keep_local=True)
    print(f"Merging Index Complete at {out_root}.")


def each_task(out_root: str, dataset_root_path: str) -> Iterable[Tuple[str, str]]:
    for dataset_path in get_files(dataset_root_path):
        arrow_path = dataset_path.split(dataset_root_path)[1]
        chunk_suffix = arrow_path.split("-")[1]
        sub_out_root = f"{out_root}/chunk_{chunk_suffix}"
        yield sub_out_root, dataset_path


def convert_to_mds(args: Tuple[str, str]) -> None:
    sub_out_root, dataset_path = args
    dataset = datasets.Dataset.from_file(dataset_path)
    with MDSWriter(
        out=sub_out_root,
        columns=COLUMNS,
        compression=COMPRESSION,
        hashes=HASHES,
    ) as out:
        for sample in dataset:
            sample["id"] = torch.tensor(sample["id"], dtype=torch.int32)
            out.write(sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process datasets and write MDS files.",
    )
    parser.add_argument("--out_root", required=True, help="Output root directory.")
    parser.add_argument("--dataset_root_path", required=True, help="Dataset root path.")
    parser.add_argument(
        "--train_suffix",
        default="/scgpt_old_dataset_train.dataset",
        help="Suffix for the train dataset.",
    )
    parser.add_argument(
        "--val_suffix",
        default="/scgpt_old_dataset_valid.dataset",
        help="Suffix for the validation dataset.",
    )
    """OUT_ROOT = "/vevo/cellxgene/cellxgene_primary_2023-05-08_legacy_MDS"
    DATASET_ROOT_PATH = "/vevo/cellxgene/scgpt_old_dataset/hf_dataset"
    TRAIN_SUFFIX = "/scgpt_old_dataset_train.dataset" VALID_SUFFIX =
    "/scgpt_old_dataset_valid.dataset"."""

    args = parser.parse_args()

    out_root_train = f"{args.out_root}/train"
    dataset_path_train = f"{args.dataset_root_path}{args.train_suffix}"
    print("Processing Training Data...")
    process_data(out_root_train, dataset_path_train)
    print("Finished Writing MDS Files for Training.")

    out_root_val = f"{args.out_root}/val"
    dataset_path_val = f"{args.dataset_root_path}{args.val_suffix}"
    print("Processing Validation Data...")
    process_data(out_root_val, dataset_path_val)
    print("Finished Writing MDS Files for Validation.")
