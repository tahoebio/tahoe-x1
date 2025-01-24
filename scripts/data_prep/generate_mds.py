# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
import argparse
import logging
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
)


def init_worker():
    pid = os.getpid()
    logging.info(f"Initialized Worker PID: {pid}")


def get_files(path: str) -> Iterable[str]:
    return [str(f.resolve()) for f in Path(path).glob("data*.arrow")]


def check_and_create_dir(path: str):
    if not os.path.exists(path):
        logging.info(f"Directory {path} does not exist. Creating it.")
        os.makedirs(path)
    else:
        logging.info(f"Directory {path} already exists.")


def process_data(out_root: str, dataset_path: str):
    check_and_create_dir(out_root)

    num_files = len(get_files(dataset_path))
    if num_files == 0:
        logging.warning(f"No files found in {dataset_path}. Exiting processing.")
        return

    arg_tuples = each_task(out_root, dataset_path)

    with Pool(initializer=init_worker, processes=num_files) as pool:
        for _ in pool.imap(convert_to_mds, arg_tuples):
            pass

    merge_index(out_root, keep_local=True)
    logging.info(f"Merging Index Complete at {out_root}.")


def each_task(out_root: str, dataset_root_path: str) -> Iterable[Tuple[str, str]]:
    for dataset_path in get_files(dataset_root_path):
        arrow_path = dataset_path.split(dataset_root_path)[1]
        chunk_suffix = arrow_path.split("-")[1]
        sub_out_root = f"{out_root}/chunk_{chunk_suffix}"
        yield sub_out_root, dataset_path


def convert_to_mds(args: Tuple[str, str]) -> None:
    sub_out_root, dataset_path = args
    check_and_create_dir(sub_out_root)
    logging.info(f"Processing file {dataset_path} into {sub_out_root}.")

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

    logging.info(f"Finished processing {dataset_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process datasets and write MDS files.",
    )
    parser.add_argument("--out_root", required=True, help="Output root directory.")
    parser.add_argument(
        "--train_dataset_path",
        required=True,
        help="Training dataset root path.",
    )
    parser.add_argument(
        "--valid_dataset_path",
        required=True,
        help="Validation dataset root path.",
    )

    args = parser.parse_args()

    logging.info("Starting processing of Training Data...")
    out_root_train = f"{args.out_root}/train"
    dataset_path_train = f"{args.train_dataset_path}"
    process_data(out_root_train, dataset_path_train)
    logging.info("Finished writing MDS files for Training.")

    logging.info("Starting processing of Validation Data...")
    out_root_val = f"{args.out_root}/val"
    dataset_path_val = f"{args.valid_dataset_path}"
    process_data(out_root_val, dataset_path_val)
    logging.info("Finished writing MDS files for Validation.")
