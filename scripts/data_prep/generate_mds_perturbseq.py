# Copyright (C) Vevo Therapeutics 2024. All rights reserved.
import logging
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Tuple

import datasets
from omegaconf import DictConfig, OmegaConf
from streaming import MDSWriter
from streaming.base.util import merge_index

# Logging setup
log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
)
logging.getLogger(__name__).setLevel("INFO")


def init_worker():
    pid = os.getpid()
    log.info(f"Initialized Worker PID: {pid}")


def get_files(path: str) -> Iterable[str]:
    return [str(f.resolve()) for f in Path(path).glob("data*.arrow")]


def check_and_create_dir(path: str):
    if not os.path.exists(path):
        logging.info(f"Directory {path} does not exist. Creating it.")
        os.makedirs(path)
    else:
        logging.info(f"Directory {path} already exists.")


def process_data(
    out_root: str,
    dataset_path: str,
    columns: dict,
    compression: str,
    hashes: Tuple[str],
):
    check_and_create_dir(out_root)

    num_files = len(get_files(dataset_path))
    if num_files == 0:
        log.warning(f"No files found in {dataset_path}. Exiting processing.")
        return

    arg_tuples = each_task(out_root, dataset_path, columns, compression, hashes)

    with Pool(initializer=init_worker, processes=num_files) as pool:
        for _ in pool.imap(convert_to_mds, arg_tuples):
            pass

    merge_index(out_root, keep_local=True)
    log.info(f"Merging Index Complete at {out_root}.")


def each_task(
    out_root: str,
    dataset_root_path: str,
    columns: dict,
    compression: str,
    hashes: Tuple[str],
) -> Iterable[Tuple[str, str, dict, str, Tuple[str]]]:
    for dataset_path in get_files(dataset_root_path):
        arrow_path = dataset_path.split(dataset_root_path)[1]
        chunk_suffix = arrow_path.split("-")[1]
        sub_out_root = f"{out_root}/chunk_{chunk_suffix}"
        yield sub_out_root, dataset_path, columns, compression, hashes


def convert_to_mds(args: Tuple[str, str, dict, str, Tuple[str]]) -> None:
    sub_out_root, dataset_path, columns, compression, hashes = args
    check_and_create_dir(sub_out_root)
    log.info(f"Processing file {dataset_path} into {sub_out_root}.")

    dataset = datasets.Dataset.from_file(dataset_path)
    with MDSWriter(
        out=sub_out_root,
        columns=columns,
        compression=compression,
        hashes=hashes,
    ) as out:
        for sample in dataset:
            out.write(sample)

    log.info(f"Finished processing {dataset_path}.")


def main(cfg: DictConfig):
    # Paths setup
    out_root = cfg.out_root
    root_dir = cfg.root_dir
    splits = cfg.splits  # Reading the split names from the configuration

    # MDSWriter configurations
    columns = cfg.columns
    compression = cfg.compression
    hashes = tuple(cfg.hashes)

    # Process each split (train, val, test, etc.)
    for split in splits:
        dataset_path = os.path.join(root_dir, split)
        out_root_split = os.path.join(out_root, split)
        log.info(f"Starting processing of {split.capitalize()} Data...")
        process_data(out_root_split, dataset_path, columns, compression, hashes)
        log.info(f"Finished writing MDS files for {split.capitalize()}.")

    # Copy metadata.json if it exists
    metadata_path = os.path.join(root_dir, "metadata.json")
    if os.path.exists(metadata_path):
        shutil.copy(metadata_path, os.path.join(out_root, "metadata.json"))
        log.info(f"Copied metadata.json to {out_root}.")


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    OmegaConf.clear_resolver("oc.env")
    log.info(f"Loading configuration from {yaml_path}...")
    # Load the YAML configuration file
    with open(yaml_path) as f:
        cfg = OmegaConf.load(f)
    OmegaConf.resolve(cfg)
    main(cfg.mds)
    log.info("Script execution completed.")
