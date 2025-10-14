# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
import logging
import os
import sys
from pathlib import Path
from typing import Generator, List

import datasets
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

# Logging setup
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def get_files(path: str) -> List[str]:
    """Retrieve dataset chunk file paths."""
    files = sorted(str(f.resolve()) for f in Path(path).glob("chunk*.dataset"))
    log.info(f"Found {len(files)} dataset chunks.")
    return files


def get_datasets(files: List[str]) -> Generator[datasets.Dataset, None, None]:
    """Lazy load datasets using a generator to prevent memory overload."""
    for file in files:
        yield datasets.load_from_disk(file)


def main(cfg: DictConfig):
    dataset_root = cfg.huggingface.output_root
    dataset_name = cfg.huggingface.dataset_name
    save_dir = cfg.huggingface.merged_dataset_root
    test_size = cfg.huggingface.split_parameters.test_size
    shuffle = cfg.huggingface.split_parameters.shuffle
    seed = cfg.huggingface.split_parameters.seed

    log.info(f"Merging dataset chunks from {dataset_root}...")
    merged_dataset = datasets.concatenate_datasets(
        list(get_datasets(get_files(dataset_root))),
    )
    log.info(f"Total {dataset_name} size: {len(merged_dataset)} samples")

    merged_dataset = merged_dataset.train_test_split(
        test_size=test_size,
        shuffle=shuffle,
        seed=seed,
    )
    train_dataset = merged_dataset["train"]
    valid_dataset = merged_dataset["test"]

    print(f"train set number of samples: {len(train_dataset)}")
    print(f"valid set number of samples: {len(valid_dataset)}")

    valid_dataset.save_to_disk(os.path.join(save_dir, "valid.dataset"))
    train_dataset.save_to_disk(os.path.join(save_dir, "train.dataset"))
    log.info("Dataset merging and saving completed successfully.")


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    log.info(f"Loading configuration from {yaml_path}...")
    with open(yaml_path) as f:
        cfg = om.load(f)
    om.resolve(cfg)
    main(cfg)
