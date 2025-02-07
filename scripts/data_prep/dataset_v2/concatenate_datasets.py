# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
import logging
import os
from pathlib import Path
from typing import Iterable

import datasets
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

# Logging setup
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def get_files(path: str) -> Iterable[str]:
    files = [str(f.resolve()) for f in Path(path).glob("chunk*.dataset")]
    return files


def get_datasets(files: Iterable[str]) -> Iterable[datasets.Dataset]:
    return [datasets.load_from_disk(file) for file in files]


def main(cfg: DictConfig):
    dataset_root = cfg.huggingface.output_root
    dataset_name = cfg.huggingface.dataset_name
    log.info(f"Merging Dataset chunks in {dataset_root}...")
    merged_dataset = datasets.concatenate_datasets(
        get_datasets(get_files(dataset_root)),
    )
    log.info(f"Total {dataset_name} length: {len(merged_dataset)}")
    merged_dataset = merged_dataset.train_test_split(
        test_size=cfg.huggingface.split_parameters.test_size,
        shuffle=cfg.huggingface.split_parameters.shuffle,
        seed=cfg.huggingface.split_parameters.seed,
    )
    train_dataset = merged_dataset["train"]
    test_dataset = merged_dataset["test"]
    log.info(f"train set number of samples: {len(train_dataset)}")
    log.info(f"test set number of samples: {len(test_dataset)}")
    save_dir = cfg.huggingface.merged_dataset_root
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        log.info(f"Created directory {save_dir}")
    train_dataset.save_to_disk(
        os.path.join(save_dir, "train.dataset"),
        num_proc=cfg.huggingface.num_proc,
    )
    test_dataset.save_to_disk(
        os.path.join(save_dir, "valid.dataset"),
        num_proc=cfg.huggingface.num_proc,
    )
    log.info("Script execution completed.")


if __name__ == "__main__":
    import sys

    yaml_path = sys.argv[1]
    log.info(f"Loading configuration from {yaml_path}...")
    with open(yaml_path) as f:
        cfg = om.load(f)
    om.resolve(cfg)
    main(cfg)
