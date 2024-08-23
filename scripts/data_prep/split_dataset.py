# Copyright (C) Vevo Therapeutics 2024. All rights reserved.
import json
import logging
import os
import sys

import numpy as np
from datasets import DatasetDict, load_from_disk
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import GroupShuffleSplit

# Logging setup
log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
)
logging.getLogger(__name__).setLevel("INFO")


def load_dataset(dataset_path: str):
    log.info(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    log.info(f"Dataset loaded with {len(dataset)} records.")
    return dataset


def log_split_info(split_name, split_dataset, groups_column):
    unique_groups = np.unique(split_dataset[groups_column])
    achieved_proportion = len(split_dataset) / len(split_dataset.data)

    log.info(
        f"{split_name} Split - Size: {len(split_dataset)}, Achieved Proportion: {achieved_proportion:.4f}",
    )
    log.info(
        f"{split_name} Split - Unique Groups: {len(unique_groups)}, Groups: {list(unique_groups)}",
    )
    return list(unique_groups)


def split_dataset(
    dataset,
    split_column,
    train_proportion,
    val_proportion,
    test_proportion,
    random_seed,
):
    log.info(
        f"Splitting dataset by column '{split_column}' with proportions {train_proportion}/{val_proportion}/{test_proportion}...",
    )

    # Validate proportions
    assert (
        train_proportion + val_proportion + test_proportion == 1.0
    ), "Proportions must sum to 1."

    # Prepare GroupShuffleSplit
    gss = GroupShuffleSplit(
        n_splits=1,
        train_size=train_proportion,
        random_state=random_seed,
    )

    # Extract the column by which we are splitting
    groups = dataset[split_column]

    # Split the dataset into train and temp (val + test)
    train_idx, temp_idx = next(gss.split(dataset, groups=groups))
    train_dataset = dataset.select(train_idx)
    temp_dataset = dataset.select(temp_idx)

    # Log train split info
    train_groups = log_split_info("Train", train_dataset, split_column)

    # Further split the temp dataset into validation and test
    remaining_proportion = 1.0 - train_proportion
    val_proportion_adjusted = val_proportion / remaining_proportion

    gss_val_test = GroupShuffleSplit(
        n_splits=1,
        train_size=val_proportion_adjusted,
        random_state=random_seed,
    )

    val_idx, test_idx = next(
        gss_val_test.split(temp_dataset, groups=temp_dataset[split_column]),
    )
    val_dataset = temp_dataset.select(val_idx)
    test_dataset = temp_dataset.select(test_idx)

    # Log validation and test split info
    val_groups = log_split_info("Validation", val_dataset, split_column)
    test_groups = log_split_info("Test", test_dataset, split_column)

    # Ensure there is no overlap between the unique groups in different splits
    assert not set(train_groups).intersection(
        val_groups,
    ), "Overlap detected between Train and Validation groups!"
    assert not set(train_groups).intersection(
        test_groups,
    ), "Overlap detected between Train and Test groups!"
    assert not set(val_groups).intersection(
        test_groups,
    ), "Overlap detected between Validation and Test groups!"

    # Return the split datasets and the groups for each split
    return (
        DatasetDict(
            {
                "train": train_dataset,
                "validation": val_dataset,
                "test": test_dataset,
            },
        ),
        train_groups,
        val_groups,
        test_groups,
    )


def save_splits(
    output_dir,
    split_datasets,
    train_groups,
    val_groups,
    test_groups,
    num_proc=32,
):
    log.info(f"Saving splits to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train")
    val_path = os.path.join(output_dir, "validation")
    test_path = os.path.join(output_dir, "test")

    split_datasets["train"].save_to_disk(train_path, num_proc=num_proc)
    split_datasets["validation"].save_to_disk(val_path, num_proc=num_proc)
    split_datasets["test"].save_to_disk(test_path, num_proc=num_proc)

    # Save metadata.json
    metadata = {
        "train_groups": train_groups,
        "validation_groups": val_groups,
        "test_groups": test_groups,
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    log.info(f"Splits and metadata saved successfully at {output_dir}")


def main(cfg: DictConfig):
    # Load dataset
    dataset = load_dataset(cfg.dataset_save_path)

    # Split dataset
    split_datasets, train_groups, val_groups, test_groups = split_dataset(
        dataset,
        split_column=cfg.split_column,
        train_proportion=cfg.train_proportion,
        val_proportion=cfg.val_proportion,
        test_proportion=cfg.test_proportion,
        random_seed=cfg.random_seed,
    )

    # Save splits and metadata
    save_splits(
        cfg.split_save_path,
        split_datasets,
        train_groups,
        val_groups,
        test_groups,
        cfg.get("num_proc", 32),
    )


if __name__ == "__main__":
    yaml_path = sys.argv[1]

    OmegaConf.clear_resolver("oc.env")
    log.info(f"Loading configuration from {yaml_path}...")
    with open(yaml_path) as f:
        cfg = OmegaConf.load(f)
    OmegaConf.resolve(cfg)
    main(cfg)
    log.info("Script execution completed.")
