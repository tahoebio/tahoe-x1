# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
import logging
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
from scipy.stats import pearsonr


def add_file_handler(logger: logging.Logger, log_file_path: Path):
    """Add a file handler to the logger."""
    h = logging.FileHandler(log_file_path)

    # format showing time, name, function, and message
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(funcName)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    h.setFormatter(formatter)
    h.setLevel(logger.level)
    logger.addHandler(h)


def download_file_from_s3_url(s3_url, local_file_path):
    """Downloads a file from an S3 URL to the specified local path.

    :param local_file_path: Local path where the file will be saved. :return:
    The local path to the downloaded file.
    """
    # Validate the S3 URL format
    assert s3_url.startswith("s3://"), "URL must start with 's3://'"

    # Parse the S3 URL
    parsed_url = urlparse(s3_url)
    assert parsed_url.scheme == "s3", "URL scheme must be 's3'"

    bucket_name = parsed_url.netloc
    s3_file_key = parsed_url.path.lstrip("/")

    # Ensure bucket name and file key are not empty
    assert bucket_name, "Bucket name cannot be empty"
    assert s3_file_key, "S3 file key cannot be empty"

    # Create an S3 client
    s3 = boto3.client("s3")

    try:
        # Download the file
        s3.download_file(bucket_name, s3_file_key, local_file_path)
        print(f"File downloaded successfully to {local_file_path}")
        return local_file_path
    except Exception as e:
        print(f"Error downloading the file from {s3_url}: {e}")
        return None


def calc_pearson_metrics(preds, targets, conditions, mean_ctrl):

    conditions_unique = np.unique(conditions)
    condition2idx = {c: np.where(conditions == c)[0] for c in conditions_unique}

    targets_mean_perturbed_by_condition = np.array(
        [targets[condition2idx[c]].mean(0) for c in conditions_unique],
    )  # (n_conditions, n_genes)

    preds_mean_perturbed_by_condition = np.array(
        [preds[condition2idx[c]].mean(0) for c in conditions_unique],
    )  # (n_conditions, n_genes)

    pearson = []
    for cond, t, p in zip(
        conditions_unique,
        targets_mean_perturbed_by_condition,
        preds_mean_perturbed_by_condition,
    ):
        print(cond, pearsonr(t, p))
        pearson.append(pearsonr(t, p)[0])

    pearson_delta = []
    for cond, t, p in zip(
        conditions_unique,
        targets_mean_perturbed_by_condition,
        preds_mean_perturbed_by_condition,
    ):
        tm, pm = t, p
        tm -= mean_ctrl
        pm -= mean_ctrl

        print(cond, pearsonr(tm, pm))
        pearson_delta.append(pearsonr(tm, pm)[0])

    return {
        "pearson": np.mean(pearson),
        "pearson_delta": np.mean(pearson_delta),
    }
