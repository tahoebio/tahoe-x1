#!/usr/bin/env python
# Copyright (C) Vevo Therapeutics 2025. All rights reserved.

import argparse
import logging
import os
import warnings

import anndata
import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
from sig_predictor import GeneSigDataset, SigPredictor
from sklearn.metrics import log_loss
from torch import multiprocessing
from torch.utils.data import Subset

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", ".*does not have many workers.*")




def init_device(device_id=None):
    """Initialize CUDA device in the subprocess if necessary."""
    if torch.cuda.is_available() and device_id is not None:
        torch.cuda.set_device(device_id)


def generate_partitions(total_size, sizes=(0.7, 0.05, 0.05), seed=0):
    np.random.seed(seed)
    partitions = (np.cumsum(sizes) * total_size).astype(int)
    idx = np.random.permutation(total_size)
    return {
        "train": idx[: partitions[0]],
        "val": idx[partitions[0] : partitions[1]],
        "test": idx[partitions[1] : partitions[2]],
        "benchmark": idx[partitions[2] :],
    }


def generate_grid_search_configs(param_dict, current_config=None, all_configs=None):
    if current_config is None:
        current_config = {}
    if all_configs is None:
        all_configs = []
    if not param_dict:
        all_configs.append(current_config)
        return
    param, values = next(iter(param_dict.items()))
    rest_params = {k: v for k, v in param_dict.items() if k != param}
    for value in values:
        new_config = current_config.copy()
        new_config[param] = value
        generate_grid_search_configs(rest_params, new_config, all_configs)
    return all_configs


def evaluate_config(
    config,
    train_ds,
    val_ds,
    test_ds,
    sigs,
    sig_names,
    seed=None,
    device_id=None,
):
    init_device(device_id)
    if seed is not None:
        torch.manual_seed(seed)

    model = SigPredictor(
        sig_names=sig_names,
        emb_dim=train_ds[0][0].shape[0],
        **config,
    )
    model.fit(
        train_ds,
        val_ds,
        enable_progress_bar=False,
    )
    predictions = (
        model.predict(test_ds)
        .merge(sigs.assign(label=1), on=["gene", "sig"], how="left")
        .fillna(0)
    )
    test_score = log_loss(predictions["label"], predictions["prediction"])
    return model, config, test_score


def get_optimal_config(
    dataset,
    configs,
    partitions,
    sigs,
    sig_names,
    seed=None,
    verbose=True,
):
    train_ds = Subset(dataset=dataset, indices=partitions["train"])
    val_ds = Subset(dataset=dataset, indices=partitions["val"])
    test_ds = Subset(dataset=dataset, indices=partitions["test"])
    print(f"train DS size: {len(train_ds)}, {train_ds}")

    # Setup multiprocessing pool and map the parameters and function
    num_processes = min(multiprocessing.cpu_count(), len(configs))
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=num_processes) as pool:
        tasks = [
            (
                config,
                train_ds,
                val_ds,
                test_ds,
                sigs,
                sig_names,
                seed,
                i % torch.cuda.device_count(),
            )
            for i, config in enumerate(configs)
        ]
        if verbose:
            print(
                f"Evaluating {len(tasks)} configurations, evaluate config {evaluate_config}",
            )
        results = list(pool.starmap(evaluate_config, tasks))

    best_loss = np.inf
    best_model = None
    config_scores = []

    for model, config, loss in results:
        config_values = list(config.values()) + [loss]
        config_scores.append(config_values)
        if loss < best_loss:
            best_loss = loss
            best_model = model

    return best_model, pd.DataFrame(
        config_scores,
        columns=list(configs[0].keys()) + ["loss"],
    )


def run_benchmark(adata, emb_name, cfg, verbose: bool = True, seed: int = 0):
    partitions = generate_partitions(adata.shape[0], seed=seed)
    configs = generate_grid_search_configs(cfg.get("gridsearch_configs"))
    dataset = GeneSigDataset(
        pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var.index),
        adata.obsm[emb_name],
    )
    best_model, config_scores = get_optimal_config(
        dataset=dataset,
        configs=configs,
        partitions=partitions,
        sigs=adata.uns["sigs"],
        sig_names=adata.var.sig.values,
        verbose=verbose,
        seed=seed,
    )

    if verbose:
        print("predicting on benchmark set")
    bm_predictions = best_model.predict(
        Subset(dataset=dataset, indices=partitions["benchmark"]),
    ).pivot(index="gene", columns="sig", values="prediction")

    return best_model, config_scores, bm_predictions


def run_all(adata, output_path, cfg, seed=0):
    verbose = cfg.get("verbose", True)
    emb_names = cfg.get("emb_names", None)

    if emb_names is None:
        emb_names = adata.obsm.keys()
    for emb_name in tqdm.tqdm(emb_names):
        if verbose:
            print(f"Running benchmark for {emb_name}")
        best_model, config_scores, bm_predictions = run_benchmark(
            adata,
            emb_name,
            cfg,
            verbose=verbose,
            seed=seed,
        )
        if verbose:
            print(f"saving results to {output_path}")

        config_scores.to_csv(os.path.join(output_path, f"{emb_name}_config_scores.csv"))
        bm_predictions.to_csv(
            os.path.join(output_path, f"{emb_name}_benchmark_predictions.csv"),
            index=True,
        )


def load_config(path: str) -> dict:
    with open(path, "r") as fin:
        return yaml.safe_load(fin)


def run_from_config(cfg: dict):
    if "h5ad" in cfg["adata_path"]:
        adata = anndata.read_h5ad(cfg["adata_path"])
    elif "zarr" in cfg["adata_path"]:
        adata = anndata.read_zarr(cfg["adata_path"])
    else:
        raise ValueError("Unsupported adata format")
    
    bench_cfg = cfg.get("benchmark", {})
    seeds = bench_cfg.get("seeds", [0])
    base_output = bench_cfg.get("output_dir", "benchmark_results")

    os.makedirs(base_output, exist_ok=True)
    for seed in seeds:
        out_dir = os.path.join(base_output, f"seed_{seed}")
        os.makedirs(out_dir, exist_ok=True)
        run_all(adata, out_dir, bench_cfg)


def main():
    parser = argparse.ArgumentParser(description="Run MLP benchmark from YAML config")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_from_config(cfg)


if __name__ == "__main__":
    main()