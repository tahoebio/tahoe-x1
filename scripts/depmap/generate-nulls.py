# Copyright (C) Vevo Therapeutics 2024. All rights reserved.
"""Create null model predictions for both the marginal and cell line specific
essentiality tasks."""

import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
import scanpy as sc

# set up logging
log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
)
logging.getLogger(__name__).setLevel("INFO")


# save null predictions for marginal task
def null_marginal_task(base_path):
    obs = sc.read_h5ad(
        os.path.join(base_path, "gene-embs/loading15-lt5gt70-bin.h5ad"),
    ).obs
    splits = pd.read_csv(os.path.join(base_path, "misc/split-genes-lt5gt70.csv"))
    for fold in range(5):
        train_genes = splits[splits[f"fold-{fold}"] == "train"]["gene"].to_numpy()
        train_obs = obs[obs["gene"].isin(train_genes)]
        mean_score = train_obs["score"].mean()
        val_genes = splits[splits[f"fold-{fold}"] == "val"]["gene"].to_numpy()
        val_obs = obs[obs["gene"].isin(val_genes)]
        val_labels = val_obs["score"].to_numpy()
        val_probas = np.array([np.array([1 - mean_score, mean_score])] * len(val_obs))
        val_preds = np.array([np.argmax(pair) for pair in val_probas])
        np.save(
            os.path.join(
                base_path,
                f"gene-embs/results/null-lt5gt70-bin-fold-{fold}-val-true.npy",
            ),
            val_labels,
        )
        np.save(
            os.path.join(
                base_path,
                f"gene-embs/results/null-lt5gt70-bin-fold-{fold}-val-proba.npy",
            ),
            val_probas,
        )
        np.save(
            os.path.join(
                base_path,
                f"gene-embs/results/null-lt5gt70-bin-fold-{fold}-val-pred.npy",
            ),
            val_preds,
        )


# save null predictions for cell line specific task (for given subset)
def null_cell_line_specific_task(base_path, null_dict, subset):
    obs = sc.read_h5ad(
        os.path.join(base_path, f"gene-embs/gene_idx+pca15-{subset}.h5ad"),
    ).obs
    splits = pd.read_csv(os.path.join(base_path, "misc/split-cls.csv"))
    for fold in range(5):
        val_cls = splits[splits[f"fold-{fold}"] == "val"]["cell-line"].to_numpy()
        val_obs = obs[obs["cell-line"].isin(val_cls)]
        val_labels = val_obs["score"].to_numpy()
        genes = val_obs["gene"].tolist()
        val_preds = np.zeros(len(genes))
        for i, gene in enumerate(genes):
            val_preds[i] = null_dict[fold][gene]
        np.save(
            os.path.join(
                base_path,
                f"gene-embs/results/null-{subset}-fold-{fold}-val-true.npy",
            ),
            val_labels,
        )
        np.save(
            os.path.join(
                base_path,
                f"gene-embs/results/null-{subset}-fold-{fold}-val-pred.npy",
            ),
            val_preds,
        )


# main function
def main(base_path):

    # load dependency scores
    scores = pd.read_csv(os.path.join(base_path, "raw/depmap-gene-dependencies.csv"))
    col_map = {}
    for col in scores.columns:
        if col == "Unnamed: 0":
            col_map[col] = "cell-line"
        else:
            col_map[col] = col.split(" ")[0]
    scores = scores.rename(columns=col_map).set_index("cell-line")
    log.info("loaded dependency scores")

    # create dictionary of null predictions for cell line specific task
    splits = pd.read_csv(os.path.join(base_path, "misc/split-cls.csv"))
    null_dict = {}
    for fold in range(5):
        train_cls = splits[splits[f"fold-{fold}"] == "train"]["cell-line"].to_numpy()
        sub_scores = scores[scores.index.isin(train_cls)]
        null_dict[fold] = sub_scores.mean(axis=0).to_dict()

    # save to disk for future reference
    outpath = os.path.join(base_path, "misc/null-dict-split-cls.pkl")
    log.info(
        f"saving dicitonary of null predictions for cell line specific task to {outpath}",
    )
    with open(outpath, "wb") as f:
        pickle.dump(null_dict, f)

    # run null model for marginal task
    null_marginal_task(base_path)
    log.info("saved null predictions for marginal task")

    # run null model for each subset of cell line specific task
    subsets = [
        "0to10",
        "10to20",
        "20to30",
        "30to40",
        "40to50",
        "50to60",
        "60to70",
        "70to80",
        "80to90",
        "90to100",
    ]
    for s in subsets:
        null_cell_line_specific_task(base_path, null_dict, s)
        log.info(f"saved null predictions for cell line specific task (subset: {s})")


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path",
        type=str,
        required=True,
        help="Path to DepMap benchmark base directory.",
    )
    args = parser.parse_args()

    # run main function
    main(args.base_path)
