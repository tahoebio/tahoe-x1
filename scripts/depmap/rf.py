# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
"""Given embeddings from a model (processed into AnnData format for a DepMap
task), this script will train the specified random forest model and save results
for further analysis."""

import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--base-path", type=str, required=True)
parser.add_argument("--model-type", type=str, required=True)
parser.add_argument("--emb", type=str, required=True)
parser.add_argument("--add-label", type=str, default="")
parser.add_argument("--split-file", type=str, required=True)
parser.add_argument("--split-col", type=str, required=True)
parser.add_argument("--n-jobs", type=int, default=16)
parser.add_argument("--fold", type=int, required=True)
args = parser.parse_args()

# print header
print(f"\n===== emb: {args.emb} | fold: {args.fold} =====\n")

# load correct embeddings
print("loading embeddings...")
embs = sc.read_h5ad(os.path.join(args.base_path, f"gene-embs/{args.emb}.h5ad"))

# get splits
print("retrieving data split...")
splits = pd.read_csv(os.path.join(args.base_path, f"misc/{args.split_file}"))
train_rows = splits[splits[f"fold-{args.fold}"] == "train"][args.split_col].to_numpy()
val_rows = splits[splits[f"fold-{args.fold}"] == "val"][args.split_col].to_numpy()

# create subset AnnDatas
print("subsetting AnnDatas...")
train_embs = embs[embs.obs[args.split_col].isin(train_rows)]
val_embs = embs[embs.obs[args.split_col].isin(val_rows)]

# get data and labels
print("preparing training inputs...")
train_data = train_embs.X
train_labels = train_embs.obs["score"].to_numpy()
val_data = val_embs.X
val_labels = val_embs.obs["score"].to_numpy()

# train model
print("training model...")
if args.model_type == "regressor":
    rf = RandomForestRegressor(n_jobs=args.n_jobs)
    prefix = "rf"
elif args.model_type == "classifier":
    rf = RandomForestClassifier(n_jobs=args.n_jobs)
    prefix = "rfc"
rf.fit(train_data, train_labels)

# save results
print("saving results...")
train_preds = rf.predict(train_data)
val_preds = rf.predict(val_data)
np.save(
    os.path.join(
        args.base_path,
        f"gene-embs/results/{prefix}-{args.emb}{args.add_label}-fold-{args.fold}-train-true.npy",
    ),
    train_labels,
)
np.save(
    os.path.join(
        args.base_path,
        f"gene-embs/results/{prefix}-{args.emb}{args.add_label}-fold-{args.fold}-train-pred.npy",
    ),
    train_preds,
)
np.save(
    os.path.join(
        args.base_path,
        f"gene-embs/results/{prefix}-{args.emb}{args.add_label}-fold-{args.fold}-val-true.npy",
    ),
    val_labels,
)
np.save(
    os.path.join(
        args.base_path,
        f"gene-embs/results/{prefix}-{args.emb}{args.add_label}-fold-{args.fold}-val-pred.npy",
    ),
    val_preds,
)
if args.model_type == "classifier":
    train_probas = rf.predict_proba(train_data)
    val_probas = rf.predict_proba(val_data)
    np.save(
        os.path.join(
            args.base_path,
            f"gene-embs/results/{prefix}-{args.emb}{args.add_label}-fold-{args.fold}-train-proba.npy",
        ),
        train_probas,
    )
    np.save(
        os.path.join(
            args.base_path,
            f"gene-embs/results/{prefix}-{args.emb}{args.add_label}-fold-{args.fold}-val-proba.npy",
        ),
        val_probas,
    )
