# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
"""Evaluate various models on DepMap tasks as specified in a config file."""

import itertools
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.neighbors import kneighbors_graph


# compute LISI scores
def compute_lisi_scores(emb, labels, k=20):
    nng = kneighbors_graph(emb, n_neighbors=k).tocoo()
    labels = pd.Categorical(labels).codes
    self_id = labels[nng.row]
    ne_id = labels[nng.col]
    _, c = np.unique(labels, return_counts=True)
    theoretic_score = ((c / c.sum()) ** 2).sum()
    return (self_id == ne_id).mean() / theoretic_score


# save current figure at given path and close
def save_current_fig(path: Path, dpi=200, tight=True):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


# get the model name from a prefix (for contextual essentiality)
def model_from_prefix(prefix):
    if prefix.startswith("null"):
        return "null"
    else:
        return prefix.split("-")[1].replace("_", "-")


# cell line separation analysis
def cell_line_separation(cfg):

    # ensure output directory exists
    base_path = Path(cfg.base_path)
    outdir = Path(cfg.output_dir) / "cell_line_separation"
    outdir.mkdir(parents=True, exist_ok=True)

    # extract configuration
    models = cfg.cell_line_separation.models
    do_umaps = bool(cfg.cell_line_separation.get("umaps", True))
    do_bar = bool(cfg.cell_line_separation.get("barplot", True))

    # compute and save UMAPs if not present
    for m in models:

        # check if embeddings exist
        adata_path = base_path / "cell-embs" / f"{m}.h5ad"
        if not adata_path.exists():
            print(f"[cell_line] MISSING: {adata_path}")
            continue

        # load and compute UMAP if not present
        adata = sc.read_h5ad(adata_path.as_posix())
        if "X_umap" not in adata.obsm_keys():
            print(f"[cell_line] {m}: UMAP missing, computing...")
            sc.pp.neighbors(adata, use_rep="X")
            sc.tl.umap(adata)
            adata.write_h5ad(adata_path.as_posix())
            print(f"[cell_line] {m}: computed and saved UMAP")
        else:
            print(f"[cell_line] {m}: UMAP present")

    # get LISI score for each model
    lisi_by_model = {}
    for m in models:

        # check if embeddings exist
        adata_path = base_path / "cell-embs" / f"{m}.h5ad"
        if not adata_path.exists():
            print(f"[cell_line] MISSING: {adata_path}")
            continue

        # load and compute LISI score
        adata = sc.read_h5ad(adata_path.as_posix())
        emb = adata.X
        labels = adata.obs["OncotreeLineage"]
        lisi = compute_lisi_scores(emb, labels)
        lisi_by_model[m] = lisi
        print(f"[cell_line] {m}: LISI = {lisi:.4f}")

    # plot and save UMAP grid if requested
    if do_umaps and len(models):

        # set up grid
        n = len(models)
        ncols = 3
        nrows = math.ceil(n / ncols)
        _, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axs = np.array(axs).reshape(nrows, ncols)

        # plot each model
        for i, m in enumerate(models):

            # check if embeddings exist
            adata_path = base_path / "cell-embs" / f"{m}.h5ad"
            if not adata_path.exists():
                print(f"[cell_line] MISSING: {adata_path}")
                continue

            # load and plot
            adata = sc.read_h5ad(adata_path.as_posix())
            ax = axs[i // ncols, i % ncols]
            sc.pl.umap(
                adata,
                color="OncotreeLineage",
                ax=ax,
                show=False,
                legend_loc=None,
                title=f"{m}: {lisi_by_model.get(m, float('nan')):.3f}",
            )

        # hide any empty axes
        for j in range(n, nrows * ncols):
            axs[j // ncols, j % ncols].axis("off")

        # save figure
        save_current_fig(outdir / "umaps.png")
        print("[cell_line] saved UMAP grid")

    # plot and save barplot if requested
    if do_bar and lisi_by_model:
        df = pd.DataFrame(
            {"model": list(lisi_by_model.keys()), "lisi": list(lisi_by_model.values())},
        ).sort_values(by="lisi")
        plt.figure(figsize=(7, max(2, 0.4 * len(df))))
        ax = sns.barplot(data=df, x="lisi", y="model")
        ax.bar_label(ax.containers[0], fontsize=9, padding=-45, color="white")
        save_current_fig(outdir / "lisi_barplot.png")
        print("[cell_line] saved LISI barplot")


# marginal essentiality analysis
def run_marginal_essentiality(cfg):

    # ensure output directory exists
    base_path = Path(cfg.base_path)
    outdir = Path(cfg.output_dir) / "marginal_essentiality"
    outdir.mkdir(parents=True, exist_ok=True)

    # extract configuration
    do_auroc = bool(cfg.marginal_essentiality.get("auroc", True))
    do_auprc = bool(cfg.marginal_essentiality.get("auprc", True))
    models = cfg.marginal_essentiality.models

    # load results
    print("[marginal] loading results")
    records = []
    for model in models:

        # build file prefix based on model name
        if model == "null":
            prefix = "null-lt5gt70-bin"
        elif model == "pcloading":
            prefix = "rfc-pcloading-lt5gt70-bin"
        else:
            prefix = f"rfc-{model.replace('-', '_')}-mean-lt5gt70-bin"

        # load results from each fold
        for fold in range(5):
            true_path = (
                base_path
                / "gene-embs"
                / "results"
                / f"{prefix}-fold-{fold}-val-true.npy"
            )
            proba_path = (
                base_path
                / "gene-embs"
                / "results"
                / f"{prefix}-fold-{fold}-val-proba.npy"
            )
            y_true = np.load(true_path.as_posix())
            y_proba = np.load(proba_path.as_posix())
            auroc = roc_auc_score(y_true, y_proba[:, 1])
            auprc = average_precision_score(y_true, y_proba[:, 1])
            records.append(
                {
                    "model": model,
                    "fold": fold,
                    "auroc": auroc,
                    "auprc": auprc,
                },
            )

    # create and save DataFrame
    df = pd.DataFrame.from_records(records)
    df.to_csv(outdir / "metrics.csv", index=False)
    print("[marginal] saved metrics")

    # plot auROC
    if do_auroc and "auroc" in df.columns:
        plt.figure(figsize=(6, max(2, 0.4 * df["model"].nunique())))
        ax = sns.boxplot(data=df, x="auroc", y="model")
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
        save_current_fig(outdir / "auroc_boxplot.png")
        print("[marginal] saved auROC boxplot")

    if do_auprc and "auprc" in df.columns:
        plt.figure(figsize=(6, max(2, 0.4 * df["model"].nunique())))
        ax = sns.boxplot(data=df, x="auprc", y="model")
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
        save_current_fig(outdir / "auprc_boxplot.png")
        print("[marginal] saved auPRC boxplot")


# contextual essentiality analysis
def run_contextual_essentiality(cfg):

    # ensure output directory exists
    base_path = Path(cfg.base_path)
    outdir = Path(cfg.output_dir) / "contextual_essentiality"
    outdir.mkdir(parents=True, exist_ok=True)

    # extract configuration
    do_auroc = bool(cfg.contextual_essentiality.get("auroc", True))
    do_auprc = bool(cfg.contextual_essentiality.get("auprc", False))
    models = cfg.contextual_essentiality.models

    # load records
    print("[contextual] loading results")
    records = []
    boundaries = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    classification_threshold = 0.5
    for model in models:

        # build file prefix based on model name
        prefix = "null" if model == "null" else f"rf-{model.replace('-', '_')}"

        # iterate over strata
        for left, right in itertools.pairwise()(boundaries, boundaries[1:]):

            # add strata to file prefix
            prefix_strata = f"{prefix}-{left}to{right}"

            # iterate over folds
            for fold in range(5):
                true_path = (
                    base_path
                    / "gene-embs"
                    / "results"
                    / f"{prefix_strata}-fold-{fold}-val-true.npy"
                )
                y_true = np.load(true_path.as_posix())
                y_true_bin = y_true > classification_threshold
                pred_path = (
                    base_path
                    / "gene-embs"
                    / "results"
                    / f"{prefix_strata}-fold-{fold}-val-pred.npy"
                )
                y_pred = np.load(pred_path.as_posix())
                r2 = r2_score(y_true, y_pred)
                auroc = roc_auc_score(y_true_bin, y_pred)
                auprc = average_precision_score(y_true_bin, y_pred)
                records.append(
                    {
                        "model": model,
                        "prefix": prefix_strata,
                        "fold": fold,
                        "r2": r2,
                        "auroc": auroc,
                        "auprc": auprc,
                    },
                )

    # create and save DataFrames
    df = pd.DataFrame.from_records(records)
    df.to_csv(outdir / "metrics_all.csv", index=False)
    avg_folds = df.groupby("prefix", as_index=False).agg(
        mean_r2=("r2", lambda x: x.mean()),
        mean_auroc=("auroc", lambda x: x.mean()),
        mean_auprc=("auprc", lambda x: x.mean()),
    )
    avg_folds["model"] = avg_folds["prefix"].apply(model_from_prefix)
    avg_folds.to_csv(outdir / "metrics_avg.csv", index=False)
    print("[contextual] saved metrics")

    # plot auROC
    if do_auroc and "mean_auroc" in avg_folds.columns:
        plt.figure(figsize=(6, max(2, 0.4 * df["model"].nunique())))
        ax = sns.boxplot(data=avg_folds, x="mean_auroc", y="model", order=models)
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
        save_current_fig(outdir / "auroc_boxplot.png")
        print("[contextual] saved auROC boxplot")

    # plot auPRC
    if do_auprc and "mean_auprc" in avg_folds.columns:
        plt.figure(figsize=(6, max(2, 0.4 * df["model"].nunique())))
        ax = sns.boxplot(data=avg_folds, x="mean_auprc", y="model", order=models)
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
        save_current_fig(outdir / "auprc_boxplot.png")
        print("[contextual] saved auPRC boxplot")


# main script
if __name__ == "__main__":

    # parse configuration
    config_file = sys.argv[1]
    cfg = OmegaConf.load(config_file)
    print(f"[main] loaded config from {config_file}")

    # create output directory
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # run analyses
    if cfg.get("cell_line_separation", None):
        cell_line_separation(cfg)
    if cfg.get("marginal_essentiality", None):
        run_marginal_essentiality(cfg)
    if cfg.get("contextual_essentiality", None):
        run_contextual_essentiality(cfg)

    print("[main] done")
