# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf as om
from sklearn.metrics import average_precision_score


def load_signatures(sigs_path):
    """Load signatures from .gmt files - same as original notebooks."""

    def _read_sigs_rec(path):
        if os.path.isdir(path):
            for fn in [f for f in os.listdir(path) if f.endswith(".gmt")]:
                for pair in _read_sigs_rec(os.path.join(path, fn)):
                    yield pair
        if os.path.isfile(path) and path.endswith(".gmt"):
            with open(path, "r") as fin:
                for li in fin:
                    line = li.strip().split("\t")
                    for gene in line[2:]:
                        yield (line[0], gene)

    return pd.DataFrame(_read_sigs_rec(sigs_path), columns=["sig", "gene"])


def compute_auprc_single_rep(results_dir, sigs):
    """Compute AUPRC for single replicate - simplified version of viz_preds.ipynb."""
    print(f"Computing AUPRC for {results_dir}")

    # Find prediction files
    pred_files = [
        f for f in os.listdir(results_dir) if f.endswith("_benchmark_predictions.csv")
    ]

    auprc_scores = {}
    for fn in pred_files:
        emb_name = fn.replace("_benchmark_predictions.csv", "")

        # Load predictions
        preds = pd.read_csv(os.path.join(results_dir, fn), index_col=0)
        preds_melted = preds.melt(
            ignore_index=False,
            var_name="sig",
            value_name="prediction",
        ).reset_index(names="gene")

        # Merge with signatures (add labels)
        with_sigs = preds_melted.merge(
            sigs.assign(observed=1),
            on=["gene", "sig"],
            how="left",
        ).fillna(0)

        # Calculate AUPRC
        auprc_scores[emb_name] = average_precision_score(
            with_sigs["observed"],
            with_sigs["prediction"],
        )

    return pd.DataFrame(
        list(auprc_scores.items()),
        columns=["emb", "AUPRC"],
    ).sort_values("AUPRC", ascending=False)


def plot_multi_rep(results_dir, sigs, title="AUPRC Across Replicates"):
    """Plot across multiple replicates - simplified version of plot_reps.ipynb."""

    # Find replicate directories or CSV files
    seed_dirs = glob.glob(os.path.join(results_dir, "seed_*"))

    all_data = []
    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        seed_data = compute_auprc_single_rep(seed_dir, sigs)
        seed_data["Seed"] = seed_name
        all_data.append(seed_data)
    combined = pd.concat(all_data)

    # Calculate mean AUPRC for sorting
    mean_auprc = combined.groupby("emb")["AUPRC"].mean().sort_values(ascending=False)
    combined = combined.set_index("emb").loc[mean_auprc.index].reset_index()

    # Define a color palette for the bar plot
    n_bars = combined["emb"].shape[0]
    palette = sns.color_palette("hsv", n_bars)

    sns.set_style("whitegrid")
    plt.style.use("ggplot")

    # Create plot - exactly like plot_reps.ipynb
    plt.figure(figsize=(12, 6))

    # Bar plot with error bars
    ax = sns.barplot(
        data=combined,
        x="emb",
        y="AUPRC",
        errorbar="sd",
        estimator=np.mean,
        palette=palette,
    )

    # Add individual points
    sns.stripplot(
        data=combined,
        x="emb",
        y="AUPRC",
        color="black",
        size=5,
        jitter=True,
        alpha=0.6,
    )

    # Add mean values as text
    for i, (_, mean_val) in enumerate(mean_auprc.items()):
        fontweight = "bold" if i == 0 else "normal"
        ax.text(
            i,
            mean_val + 0.01,
            f"{mean_val:.4f}",
            color="black",
            ha="center",
            va="bottom",
            fontweight=fontweight,
        )

    # Format
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(title)
    ax.grid(True)

    # Bold first label
    tick_labels = ax.get_xticklabels()
    if tick_labels:
        tick_labels[0].set_fontweight("bold")

    plt.tight_layout()

    plt_name = os.path.join(results_dir, "auprc_with_reps.png")
    plt.savefig(plt_name, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {plt_name}")

    plt.show()


def main():
    cfg = om.load(sys.argv[1])

    num_mand_args = 2
    cli_args = []
    for arg in sys.argv[num_mand_args:]:
        if arg.startswith("--"):
            cli_args.append(arg[2:])
        else:
            cli_args.append(arg)

    cli_cfg = om.from_cli(cli_args)
    cfg = om.merge(cfg, cli_cfg)

    om.resolve(cfg)

    print("Loading signatures...")
    sigs = load_signatures(cfg["signatures_path"])
    print(f"Loaded {len(sigs)} signature entries")

    # Set style
    plt.style.use("ggplot")
    sns.set_style("whitegrid")

    plot_multi_rep(
        cfg["vis"]["pred_dir"],
        sigs,
        cfg["vis"].get("title", "AUPRC Across Replicates"),
    )


if __name__ == "__main__":
    main()
