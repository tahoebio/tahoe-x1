# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
"""Starting from a minimal DepMap benchmark directory (containing only raw
data), this script will create the files necessary to run the benchmark tasks
for MosaicFM models.

Requires:

    [base_path]/raw/ccle-counts.gct
    [base_path]/raw/depmap-gene-dependencies.csv
    [base_path]/raw/depmap-gene-effects.csv
    [base_path]/raw/depmap-metadata.csv
    [base_path]/raw/gene-mapping.csv

Creates:

    [base_path]/counts.h5ad
    [base_path]/misc/genes-by-mean-disc.csv
    [base_path]/misc/split-cls.csv
    [base_path]/misc/split-genes-lt5gt70.csv
"""

import argparse
import logging
import os
import pickle

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold

# set up logging
log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
)
logging.getLogger(__name__).setLevel("INFO")


# main function
def main(base_path):

    # load raw counts
    counts = pd.read_csv(os.path.join(base_path, "raw/ccle-counts.gct"), sep="\t")

    # create column map
    col_map = {}
    for col in counts.columns:
        if col == "Name":
            col_map[col] = "gene_id"
        elif col == "Description":
            col_map[col] = "gene_name"

    # rename columns and simplify gene IDs
    counts = counts.rename(columns=col_map)
    counts["gene_id"] = [gene.split(".")[0] for gene in counts["gene_id"]]

    # get all genes and drop gene name
    all_genes = counts[["gene_id", "gene_name"]]
    counts = counts.drop(columns="gene_name")

    # transpose so rows are cell lines and columns are genes
    counts = (
        counts.set_index("gene_id")
        .T.reset_index()
        .rename_axis(None, axis=1)
        .rename(columns={"index": "ccl"})
    )
    log.info("loaded and cleaned count data")

    # subset to usable cell lines from DepMap data
    depmap_gene_effects = pd.read_csv(
        os.path.join(base_path, "raw/depmap-gene-effects.csv"),
    )
    depmap_cell_lines = depmap_gene_effects.iloc[:, 0]
    depmap_metadata = pd.read_csv(os.path.join(base_path, "raw/depmap-metadata.csv"))
    depmap_metadata = depmap_metadata[
        depmap_metadata["ModelID"].isin(depmap_cell_lines)
    ].reset_index(drop=True)
    counts = counts[counts["ccl"].isin(depmap_metadata["CCLEName"])]
    log.info("subset to usable cell lines")

    # subset to usable genes in common
    vocab_genes = pd.read_csv(os.path.join(base_path, "raw/gene-mapping.csv"))[
        "feature_name"
    ].tolist()
    depmap_genes = [i.split(" ")[0] for i in depmap_gene_effects.columns.tolist()[1:]]
    ccle_genes = all_genes["gene_name"].tolist()
    common_genes = list(set(vocab_genes) & set(depmap_genes) & set(ccle_genes))
    valid_genes = all_genes[all_genes["gene_name"].isin(common_genes)][
        "gene_id"
    ].tolist()
    columns_to_keep = [col for col in counts.columns if col in valid_genes]
    columns_to_keep = ["ccl", *sorted(columns_to_keep)]
    counts = counts[columns_to_keep]
    log.info("subset to usable genes")

    # build and save AnnData
    X = csr_matrix(counts.drop(columns="ccl").to_numpy(dtype=np.float32))
    obs = (
        counts[["ccl"]]
        .rename(columns={"ccl": "CCLEName"})
        .merge(depmap_metadata, on="CCLEName")
        .set_index("CCLEName")
    )
    obs = obs.drop(columns=["EngineeredModel", "ModelDerivationMaterial"])
    var = pd.DataFrame({"gene_id": counts.columns.tolist()[1:]})
    var = (
        var.merge(all_genes, on="gene_id")
        .set_index("gene_id")
        .rename(columns={"gene_name": "feature_name"})
    )
    adata = ad.AnnData(X=X, obs=obs, var=var)
    outpath = os.path.join(base_path, "counts.h5ad")
    adata.write_h5ad(outpath)
    log.info(f"saved count AnnData to {outpath}")

    # get mean discretized dependency for each gene
    depmap = pd.read_csv(
        os.path.join(base_path, "raw/depmap-gene-dependencies.csv"),
    ).iloc[:, 1:]
    col_map = {s: s.split(" ")[0] for s in depmap.columns}
    depmap = depmap.rename(columns=col_map)
    null_frac = depmap.isnull().sum() / depmap.shape[0]
    disc_depmap = depmap.fillna(value=0)
    disc_depmap_lower_threshold = 0.5
    disc_depmap_upper_threshold = 0.5
    disc_depmap[disc_depmap <= disc_depmap_lower_threshold] = 0
    disc_depmap[disc_depmap > disc_depmap_upper_threshold] = 1
    mean_disc_dep = disc_depmap.mean()
    log.info("computed mean discretized dependency score for each gene")

    # create and save DataFrame of mean discretized dependency
    available_genes = adata.obs["feature_name"].unique()
    df = pd.DataFrame(
        {
            "gene": null_frac.index,
            "mean-disc-dep": mean_disc_dep,
            "null-frac": null_frac,
        },
    )
    df = (
        df[df["gene"].isin(available_genes)]
        .sort_values(by=["mean-disc-dep", "null-frac"])
        .reset_index(drop=True)
    )
    outpath = os.path.join(base_path, "misc/genes-by-mean-disc.csv")
    df.to_csv(outpath, index=False)
    log.info(f"saved mean discretized dependnecy scores to {outpath}")

    # create five folds across cell lines
    cell_lines = adata.obs["ModelID"].unique().to_numpy()
    np.random.shuffle(cell_lines)
    columns = {"cell-line": cell_lines}
    n = len(cell_lines)
    kf = KFold(n_splits=5)
    for i, (_, val_idx) in enumerate(kf.split(cell_lines)):
        split = np.array(["train"] * n)
        split[val_idx] = "val"
        columns[f"fold-{i}"] = split

    # create and save cell line split DataFrame
    df = pd.DataFrame(columns).sort_values(by="cell-line").reset_index(drop=True)
    outpath = os.path.join(base_path, "misc/split-cls.csv")
    df.to_csv(outpath, index=False)
    log.info(f"saved five folds for cell line split to {outpath}")

    # create five folds across genes (with <5% or >70% mean discretized dependency, <10% null scores)
    null_frac_limit = 0.1
    marginal_mean_disc_lower_limit = 0.05
    marginal_mean_disc_upper_limit = 0.7
    genes_df = pd.read_csv(os.path.join(base_path, "misc/genes-by-mean-disc.csv"))
    genes_df = genes_df[genes_df["gene"].isin(adata.var["feature_name"])]

    genes_df = genes_df[genes_df["null-frac"] < null_frac_limit]
    genes_df = genes_df[
        (genes_df["mean-disc-dep"] < marginal_mean_disc_lower_limit)
        | (genes_df["mean-disc-dep"] > marginal_mean_disc_upper_limit)
    ]
    genes = np.sort(genes_df["gene"].to_numpy())
    np.random.shuffle(genes)
    columns = {"gene": genes}
    n = len(genes)
    kf = KFold(n_splits=5)
    for i, (_, val_idx) in enumerate(kf.split(genes)):
        split = np.array(["train"] * n)
        split[val_idx] = "val"
        columns[f"fold-{i}"] = split

    # create and save gene split DataFrame
    df = pd.DataFrame(columns).sort_values(by="gene").reset_index(drop=True)
    outpath = os.path.join(base_path, "misc/split-genes-lt5gt70.csv")
    df.to_csv(outpath, index=False)
    log.info(f"saved five folds for gene split to {outpath}")

    # save dictionary of new gene medians for use with Geneformer
    counts = sc.read_h5ad(os.path.join(base_path, "counts.h5ad"))
    sc.pp.normalize_total(counts)
    counts_X = np.array(counts.X.todense())
    counts_X[counts_X == 0] = np.nan
    medians = np.nanmedian(counts_X, axis=0)
    medians = np.nan_to_num(medians, nan=0.0)
    medians_dict = dict(zip(counts.var.index.tolist(), medians))
    outpath = os.path.join(base_path, "geneformer/ccle-nonzero-medians.pkl")
    with open(outpath, "wb") as f:
        pickle.dump(medians_dict, f)
    log.info(f"saved nonzero medians to {outpath}")


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
