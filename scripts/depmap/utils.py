# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
"""Functions that are reused across scripts."""

import itertools
import os

import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm


# return genes and classes for marginal essentiality task
def get_marginal_genes_scores(base_path, log):
    genes = pd.read_csv(os.path.join(base_path, "misc/split-genes-lt5gt70.csv"))[
        "gene"
    ].tolist()
    depmap = pd.read_csv(
        os.path.join(base_path, "raw/depmap-gene-dependencies.csv"),
    ).iloc[:, 1:]
    col_map = {s: s.split(" ")[0] for s in depmap.columns}
    depmap = depmap.rename(columns=col_map)
    disc_depmap = depmap.fillna(value=0)
    disc_depmap_lower_threshold = 0.5
    disc_depmap_upper_threshold = 0.5
    disc_depmap[disc_depmap <= disc_depmap_lower_threshold] = 0
    disc_depmap[disc_depmap > disc_depmap_upper_threshold] = 1
    mean_disc_dep_dict = disc_depmap.mean().to_dict()
    marginal_disc_threshold = 0.5
    scores = [
        0 if mean_disc_dep_dict[g] < marginal_disc_threshold else 1 for g in genes
    ]
    log.info("loaded genes and classes for marginal essentiality task")
    return genes, scores


# given sorted arrays of labels and contextual gene embeddings, create and save AnnDatas for tasks
def process_contextual_gene_embs(base_path, log, labels, embeddings, prefix):

    # load essentiality data
    log.info("loading essentiality data")
    scores = pd.read_csv(os.path.join(base_path, "raw/depmap-gene-dependencies.csv"))
    col_map = {}
    for col in scores.columns:
        if col == "Unnamed: 0":
            col_map[col] = "cell-line"
        else:
            col_map[col] = col.split(" ")[0]
    scores = scores.rename(columns=col_map).set_index("cell-line")
    score_dict = scores.to_dict("index")

    # make a .obs entry for each (cell line, gene) pair
    log.info("creating .obs entries")
    entries = []
    for cell_line_label in tqdm(labels):
        cell_line, gene = cell_line_label.split(" | ")
        try:
            score = score_dict[cell_line][gene]
        except KeyError:
            score = np.nan
        entries.append(
            {
                "label": cell_line_label,
                "cell-line": cell_line,
                "gene": gene,
                "score": score,
            },
        )

    # create .obs and subset to valid rows
    obs = pd.DataFrame.from_records(entries)
    valid_rows = ~obs["score"].isna()
    embeddings = embeddings[valid_rows]
    obs = obs[valid_rows]
    log.info("created .obs DataFrame and subsetted to valid rows")

    # create and save AnnData
    outpath = os.path.join(base_path, f"gene-embs/{prefix}.h5ad")
    gene_adata = ad.AnnData(
        X=embeddings,
        obs=obs.set_index("label"),
        var=pd.DataFrame(
            {"dim": [str(i) for i in range(embeddings.shape[1])]},
        ).set_index("dim"),
    )
    gene_adata.write_h5ad(outpath)
    log.info(f"saved {prefix} contextual gene embeddings to {outpath}")

    # load discretized dependency information and iterate over strata
    mean_disc = pd.read_csv(os.path.join(base_path, "misc/genes-by-mean-disc.csv"))
    null_frac_limit = 0.1
    mean_disc = mean_disc[mean_disc["null-frac"] < null_frac_limit]
    boundaries = np.arange(0, 101, 10)
    for left, right in itertools.pairwise(boundaries, boundaries[1:]):

        # get genes to keep
        subset = mean_disc[
            (mean_disc["mean-disc-dep"] >= (left / 100))
            & (mean_disc["mean-disc-dep"] <= (right / 100))
        ]
        genes_to_keep = subset["gene"].tolist()

        # subset AnnData and save
        subad = gene_adata[gene_adata.obs["gene"].isin(genes_to_keep)]
        subad.write_h5ad(
            os.path.join(base_path, f"gene-embs/{prefix}-{left}to{right}.h5ad"),
        )
        log.info(
            f"saved {prefix} contextual gene embeddings for {left}% to {right}% strata",
        )
