# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
import os
from typing import Optional

import numpy as np
import pandas as pd


def read_sigs_and_embeddings(sigs_path, emb_path, **kwargs):
    """Read and filter signatures and gene embeddings."""
    return filter_genes_and_embeddings(
        read_sigs(sigs_path),
        read_embeddings(emb_path),
        **kwargs,
    )


def read_embeddings(path):
    embs = np.load(path, allow_pickle=True)
    return pd.DataFrame(
        embs["gene_embeddings"],
        index=embs["gene_names"],
    )


def read_sigs(path):
    return pd.DataFrame(_read_sigs_rec(path), columns=["sig", "gene"])


def _read_sigs_rec(path):
    """Read signatures from a .GMT file or list of .GMT files."""
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


def filter_genes_and_embeddings(
    sigs: pd.DataFrame,
    embs: pd.DataFrame,
    min_sig_size: Optional[int] = 25,
    max_sig_size: Optional[int] = None,
    min_hits_per_gene: Optional[int] = 10,
):
    if max_sig_size is None:
        max_sig_size = np.inf
    if min_sig_size is None:
        min_sig_size = 0
    if min_hits_per_gene is None:
        min_hits_per_gene = 1

    # filter signatures
    sig_size = sigs.value_counts("sig")
    sigs = sigs.loc[
        sigs.sig.isin(
            sig_size.loc[(sig_size >= min_sig_size) & (sig_size <= max_sig_size)].index,
        )
    ]

    # filter genes
    gene_hits = sigs.value_counts("gene")
    sigs = sigs.loc[sigs.gene.isin(gene_hits.loc[gene_hits >= min_hits_per_gene].index)]

    # only keep genes that appear in both the embeddings and the signatures
    good_genes = np.intersect1d(embs.index.values, sigs.gene.unique())
    embs = embs.loc[good_genes].copy()
    sigs = sigs.loc[sigs.gene.isin(good_genes)].copy()

    return sigs, embs
