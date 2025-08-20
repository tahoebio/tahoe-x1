#!/opt/conda/bin/python

import os
import anndata
import scanpy as sc
import numpy as np
import pandas as pd
from benchmark_utils import read_sigs, read_embeddings


def read_all_embs(embs_path):
    embs = {}
    for fn in [fn for fn in os.listdir(embs_path) if fn.endswith(".npz")]:
        emb_name = os.path.splitext(fn)[0]
        emb = read_embeddings(os.path.join(embs_path, fn))
        emb.index = emb.index.astype(str)
        emb.columns = emb.columns.astype(str)
        embs[emb_name] = emb
    return embs


def create_anndata(embs, sigs):
    shared_genes = set(sigs.gene)
    for emb in embs.values():
        shared_genes = shared_genes.intersection(emb.index)
    shared_genes = list(shared_genes)

    hit_matrix = pd.crosstab(sigs.gene, sigs.sig).loc[shared_genes]
    embs = {k: v.loc[shared_genes] for k, v in embs.items()}

    return anndata.AnnData(
        X=hit_matrix,
        obs=hit_matrix.index.to_frame(),
        var=hit_matrix.columns.to_frame(),
        obsm=embs,
        uns={"sigs": sigs},
    )


def main(embs_path, sigs_path, filter, min_sig_size, max_sig_size, min_hits_per_gene):
    embs = read_all_embs(embs_path)
    sigs = read_sigs(sigs_path)
    adata = create_anndata(embs, sigs)

    if filter:
        sc.pp.filter_genes(adata, min_cells=min_sig_size, max_cells=max_sig_size)
        sc.pp.filter_cells(adata, min_genes=min_hits_per_gene)


    try:
        adata.write(os.path.join(embs_path, "embs_adata.h5ad.gz"), compression="gzip")
    except:
        adata.write_zarr(os.path.join(embs_path, "embs_adata.zarr"))
    (
        open(os.path.join(embs_path, "gene_names.txt"), "w").write(
            "\n".join(adata.obs.gene.values) + "\n"
        )
    )



# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("embs_path", type=str)
    parser.add_argument("sigs_path", type=str)
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--min_sig_size", type=int, default=25)
    parser.add_argument("--max_sig_size", type=int, default=None)
    parser.add_argument("--min_hits_per_gene", type=int, default=10)
    args = parser.parse_args()

    main(
        args.embs_path,
        args.sigs_path,
        args.filter,
        args.min_sig_size,
        args.max_sig_size,
        args.min_hits_per_gene,
    )
