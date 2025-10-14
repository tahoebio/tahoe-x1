# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
import os
import sys

import anndata
import pandas as pd
import scanpy as sc
from benchmark_utils import read_embeddings, read_sigs
from omegaconf import OmegaConf as om


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


def main(cfg):
    embs = read_all_embs(cfg["embeddings_path"])
    sigs = read_sigs(cfg["signatures_path"])
    adata = create_anndata(embs, sigs)

    if cfg.get("filter", False):
        sc.pp.filter_genes(
            adata,
            min_cells=cfg.get("min_sig_size", 25),
            max_cells=cfg.get("max_sig_size", None),
        )
        sc.pp.filter_cells(adata, min_genes=cfg.get("min_hits_per_gene", 10))

    out_path = os.path.join(cfg["embeddings_path"], "embs_adata")

    try:
        adata.write(out_path + ".h5ad.gz", compression="gzip")
    except Exception:
        adata.write_zarr(out_path + ".zarr")

    with open(os.path.join(cfg["embeddings_path"], "gene_names.txt"), "w") as fout:
        fout.write("\n".join(adata.obs.gene.values) + "\n")


if __name__ == "__main__":
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
    main(cfg)
