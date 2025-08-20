import argparse
import os
from pathlib import Path
from typing import List

import anndata

from benchmark_mlp import run_all


def load_adata(path: str) -> anndata.AnnData:
    if path.endswith(".h5ad") or path.endswith(".h5ad.gz"):
        return anndata.read_h5ad(path)
    if path.endswith(".zarr"):
        return anndata.read_zarr(path)
    raise ValueError(f"Unsupported file format: {path}")


def run_for_path(h5ad_path: str, output_dir: str, seeds: List[int]):
    adata = load_adata(h5ad_path)
    base = Path(h5ad_path).stem
    for i, seed in enumerate(seeds, 1):
        rep_dir = os.path.join(output_dir, base, f"rep_{i}")
        os.makedirs(rep_dir, exist_ok=True)
        run_all(adata, rep_dir, seed=seed, verbose=True)


def main(h5ad_path: str, output_dir: str, seeds: List[int], gene_sets: List[str]):
    if gene_sets:
        for gs in gene_sets:
            path = os.path.join(h5ad_path, gs, "embs_adata.h5ad.gz")
            run_for_path(path, os.path.join(output_dir, gs), seeds)
    else:
        run_for_path(h5ad_path, output_dir, seeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark replicates for embeddings")
    parser.add_argument("h5ad_path", help="Path to AnnData file or base directory")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--gene-sets", nargs="*", default=None, help="Optional gene set subdirectories")
    args = parser.parse_args()

    main(args.h5ad_path, args.output_dir, args.seeds, args.gene_sets)

