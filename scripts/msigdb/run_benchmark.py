#!/usr/bin/env python
import argparse
import os
import yaml
import anndata

from benchmark_mlp import run_all


def load_config(path: str) -> dict:
    with open(path, 'r') as fin:
        return yaml.safe_load(fin)


def run_from_config(cfg: dict):
    if 'h5ad' in cfg['aadata_path']:
        adata = anndata.read_h5ad(cfg['aadata_path'])
    elif 'zarr' in cfg['aadata_path']:
        adata = anndata.read_zarr(cfg['aadata_path'])
    else:
        raise ValueError('Unsupported adata format')
    bench_cfg = cfg.get('benchmark', {})
    seeds = bench_cfg.get('seeds', [0])
    reps = bench_cfg.get('reps')
    gene_set_paths = bench_cfg.get('gene_set_paths', [])
    base_output = bench_cfg.get('output_dir', 'benchmark_results')
    os.makedirs(base_output, exist_ok=True)
    for seed in seeds:
        out_dir = os.path.join(base_output, f'seed_{seed}')
        os.makedirs(out_dir, exist_ok=True)
        run_all(adata, out_dir, emb_names=reps, seed=seed, verbose=True)


def main():
    parser = argparse.ArgumentParser(description='Run MLP benchmark from YAML config')
    parser.add_argument('config', type=str, help='Path to YAML config file')
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_from_config(cfg)


if __name__ == '__main__':
    main()
