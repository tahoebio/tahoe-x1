# Copyright (C) Vevo Therapeutics 2025. All rights reserved.

import logging
import os
import sys
from typing import Any, Iterator, Sequence

import cellxgene_census
import scanpy as sc
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from tqdm.autonotebook import tqdm

# Logging setup
log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)


def chunker(seq: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def main(cfg: DictConfig) -> None:
    census_version: str = cfg.cellxgene.version
    with cellxgene_census.open_soma(census_version=census_version) as census:
        cell_metadata = census["census_data"]["homo_sapiens"].obs.read(
            column_names=["is_primary_data", "soma_joinid", "suspension_type"],
        )
        gene_metadata = census["census_data"]["homo_sapiens"].ms["RNA"].var.read()
        gene_metadata = gene_metadata.concat().to_pandas()
        cell_metadata = cell_metadata.concat().to_pandas()

    obs_coords = cell_metadata[
        (cell_metadata["is_primary_data"]) & (cell_metadata["suspension_type"] != "na")
    ]["soma_joinid"].tolist()
    log.info(f"Number of unique cells in {census_version} data: {len(obs_coords)}")
    adata_reference = sc.read_h5ad(cfg.vocab.reference_adata)
    reference_ids = adata_reference.var[cfg.vocab.use_col]
    cellxgene_ids = gene_metadata[
        (gene_metadata["n_measured_obs"] >= cfg.cellxgene.min_gene_measured_obs)
        & (gene_metadata["nnz"] >= cfg.cellxgene.min_gene_nnz)
    ][cfg.cellxgene.use_col].values
    cellxgene_feature_ids = set(cellxgene_ids) & set(reference_ids.values)
    soma_joinid_list = gene_metadata[
        gene_metadata[cfg.cellxgene.use_col].isin(cellxgene_feature_ids)
    ]["soma_joinid"].values

    chunk_size: int = cfg.cellxgene.chunk_size
    dataset_size: int = len(obs_coords)
    min_count_per_gene: int = cfg.cellxgene.min_count_per_gene

    with cellxgene_census.open_soma(census_version=census_version) as census:
        for chunk_id, chunk_indices in tqdm(
            enumerate(chunker(obs_coords, chunk_size)),
            total=dataset_size // chunk_size + 1,
        ):
            save_path = os.path.join(
                cfg.cellxgene.output_root,
                f"chunk_{chunk_id}.h5ad",
            )
            adata = cellxgene_census.get_anndata(
                census,
                organism="Homo sapiens",
                obs_coords=chunk_indices,
                var_coords=soma_joinid_list,
            )
            sc.pp.filter_genes(adata, min_counts=min_count_per_gene)
            adata.write_h5ad(save_path)
            log.info(f"Chunk {chunk_id} saved to {save_path}")


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    log.info(f"Loading configuration from {yaml_path}...")

    with open(yaml_path) as f:
        cfg = om.load(f)
    om.resolve(cfg)

    main(cfg)
    log.info("Script execution completed.")
