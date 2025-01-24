# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
import json
import logging
import os
import sys

import cellxgene_census
import pandas as pd
import scanpy as sc
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from tqdm.autonotebook import tqdm


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
)
logging.getLogger(__name__).setLevel("INFO")


def main(cfg: DictConfig):
    version = cfg.get("version")
    output_root = cfg.get("output_root")

    # Ensure output directory exists
    os.makedirs(output_root, exist_ok=True)

    with cellxgene_census.open_soma(census_version=version) as census:
        cell_metadata = census["census_data"]["homo_sapiens"].obs.read(
            column_names=["is_primary_data", "soma_joinid", "suspension_type"],
        )
        gene_metadata = census["census_data"]["homo_sapiens"].ms["RNA"].var.read()
        gene_metadata = gene_metadata.concat().to_pandas()
        cell_metadata = cell_metadata.concat().to_pandas()

    base_vocabulary_path = cfg.get(
        "base_vocabulary_path",
        "/vevo/cellxgene/cellxgene_primary_2023-12-15_vocab.json",
    )
    with open(base_vocabulary_path, "r") as f:
        old_gene_dict = json.load(f)

    master_gene_to_id_path = cfg.get(
        "gene_to_id_path",
        "https://github.com/bowang-lab/scGPT/files/13243634/gene_info.csv",
    )
    gene_to_id_table = pd.read_csv(master_gene_to_id_path)
    id_to_gene_old = {
        gene_id: gene_name
        for gene_name, gene_id in zip(
            gene_to_id_table["feature_name"],
            gene_to_id_table["feature_id"],
        )
    }

    new_gene_ids = gene_metadata["feature_id"].to_list()
    id_to_gene_new = {
        gene_id: gene_name
        for gene_name, gene_id in zip(
            gene_metadata["feature_name"],
            gene_metadata["feature_id"],
        )
    }

    new_gene_list = [
        id_to_gene_old.get(gene_id, id_to_gene_new[gene_id]) for gene_id in new_gene_ids
    ]
    master_gene_to_id = dict(zip(new_gene_list, new_gene_ids))
    master_id_to_gene = dict(zip(new_gene_ids, new_gene_list))

    log.info(f"old gene list length: {len(old_gene_dict)}")
    expanded_dict = old_gene_dict.copy()
    starting_num = max(old_gene_dict.values()) + 1
    for new_gene in new_gene_list:
        if new_gene not in old_gene_dict:
            expanded_dict[new_gene] = starting_num
            starting_num += 1
    log.info(
        f"new gene dict length: {len(expanded_dict)}",
    )

    new_vocab_path = os.path.join(
        output_root,
        f"cellxgene_primary_{version}_vocab.json",
    )
    with open(new_vocab_path, "w") as f:
        json.dump(expanded_dict, f, indent=2)
    log.info(f"Vocabulary saved to {new_vocab_path}")

    new_gene_to_id_path = os.path.join(output_root, f"gene_info_{version}.json")
    with open(new_gene_to_id_path, "w") as f:
        json.dump(master_gene_to_id, f, indent=2)
    log.info(f"Gene to ID mapping saved to {new_gene_to_id_path}")

    obs_coords = cell_metadata[
        (cell_metadata["is_primary_data"]) & (cell_metadata["suspension_type"] != "na")
    ]["soma_joinid"].tolist()
    log.info(f"Number of unique cells in {version} data: {len(obs_coords)}")

    min_count_per_gene = cfg.get("min_count_per_gene", 3)
    chunk_size = cfg.get("chunk_size", 200000)
    dataset_size = len(obs_coords)

    with cellxgene_census.open_soma(census_version=version) as census:
        for chunk_id, chunk_indices in tqdm(
            enumerate(chunker(obs_coords, chunk_size)),
            total=dataset_size // chunk_size + 1,
        ):
            save_path = os.path.join(output_root, f"chunk_{chunk_id}.h5ad")
            adata = cellxgene_census.get_anndata(
                census,
                organism="Homo sapiens",
                obs_coords=chunk_indices,
            )
            sc.pp.filter_genes(adata, min_counts=min_count_per_gene)
            updated_feature_name = [
                master_id_to_gene[gene_id] for gene_id in adata.var["feature_id"]
            ]
            adata.var["feature_name"] = updated_feature_name
            adata.write_h5ad(save_path)
            log.info(f"Chunk {chunk_id} saved to {save_path}")


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    # Disable resolving environment variables through omegaconf.
    om.clear_resolver("oc.env")
    # Load yaml file.
    with open(yaml_path) as f:
        cfg = om.load(f)
    om.resolve(cfg)
    main(cfg)
