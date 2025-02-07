# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import scanpy as sc
from omegaconf import DictConfig, OmegaConf

from mosaicfm.tokenizer import GeneVocab

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def find_h5ad_files(directory: str) -> list:
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files
        if file.endswith(".h5ad")
    ]


def load_json_file(file_path: str) -> dict:
    with open(file_path, "r") as file:
        return json.load(file)


def check_matches(ids: List[str], vocab: Dict[str, int]) -> Tuple[int, float]:
    matches = sum(id in vocab for id in ids)
    return matches, (matches / len(ids)) * 100


def process_dataset(
    file_path: str,
    vocab: Dict[str, int],
    gene_to_id: Dict[str, str],
    possible_keys: List[str],
    min_percentage: int,
) -> Tuple[int, Optional[np.ndarray], Optional[sc.AnnData]]:
    adata = sc.read_h5ad(file_path, backed="r")
    feature_IDs, matched = None, False

    for key in possible_keys:
        if key in adata.var.columns:
            matches, percentage_matched = check_matches(adata.var[key].values, vocab)
            if percentage_matched > min_percentage:
                feature_IDs = adata.var[key].values
                matched = True
                break

    if not matched:
        remapped_ids = [gene_to_id.get(gene) for gene in adata.var_names]
        matches, percentage_matched = check_matches(remapped_ids, vocab)
        if matches > 0 and percentage_matched > min_percentage:
            feature_IDs = remapped_ids
            matched = True

    if matched:
        return percentage_matched, feature_IDs, adata
    return 0, None, None


def save_processed_data(
    adata: sc.AnnData,
    feature_IDs: List[str],
    file_path: str,
    output_dir: str,
    vocab: Dict[str, int],
) -> str:
    output_file = os.path.join(output_dir, os.path.basename(file_path))
    mask = [id in vocab for id in feature_IDs]
    adata_subset = adata[:, mask].to_memory()
    adata_subset.var["feature_id"] = np.array(feature_IDs)[mask]
    adata_subset = adata_subset[:, ~adata_subset.var["feature_id"].duplicated()]
    adata_subset.write_h5ad(output_file)
    return output_file


def main(cfg: DictConfig):
    log.info("Starting dataset processing.")
    files = find_h5ad_files(cfg.scperturb.adata_root)
    vocab = GeneVocab.from_file(
        os.path.join(cfg.vocab.output_root, cfg.vocab.output_file),
    ).get_stoi()
    id_to_gene = load_json_file(
        os.path.join(cfg.vocab.output_root, cfg.vocab.id_to_gene_output_file),
    )
    gene_to_id = dict(zip(id_to_gene.values(), id_to_gene.keys()))
    results = []
    for file_path in files:
        percentage_matched, feature_IDs, adata = process_dataset(
            file_path,
            vocab,
            gene_to_id,
            cfg.scperturb.possible_gene_id_keys,
            cfg.scperturb.min_percentage_matched,
        )
        if adata:
            output_file = save_processed_data(
                adata,
                feature_IDs,
                file_path,
                cfg.scperturb.output_dir,
                vocab,
            )
            results.append((output_file, percentage_matched))
        else:
            log.info(f"No valid gene IDs found or matched for {file_path}.")

    log.info(f"Script completed with {len(results)} datasets processed successfully.")
    return results


if __name__ == "__main__":
    import sys

    yaml_path = sys.argv[1]
    log.info(f"Loading configuration from {yaml_path}...")
    cfg = OmegaConf.load(yaml_path)
    OmegaConf.resolve(cfg)
    main(cfg)
