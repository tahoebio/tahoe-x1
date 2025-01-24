# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
import argparse
import logging
import os
from typing import Dict, List, Optional

import datasets
import numpy as np
import scanpy as sc

from mosaicfm.data import CountDataset
from mosaicfm.tokenizer import GeneVocab


def find_h5ad_files(directory: str, ignore_subdirs: Optional[List] = None) -> List[str]:
    h5_files = []
    if ignore_subdirs is None:
        ignore_subdirs = []
    # Walk through all directories and files in the provided directory
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in ignore_subdirs]
        for file in files:
            # Check if the file ends with .h5
            if file.endswith(".h5ad"):
                # Append the full path of the file to the list
                h5_files.append(os.path.join(root, file))
    return h5_files


def dataset_generator(adata_files: List[str], vocab: GeneVocab, gene_col: str) -> Dict:
    for chunk_id, file in enumerate(adata_files):
        adata = sc.read_h5ad(file)
        adata.var["id_in_vocab"] = [vocab.get(gene, -1) for gene in adata.var[gene_col]]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        sc.pp.filter_genes(adata, min_counts=3)
        count_matrix = (
            adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
        )
        torch_dataset = CountDataset(
            count_matrix,
            gene_ids_in_vocab,
            cls_token_id=vocab["<cls>"],
            pad_value=-2,
        )
        for item in torch_dataset:
            yield item


def process_data(args):
    # Configure environment and logging
    os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = args.mem_size
    logging.basicConfig(
        format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
        level=logging.INFO,
    )
    log = logging.getLogger(__name__)

    adata_files = find_h5ad_files(args.adata_dir, args.ignore_dir)
    vocab = GeneVocab.from_file(args.vocab_path)
    gene_col = "feature_name"

    datasets.disable_caching()
    chunks = np.array_split(adata_files, 10)
    for i, chunk in enumerate(chunks):
        save_path = os.path.join(args.output_dir, f"chunk_{i}.dataset")
        if os.path.exists(save_path):
            log.info(f"Chunk {i} dataset already exists. Skipping.")
            continue
        log.info(f"Processing chunk {i} with  {len(chunk)} files")
        chunk_dataset = datasets.Dataset.from_generator(
            dataset_generator,
            gen_kwargs={
                "adata_files": chunk.tolist(),
                "vocab": vocab,
                "gene_col": gene_col,
            },
            num_proc=len(chunk),
            keep_in_memory=True,
        )
        chunk_dataset.save_to_disk(save_path)
        log.info(f"Chunk {i} dataset saved to disk with length: {len(chunk_dataset)}")
        chunk_dataset.cleanup_cache_files()
        del chunk_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process h5ad files and create datasets.",
    )
    parser.add_argument(
        "--adata_dir",
        type=str,
        required=True,
        help="Directory containing h5ad files.",
    )
    parser.add_argument(
        "--ignore_dir",
        nargs="*",
        default=None,
        help="Directories to ignore.",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        required=True,
        help="Path to the vocabulary JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output datasets.",
    )
    parser.add_argument(
        "--mem_size",
        type=str,
        default="500000000000",
        help="Maximum in-memory size for datasets.",
    )
    args = parser.parse_args()
    process_data(args)
