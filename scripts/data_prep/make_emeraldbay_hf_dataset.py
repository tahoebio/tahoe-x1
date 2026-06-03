# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
"""
Generate HuggingFace dataset with extended vocabulary and metadata tables.

Designed for datasets that need vocabulary extension beyond the base
Tahoe-100M vocab, cell line ID mapping, and metadata table generation.

Usage:
    python make_emeraldbay_hf_dataset.py <config.yaml>

Example:
    python make_emeraldbay_hf_dataset.py emeraldbay.yaml
"""

import gc
import json
import logging
import math
import os
import sys
from typing import Dict, Generator, List, Optional

import datasets
import numpy as np
import pandas as pd
import scanpy as sc
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from scipy.sparse import csc_matrix, csr_matrix

logging.basicConfig(
    format=r"%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ── Vocabulary ──────────────────────────────────────────────────────────────


def build_extended_vocab(
    adata_var: pd.DataFrame,
    vocab_json_path: str,
    gene_col: str = "gene_id",
) -> tuple:
    """Build an extended vocabulary that includes all genes from the dataset.

    Starts from an existing base vocab (JSON), keeps all existing gene->token
    mappings intact, and appends new genes not found in the base vocab.
    Replaces junk padding tokens and re-pads to a multiple of 64.

    Args:
        adata_var: The .var DataFrame from the AnnData file.
        vocab_json_path: Path to the base vocabulary JSON file.
        gene_col: Column in adata_var containing Ensembl gene IDs.

    Returns:
        extended_vocab: dict mapping token_name -> token_id
        gene_metadata_rows: list of dicts with gene_symbol, ensembl_id, token_id
    """
    with open(vocab_json_path) as f:
        base_vocab = json.load(f)

    # Separate real tokens from junk/special
    special_tokens = {
        k: v for k, v in base_vocab.items() if k.startswith("<") and "junk" not in k
    }
    junk_tokens = {k: v for k, v in base_vocab.items() if "junk" in k}
    gene_tokens = {
        k: v for k, v in base_vocab.items() if not k.startswith("<") and "junk" not in k
    }

    log.info(
        f"Base vocab: {len(special_tokens)} special, "
        f"{len(gene_tokens)} genes, {len(junk_tokens)} junk",
    )

    # Find genes in the dataset that are NOT in the existing vocab
    dataset_genes = list(zip(adata_var.index, adata_var[gene_col].values))
    existing_ensembl = set(gene_tokens.keys())
    new_genes = [
        (symbol, eid) for symbol, eid in dataset_genes if eid not in existing_ensembl
    ]
    log.info(
        f"Dataset genes: {len(dataset_genes)}, "
        f"already in vocab: {len(dataset_genes) - len(new_genes)}, "
        f"new: {len(new_genes)}",
    )

    # Sort junk tokens by token_id to know which IDs to reclaim
    junk_sorted = sorted(junk_tokens.items(), key=lambda x: x[1])
    junk_ids = [v for _, v in junk_sorted]

    # Assign token IDs to new genes:
    #  - First, reclaim junk token IDs
    #  - Then, append after the last used ID
    extended_vocab = dict(base_vocab)
    for k in junk_tokens:
        del extended_vocab[k]

    next_id = max(base_vocab.values()) + 1
    available_ids = list(junk_ids)

    for _symbol, eid in new_genes:
        if available_ids:
            token_id = available_ids.pop(0)
        else:
            token_id = next_id
            next_id += 1
        extended_vocab[eid] = token_id

    # Re-pad to next multiple of 64
    current_size = len(extended_vocab)
    target_size = math.ceil(current_size / 64) * 64
    num_new_junk = target_size - current_size
    for i in range(num_new_junk):
        extended_vocab[f"<junk{i}>"] = next_id
        next_id += 1

    log.info(f"Extended vocab size: {len(extended_vocab)} (padded to {target_size})")

    # Build gene_metadata rows (all genes, not junk/special)
    ensembl_to_symbol = {eid: sym for sym, eid in dataset_genes}
    gene_metadata_rows = []
    for k, v in sorted(extended_vocab.items(), key=lambda x: x[1]):
        if k.startswith("<"):
            continue
        symbol = ensembl_to_symbol.get(k, k)
        gene_metadata_rows.append(
            {
                "gene_symbol": symbol,
                "ensembl_id": k,
                "token_id": v,
            },
        )

    return extended_vocab, gene_metadata_rows


# ── Expression data generator ───────────────────────────────────────────────


def expression_data_generator(
    adata_path: str,
    extended_vocab: dict,
    cfg: DictConfig,
    cell_line_map: Optional[dict] = None,
) -> Generator[Dict, None, None]:
    """Generator yielding one dict per cell for expression_data.

    Args:
        adata_path: Path to the h5ad file.
        extended_vocab: dict mapping token_name -> token_id.
        cfg: The huggingface section of the config.
        cell_line_map: Optional dict mapping internal cell line IDs to CVCL.

    Yields:
        Dict with genes, expressions, and metadata columns.
    """
    gene_col = cfg.gene_col
    obs_metadata_columns = list(cfg.get("obs_metadata_columns", []))
    cl_mapping = cfg.get("cell_line_mapping", None)

    log.info(f"Loading adata from {adata_path}")
    adata = sc.read_h5ad(adata_path, backed="r")
    log.info(f"Loaded adata: {adata.shape[0]} cells, {adata.shape[1]} genes")

    # Map genes to token IDs
    adata.var.reset_index(inplace=True)
    adata.var["token_id"] = [extended_vocab.get(g, -1) for g in adata.var[gene_col]]
    n_unmapped = (adata.var["token_id"] == -1).sum()
    if n_unmapped > 0:
        log.warning(f"{n_unmapped} genes not in extended vocab")
    adata = adata[:, adata.var["token_id"] >= 0]
    gene_token_ids = np.array(adata.var["token_id"])

    # Load count matrix
    count_matrix = adata.X
    if isinstance(count_matrix, np.ndarray):
        count_matrix = csr_matrix(count_matrix)
    elif isinstance(count_matrix, csc_matrix):
        count_matrix = count_matrix.tocsr()
    elif hasattr(count_matrix, "to_memory"):
        count_matrix = count_matrix.to_memory().tocsr()

    obs = adata.obs.copy()
    index_name = obs.index.name if obs.index.name is not None else "index"
    obs.reset_index(inplace=True)

    n_cells = count_matrix.shape[0]
    for idx in range(n_cells):
        if idx % 100_000 == 0:
            log.info(f"Processing cell {idx}/{n_cells}")

        row = count_matrix.getrow(idx)
        nonzero_idx = row.indices
        values = row.data
        genes = gene_token_ids[nonzero_idx]

        item = {
            "genes": genes.tolist(),
            "expressions": values.astype(np.float32).tolist(),
        }

        # Add metadata columns
        for col in obs_metadata_columns:
            item[col] = str(obs.iloc[idx][col])

        # Add cell line with optional mapping
        if cl_mapping:
            adata_col = cl_mapping.adata_col
            raw_cl = str(obs.iloc[idx][adata_col])
            item["cell_line"] = (
                cell_line_map.get(raw_cl, raw_cl) if cell_line_map else raw_cl
            )

        # Add obs index as cell barcode
        item[index_name] = str(obs.iloc[idx][index_name])

        yield item

    del adata, count_matrix
    gc.collect()


# ── Metadata table builders ─────────────────────────────────────────────────


def build_drug_metadata(adata_path: str, columns: List[str]) -> List[dict]:
    """Extract unique drug metadata from adata obs."""
    adata = sc.read_h5ad(adata_path, backed="r")
    obs = adata.obs[columns].copy()
    obs = obs.drop_duplicates()
    obs = obs.sort_values(columns[0]).reset_index(drop=True)
    records = obs.to_dict("records")
    for r in records:
        for col in columns:
            r[col] = str(r[col])
    del adata
    gc.collect()
    return records


def build_cell_line_metadata(
    csv_path: str,
    adata_path: str,
    filter_col: str,
    drop_columns: Optional[List[str]] = None,
) -> List[dict]:
    """Build cell_line_metadata from mapping CSV, filtered to dataset cell lines."""
    cl_df = pd.read_csv(csv_path)

    # Get the set of cell lines in the dataset
    adata = sc.read_h5ad(adata_path, backed="r")
    dataset_cell_lines = set(adata.obs["cell_line"].unique())
    del adata
    gc.collect()

    cl_df = cl_df[cl_df[filter_col].isin(dataset_cell_lines)].copy()
    log.info(f"Cell line metadata: {len(cl_df)} lines")

    if drop_columns:
        cl_df = cl_df.drop(columns=[c for c in drop_columns if c in cl_df.columns])

    # Fix mixed-type columns
    str_cols = cl_df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        cl_df[col] = cl_df[col].fillna("").astype(str)

    return cl_df.to_dict("records")


def build_summary_statistics(
    parquet_path: str,
    cell_line_map: Optional[dict] = None,
) -> List[dict]:
    """Build summary_statistics from a parquet file, with optional cell line mapping."""
    df = pd.read_parquet(parquet_path)
    if cell_line_map:
        df["cell_line"] = df["cell_line"].map(cell_line_map).fillna(df["cell_line"])
    df["condition"] = df["condition"].astype(str)
    return df.to_dict("records")


# ── Helpers ─────────────────────────────────────────────────────────────────


def save_dataset(records: List[dict], path: str, name: str) -> None:
    """Save a list of dicts as a HuggingFace Dataset to disk."""
    ds = datasets.Dataset.from_list(records)
    ds.save_to_disk(path)
    log.info(f"{name}: {len(ds)} rows -> {path}")
    del ds
    gc.collect()


# ── Main ────────────────────────────────────────────────────────────────────


def main(cfg: DictConfig) -> None:
    output_root = cfg.output_root
    os.makedirs(output_root, exist_ok=True)

    hf_cfg = cfg.huggingface
    vocab_cfg = cfg.vocab
    meta_cfg = cfg.get("metadata", {})
    adata_path = hf_cfg.adata_path

    # ── Cell line mapping ────────────────────────────────────────────────
    cell_line_map = None
    cl_mapping = hf_cfg.get("cell_line_mapping", None)
    if cl_mapping:
        cl_df = pd.read_csv(cl_mapping.csv_path)
        cell_line_map = dict(
            zip(cl_df[cl_mapping.source_col], cl_df[cl_mapping.target_col]),
        )
        log.info(f"Loaded cell line mapping: {len(cell_line_map)} entries")

    # ── Vocabulary ───────────────────────────────────────────────────────
    if vocab_cfg.get("extend_vocab", False):
        log.info("Building extended vocabulary...")
        adata_tmp = sc.read_h5ad(adata_path, backed="r")
        extended_vocab, gene_metadata_rows = build_extended_vocab(
            adata_tmp.var,
            vocab_cfg.base_vocab_path,
            gene_col=hf_cfg.gene_col,
        )
        del adata_tmp
        gc.collect()

        vocab_out_path = os.path.join(output_root, vocab_cfg.output_file)
        with open(vocab_out_path, "w") as f:
            json.dump(extended_vocab, f, indent=2)
        log.info(f"Saved extended vocab to {vocab_out_path}")
    else:
        with open(vocab_cfg.base_vocab_path) as f:
            extended_vocab = json.load(f)
        gene_metadata_rows = None
        log.info(f"Loaded base vocab: {len(extended_vocab)} tokens")

    # ── Metadata tables ──────────────────────────────────────────────────
    gm_cfg = meta_cfg.get("gene_metadata", {})
    if gm_cfg.get("enabled", False) and gene_metadata_rows is not None:
        save_dataset(
            gene_metadata_rows,
            os.path.join(output_root, "gene_metadata"),
            "gene_metadata",
        )

    dm_cfg = meta_cfg.get("drug_metadata", {})
    if dm_cfg.get("enabled", False):
        drug_records = build_drug_metadata(adata_path, list(dm_cfg.columns))
        save_dataset(
            drug_records,
            os.path.join(output_root, "drug_metadata"),
            "drug_metadata",
        )

    cl_cfg = meta_cfg.get("cell_line_metadata", {})
    if cl_cfg.get("enabled", False):
        cl_records = build_cell_line_metadata(
            cl_cfg.csv_path,
            adata_path,
            cl_cfg.filter_col,
            drop_columns=list(cl_cfg.get("drop_columns", [])),
        )
        save_dataset(
            cl_records,
            os.path.join(output_root, "cell_line_metadata"),
            "cell_line_metadata",
        )

    ss_cfg = meta_cfg.get("summary_statistics", {})
    if ss_cfg.get("enabled", False):
        ss_records = build_summary_statistics(
            ss_cfg.parquet_path,
            cell_line_map=cell_line_map,
        )
        save_dataset(
            ss_records,
            os.path.join(output_root, "summary_statistics"),
            "summary_statistics",
        )

    # ── Expression data ──────────────────────────────────────────────────
    log.info("Generating expression_data (this will take a while)...")
    expr_ds = datasets.Dataset.from_generator(
        expression_data_generator,
        gen_kwargs={
            "adata_path": adata_path,
            "extended_vocab": extended_vocab,
            "cfg": hf_cfg,
            "cell_line_map": cell_line_map,
        },
        keep_in_memory=False,
    )
    log.info(f"expression_data generated: {len(expr_ds)} rows")

    train_path = os.path.join(output_root, "expression_data", "train")
    os.makedirs(train_path, exist_ok=True)
    expr_ds.save_to_disk(train_path)
    log.info(f"expression_data train: {len(expr_ds)} rows -> {train_path}")

    del expr_ds
    gc.collect()
    log.info("Done! All datasets saved to %s", output_root)


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    with open(yaml_path) as f:
        cfg = om.load(f)
    om.resolve(cfg)
    main(cfg)
