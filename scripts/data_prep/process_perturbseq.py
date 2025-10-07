# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
import json
import logging
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from scipy import sparse

from tahoex.tokenizer import GeneVocab

# Logging setup
log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
)
logging.getLogger(__name__).setLevel("INFO")


def generate_gene_aliases(
    gene_info_path: str,
    custom_aliases: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    log.info(f"Generating gene aliases from {gene_info_path}...")
    raw_df = pd.read_csv(gene_info_path, sep="\t")
    synonym_to_symbol = {
        synonym: row["Symbol"]
        for row in raw_df.to_dict(orient="records")
        for synonym in row["Synonyms"].split("|")
    }

    # Ensure all symbols are included as their own synonym
    synonym_to_symbol.update(
        {
            value: value
            for value in synonym_to_symbol.values()
            if value not in synonym_to_symbol
        },
    )

    # Apply any custom gene aliases (e.g., "KIAA1804" -> "MAP3K21")
    if custom_aliases:
        synonym_to_symbol.update(custom_aliases)

    # Convert dictionary to DataFrame
    df = pd.DataFrame(
        list(synonym_to_symbol.items()),
        columns=["synonym", "gene_symbol"],
    )
    df.set_index("synonym", inplace=True)

    log.info(f"Generated gene alias DataFrame with {len(df)} entries.")
    return df


def map_gene_names_to_vocab(
    gene_name_list: List[str],
    vocab: GeneVocab,
    gene_alias_dict: Dict[str, str],
) -> Optional[List[int]]:
    vocab_map_per_row = []
    for gene_name in gene_name_list:
        gene_alias = gene_alias_dict.get(gene_name, "None")
        if gene_name in vocab:
            vocab_map_per_row.append(vocab[gene_name])
        elif gene_alias in vocab:
            vocab_map_per_row.append(vocab[gene_alias])
        else:
            return None
    if len(vocab_map_per_row) == len(gene_name_list):
        return vocab_map_per_row
    return None


def map_gene_name_to_dep(
    gene_name_list: List[str],
    dep_scores: Dict[str, float],
    gene_alias_dict: Dict[str, str],
) -> Optional[List[float]]:
    dep_scores_per_row = []
    for gene_name in gene_name_list:
        gene_alias = gene_alias_dict.get(gene_name, "None")
        if gene_name in dep_scores:
            dep_scores_per_row.append(dep_scores[gene_name])
        elif gene_alias in dep_scores:
            dep_scores_per_row.append(dep_scores[gene_alias])
        else:
            return None
    if len(dep_scores_per_row) == len(gene_name_list):
        return dep_scores_per_row
    return None


def map_gene_name_or_ensembl_to_vocab(
    gene_name: str,
    ensembl_id: str,
    vocab: GeneVocab,
    gene_alias_dict: Dict[str, str],
    ensembl_to_gene_name: Dict[str, str],
) -> int:
    if gene_name in vocab:
        return vocab[gene_name]
    gene_alias = gene_alias_dict.get(gene_name, "None")
    if gene_alias in vocab:
        return vocab[gene_alias]
    gene_name_from_id = ensembl_to_gene_name.get(ensembl_id, "None")
    if gene_name_from_id in vocab:
        return vocab[gene_name_from_id]
    else:
        return vocab["<pad>"]


def record_generator(
    adata: sc.AnnData,
    perturbation_metadata: pd.DataFrame,
    cfg: DictConfig,
    include_depmap: bool,
) -> Dict:
    ctrl_adata = adata[adata.obs[cfg.perturbation_col] == cfg.control_value]
    num_ctrl_samples = len(ctrl_adata)
    perturbation_list = list(set(adata.obs[cfg.perturbation_col]) - {cfg.control_value})
    log.info(f"Using {len(perturbation_list)} perturbations")
    gene_ids = np.array(adata.var["id_in_vocab"], dtype=np.int32)
    control_counts = ctrl_adata.X.toarray()
    cell_line_name = cfg.cell_line_name

    for perturbation_name in perturbation_list:
        perturb_adata = adata[adata.obs[cfg.perturbation_col] == perturbation_name]
        log.info(f"Retrieved {len(perturb_adata)} cells for {perturbation_name}")
        perturbation_edist = np.float32(
            perturbation_metadata.loc[perturbation_name, cfg.edist_col],
        )
        if include_depmap:
            depmap_dependency = np.array(
                perturbation_metadata.loc[perturbation_name, cfg.depmap_col],
                np.float32,
            )
        else:
            depmap_dependency = None
        perturbation_targets = np.array(
            perturbation_metadata.loc[perturbation_name, "target_gene_vocab_id"],
            dtype=np.int32,
        )
        perturb_counts = perturb_adata.X.toarray()

        assert all(target in gene_ids for target in perturbation_targets)

        for idx in range(len(perturb_adata)):
            expressions_perturbed = perturb_counts[idx]

            random_ids = np.random.randint(
                low=0,
                high=num_ctrl_samples,
                size=cfg.num_ctrl_samples_to_pair,
            )

            for ctrl_id in random_ids:
                expressions_ctrl = control_counts[ctrl_id]
                record = {
                    "perturbation_edist": perturbation_edist,
                    "perturbation_target_genes": perturbation_targets,
                    "expressions_ctrl_raw": expressions_ctrl,
                    "expressions_perturbed_raw": expressions_perturbed,
                    "genes": gene_ids,
                    "cell_line": cell_line_name,
                    "perturbation_name": perturbation_name,
                }
                if depmap_dependency is not None:
                    record["depmap_dependency"] = depmap_dependency
                yield record


def main(cfg: DictConfig) -> Dataset:
    dataset_name = cfg.dataset_name
    log.info(f"Starting processing for {dataset_name} Dataset...")

    # Load vocab and gene mapping
    log.info(f"Loading vocab from {cfg.vocab_path}...")
    vocab = GeneVocab.from_file(cfg.vocab_path)
    log.info(f"Vocab loaded with size {len(vocab)}")

    log.info(f"Loading gene-to-ensembl mapping from {cfg.gene_to_ensembl_path}...")
    with open(cfg.gene_to_ensembl_path) as f:
        gene_to_ensembl = json.load(f)
    log.info(f"Gene-to-ensembl mapping loaded with {len(gene_to_ensembl)} entries")

    ensembl_to_gene_name = {v: k for k, v in gene_to_ensembl.items()}

    # Generate gene aliases
    gene_alias_df = generate_gene_aliases(
        cfg.gene_info_path,
        cfg.get("custom_aliases", None),
    )

    gene_alias_to_gene_symbol = gene_alias_df["gene_symbol"].to_dict()

    # Load in raw data
    log.info(f"Loading raw data from {cfg.raw_data_path}...")
    adata = sc.read_h5ad(cfg.raw_data_path)
    if isinstance(adata.X, np.ndarray):
        log.warning(
            "The adata.X object is dense. Converting to sparse matrix format...",
        )
        adata.X = sparse.csr_matrix(adata.X)
        log.info("Conversion to sparse matrix format completed.")
    adata.var["gene_name"] = adata.var.index
    adata.var = adata.var.rename(columns={cfg.ensembl_col: "ensembl_id"})
    log.info(f"Raw data loaded. Data shape: {adata.shape}")

    # Load metadata
    log.info(f"Loading {dataset_name} metadata from {cfg.metadata_path}...")
    perturbation_meta_df = pd.read_csv(cfg.metadata_path)
    # Remove metadata for ctrl perturbation (most fields are NaN)
    perturbation_meta_df = perturbation_meta_df[
        perturbation_meta_df[cfg.perturbation_col] != cfg.control_value
    ]
    perturbation_meta_df["target_gene_names"] = perturbation_meta_df[
        cfg.perturbation_col
    ].apply(
        lambda x: [
            gene_name.strip()
            for gene_name in x.split("+")
            if gene_name != cfg.control_value
        ],
    )
    perturbation_meta_df["target_gene_vocab_id"] = perturbation_meta_df[
        "target_gene_names"
    ].apply(
        lambda gene_name_list: map_gene_names_to_vocab(
            gene_name_list,
            vocab,
            gene_alias_to_gene_symbol,
        ),
    )
    perturbation_meta_df = perturbation_meta_df.set_index(
        cfg.perturbation_col,
        drop=False,
    )

    unmapped_targets = perturbation_meta_df[
        perturbation_meta_df["target_gene_vocab_id"].isna()
    ][cfg.perturbation_col]
    assert (
        len(unmapped_targets) == 0
    ), f"Couldn't map these genes to a vocab gene: {unmapped_targets.values}"

    log.info(
        f"{dataset_name} metadata loaded with {len(perturbation_meta_df)} records.",
    )

    include_depmap = False
    if "depmap_col" in cfg and "cell_line_depmap_id" in cfg:
        include_depmap = True
        # Load DepMap dependency score
        log.info(f"Loading DepMap dependency scores from {cfg.depmap_scores_path}...")
        depmap_df = pd.read_csv(cfg.depmap_scores_path)
        depmap_df = depmap_df.rename(columns={"Unnamed: 0": "cell_line_depmap_id"})

        cell_line_depmap_id = cfg.cell_line_depmap_id
        cell_line_dependency_scores = (
            depmap_df.set_index("cell_line_depmap_id")
            .loc[cell_line_depmap_id, :]
            .transpose()
            .to_dict()
        )
        cell_line_dependency_scores = {
            k.split(" (")[0]: v for k, v in cell_line_dependency_scores.items()
        }

        perturbation_meta_df["depmap_dependency"] = perturbation_meta_df[
            "target_gene_names"
        ].apply(
            lambda gene_name_list: map_gene_name_to_dep(
                gene_name_list,
                cell_line_dependency_scores,
                gene_alias_to_gene_symbol,
            ),
        )
        log.info(f"DepMap dependency scores added to {dataset_name} metadata.")

        depmap_nan_perts = perturbation_meta_df[
            perturbation_meta_df["depmap_dependency"].isna()
        ][cfg.perturbation_col].values
        log.info(
            f"Found {len(depmap_nan_perts)} perturbations with missing DepMap dependency scores. Removing them.",
        )
        perturbation_meta_df = perturbation_meta_df[
            ~perturbation_meta_df[cfg.perturbation_col].isin(depmap_nan_perts)
        ]
        adata = adata[~adata.obs[cfg.perturbation_col].isin(depmap_nan_perts)]

    # Data preparation
    log.info("Starting data preparation...")
    gene_ids_in_vocab = np.array(
        [
            map_gene_name_or_ensembl_to_vocab(
                gene_name,
                gene_id,
                vocab,
                gene_alias_to_gene_symbol,
                ensembl_to_gene_name,
            )
            for gene_name, gene_id in zip(adata.var.index, adata.var["ensembl_id"])
        ],
    )

    adata.var["id_in_vocab"] = gene_ids_in_vocab
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    filter_vocab = adata.var["id_in_vocab"] != vocab["<pad>"]
    log.info(
        f"Matched {np.sum(filter_vocab)} / {len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}",
    )
    adata = adata[:, filter_vocab]

    # Optionally remove perturbations with missing target genes
    # Only do this when this is expected from the dataset
    # Replogle RPE1 for example, has around 15% perturbations that are missing in the adata
    if cfg.get("remove_missing_target_perturbations", False):
        perturbations_to_remove = [
            pert
            for pert in perturbation_meta_df[cfg.perturbation_col]
            if not all(
                gene in set(adata.var["id_in_vocab"])
                for gene in perturbation_meta_df.loc[pert, "target_gene_vocab_id"]
            )
        ]
        log.info(
            f"Found {len(perturbations_to_remove)} perturbations with missing target genes. Removing them.",
        )
        perturbation_meta_df = perturbation_meta_df[
            ~perturbation_meta_df[cfg.perturbation_col].isin(perturbations_to_remove)
        ]
        filter_missing_perturb = ~adata.obs[cfg.perturbation_col].isin(
            perturbations_to_remove,
        )
        adata = adata[filter_missing_perturb, :].copy()

    # Create set of target genes, after filtering out missing perturbations
    target_gene_vocab_id_set = {
        gene
        for gene_list in perturbation_meta_df["target_gene_vocab_id"]
        for gene in gene_list
    }

    log.info("Identifying highly variable genes...")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=cfg.n_top_genes,
        subset=False,
        flavor="seurat_v3",
    )

    filter_hvg = (
        adata.var["id_in_vocab"].isin(target_gene_vocab_id_set)
        | adata.var["highly_variable"]
    )
    log.info(
        f"Subset to {np.sum(filter_hvg)} / {len(adata.var)} after HVG selection and adding back target genes",
    )
    adata = adata[:, filter_hvg]

    missing_gene_ids = [
        gene_id
        for gene_id in target_gene_vocab_id_set
        if gene_id not in set(adata.var["id_in_vocab"])
    ]
    assert (
        not missing_gene_ids
    ), f"{len(missing_gene_ids)} target genes are missing in adata: {missing_gene_ids}"

    missing_perturbations = [
        perturbation_name
        for perturbation_name in (
            set(adata.obs[cfg.perturbation_col]) - {cfg.control_value}
        )
        if perturbation_name not in set(perturbation_meta_df[cfg.perturbation_col])
    ]
    assert (
        not missing_perturbations
    ), f"The following perturbations are missing metadata: {missing_perturbations}"

    # Create Dataset using from_generator
    expected_samples = (
        len(adata[adata.obs[cfg.perturbation_col] != cfg.control_value])
        * cfg.num_ctrl_samples_to_pair
    )
    log.info(f"Creating HF dataset with {expected_samples} samples...")
    hf_dataset = Dataset.from_generator(
        lambda: record_generator(
            adata=adata,
            perturbation_metadata=perturbation_meta_df,
            cfg=cfg,
            include_depmap=include_depmap,
        ),
        cache_dir=cfg.get("cache_dir"),
    )
    hf_dataset.set_format(type="torch")
    log.info(f"Generated {dataset_name} dataset with {len(hf_dataset)} records.")

    dataset_save_path = cfg.dataset_save_path
    log.info(f"Saving {dataset_name} dataset to {dataset_save_path}...")
    hf_dataset.save_to_disk(
        dataset_save_path,
        max_shard_size=cfg.get("max_shard_size", "200MB"),
    )
    log.info(f"Saved {dataset_name} dataset to {dataset_save_path}")
    return hf_dataset


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    OmegaConf.clear_resolver("oc.env")
    log.info(f"Loading configuration from {yaml_path}...")
    with open(yaml_path) as f:
        cfg = OmegaConf.load(f)
    OmegaConf.resolve(cfg)
    main(cfg)
    log.info("Script execution completed.")
