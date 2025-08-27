# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
import json
import logging
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import scanpy as sc
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from mosaicfm.tokenizer import GeneVocab

# Logging setup
log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
)
logging.getLogger(__name__).setLevel("INFO")


def record_generator(
    adata: sc.AnnData,
    sensitivity_df: pd.DataFrame,
    drug_targets_df: pd.DataFrame,
    cfg: DictConfig,
) -> Dict:
    cell_line_list = list(set(adata.obs["cell_line"]))
    log.info(f"Using {len(cell_line_list)} cell-lines")

    for cell_line in tqdm(cell_line_list):
        log.info(f"Processing cell line: {cell_line}")
        ctrl_adata = adata[
            (adata.obs["cell_line"] == cell_line) & (adata.obs["drug"] == "DMSO_TF")
        ]
        log.info(f"Retrieved {len(ctrl_adata)} DMSO_TF cells for {cell_line}")
        drug_list_for_cell_line = set(
            adata[(adata.obs["cell_line"] == cell_line)].obs["drug"],
        ) - {"DMSO_TF"}
        log.info(f"Using {len(drug_list_for_cell_line)} drugs for {cell_line}")
        for drug in drug_list_for_cell_line:
            log.info(f"Processing drug: {drug} for cell line: {cell_line}")
            drug_adata = adata[
                (adata.obs["cell_line"] == cell_line) & (adata.obs["drug"] == drug)
            ]
            log.info(f"Retrieved {len(drug_adata)} {drug} cells for {cell_line}")
            drugname_drugconc = set(drug_adata.obs["drugname_drugconc"].values)
            assert (
                len(drugname_drugconc) == 1
            ), f"Only one drug concentration should be present, found {len(drugname_drugconc)}: {drugname_drugconc}"
            drugname_drugconc = next(iter(drugname_drugconc))
            sensitivity_data = sensitivity_df[
                (sensitivity_df["condition"] == drugname_drugconc)
                & (sensitivity_df["cell_line"] == cell_line)
            ]
            assert (
                len(sensitivity_data) == 1
            ), f"Sensitivity data must match exactly one row, found {len(sensitivity_data)}"

            growth_rate = sensitivity_data["growth_rate"].values[0]
            growth_rate_mdn = sensitivity_data["growth_rate_mdn"].values[0]
            growth_rate_bin = sensitivity_data["growth_rate_bin"].values[0]

            for cell in drug_adata:
                expressions_perturbed = cell.X.A[0]
                genes_pert = cell.var["id_in_vocab"].values
                perturbation_targets = drug_targets_df.loc[drug, "target_id"]

                random_ids = np.random.randint(
                    low=0,
                    high=len(ctrl_adata),
                    size=cfg.num_ctrl_samples_to_pair,
                )
                for ctrl_id in random_ids:
                    expressions_ctrl = ctrl_adata[ctrl_id].X.A[0]
                    genes_ctrl = ctrl_adata[ctrl_id].var["id_in_vocab"].values
                    assert all(genes_pert == genes_ctrl)

                    yield {
                        "growth_rate": np.float32(growth_rate),
                        "growth_rate_mdn": np.float32(growth_rate_mdn),
                        "growth_rate_bin": np.int64(growth_rate_bin),
                        "expressions_ctrl_raw": np.array(
                            expressions_ctrl,
                            dtype=np.float32,
                        ),
                        "expressions_perturbed_raw": np.array(
                            expressions_perturbed,
                            dtype=np.float32,
                        ),
                        "perturbation_target_genes": np.array(
                            perturbation_targets,
                            dtype=np.int64,
                        ),
                        "genes": np.array(genes_pert, dtype=np.int64),
                        "cell_line": cell_line,
                        "drug": drug,
                        "cell_key": cell.obs.index.values[0],
                        "cell_key_ctrl": ctrl_adata[ctrl_id].obs.index[0],
                    }


def main(cfg: DictConfig) -> None:
    log.info("Starting main script execution...")

    # Load in raw data
    log.info(f"Loading raw data from {cfg.raw_data_path}...")
    adata = sc.read_h5ad(cfg.raw_data_path)
    log.info(f"Raw data loaded. Data shape: {adata.shape}")

    dataset_save_path = (
        cfg.dataset_save_path
    )  # Ensure save path is specified in the config

    # Ensure the directory exists
    log.info(f"Ensuring directory exists for dataset save path: {dataset_save_path}")
    os.makedirs(os.path.dirname(dataset_save_path), exist_ok=True)

    # Load vocab and gene mapping
    log.info(f"Loading vocab from {cfg.vocab_path}...")
    vocab = GeneVocab.from_file(cfg.vocab_path)
    log.info(f"Vocab loaded with size {len(vocab)}")

    log.info(f"Loading gene-to-ensembl mapping from {cfg.gene_to_ensembl_path}...")
    with open(cfg.gene_to_ensembl_path) as f:
        gene_to_ensembl = json.load(f)
    log.info(f"Gene-to-ensembl mapping loaded with {len(gene_to_ensembl)} entries")

    ensembl_to_gene_name = {v: k for k, v in gene_to_ensembl.items()}

    # Load metadata
    log.info(
        f"Loading metadata from {cfg.metadata_path}, {cfg.sensitivity_path}, and {cfg.drug_targets_path}...",
    )
    metadata_df = pd.read_csv(cfg.metadata_path, sep="\t")
    sensitivity_df = pd.read_csv(cfg.sensitivity_path, sep="\t")
    drug_targets = pd.read_csv(cfg.drug_targets_path, sep="\t")
    log.info("Metadata loaded.")

    # Processing drug targets
    log.info("Processing drug targets...")
    drug_targets["target_id"] = drug_targets["Target_GeneSymbol"].apply(
        lambda x: [vocab[gene_name.strip()] for gene_name in x.split(",")],
    )
    drug_targets = drug_targets.rename(columns={"Drug": "drug"})
    drug_targets = drug_targets.set_index("drug")
    drug_targets["drug_name"] = drug_targets.index
    log.info("Drug targets processed.")

    # Data preparation
    log.info("Starting data preparation...")
    adata.obs["n_cell_cond"] = metadata_df["N_cell_cond"].values
    adata.obs["select_drug"] = metadata_df["Select_drug"].values
    adata.var["gene_name"] = adata.var.index

    target_genesymbol_set = {
        gene.strip()
        for d in drug_targets["Target_GeneSymbol"]
        for gene in d.split(", ")
    }

    log.info("Mapping genes to vocab IDs...")
    adata.var["id_in_vocab"] = [
        vocab[ensembl_to_gene_name.get(gene_id, "<pad>")]
        for gene_id in adata.var["gene_id"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    filter_vocab = adata.var["id_in_vocab"] != vocab["<pad>"]
    log.info(
        f"Matched {np.sum(filter_vocab)} / {len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}",
    )
    adata = adata[:, filter_vocab]

    log.info("Identifying highly variable genes...")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=cfg.n_top_genes,
        subset=False,
        flavor="seurat_v3",
    )
    log.info("Highly variable genes identified.")

    filter_hvg = (
        adata.var["gene_name"].isin(target_genesymbol_set)
        | adata.var["highly_variable"]
    )
    log.info(
        f"Subset to {np.sum(filter_hvg)} / {len(adata.var)} after HVG selection and adding back target genes",
    )
    adata = adata[:, filter_hvg]

    # Filtering data based on the conditions
    log.info("Filtering data based on drug conditions...")
    targeted_drug_list = drug_targets["drug_name"].values.tolist()
    adata = adata[adata.obs["drug"].isin([*targeted_drug_list, "DMSO_TF"])]
    adata = adata[adata.obs["n_cell_cond"] >= cfg.min_n_cell_cond]
    adata = adata[(adata.obs["drug"] == "DMSO_TF") | (adata.obs["select_drug"] == 1)]
    log.info(
        f"Dataset has {len(adata.obs)} cells and {len(adata.var)} genes after filtering",
    )

    # Create Dataset using from_generator
    log.info("Creating Hugging Face dataset using from_generator...")
    mosaic_dataset = Dataset.from_generator(
        lambda: record_generator(adata, sensitivity_df, drug_targets, cfg),
    )
    mosaic_dataset.set_format(type="torch")
    log.info(f"Generated mosaic dataset with {len(mosaic_dataset)} records.")

    log.info(f"Saving mosaic dataset to {dataset_save_path}...")
    mosaic_dataset.save_to_disk(
        dataset_save_path,
        max_shard_size=cfg.get("max_shard_size", "200MB"),
    )
    log.info(f"Saved mosaic dataset to {dataset_save_path}")


if __name__ == "__main__":
    yaml_path = sys.argv[1]

    # Disable resolving environment variables through omegaconf.
    OmegaConf.clear_resolver("oc.env")

    # Load yaml file.
    log.info(f"Loading configuration from {yaml_path}...")
    with open(yaml_path) as f:
        cfg = OmegaConf.load(f)
    OmegaConf.resolve(cfg)
    main(cfg)
    log.info("Script execution completed.")
