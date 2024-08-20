# Copyright (C) Vevo Therapeutics 2024. All rights reserved.
import argparse
import json
import os

import numpy as np
import torch
from datasets import Dataset
from gears import PertData
from tqdm.auto import tqdm

from mosaicfm.tokenizer import GeneVocab


def prepare_data(
    data_path: str,
    dataset_name: str,
    vocab_path: str,
    gene_info_path: str,
    batch_size: int = 1,
):

    data_dir = os.path.join(data_path, dataset_name)
    pert_data = PertData(data_dir)
    pert_data.load(data_name=dataset_name)
    pert_data.prepare_split(split="simulation", seed=1)

    # remove control conditions
    if "ctrl" in pert_data.set2conditions["train"]:
        pert_data.set2conditions["train"].remove("ctrl")
    if "ctrl" in pert_data.set2conditions["val"]:
        pert_data.set2conditions["val"].remove("ctrl")

    """
    Data Format
    - one sample in the data_loader: DataBatch(x=[5060, 2], y=[1, 5060], de_idx=[1], pert=[1], batch=[5060], ptr=[2])
    - n_genes: 5060
    - x[:, 0] --> preperturbation, x[:, 1]--> one-hot perturbation encoding (only one gene is perturbed)
    - y[0] --> post-perturbation
    - x[:, 0], y[0] are saved as log1p values and not raw counts!
    - de_idx: indices of differentially expressed (DE) genes ( genes whose expression levels are significantly different between control vs perturbed conditions)
    - pert: 
    - length data_loader: 30096
    
    """
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=batch_size)
    """Adta Format.

    - AnnData object with n_obs X n_vars = 68603 X 5060
    - obs: 'condition', 'cell_type', 'dose_val', 'control', 'condition_name'
    - var: 'gene_name' --> 5060 genes
    - uns: 'non_dropout_gene_idx', 'non_zeros_gene_idx', 'rank_genes_groups_cov_all', 'top_non_dropout_de_20', 'top_non_zero_de_20'
    """

    # map dataset genes to scgpt gene_ids
    # Mapping based on gene ensemble ids

    vocab = GeneVocab.from_file(vocab_path)
    with open(gene_info_path) as f:
        gene_to_ensembl = json.load(f)
    ensembl_to_gene_name = {v: k for k, v in gene_to_ensembl.items()}
    gene_id_list = pert_data.adata.var.index
    mapped_gene_ids = np.array(
        [vocab[ensembl_to_gene_name.get(gene_id, "<pad>")] for gene_id in gene_id_list],
        dtype=int,
    )

    # compute mean control gene expression
    adata = pert_data.adata
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    mean_ctrl = np.array(ctrl_adata.X.mean(0)).flatten()  # (1, n_genes) = (1, 5060)
    np.savez(
        os.path.join(data_dir, "mean_ctrl_log1p.npz"),
        gene_names=list(adata.var.gene_name),
        mean_ctrl=mean_ctrl,
    )

    # save all splits in torch dataset format
    splits = ["train", "val", "test"]
    for split in splits:
        data_loader = pert_data.dataloader[f"{split}_loader"]
        save_to_torch_dataset(
            data_loader,
            data_dir,
            dataset_name,
            split,
            mapped_gene_ids,
        )


def yield_examples(dataloader, mapped_gene_ids: np.array):
    for batch_data in dataloader:
        x = batch_data.x  # (batch_size * n_genes, 2)
        batch_size = len(batch_data.y)
        assert batch_size == 1
        n_genes = int(x.shape[0] / batch_size)
        ori_gene_values = x[:, 0].view(batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, n_genes)
        target_gene_values = batch_data.y  # (batch_size, n_genes)
        de_flag = torch.zeros(size=(n_genes,), dtype=torch.bool)
        de_idx = batch_data.de_idx[0]
        de_flag[de_idx] = True  # boolean tensor:(n_genes)
        output_dict = {
            "genes": torch.tensor(mapped_gene_ids, dtype=torch.long),
            "expressions_ctrl": ori_gene_values[0],
            "expressions_perturbed": target_gene_values[0],
            "perturb_flag": pert_flags[0],
            "perturb_name": batch_data.pert[0],
            "de_flag": de_flag,
        }
        yield output_dict


def save_to_torch_dataset(
    data_loader,
    data_dir: str,
    dataset_name: str,
    split: str,
    mapped_gene_ids: dict,
):

    records = {
        "genes": [],
        "expressions_ctrl": [],
        "expressions_perturbed": [],
        "perturb_flag": [],
        "perturb_name": [],
        "de_flag": [],
    }
    for example in tqdm(yield_examples(data_loader, mapped_gene_ids)):
        for key in example:
            records[key].append(example[key])

    dataset = Dataset.from_dict(records)
    dataset.set_format(type="torch")
    dataset.save_to_disk(os.path.join(data_dir, f"{dataset_name}_{split}.dataset"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load dataset and save as .hf format")
    parser.add_argument("--data_path", required=True, help="Where to save data.")
    parser.add_argument("--dataset_name", required=True, help="adamson or norman")
    parser.add_argument(
        "--vocab_path",
        required=True,
        help="path to model's vocab.json.",
    )
    parser.add_argument(
        "--gene_info_path",
        required=True,
        help="path to cellxgene gene_info.json.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing data",
    )
    """DATA_PATH = "/vevo/datasets/perturbation_datasets/" DATASET_NAME =
    "adamson" VOCAB_PATH = "/vevo/scgpt/checkpoints/release/scgpt-70m-1024-fix-
    norm-apr24-data/vocab.json" GENE_INFO_PATH = "/vevo/datasets/cellxgene/cellx
    gene_primary_2024-04-29_MDS/gene_info_2024-04-29.json"."""

    args = parser.parse_args()

    print("Processing Data...")
    prepare_data(
        args.data_path,
        args.dataset_name,
        args.vocab_path,
        args.gene_info_path,
        args.batch_size,
    )
    print(f"Data Splits saved in {args.data_path}/{args.dataset_name}")

# def convert_hf_to_mds() TODO
