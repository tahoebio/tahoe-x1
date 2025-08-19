# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
import logging
from pathlib import Path
from typing import List
from urllib.parse import urlparse
import torch
import boto3
from git import Optional
import numpy as np
from scanpy import AnnData
from scipy.stats import pearsonr
from scipy.sparse import csc_matrix, csr_matrix
from mosaicfm.tokenizer import GeneVocab


from omegaconf import DictConfig

def finalize_embeddings(
    cell_embs: List[torch.Tensor],
    gene_embs: List[torch.Tensor],
    gene_ids_list: List[torch.Tensor],
    vocab: GeneVocab,
    pad_token_id: int = -1,
    return_gene_embeddings: bool = True,
):
    

    """Concatenate and finalize cell and gene embeddings."""
    cell_array = torch.cat(cell_embs, dim=0).to(torch.float32)
    cell_array = cell_array.to("cpu").numpy()
    cell_array = cell_array / np.linalg.norm(
                cell_array,
                axis=1,
                keepdims=True,
                )


    gene_array = None
    if return_gene_embeddings:
        gene_embs = torch.cat(gene_embs, dim=0)
        gene_ids = torch.cat(gene_ids_list, dim=0)

        flat_ids = gene_ids.flatten()
        flat_embs = gene_embs.flatten(0, 1)

        valid = flat_ids != pad_token_id
        flat_ids = flat_ids[valid]
        flat_embs = flat_embs[valid].to(torch.float32)

        sums = torch.zeros(len(vocab), flat_embs.size(-1), dtype=torch.float32, device=flat_embs.device)
        counts = torch.zeros(len(vocab),  dtype=torch.float32, device=flat_embs.device)

        sums.index_add_(0, flat_ids, flat_embs)
        counts.index_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=torch.float32))

        means = sums / counts.unsqueeze(1)

        means = means.to("cpu").to(torch.float32).numpy()
        sums = sums.to("cpu").to(torch.float32).numpy()
        counts = np.expand_dims(counts.to("cpu").to(torch.float32).numpy(), axis=1)

        means = np.divide(
            sums,
            counts,
            out=np.ones_like(sums) * np.nan,
            where=counts != 0,
        )

        gene2idx = vocab.get_stoi()
        all_gene_ids = np.array(list(gene2idx.values()))
        gene_array = means[all_gene_ids, :]


    return cell_array, gene_array
    
def loader_from_adata( adata: AnnData,
                      collator_cfg: DictConfig,
                      vocab: GeneVocab,
                      batch_size: int = 50,
                      max_length: Optional[int] = None,
                      gene_ids: Optional[np.ndarray] = None,
                      num_workers: int = 8,
       ):
    count_matrix = adata.X
    if isinstance(count_matrix, np.ndarray):
        count_matrix = csr_matrix(count_matrix)
    elif isinstance(count_matrix, csc_matrix):
        count_matrix = count_matrix.tocsr()
    elif hasattr(count_matrix, "to_memory"):
        count_matrix = count_matrix.to_memory().tocsr()

    if gene_ids is None:
        gene_ids = np.array(adata.var["id_in_vocab"])
        assert np.all(gene_ids >= 0)

    if max_length is None:
        max_length = len(gene_ids)
    
    from mosaicfm.data import CountDataset, DataCollator

    dataset = CountDataset(
        count_matrix,
        gene_ids,
        cls_token_id=vocab["<cls>"],
        pad_value=collator_cfg["pad_value"],
    )
    collate_fn = DataCollator(
        vocab=vocab,
        drug_to_id_path=collator_cfg.get("drug_to_id_path", None),
        do_padding=collator_cfg.get("do_padding", True),
        unexp_padding=False,  # Disable padding with random unexpressed genes for inference
        pad_token_id=collator_cfg.pad_token_id,
        pad_value=collator_cfg.pad_value,
        do_mlm=False,  # Disable masking for inference
        do_binning=collator_cfg.get("do_binning", True),
        log_transform=collator_cfg.get("log_transform", False),
        target_sum=collator_cfg.get("target_sum"),
        mlm_probability=collator_cfg.mlm_probability,  # Not used
        mask_value=collator_cfg.mask_value,
        max_length=max_length,
        sampling=collator_cfg.sampling,  # Turned on since max-length can be less than the number of genes
        num_bins=collator_cfg.get("num_bins", 51),
        right_binning=collator_cfg.get("right_binning", False),
        keep_first_n_tokens=collator_cfg.get("keep_first_n_tokens", 1),
        use_chem_token=collator_cfg.get("use_chem_token", False),
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=48,
    )

    return data_loader



def add_file_handler(logger: logging.Logger, log_file_path: Path):
    """Add a file handler to the logger."""
    h = logging.FileHandler(log_file_path)

    # format showing time, name, function, and message
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(funcName)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    h.setFormatter(formatter)
    h.setLevel(logger.level)
    logger.addHandler(h)


def download_file_from_s3_url(s3_url, local_file_path):
    """Downloads a file from an S3 URL to the specified local path.

    :param s3_url: S3 URL in the form s3://bucket-name/path/to/file
    :param local_file_path: Local path where the file will be saved.
    :return: The local path to the downloaded file, or None if download fails.
    """
    # Validate the S3 URL format
    assert s3_url.startswith("s3://"), "URL must start with 's3://'"

    # Parse the S3 URL
    parsed_url = urlparse(s3_url)
    assert parsed_url.scheme == "s3", "URL scheme must be 's3'"

    bucket_name = parsed_url.netloc
    s3_file_key = parsed_url.path.lstrip("/")

    # Ensure bucket name and file key are not empty
    assert bucket_name, "Bucket name cannot be empty"
    assert s3_file_key, "S3 file key cannot be empty"

    # Ensure the directory for local_file_path exists (if any)
    local_path = Path(local_file_path)
    if local_path.parent != Path("."):
        local_path.parent.mkdir(parents=True, exist_ok=True)

    # Create an S3 client
    s3 = boto3.client("s3")

    try:
        # Download the file
        s3.download_file(bucket_name, s3_file_key, str(local_path))
        print(f"File downloaded successfully to {local_path}")
        return str(local_path)
    except Exception as e:
        print(f"Error downloading the file from {s3_url}: {e}")
        return None


def calc_pearson_metrics(preds, targets, conditions, mean_ctrl):

    conditions_unique = np.unique(conditions)
    condition2idx = {c: np.where(conditions == c)[0] for c in conditions_unique}

    targets_mean_perturbed_by_condition = np.array(
        [targets[condition2idx[c]].mean(0) for c in conditions_unique],
    )  # (n_conditions, n_genes)

    preds_mean_perturbed_by_condition = np.array(
        [preds[condition2idx[c]].mean(0) for c in conditions_unique],
    )  # (n_conditions, n_genes)

    pearson = []
    for cond, t, p in zip(
        conditions_unique,
        targets_mean_perturbed_by_condition,
        preds_mean_perturbed_by_condition,
    ):
        print(cond, pearsonr(t, p))
        pearson.append(pearsonr(t, p)[0])

    pearson_delta = []
    for cond, t, p in zip(
        conditions_unique,
        targets_mean_perturbed_by_condition,
        preds_mean_perturbed_by_condition,
    ):
        tm, pm = t, p
        tm -= mean_ctrl
        pm -= mean_ctrl

        print(cond, pearsonr(tm, pm))
        pearson_delta.append(pearsonr(tm, pm)[0])

    return {
        "pearson": np.mean(pearson),
        "pearson_delta": np.mean(pearson_delta),
    }
