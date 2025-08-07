# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.amp
import torch.utils.data
from anndata import AnnData
from omegaconf import DictConfig
from scipy.sparse import csc_matrix, csr_matrix
from tqdm.auto import tqdm

from mosaicfm.data import CountDataset, DataCollator
from mosaicfm.model import SCGPTModel
from mosaicfm.tokenizer import GeneVocab


def get_batch_embeddings(
    adata: AnnData,
    model: SCGPTModel,
    vocab: GeneVocab,
    model_cfg: DictConfig,
    collator_cfg: DictConfig,
    gene_ids: Optional[np.ndarray] = None,
    batch_size: int = 8,
    num_workers: int = 8,
    max_length: Optional[int] = None,
    return_gene_embeddings: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Get the cell embeddings for a batch of cells.

    Args:
        adata (AnnData): The AnnData object.
        model (SCGPTModel): The model.
        vocab (GeneVocab): The gene-to-ID vocabulary
        model_cfg (DictConfig, optional): The model configuration dictionary.
        collator_cfg (DictConfig, optional): The collator configuration dictionary.
        gene_ids (np.ndarray, optional): The gene vocabulary ids.
            Defaults to None, in which case the gene IDs are taken from adata.var["id_in_vocab"].
        batch_size (int): The batch size for inference. Defaults to 8.
        num_workers (int): The number of workers for the data loader. Defaults to 8.
        max_length (int, optional): The maximum context length. Defaults to number of genes in the adata.
        return_gene_embeddings (bool): Whether to return the mean gene embeddings as well. Defaults to False.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            - If `return_gene_embeddings` is False, returns a NumPy array of cell embeddings.
            - If `return_gene_embeddings` is True, returns a tuple of cell embeddings and
              gene embeddings as NumPy arrays.
    """
    count_matrix = adata.X
    if isinstance(count_matrix, np.ndarray):
        count_matrix = csr_matrix(count_matrix)
    elif isinstance(count_matrix, csc_matrix):
        count_matrix = count_matrix.tocsr()
    elif hasattr(count_matrix, "to_memory"):
        count_matrix = count_matrix.to_memory().tocsr()

    if gene_ids is None:
        gene_ids = np.array(adata.var["id_in_vocab"])
    assert gene_ids is not None and np.all(gene_ids >= 0)

    if max_length is None:
        max_length = len(gene_ids)

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
        data_style="pcpt",  # Disable splitting of genes into pcpt and gen for inference
        num_bins=collator_cfg.get("num_bins", 51),
        right_binning=collator_cfg.get("right_binning", False),
        keep_first_n_tokens=collator_cfg.get("keep_first_n_tokens", 1),
        use_chem_token=collator_cfg.get("use_chem_token", False),
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=48,
    )

    device = next(model.parameters()).device
    cell_embeddings = np.zeros((len(dataset), model_cfg["d_model"]), dtype=np.float32)

    # Initialize gene embedding variables (will be used if return_gene_embeddings is True)
    gene_embeddings = torch.zeros(
        len(vocab),
        model_cfg["d_model"],
        dtype=torch.float32,
        device=device,
    )
    gene_embedding_counts = torch.zeros(
        len(vocab),
        dtype=torch.float32,
        device=device,
    )

    dtype_from_string = {
        "fp32": torch.float32,
        "amp_bf16": torch.bfloat16,
        "amp_fp16": torch.float16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    with torch.no_grad(), torch.amp.autocast(
        enabled=True,
        dtype=dtype_from_string[model_cfg["precision"]],
        device_type=device.type,
    ):
        count = 0
        pbar = tqdm(total=len(dataset), desc="Embedding cells")
        for data_dict in data_loader:
            input_gene_ids = data_dict["gene"].to(device)
            src_key_padding_mask = ~input_gene_ids.eq(collator_cfg["pad_token_id"])

            embeddings = model._encode(
                src=input_gene_ids,
                values=data_dict["expr"].to(device),
                src_key_padding_mask=src_key_padding_mask,
            )

            if return_gene_embeddings:
                flat_gene_ids = input_gene_ids.view(-1)
                flat_embeddings = embeddings.view(-1, embeddings.shape[-1])

                valid = flat_gene_ids != collator_cfg["pad_token_id"]
                flat_gene_ids = flat_gene_ids[valid]
                flat_embeddings = flat_embeddings[valid]
                flat_embeddings = flat_embeddings.to(gene_embeddings.dtype)

                gene_embeddings.index_add_(0, flat_gene_ids, flat_embeddings)
                gene_embedding_counts.index_add_(
                    0,
                    flat_gene_ids,
                    torch.ones_like(flat_gene_ids, dtype=torch.float32),
                )

            cls_embeddings = embeddings[:, 0, :]
            if not isinstance(cls_embeddings, np.ndarray):
                cls_embeddings = cls_embeddings.to("cpu").to(torch.float32).numpy()
            cell_embeddings[count : count + len(embeddings)] = cls_embeddings
            count += len(embeddings)
            pbar.update(len(embeddings))
    cell_embeddings = cell_embeddings / np.linalg.norm(
        cell_embeddings,
        axis=1,
        keepdims=True,
    )
    if return_gene_embeddings:
        gene_embeddings = gene_embeddings.to("cpu").numpy()
        gene_embedding_counts = gene_embedding_counts.to("cpu").numpy()
        gene_embedding_counts = np.expand_dims(gene_embedding_counts, axis=1)

        gene_embeddings = np.divide(
            gene_embeddings,
            gene_embedding_counts,
            out=np.ones_like(gene_embeddings) * np.nan,
            where=gene_embedding_counts != 0,
        )

        gene2idx = vocab.get_stoi()
        all_gene_ids = np.array(list(gene2idx.values()))
        gene_embeddings = gene_embeddings[all_gene_ids, :]
        return cell_embeddings, gene_embeddings
    else:
        return cell_embeddings
