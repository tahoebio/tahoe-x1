# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
from typing import List, Optional, Tuple, Union

import logging
import composer
import numpy as np
import torch
from anndata import AnnData
from omegaconf import DictConfig
from scipy.sparse import csc_matrix, csr_matrix
from tqdm.auto import tqdm

from mosaicfm.data import CountDataset, DataCollator
from mosaicfm.model import SCGPTModel
from mosaicfm.tokenizer import GeneVocab
from mosaicfm.model import ComposerSCGPTModel

from mosaicfm.utils.util import finalize_embeddings, loader_from_adata


log = logging.getLogger(__name__)

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
):
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
    device = next(model.parameters()).device

    collator_cfg["do_mlm"] = False
    data_loader = loader_from_adata(
        adata=adata,
        collator_cfg=collator_cfg,
        vocab=vocab,
        batch_size=batch_size,
        max_length=max_length,
        gene_ids=gene_ids,
        num_workers=num_workers,
    )


    cell_embs: List[torch.Tensor] = []
    gene_embs: List[torch.Tensor] = []
    gene_ids_list: List[torch.Tensor] = []

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
        pbar = tqdm(total=len(data_loader), desc="Embedding cells")

        for data_dict in data_loader:
            input_gene_ids = data_dict["gene"].to(device)
            src_key_padding_mask = ~input_gene_ids.eq(collator_cfg["pad_token_id"])

            output = model(
                genes=input_gene_ids,
                values=data_dict["expr"].to(device),
                gen_masks=data_dict["gen_mask"].to(device),
                key_padding_mask=src_key_padding_mask,
                drug_ids=data_dict["drug_ids"].to(device),
                inference_mode=True,
            )

            cell_embs.append(output["cell_emb"])
            gene_embs.append(output["gene_emb"])
            gene_ids_list.append(output["gene_ids"])
            pbar.update(len(input_gene_ids))



    cell_array, gene_array = finalize_embeddings(
        cell_embs=cell_embs,
        gene_embs=gene_embs,
        gene_ids_list=gene_ids_list,
        vocab=vocab,
        pad_token_id=collator_cfg["pad_token_id"],
        return_gene_embeddings=return_gene_embeddings,
    )

    log.info(f"Extracted  cell embeddings of shape {cell_array.shape}.  ")

    if return_gene_embeddings:
        return cell_array, gene_array
    else:
        return cell_array

