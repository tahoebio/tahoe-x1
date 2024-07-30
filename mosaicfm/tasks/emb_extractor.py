# Copyright (C) Vevo Therapeutics 2024. All rights reserved.
from typing import Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm

from mosaicfm.data import CountDataset, DataCollator
from mosaicfm.model import SCGPTModel
from mosaicfm.tokenizer import GeneVocab


def get_batch_embeddings(
    adata,
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
        np.ndarray: The cell embeddings.
    """

    count_matrix = adata.X
    count_matrix = (
        count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
    )

    # gene vocabulary ids
    if gene_ids is None:
        gene_ids = np.array(adata.var["id_in_vocab"])
        assert np.all(gene_ids >= 0)

    # Max context length is set to the number of genes unless provided
    if max_length is None:
        max_length = len(gene_ids)

    dataset = CountDataset(
        count_matrix,
        gene_ids,
        cls_token_id=vocab["<cls>"],
        pad_value=collator_cfg["pad_value"],
    )
    collate_fn = DataCollator(
        do_padding=collator_cfg.get("do_padding", True),
        pad_token_id=collator_cfg.pad_token_id,
        pad_value=collator_cfg.pad_value,
        do_mlm=False,  # Disable masking for inference
        do_binning=collator_cfg.get("do_binning", True),
        mlm_probability=collator_cfg.mlm_probability,  # Not used
        mask_value=collator_cfg.mask_value,
        max_length=max_length,
        sampling=collator_cfg.sampling,  # Turned on since max-length can be less than the number of genes
        data_style="pcpt",  # Disable splitting of genes into pcpt and gen for inference
        num_bins=collator_cfg.get("num_bins", 51),
        right_binning=collator_cfg.get("right_binning", False),
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
    if return_gene_embeddings:
        # Instantiate empty gene embeddings
        gene_embeddings = np.zeros((len(vocab), model_cfg["d_model"]), dtype=np.float32)
        gene_embedding_counts = np.zeros((len(vocab)), dtype=np.float32)

    dtype_from_string = {
        "fp32": torch.float32,
        "amp_bf16": torch.bfloat16,
        "amp_fp16": torch.float16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    with torch.no_grad(), torch.cuda.amp.autocast(
        enabled=True,
        dtype=dtype_from_string[model_cfg["precision"]],
    ):
        count = 0
        pbar = tqdm(total=len(dataset), desc="Embedding cells")
        for data_dict in data_loader:
            input_gene_ids = data_dict["gene"].to(device)
            src_key_padding_mask = ~input_gene_ids.eq(
                collator_cfg["pad_token_id"],
            )  # Note the negation here compared to the public scGPT implementation!
            embeddings = model._encode(
                src=input_gene_ids,
                values=data_dict["expr"].to(device),
                src_key_padding_mask=src_key_padding_mask,
            )
            if return_gene_embeddings:
                input_gene_ids = input_gene_ids.to("cpu").numpy()
                embeddings = embeddings.to(torch.float32).to("cpu").numpy()
                for emb, genes in zip(embeddings, input_gene_ids):
                    gene_embeddings[genes] += emb
                    gene_embedding_counts[genes] += 1

            cls_embeddings = embeddings[:, 0, :]  # get the <cls> position embedding

            # Casting to float 32 avoids issues with bfloat16 -> numpy conversion for some models
            # https://github.com/pytorch/pytorch/issues/110285
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
