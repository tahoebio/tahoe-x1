import numpy as np
import torch
from scgpt.data import DataCollator, CountDataset
from tqdm.auto import tqdm
from scgpt.model import SCGPTModel
from scgpt.tokenizer import GeneVocab
from typing import Optional
from omegaconf import DictConfig


def get_batch_cell_embeddings(
    adata,
    model: SCGPTModel,
    vocab: GeneVocab,
    model_cfg: DictConfig,
    collator_cfg: DictConfig,
    gene_ids: Optional[np.ndarray] = None,
    batch_size: int = 8,
    num_workers: int = 8,
    max_length: Optional[int] = None,
) -> np.ndarray:
    """
    Get the cell embeddings for a batch of cells.

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
        mlm_probability=collator_cfg.mlm_probability, # Not used
        mask_value=collator_cfg.mask_value,
        max_length=max_length,
        sampling=collator_cfg.sampling, # Turned on since max-length can be less than the number of genes
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
    )

    device = next(model.parameters()).device
    cell_embeddings = np.zeros((len(dataset), model_cfg["d_model"]), dtype=np.float32)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=model_cfg["precision"]):
        count = 0
        for data_dict in tqdm(data_loader, desc="Embedding cells"):
            input_gene_ids = data_dict["gene"].to(device)
            src_key_padding_mask = ~input_gene_ids.eq(
                collator_cfg["pad_token_id"]
            )  # Not the negation here compared to the public scGPT implementation!
            embeddings = model._encode(
                src=input_gene_ids,
                values=data_dict["expr"].to(device),
                src_key_padding_mask=src_key_padding_mask,
            )

            embeddings = embeddings[:, 0, :]  # get the <cls> position embedding

            # Casting to float 32 avoids issues with bfloat16 -> numpy conversion for some models
            # https://github.com/pytorch/pytorch/issues/110285
            embeddings = embeddings.to("cpu").to(torch.float32).numpy()
            cell_embeddings[count : count + len(embeddings)] = embeddings
            count += len(embeddings)
    cell_embeddings = cell_embeddings / np.linalg.norm(
        cell_embeddings, axis=1, keepdims=True
    )
    return cell_embeddings
