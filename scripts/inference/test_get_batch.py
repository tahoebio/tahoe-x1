# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
"""Generate cell and gene embeddings using ``composer.Trainer.predict``.

This script loads a trained :class:`~mosaicfm.model.ComposerTXModel` and
produces embeddings for an input AnnData file. Configuration is provided via a
YAML file with two top-level sections:

``paths``
    ``model_config`` – path to the model configuration YAML.
    ``collator_config`` – path to the data collator configuration YAML.
    ``vocab`` – path to the gene vocabulary JSON file.
    ``checkpoint`` – path to the model checkpoint.
    ``adata_input`` – input AnnData (``.h5ad``) file.
    ``cell_output`` – where to save the AnnData with cell embeddings stored in
    ``obsm['X_scGPT']``.
    ``gene_output`` – path to ``.npy`` file for gene embeddings ordered by
    vocabulary index.

``predict``
    Runtime options such as ``batch_size``, ``max_length``, ``precision`` and
    ``num_workers``.

Example usage:

.. code-block:: bash

    python scripts/inference/predict_embeddings.py configs/predict.yaml

"""

import logging
import sys
from typing import Dict, List, Union

import numpy as np
import torch
from anndata import read_h5ad
from composer import Trainer
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch.utils.data import DataLoader
import scanpy as sc

from mosaicfm.tasks import get_batch_embeddings

from mosaicfm.data import CountDataset, DataCollator
from mosaicfm.model import ComposerTXModel
from mosaicfm.tokenizer import GeneVocab

from mosaicfm.utils.util import  loader_from_adata, load_model
from composer.utils import model_eval_mode
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)


@staticmethod
def compute_lisi_scores(
    emb: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    k: int,
) -> float:
    """Computes a LISI score. Accepts numpy arrays or torch tensors.

    Args:
        emb (Union[np.ndarray, torch.Tensor]): (n_samples, n_features) embedding matrix.
        labels (Union[np.ndarray, torch.Tensor]): (n_samples,) label vector, can be strings or ints.
        k (int): Number of neighbors.
    Returns:
        float: The LISI score.
    """
    # Convert to torch tensors
    emb = torch.from_numpy(emb).float()
    _, inverse_labels = np.unique(labels, return_inverse=True)
    labels = torch.from_numpy(inverse_labels).long()

    # Compute pairwise distances
    distances = torch.cdist(emb, emb, p=2)

    # Get k nearest neighbors for each point (excluding itself)
    _, knn_indices = torch.topk(distances, k + 1, largest=False)
    knn_indices = knn_indices[:, 1:]  # exclude self

    # Self vs neighbor labels
    self_labels = labels.unsqueeze(1).expand(-1, k)
    neighbor_labels = labels[knn_indices]

    # Compute label agreement
    same_label = (self_labels == neighbor_labels).float().mean()

    # Theoretical LISI normalization
    label_counts = torch.bincount(labels)
    theoretic_score = ((label_counts / label_counts.sum()) ** 2).sum()

    return (same_label / theoretic_score).item()


def main(cfg: DictConfig) -> None:

    device = 'cuda'
    cell_type_key = cfg.data.cell_type_key
    gene_id_key = cfg.data.gene_id_key
    return_gene_embeddings = cfg.predict.get("return_gene_embeddings", False)
    batch_size = cfg.predict.get("batch_size", 64)
    max_length = cfg.predict.get("seq_len", 2048)    
    num_workers = cfg.predict.get("num_workers", 8)

    model_dir = cfg.paths.model_dir


    log.info("Loading vocabulary and collator configuration and model checkpoints")
    model, vocab, model_cfg, coll_cfg = load_model(model_dir, device=device, return_gene_embeddings=return_gene_embeddings)
    print(f"Model is loaded with {model.model.n_layers} transformer layers.")


    log.info("Loading AnnData file…")
    adata = sc.read_h5ad(cfg.paths.adata_input)
    adata = adata[~adata.obs[cell_type_key].isna(), :]

    adata.var["id_in_vocab"] = [
        vocab[gene] if gene in vocab else -1 for gene in adata.var[gene_id_key]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    print(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}.",
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    genes = adata.var[gene_id_key].tolist()
    gene_ids = np.array([vocab[gene] for gene in genes], dtype=int)
    assert np.all(gene_ids >= 0), "Some genes are not in the vocabulary."

    with model_eval_mode(
        model.model,
    ), torch.no_grad(), FSDP.summon_full_params(model.model, writeback=False):
        if not return_gene_embeddings:
            cell_array = get_batch_embeddings(
                adata=adata,
                model=model.model.cuda(),
                vocab=vocab,
                gene_ids=gene_ids,
                model_cfg=model_cfg,
                collator_cfg=coll_cfg,
                batch_size=64,
                max_length=2048,
                return_gene_embeddings=False,
            )
        else:
            cell_array, gene_array = get_batch_embeddings(
                adata=adata,
                model=model.model.cuda(),
                vocab=vocab,
                gene_ids=gene_ids,
                model_cfg=model_cfg,
                collator_cfg=coll_cfg,
                batch_size=64,
                max_length=2048,
                return_gene_embeddings=True,
            )
    
    gene_array = gene_array[gene_ids]
    mask = ~np.isnan(gene_array).all(axis=1)
    valid_gene_array = gene_array[mask]

    print("not nan gene arrays", valid_gene_array.shape, valid_gene_array)
    print("gene array", gene_array, gene_array)
    print("gene ids", gene_ids.shape, gene_ids)


    log.info("Finished writing embeddings")

    lisi_score = compute_lisi_scores(
        cell_array,
        adata.obs[cell_type_key].values.to_numpy(dtype="str"),
        20,
    )
    print("LISI score:", lisi_score)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: predict_embeddings.py <config.yaml>")
    cfg = om.load(sys.argv[1])
    om.resolve(cfg)
    main(cfg)


