# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
"""Generate cell and gene embeddings using ``composer.Trainer.predict``.

This script loads a trained :class:`~mosaicfm.model.ComposerSCGPTModel` and
produces embeddings for an input AnnData file. Configuration is provided via a
YAML file.

Example usage:

.. code-block:: bash

    python scripts/inference/predict_embeddings.py configs/predict.yaml [--key=value ...]
"""

import logging
import sys
from typing import List

import numpy as np
import scanpy as sc
import torch
from composer import Trainer
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from mosaicfm.utils.util import load_model, loader_from_adata

log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)


def predict_embeddings(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cell_type_key = cfg.data.cell_type_key
    gene_id_key = cfg.data.gene_id_key
    return_genes = cfg.predict.get("return_genes", False)
    batch_size = cfg.predict.get("batch_size", 64)
    max_length = cfg.predict.get("seq_len", 2048)
    num_workers = cfg.predict.get("num_workers", 8)
    save = cfg.predict.get("save", True)

    log.info("Loading vocabulary and collator configuration and model checkpoints")
    model, vocab, _, coll_cfg = load_model(cfg.paths.model_dir, device=device, return_genes=return_genes)
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

    log.info("Creating data loader…")
    loader = loader_from_adata(
        adata=adata,
        collator_cfg=coll_cfg,
        vocab=vocab,
        batch_size=batch_size,
        max_length=max_length,
        gene_ids=gene_ids,
        num_workers=num_workers,
    )

    trainer = Trainer(
        model=model,
        device="gpu" if torch.cuda.is_available() else "cpu",
    )

    predictions = trainer.predict(loader, return_outputs=True)

    log.info("Aggregating embeddings…")
    cell_embs: List[torch.Tensor] = []

    if return_genes:
        gene_embs: List[torch.Tensor] = []
        gene_ids_list: List[torch.Tensor] = []

    for out in predictions:
        cell_embs.append(out["cell_emb"].cpu())

        if return_genes:
            gene_embs.append(out["gene_emb"].cpu())
            gene_ids_list.append(out["gene_ids"].cpu())

    """Concatenate and finalize cell and gene embeddings."""
    cell_array = torch.cat(cell_embs, dim=0).numpy()
    cell_array = cell_array / np.linalg.norm(
        cell_array,
        axis=1,
        keepdims=True,
    )

    if return_genes:
        gene_embs = torch.cat(gene_embs, dim=0)
        gene_ids = torch.cat(gene_ids_list, dim=0)

        flat_ids = gene_ids.flatten()
        flat_embs = gene_embs.flatten(0, 1)

        valid = flat_ids != coll_cfg["pad_token_id"]
        flat_ids = flat_ids[valid]
        flat_embs = flat_embs[valid]

        sums = torch.zeros(
            len(vocab),
            flat_embs.size(-1),
            dtype=torch.float32,
            device=flat_embs.device,
        )
        counts = torch.zeros(len(vocab), dtype=torch.float32, device=flat_embs.device)

        sums.index_add_(0, flat_ids, flat_embs)
        counts.index_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=torch.float32))

        sums = sums.numpy()
        counts = np.expand_dims(counts.numpy(), axis=1)

        means = np.divide(
            sums,
            counts,
            out=np.ones_like(sums) * np.nan,
            where=counts != 0,
        )

        gene2idx = vocab.get_stoi()
        all_gene_ids = np.array(list(gene2idx.values()))
        gene_array = means[all_gene_ids, :]

    if save:
        log.info("Saving outputs…")

        np.save(cfg.paths.cell_output, cell_array)

        if return_genes:
            np.savez(cfg.paths.gene_output, gene_array=gene_array, gene_ids=gene_ids)

        log.info("Finished writing embeddings")

    return adata, cell_array, gene_array if return_genes else None


if __name__ == "__main__":

    num_mand_args = 2
    if len(sys.argv) < num_mand_args:
        raise SystemExit("Usage: predict_embeddings.py <config.yaml> [--key=value ...]")

    # Load base config from YAML file
    cfg = om.load(sys.argv[1])

    # Merge with command line arguments

    cli_args = []
    for arg in sys.argv[num_mand_args:]:
        # Convert --key=value to key=value format for OmegaConf
        if arg.startswith("--"):
            cli_args.append(arg[2:])
        else:
            cli_args.append(arg)

    cli_cfg = om.from_cli(cli_args)
    cfg = om.merge(cfg, cli_cfg)

    om.resolve(cfg)
    predict_embeddings(cfg)
