# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
"""Generate cell and gene embeddings using ``composer.Trainer.predict``.

This script loads a trained :class:`~mosaicfm.model.ComposerSCGPTModel` and
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
from typing import Dict, List

import numpy as np
import torch
from anndata import read_h5ad
from composer import Trainer
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch.utils.data import DataLoader

from mosaicfm.data import CountDataset, DataCollator
from mosaicfm.model import ComposerSCGPTModel
from mosaicfm.tokenizer import GeneVocab


log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)


def _compute_mean_gene_embeddings(
    gene_ids: torch.Tensor, gene_embs: torch.Tensor, vocab_size: int
) -> np.ndarray:
    """Aggregate gene embeddings by taking the mean per vocabulary id."""
    flat_ids = gene_ids.flatten()
    flat_embs = gene_embs.flatten(0, 1)

    valid = flat_ids != -1
    flat_ids = flat_ids[valid]
    flat_embs = flat_embs[valid]

    sums = torch.zeros(vocab_size, flat_embs.size(-1), device=flat_embs.device)
    counts = torch.zeros(vocab_size, device=flat_embs.device)

    sums.index_add_(0, flat_ids, flat_embs)
    counts.index_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=torch.float32))

    means = sums / counts.unsqueeze(1)
    means = means.to("cpu").numpy()
    return means


def main(cfg: DictConfig) -> None:
    log.info("Loading vocabulary and collator configuration…")
    vocab = GeneVocab.from_file(cfg.paths.vocab)
    coll_cfg = om.load(cfg.paths.collator_config)

    collator = DataCollator(
        vocab=vocab,
        do_padding=coll_cfg.get("do_padding", True),
        unexp_padding=False,
        pad_token_id=coll_cfg.pad_token_id,
        pad_value=coll_cfg.pad_value,
        do_mlm=False,
        do_binning=coll_cfg.get("do_binning", True),
        log_transform=coll_cfg.get("log_transform", False),
        target_sum=coll_cfg.get("target_sum"),
        mlm_probability=coll_cfg.mlm_probability,
        mask_value=coll_cfg.mask_value,
        max_length=cfg.predict.max_length,
        sampling=coll_cfg.sampling,
        data_style="pcpt",
        num_bins=coll_cfg.get("num_bins", 51),
        right_binning=coll_cfg.get("right_binning", False),
        reserve_keys=coll_cfg.get("reserve_keys"),
        keep_first_n_tokens=coll_cfg.get("keep_first_n_tokens", 1),
        use_chem_token=coll_cfg.get("use_chem_token", False),
    )

    log.info("Loading AnnData file…")
    adata = read_h5ad(cfg.paths.adata_input)
    if "id_in_vocab" in adata.var:
        gene_ids = adata.var["id_in_vocab"].to_numpy()
    else:
        gene_ids = np.array([vocab[g] if g in vocab else -1 for g in adata.var_names])
        adata.var["id_in_vocab"] = gene_ids

    keep = gene_ids >= 0
    adata = adata[:, keep]
    gene_ids = gene_ids[keep]
    count_matrix = adata.X

    dataset = CountDataset(
        count_matrix,
        gene_ids,
        cls_token_id=vocab["<cls>"],
        pad_value=coll_cfg.pad_value,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.predict.batch_size,
        collate_fn=collator,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.predict.num_workers,
        pin_memory=True,
    )

    log.info("Initialising model and loading checkpoint…")
    model_cfg = om.load(cfg.paths.model_config)
    model = ComposerSCGPTModel(model_cfg, coll_cfg)
    state = torch.load(cfg.paths.checkpoint, map_location="cpu")["state"]["model"]
    model.load_state_dict(state, strict=True)

    trainer = Trainer(
        model=model,
        precision=cfg.predict.precision,
        device="gpu" if torch.cuda.is_available() else "cpu",
    )

    log.info("Running prediction…")
    predictions = trainer.predict(loader, return_outputs=True)

    log.info("Aggregating embeddings…")
    cell_embs: List[torch.Tensor] = []
    gene_embs: List[torch.Tensor] = []
    gene_ids_list: List[torch.Tensor] = []
    for out in predictions:
        cell_embs.append(out["cell_emb"].cpu())
        gene_embs.append(out["gene_emb"].cpu())
        gene_ids_list.append(out["gene_ids"].cpu())

    cell_array = torch.cat(cell_embs, dim=0).numpy()
    all_gene_embs = torch.cat(gene_embs, dim=0)
    all_gene_ids = torch.cat(gene_ids_list, dim=0)

    gene_array = _compute_mean_gene_embeddings(
        all_gene_ids, all_gene_embs, vocab_size=len(vocab)
    )

    log.info("Saving outputs…")
    adata.obsm["X_scGPT"] = cell_array
    adata.write(cfg.paths.cell_output)
    np.save(cfg.paths.gene_output, gene_array)

    log.info("Finished writing embeddings")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: predict_embeddings.py <config.yaml>")
    cfg = om.load(sys.argv[1])
    om.resolve(cfg)
    main(cfg)
