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
from mosaicfm.model import ComposerSCGPTModel
from mosaicfm.tokenizer import GeneVocab

from mosaicfm.utils.util import finalize_embeddings, loader_from_adata


log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)


def main(cfg: DictConfig) -> None:
    log.info("Loading vocabulary and collator configuration…")
    vocab = GeneVocab.from_file(cfg.paths.vocab)
    coll_cfg = om.load(cfg.paths.collator_config)
    model_cfg = om.load(cfg.paths.model_config)

    cell_type_key = cfg.data.cell_type_key
    gene_id_key = cfg.data.gene_id_key
    return_genes = cfg.predict.get("return_genes", True)
    batch_size = cfg.predict.get("batch_size", 64)
    max_length = cfg.predict.get("seq_len", 2048)    
    num_workers = cfg.predict.get("num_workers", 8)

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
    

    log.info("Initialising model and loading checkpoint…")

    if model_cfg["attn_config"]["attn_impl"] == "triton":
        model_cfg["attn_config"]["attn_impl"] = "flash"
        model_cfg["attn_config"]["use_attn_mask"] = False


    model = ComposerSCGPTModel(model_cfg, coll_cfg)
    state = torch.load(cfg.paths.checkpoint, map_location="cpu")["state"]["model"]
    model.load_state_dict(state, strict=True)

    print(f"Model is loaded with {model.model.n_layers} transformer layers.")

    trainer = Trainer(
        model=model,
        device="gpu" if torch.cuda.is_available() else "cpu",
    )


    predictions = trainer.predict(loader, return_outputs=True)    

    log.info("Aggregating embeddings…")
    cell_embs: List[torch.Tensor] = []
    gene_embs: List[torch.Tensor] = []
    gene_ids_list: List[torch.Tensor] = []
    for out in predictions:
        cell_embs.append(out["cell_emb"].cpu())
        gene_embs.append(out["gene_emb"].cpu())
        gene_ids_list.append(out["gene_ids"].cpu()) 

    cell_array, gene_array = finalize_embeddings(
        cell_embs=cell_embs,
        gene_embs=gene_embs,
        gene_ids_list=gene_ids_list,
        vocab=vocab,
        pad_token_id=coll_cfg["pad_token_id"],
        return_gene_embeddings=return_genes,
    )
    
    
    log.info("Saving outputs…")
    
    np.save(cfg.paths.cell_output, cell_array)

    if return_genes:
        np.savez(cfg.paths.gene_output, gene_array=gene_array, gene_ids=gene_ids)




    log.info("Finished writing embeddings")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: predict_embeddings.py <config.yaml> [--key=value ...]")
    
    # Load base config from YAML file
    cfg = om.load(sys.argv[1])
    
    # Merge with command line arguments
    cli_args = []
    for arg in sys.argv[2:]:
        # Convert --key=value to key=value format for OmegaConf
        if arg.startswith('--'):
            cli_args.append(arg[2:])
        else:
            cli_args.append(arg)
    
    cli_cfg = om.from_cli(cli_args)
    cfg = om.merge(cfg, cli_cfg)

    
    om.resolve(cfg)
    main(cfg)


