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
import scanpy as sc
from scipy.sparse import csc_matrix, csr_matrix


from mosaicfm.data import CountDataset, DataCollator
from mosaicfm.model import ComposerSCGPTModel
from mosaicfm.tokenizer import GeneVocab


log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)
def compute_lisi_scores(
        emb ,
        labels ,
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

def _compute_mean_gene_embeddings(
    gene_ids: torch.Tensor, gene_embs: torch.Tensor, vocab: GeneVocab, pad_token_id: int = -1
) -> np.ndarray:
    """Aggregate gene embeddings by taking the mean per vocabulary id."""
    flat_ids = gene_ids.flatten()
    flat_embs = gene_embs.flatten(0, 1)

    valid = flat_ids != pad_token_id
    flat_ids = flat_ids[valid]
    flat_embs = flat_embs[valid]

    sums = torch.zeros(len(vocab), flat_embs.size(-1), device=flat_embs.device)
    counts = torch.zeros(len(vocab), device=flat_embs.device)

    sums.index_add_(0, flat_ids, flat_embs)
    counts.index_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=torch.float32))

    means = sums / counts.unsqueeze(1)
    means = means.to("cpu").numpy()


    counts = np.expand_dims(counts, axis=1)

    means = np.divide(
        sums,
        counts,
        out=np.ones_like(sums) * np.nan,
        where=counts != 0,
    )

    gene2idx = vocab.get_stoi()
    all_gene_ids = np.array(list(gene2idx.values()))
    means = means[all_gene_ids, :]
 
    return means


def main(cfg: DictConfig) -> None:
    log.info("Loading vocabulary and collator configuration…")
    vocab = GeneVocab.from_file(cfg.paths.vocab)
    coll_cfg = om.load(cfg.paths.collator_config)
    model_cfg = om.load(cfg.paths.model_config)

    cell_type_key = cfg.data.cell_type_key
    gene_id_key = cfg.data.gene_id_key
    return_genes = cfg.predict.get("return_genes", True)


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


    count_matrix = adata.X
    if isinstance(count_matrix, np.ndarray):
        count_matrix = csr_matrix(count_matrix)
    elif isinstance(count_matrix, csc_matrix):
        count_matrix = count_matrix.tocsr()
    elif hasattr(count_matrix, "to_memory"):
        count_matrix = count_matrix.to_memory().tocsr()


    dataset = CountDataset(
        count_matrix,
        gene_ids,
        cls_token_id=vocab["<cls>"],
        pad_value=coll_cfg.pad_value,
    )
    print(f"Dataset is loaded with {len(dataset)} cells and {len(vocab)} genes.")

    collator = DataCollator(
        vocab=vocab,
        drug_to_id_path=coll_cfg.get("drug_to_id_path", None),
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
        max_length=cfg.predict.seq_len_dataset,
        sampling=coll_cfg.sampling,
        num_bins=coll_cfg.get("num_bins", 51),
        right_binning=coll_cfg.get("right_binning", False),
        reserve_keys=coll_cfg.get("reserve_keys"),
        keep_first_n_tokens=coll_cfg.get("keep_first_n_tokens", 1),
        use_chem_token=coll_cfg.get("use_chem_token", False),
    )


    loader = DataLoader(
        dataset,
        batch_size=cfg.predict.batch_size,
        collate_fn=collator,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.predict.num_workers,
        pin_memory=True,
        prefetch_factor=48,
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
    cell_array = cell_array / np.linalg.norm(
    cell_array,
    axis=1,
    keepdims=True,
)
    
    log.info("Saving outputs…")
    np.save(cfg.paths.cell_output, cell_array)
    
    print(f"Cell embeddings shape: {cell_array.shape}")

    if return_genes:
        all_gene_embs = torch.cat(gene_embs, dim=0)
        all_gene_ids = torch.cat(gene_ids_list, dim=0)

        gene_array = _compute_mean_gene_embeddings(
            all_gene_ids, all_gene_embs, vocab, coll_cfg["pad_token_id"]
        )
        print(f"Gene embeddings shape: {gene_array.shape}")

        np.save(cfg.paths.gene_output, gene_array)




    log.info("Finished writing embeddings")


    lisi_score = compute_lisi_scores(
        cell_array,
        adata.obs[cell_type_key].values.to_numpy(dtype="str"),
        20,
    )
    print("LiSI score:", lisi_score)
if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: predict_embeddings.py <config.yaml>")
    cfg = om.load(sys.argv[1])
    om.resolve(cfg)
    main(cfg)
