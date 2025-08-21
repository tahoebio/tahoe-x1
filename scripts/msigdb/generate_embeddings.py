import argparse
import logging
import os
from typing import Sequence

import numpy as np
import scanpy as sc
import torch
import yaml
from omegaconf import OmegaConf as om

from mosaicfm.model import ComposerSCGPTModel
from mosaicfm.tasks import get_batch_embeddings
from mosaicfm.tokenizer import GeneVocab

log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
)
logging.getLogger(__name__).setLevel("INFO")


def load_config(path: str) -> dict:
    with open(path, "r") as fin:
        return yaml.safe_load(fin)


def _create_context_free_embeddings(model, vocab, device):
    """Return transformer based (TE) and gene encoder (GE) embeddings."""
    gene2idx = vocab.get_stoi()
    all_gene_ids = np.array([list(gene2idx.values())])
    chunk_size = 30000
    num_genes = all_gene_ids.shape[1]
    te = np.ones((num_genes, model.config["d_model"])) * np.nan

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        for i in range(0, num_genes, chunk_size):
            chunk_gene_ids = all_gene_ids[:, i : i + chunk_size]
            chunk_gene_ids_tensor = torch.tensor(chunk_gene_ids, dtype=torch.long).to(device)
            token_embs = model.model.gene_encoder(chunk_gene_ids_tensor)
            flag_embs = model.model.flag_encoder(torch.tensor(1, device=token_embs.device)).expand(
                chunk_gene_ids_tensor.shape[0], chunk_gene_ids_tensor.shape[1], -1
            )
            total_embs = token_embs + flag_embs
            chunk_embeddings = model.model.transformer_encoder(total_embs)
            te[i : i + chunk_size] = chunk_embeddings.to("cpu").to(torch.float32).numpy()

        ge = model.model.gene_encoder(torch.tensor(all_gene_ids, dtype=torch.long).to(device))
        ge = ge.to("cpu").to(torch.float32).numpy()[0, :, :]

    return te, ge, list(gene2idx.keys()), list(gene2idx.values())


def _create_expression_aware_embeddings(adata, model, vocab, gene_ids, model_cfg, collator_cfg):
    cell_embeddings, gene_embeddings = get_batch_embeddings(
        adata=adata,
        model=model.model,
        vocab=vocab,
        gene_ids=gene_ids,
        model_cfg=model_cfg,
        collator_cfg=collator_cfg,
        batch_size=32,
        max_length=8192,
        return_gene_embeddings=True,
    )
    adata.obsm["X_scGPT"] = cell_embeddings
    return gene_embeddings


def generate_embeddings(config: dict, modes: Sequence[str]):
    model_paths = config.get("model_paths", {})
    model_name = config["model_name"]
    model_dir = model_paths.get(model_name, model_name)
    model_config_path = os.path.join(model_dir, "model_config.yml")
    vocab_path = os.path.join(model_dir, "vocab.json")
    collator_config_path = os.path.join(model_dir, "collator_config.yml")
    model_file = os.path.join(model_dir, "best-model.pt")

    model_config = om.load(model_config_path)
    if model_config["attn_config"]["attn_impl"] == "triton":
        model_config["attn_config"]["attn_impl"] = "flash"
        model_config["attn_config"]["use_attn_mask"] = False
    collator_config = om.load(collator_config_path)
    vocab = GeneVocab.from_file(vocab_path)

    model = ComposerSCGPTModel(model_config=model_config, collator_config=collator_config)
    model.load_state_dict(torch.load(model_file)["state"]["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    log.info(f"Model loaded from {model_file}")

    output_path = config["output_dir"]
    os.makedirs(output_path, exist_ok=True)

    if "TE" in modes or "GE" in modes:
        te, ge, gene_names, gene_ids = _create_context_free_embeddings(model, vocab, device)
        if "TE" in modes:
            np.savez_compressed(
                os.path.join(output_path, f"{model_name}_TE.npz"),
                gene_embeddings=te,
                gene_names=gene_names,
                gene_ids=gene_ids,
            )
            log.info("Saved TE embeddings")
        if "GE" in modes:
            np.savez_compressed(
                os.path.join(output_path, f"{model_name}_GE.npz"),
                gene_embeddings=ge,
                gene_names=gene_names,
                gene_ids=gene_ids,
            )
            log.info("Saved GE embeddings")

    if "EA" in modes:
        input_path = config["input_path"]
        gene_col = config.get("gene_col", "feature_name")
        n_hvg = config.get("n_hvg")
        adata = sc.read_h5ad(input_path)
        if n_hvg is not None:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3")
            adata = adata[:, adata.var["highly_variable"]]
        sc.pp.filter_cells(adata, min_genes=3)
        adata.var["id_in_vocab"] = [vocab[g] if g in vocab else -1 for g in adata.var[gene_col]]
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
        gene_ids = np.array(adata.var["id_in_vocab"], dtype=int)
        ea = _create_expression_aware_embeddings(
            adata, model, vocab, gene_ids, model_config, collator_config
        )
        nan_genes = np.where(np.any(np.isnan(ea), axis=-1))[0]
        if "TE" not in locals():
            te, _, gene_names, gene_ids_all = _create_context_free_embeddings(model, vocab, device)
        ea[nan_genes] = te[nan_genes]
        np.savez_compressed(
            os.path.join(output_path, f"{model_name}_EA.npz"),
            gene_embeddings=ea,
            gene_names=gene_names,
            gene_ids=gene_ids_all,
            genes_not_expressed=nan_genes,
        )
        log.info("Saved EA embeddings")


def main():
    parser = argparse.ArgumentParser(description="Generate MSigDB embeddings from config")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--modes",
        nargs="*",
        default=["GE", "TE", "EA"],
        help="Embedding types to generate",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    generate_embeddings(config, args.modes)


if __name__ == "__main__":
    main()
