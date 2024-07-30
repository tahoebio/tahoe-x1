# Copyright (C) Vevo Therapeutics 2024. All rights reserved.
import argparse
import logging
import os

import numpy as np
import scanpy as sc
import torch
from omegaconf import OmegaConf as om

from mosaicfm.model import ComposerSCGPTModel
from mosaicfm.tasks import get_batch_embeddings
from mosaicfm.tokenizer import GeneVocab

log = logging.getLogger(__name__)
logging.basicConfig(
    # Example of format string
    # 2022-06-29 11:22:26,152: [822018][MainThread]: INFO: Message here
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
)
logging.getLogger(__name__).setLevel("INFO")  # Train script


def main(model_name, input_path, output_path, gene_col, n_hvg):
    model_paths = {
        "scgpt-70m-2048": "/vevo/scgpt/checkpoints/release/scgpt-70m-2048/",
        "scgpt-70m-1024": "/vevo/scgpt/checkpoints/release/scgpt-70m-1024/",
        "scgpt-70m-1024-cell-cond": "/vevo/scgpt/checkpoints/release/scgpt-70m-1024-cell-cond/",
        "scgpt-70m-1024-right-bin": "/vevo/scgpt/checkpoints/release/scgpt-70m-1024-right-bin",
        "scgpt-1_3b-2048": "/vevo/scgpt/checkpoints/release/scgpt-1_3b-2048/",
    }
    # At the moment only these 4 models have been prepared for inference
    # For a new model, the wandb config needs to be split into model and collator configs and the latest
    # checkpoint needs to be saved in the folder as best-model.pt
    model_config_path = os.path.join(model_paths[model_name], "model_config.yml")
    vocab_path = os.path.join(model_paths[model_name], "vocab.json")
    collator_config_path = os.path.join(model_paths[model_name], "collator_config.yml")
    model_file = os.path.join(model_paths[model_name], "best-model.pt")

    model_config = om.load(model_config_path)
    collator_config = om.load(collator_config_path)
    vocab = GeneVocab.from_file(vocab_path)

    model = ComposerSCGPTModel(
        model_config=model_config,
        collator_config=collator_config,
    )

    model.load_state_dict(torch.load(model_file)["state"]["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    log.info(f"Model loaded from {model_file}")
    # First create context free embeddings as the "default" per gene
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        gene2idx = vocab.get_stoi()
        all_gene_ids = np.array([list(gene2idx.values())])
        chunk_size = 30000  # Size of each chunk, >30000 OOMs

        # Initialize an empty array to hold the final embeddings.
        # Assuming 'num_genes' is the total number of genes.
        # This should be equivalent to len(all_gene_ids.flatten()) in your case.
        num_genes = all_gene_ids.shape[1]
        gene_embeddings_ctx_free = (
            np.ones((num_genes, model_config["d_model"])) * np.nan
        )
        # Update output_size accordingly

        for i in range(0, num_genes, chunk_size):
            chunk_gene_ids = all_gene_ids[:, i : i + chunk_size]
            chunk_gene_ids_tensor = torch.tensor(chunk_gene_ids, dtype=torch.long).to(
                device,
            )

            token_embs = model.model.gene_encoder(chunk_gene_ids_tensor)
            flag_embs = model.model.flag_encoder(
                torch.tensor(1, device=token_embs.device),
            ).expand(chunk_gene_ids_tensor.shape[0], chunk_gene_ids_tensor.shape[1], -1)

            total_embs = token_embs + flag_embs
            chunk_embeddings = model.model.transformer_encoder(total_embs)
            chunk_embeddings_cpu = chunk_embeddings.to("cpu").to(torch.float32).numpy()

            # Assigning the chunk embeddings to the correct place in the full array.
            gene_embeddings_ctx_free[i : i + chunk_size] = chunk_embeddings_cpu
    torch.cuda.empty_cache()
    log.info("Context free embeddings created.")
    adata = sc.read_h5ad(input_path)
    log.info(
        f"Loaded {adata.shape[0]} cells and {adata.shape[1]} genes from {input_path}",
    )
    if n_hvg is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3")
        adata = adata[:, adata.var["highly_variable"]]
        log.info(f"Performed HVG selection with n_top_genes = {n_hvg}")
    sc.pp.filter_cells(adata, min_genes=3)
    log.info("Filtered cells with min_genes = 3")
    adata.var["id_in_vocab"] = [vocab.get(gene, -1) for gene in adata.var[gene_col]]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    log.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}.",
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    vocab.set_default_index(vocab["<pad>"])
    genes = adata.var[gene_col].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)
    assert np.all(gene_ids == np.array(adata.var["id_in_vocab"]))

    cell_embeddings, gene_embeddings = get_batch_embeddings(
        adata=adata,
        model=model.model,
        vocab=vocab,
        gene_ids=gene_ids,
        model_cfg=model_config,
        collator_cfg=collator_config,
        batch_size=32,
        max_length=8192,
        return_gene_embeddings=True,
    )
    adata.obsm["X_scGPT"] = cell_embeddings
    nan_genes = np.where(np.any(np.isnan(gene_embeddings), axis=-1))[0]
    log.info(f"Found {len(nan_genes)} genes with NaN embeddings.")
    gene_embeddings[nan_genes] = gene_embeddings_ctx_free[nan_genes]
    gene_emb_save_path = os.path.join(output_path, f"gene_embeddings_{model_name}.npz")
    np.savez(
        gene_emb_save_path,
        gene_embeddings=gene_embeddings,
        genes_not_expressed=nan_genes,
        gene_embeddings_context_free=gene_embeddings_ctx_free,
        gene_names=list(gene2idx.keys()),
        gene_ids=list(gene2idx.values()),
    )
    log.info(f"Saved gene embeddings to {gene_emb_save_path}")
    adata_save_path = os.path.join(output_path, f"adata_{model_name}.h5ad")
    adata.write_h5ad(adata_save_path, compression="gzip")
    log.info(f"Saved adata with cell_embs to {adata_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate cell and gene embeddings for scGPT models",
    )

    # Required arguments
    parser.add_argument("--model_name", type=str, help="Name of the model to be used")
    parser.add_argument("--input_path", type=str, help="Path to the input data file")
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path where the output should be stored",
    )

    # Optional arguments
    parser.add_argument(
        "--gene_col",
        type=str,
        default="feature_name",
        help="Name of the column to be treated as gene identifier (default: feature_name)",
    )
    parser.add_argument(
        "--n_hvg",
        type=int,
        default=None,
        help="Number of highly variable genes to subset data ,default: None (no HVG selection)",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Pass the parsed arguments to the main function
    main(
        model_name=args.model_name,
        input_path=args.input_path,
        output_path=args.output_path,
        gene_col=args.gene_col,
        n_hvg=args.n_hvg,
    )
