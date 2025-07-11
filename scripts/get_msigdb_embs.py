# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf as om

from mosaicfm.model import ComposerSCGPTModel
from mosaicfm.tokenizer import GeneVocab

log = logging.getLogger(__name__)
logging.basicConfig(
    # Example of format string
    # 2022-06-29 11:22:26,152: [822018][MainThread]: INFO: Message here
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
)
logging.getLogger(__name__).setLevel("INFO")

model_names = os.listdir("/tahoe/data/ckpts/MFM-v2/release/")
for model_name in model_names:
    model_dir = os.path.join("/tahoe/data/ckpts/MFM-v2/release/", model_name)

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

    model = ComposerSCGPTModel(
        model_config=model_config,
        collator_config=collator_config,
    )
    model.load_state_dict(torch.load(model_file)["state"]["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    log.info(f"Model loaded from {model_file}")

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

        gene_embeddings_ctx_free_old = model.model.gene_encoder(
            torch.tensor(all_gene_ids, dtype=torch.long).to(device),
        )
        gene_embeddings_ctx_free_old = (
            gene_embeddings_ctx_free_old.to("cpu").to(torch.float32).numpy()
        )
        gene_embeddings_ctx_free_old = gene_embeddings_ctx_free_old[0, :, :]
    torch.cuda.empty_cache()
    log.info("Context free embeddings created.")
    gene_emb_save_path = os.path.join(
        f"/tahoe/data/msigdb/gene_embeddings_new/gene_embeddings_{model_name}.npz",
    )
    np.savez(
        gene_emb_save_path,
        gene_embeddings_context_free=gene_embeddings_ctx_free,
        gene_embeddings_context_free_old=gene_embeddings_ctx_free_old,
        gene_names=list(gene2idx.keys()),
        gene_ids=list(gene2idx.values()),
    )
    log.info(f"Saved gene embeddings to {gene_emb_save_path}")
