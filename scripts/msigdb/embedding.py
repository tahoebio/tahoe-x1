import argparse
import logging
import os
from typing import Tuple

import numpy as np
import torch
from omegaconf import OmegaConf as om

from mosaicfm.model import ComposerSCGPTModel
from mosaicfm.tokenizer import GeneVocab

log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)


def load_model(model_dir: str) -> Tuple[ComposerSCGPTModel, GeneVocab]:
    model_cfg = om.load(os.path.join(model_dir, "model_config.yml"))
    coll_cfg = om.load(os.path.join(model_dir, "collator_config.yml"))
    vocab = GeneVocab.from_file(os.path.join(model_dir, "vocab.json"))
    if model_cfg["attn_config"]["attn_impl"] == "triton":
        model_cfg["attn_config"]["attn_impl"] = "flash"
        model_cfg["attn_config"]["use_attn_mask"] = False
    model_cfg["return_genes"] = True
    model = ComposerSCGPTModel(model_cfg, coll_cfg)
    state = torch.load(os.path.join(model_dir, "best-model.pt"))["state"]["model"]
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, vocab


def generate_context_free_embeddings(model, vocab, chunk_size: int = 30000):
    gene2idx = vocab.get_stoi()
    all_gene_ids = np.array(list(gene2idx.values()))
    device = next(model.parameters()).device
    flag = model.model.flag_encoder(torch.tensor(1, device=device)).reshape(1, 1, -1)
    ge_chunks = []
    te_chunks = []
    with torch.no_grad():
        for i in range(0, len(all_gene_ids), chunk_size):
            ids = torch.tensor(all_gene_ids[i : i + chunk_size], dtype=torch.long, device=device).unsqueeze(0)
            token_embs = model.model.gene_encoder(ids)
            ge_chunks.append(token_embs[0].cpu().to(torch.float32).numpy())
            total = token_embs + flag
            te = model.model.transformer_encoder(total)
            te_chunks.append(te[0].cpu().to(torch.float32).numpy())
    ge = np.concatenate(ge_chunks, axis=0)
    te = np.concatenate(te_chunks, axis=0)
    return ge, te, list(gene2idx.keys()), list(gene2idx.values())


def save_parsed_embeddings(ge, te, genes, ids, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{prefix}_GE.npy"), ge)
    np.save(os.path.join(out_dir, f"{prefix}_TE.npy"), te)
    np.savez(
        os.path.join(out_dir, f"{prefix}_metadata.npz"),
        gene_names=genes,
        gene_ids=ids,
    )


def main(model_dirs, out_dir, chunk_size):
    for model_dir in model_dirs:
        model_name = os.path.basename(os.path.abspath(model_dir))
        log.info(f"Processing {model_name}")
        model, vocab = load_model(model_dir)
        ge, te, genes, ids = generate_context_free_embeddings(model, vocab, chunk_size)
        save_parsed_embeddings(ge, te, genes, ids, out_dir, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate context-free gene embeddings")
    parser.add_argument("model_dirs", nargs="+", help="One or more model directories")
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=30000)
    args = parser.parse_args()

    main(args.model_dirs, args.out_dir, args.chunk_size)

