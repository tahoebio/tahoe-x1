# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
import json
import logging
import os
import sys
from typing import Sequence

import numpy as np
import scanpy as sc
import torch
from omegaconf import OmegaConf as om

from mosaicfm.tasks import get_batch_embeddings
from mosaicfm.utils.util import load_model

log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
)
logging.getLogger(__name__).setLevel("INFO")


def _create_context_free_embeddings(cfg, model, all_gene_ids, device):
    """Return transformer based (TE) and gene encoder (GE) embeddings."""

    chunk_size = cfg.get("chunk_size", 30000)
    num_genes = all_gene_ids.shape[1]
    te = np.ones((num_genes, model.model_config["d_model"])) * np.nan
    ge = np.zeros((num_genes, model.model_config["d_model"]))

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        for i in range(0, num_genes, chunk_size):
            chunk_gene_ids = all_gene_ids[:, i : i + chunk_size]
            chunk_gene_ids_tensor = torch.tensor(chunk_gene_ids, dtype=torch.long).to(
                device,
            )
            token_embs = model.model.gene_encoder(chunk_gene_ids_tensor)
            ge[i : i + chunk_size] = (
                token_embs.to("cpu").to(torch.float32).numpy()[0, :, :]
            )
            flag_embs = model.model.flag_encoder(
                torch.tensor(1, device=token_embs.device),
            ).expand(
                chunk_gene_ids_tensor.shape[0],
                chunk_gene_ids_tensor.shape[1],
                -1,
            )
            total_embs = token_embs + flag_embs
            chunk_embeddings = model.model.transformer_encoder(total_embs)
            te[i : i + chunk_size] = (
                chunk_embeddings.to("cpu").to(torch.float32).numpy()
            )

    torch.cuda.empty_cache()

    return te, ge  # , list(gene2idx.keys()), list(gene2idx.values())


def generate_embeddings(config, modes: Sequence[str]):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir_path = os.listdir(config["models_dir_path"])

    for model_name in models_dir_path:

        model_dir = os.path.join(config["models_dir_path"], model_name)

        composer_model, vocab, model_cfg, coll_cfg = load_model(model_dir, device)

        output_path = config["embeddings_path"]
        os.makedirs(output_path, exist_ok=True)

        gene2idx = vocab.get_stoi()
        all_gene_ids = np.array([list(gene2idx.values())])

        with open(config["ensembl_to_gene_path"], "r") as f:
            ensemble_to_name = json.load(f)

        gene_ensembles = list(gene2idx.keys())
        gene_names = [ensemble_to_name.get(ens, ens) for ens in gene_ensembles]
        gene_ids = list(gene2idx.values())

        if "TE" in modes or "GE" in modes:

            te, ge = _create_context_free_embeddings(
                config,
                composer_model,
                all_gene_ids,
                device,
            )
            if "TE" in modes:
                np.savez_compressed(
                    os.path.join(output_path, f"{model_name}_TE.npz"),
                    gene_embeddings=te,
                    gene_names=gene_names,
                    gene_ids=gene_ids,
                )
                print("TE embeddings shape:", te.shape)
                log.info(f"Saved TE embeddings at {output_path}")

            if "GE" in modes:
                np.savez_compressed(
                    os.path.join(output_path, f"{model_name}_GE.npz"),
                    gene_embeddings=ge,
                    gene_names=gene_names,
                    gene_ids=gene_ids,
                )
                print("GE embeddings shape:", ge.shape)
                log.info(f"Saved GE embeddings at {output_path}")

        if "EA" in modes:
            dataset_path = config["dataset_path"]
            gene_col = config.get("gene_col", "feature_name")
            n_hvg = config.get("n_hvg")
            adata = sc.read_h5ad(dataset_path)
            if n_hvg is not None:
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=n_hvg,
                    flavor="seurat_v3",
                )
                adata = adata[:, adata.var["highly_variable"]]
            sc.pp.filter_cells(adata, min_genes=3)
            adata.var["id_in_vocab"] = [
                vocab[g] if g in vocab else -1 for g in adata.var[gene_col]
            ]
            adata = adata[:, adata.var["id_in_vocab"] >= 0]
            gene_ids = np.array(adata.var["id_in_vocab"], dtype=int)
            _, ea = get_batch_embeddings(
                adata=adata,
                model=composer_model.model,
                vocab=vocab,
                gene_ids=gene_ids,
                model_cfg=model_cfg,
                collator_cfg=coll_cfg,
                batch_size=config.get("batch_size", 32),
                max_length=config.get("max_length", 8192),
                return_gene_embeddings=True,
            )  # return ea in the order of vocabulary
            nan_genes = np.where(np.any(np.isnan(ea), axis=-1))[0]
            if "TE" not in locals():
                te, _ = _create_context_free_embeddings(
                    config,
                    composer_model,
                    all_gene_ids,
                    device,
                )

            ea[nan_genes] = te[nan_genes]
            print("EA embeddings shape:", ea.shape)
            np.savez_compressed(
                os.path.join(output_path, f"{model_name}_EA.npz"),
                gene_embeddings=ea,
                gene_names=gene_names,
                gene_ids=gene_ids,
                genes_not_expressed=nan_genes,
            )
            log.info("Saved EA embeddings")


def main():
    cfg = om.load(sys.argv[1])

    num_mand_args = 2
    cli_args = []
    for arg in sys.argv[num_mand_args:]:
        if arg.startswith("--"):
            cli_args.append(arg[2:])
        else:
            cli_args.append(arg)

    cli_cfg = om.from_cli(cli_args)
    cfg = om.merge(cfg, cli_cfg)

    om.resolve(cfg)

    modes = cfg.get("mods", ["GE", "TE"])
    print(f"config file {cfg}")
    generate_embeddings(cfg, modes)


if __name__ == "__main__":
    main()
