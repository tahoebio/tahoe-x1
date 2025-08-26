# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
"""Given a model, this script will create and save the cell line embeddings,
mean gene embeddings, and contextual gene embeddings needed to run the DepMap
benchmarks."""

import argparse
import logging
import os

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import utils
from omegaconf import OmegaConf as om
from tqdm import tqdm

from mosaicfm.data import CountDataset, DataCollator
from mosaicfm.model import ComposerSCGPTModel
from mosaicfm.tasks import get_batch_embeddings
from mosaicfm.tokenizer import GeneVocab

# set up logging
log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
)
logging.getLogger(__name__).setLevel("INFO")


# generate embeddings for a MosaicFM model
def run_mosaicfm(base_path, model_path, model_name):

    # create paths
    model_config_path = os.path.join(model_path, "model_config.yml")
    vocab_path = os.path.join(model_path, "vocab.json")
    collator_config_path = os.path.join(model_path, "collator_config.yml")
    model_file = os.path.join(model_path, "best-model.pt")

    # load configurations and vocabulary
    model_config = om.load(model_config_path)
    collator_config = om.load(collator_config_path)
    vocab = GeneVocab.from_file(vocab_path)
    gene2idx = vocab.get_stoi()

    # load model
    model = ComposerSCGPTModel(
        model_config=model_config,
        collator_config=collator_config,
    )
    model.load_state_dict(torch.load(model_file)["state"]["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    log.info(f"model loaded from {model_file}")

    # load and process AnnData of CCLE counts
    input_path = os.path.join(base_path, "counts.h5ad")
    adata = sc.read_h5ad(input_path)
    log.info(f"loaded CCLE AnnData from {input_path} for mean gene embeddings")
    adata.var["id_in_vocab"] = [
        vocab[gene] if gene in vocab else -1 for gene in adata.var["feature_id"]
    ]
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    log.info(
        f"matched {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}",
    )

    # make sure all remaining genes are in vocabulary
    genes = adata.var["feature_id"].tolist()
    gene_ids = np.array([vocab[gene] for gene in genes], dtype=int)
    assert np.all(gene_ids == np.array(adata.var["id_in_vocab"]))

    # get cell line and mean gene embeddings
    cell_embeddings, gene_embeddings = get_batch_embeddings(
        adata=adata,
        model=model.model,
        vocab=vocab,
        gene_ids=gene_ids,
        model_cfg=model_config,
        collator_cfg=collator_config,
        batch_size=16,
        max_length=17000,
        return_gene_embeddings=True,
    )

    # save cell line embeddings
    outpath = os.path.join(base_path, f"cell-embs/{model_name}.h5ad")
    cl_embs = ad.AnnData(
        X=cell_embeddings,
        obs=adata.obs,
        var=pd.DataFrame(
            {"dim": [str(i) for i in range(cell_embeddings.shape[1])]},
        ).set_index("dim"),
    )
    cl_embs.write_h5ad(outpath)
    log.info(f"saved cell line embeddings to {outpath}")

    # record genes with NaN embeddings
    nan_genes = np.where(np.any(np.isnan(gene_embeddings), axis=-1))[0]
    log.info(
        f"found {len(nan_genes)} genes with NaN embeddings",
    )

    # save NPZ of gene embeddings
    outpath = os.path.join(base_path, f"gene-embs/npz/{model_name}-gene-embs.npz")
    np.savez(
        outpath,
        gene_embeddings=gene_embeddings,
        genes_not_expressed=nan_genes,
        gene_names=list(gene2idx.keys()),
        gene_ids=list(gene2idx.values()),
    )
    log.info(f"saved gene embeddings to {outpath}")

    # load genes and scores for marginal essentiality task
    genes, scores = utils.get_marginal_genes_scores(base_path, log)

    # reload embeddings
    embs_npz = np.load(
        os.path.join(base_path, f"gene-embs/npz/{model_name}-gene-embs.npz"),
    )
    mean_embs_all = embs_npz["gene_embeddings"]
    gene_names = embs_npz["gene_names"]
    gene_info_df = pd.read_csv(os.path.join(base_path, "raw/scgpt-genes.csv"))
    scgpt_gene_mapping = dict(
        zip(gene_info_df["feature_id"], gene_info_df["feature_name"]),
    )
    valid_gene_ids = scgpt_gene_mapping.keys()
    gene_names = np.array(
        [
            scgpt_gene_mapping[gene_id] if gene_id in valid_gene_ids else gene_id
            for gene_id in gene_names
        ],
    )
    invalid_indices = [i for i, g in enumerate(genes) if g not in gene_names]
    genes = [g for i, g in enumerate(genes) if i not in invalid_indices]
    scores = [s for i, s in enumerate(scores) if i not in invalid_indices]
    log.info("loaded mean gene embeddings for processing")

    # get mean embeddings for each gene
    mean_embs = np.zeros((len(genes), mean_embs_all.shape[1]))
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    for i, g in enumerate(tqdm(genes)):
        mean_embs[i] = mean_embs_all[gene_to_idx[g]]

    # create AnnData
    mean_embs_ad = ad.AnnData(
        X=mean_embs,
        obs=pd.DataFrame({"gene": genes, "score": scores}),
        var=pd.DataFrame({"dim": np.arange(mean_embs.shape[1])}),
    )

    # write to disk
    outpath = os.path.join(
        base_path,
        f"gene-embs/{'_'.join(model_name.split('-'))}-mean-lt5gt70-bin.h5ad",
    )
    mean_embs_ad.write_h5ad(outpath)
    log.info(
        f"saved mean gene embedding AnnData for marginal essentiality task to {outpath}",
    )

    # reprocess AnnData of CCLE counts for contextual gene embeddings
    adata = sc.read_h5ad(os.path.join(base_path, "counts.h5ad"))
    gene_info_df = pd.read_csv(os.path.join(base_path, "raw/scgpt-genes.csv"))
    scgpt_gene_mapping = dict(
        zip(gene_info_df["feature_id"], gene_info_df["feature_name"]),
    )
    scgpt_vocab_ids = set(gene_info_df["feature_id"])
    adata.var["in_scgpt_vocab"] = adata.var.index.map(lambda x: x in scgpt_vocab_ids)
    adata = adata[:, adata.var["in_scgpt_vocab"]]
    adata.var["gene_name_scgpt"] = adata.var.index.map(
        lambda gene_id: scgpt_gene_mapping[gene_id],
    )
    adata.var = adata.var.drop(columns=["in_scgpt_vocab"])
    adata.layers["counts"] = adata.X.copy()
    log.info("processed CCLE AnnData for contextual gene embeddings")

    # subset to available genes
    adata.var["id_in_vocab"] = [
        vocab[gene] if gene in vocab else -1 for gene in adata.var["feature_id"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    log.info(
        f"matched {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}",
    )

    # get gene IDs
    genes = adata.var["feature_id"].tolist()
    gene_ids = np.array([vocab[gene] for gene in genes], dtype=int)

    # get count matrix
    count_matrix = adata.X
    count_matrix = (
        count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.toarray()
    )

    # verify gene IDs
    if gene_ids is None:
        gene_ids = np.array(adata.var["id_in_vocab"])
        assert np.all(gene_ids >= 0)

    # set up dataset
    dataset = CountDataset(
        count_matrix,
        gene_ids,
        cls_token_id=vocab["<cls>"],
        pad_value=collator_config["pad_value"],
    )

    # set up collator
    max_length = len(gene_ids)
    collator = DataCollator(
        vocab=vocab,
        drug_to_id_path=collator_config.get("drug_to_id_path", None),
        do_padding=collator_config.get("do_padding", True),
        pad_token_id=collator_config.pad_token_id,
        pad_value=collator_config.pad_value,
        do_mlm=False,
        do_binning=collator_config.get("do_binning", True),
        mlm_probability=collator_config.mlm_probability,
        mask_value=collator_config.mask_value,
        max_length=max_length,
        sampling=False,
        num_bins=collator_config.get("num_bins", 51),
        right_binning=collator_config.get("right_binning", False),
        keep_first_n_tokens=collator_config.get("keep_first_n_tokens", 1),
        use_chem_token=collator_config.get("use_chem_token", False),
    )

    # set up data loader
    batch_size = 4
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SequentialSampler(dataset),
        collate_fn=collator,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), batch_size),
        pin_memory=True,
    )

    # get lists for indexing
    cell_lines = adata.obs["ModelID"].tolist()
    genes = {v: k for k, v in vocab.get_stoi().items()}

    # make empty objects to fill
    labels = []
    embeddings = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):

        # keep track of cell line
        count = 0
        pbar = tqdm(total=len(dataset))

        # iterate through data loader
        for data_dict in data_loader:

            # get batch embeddings
            input_gene_ids = data_dict["gene"].to(device)
            src_key_padding_mask = ~input_gene_ids.eq(collator_config["pad_token_id"])
            batch_embeddings = model.model._encode(
                src=input_gene_ids,
                values=data_dict["expr"].to(device),
                src_key_padding_mask=src_key_padding_mask,
            )

            # bring back to CPU
            input_gene_ids = input_gene_ids.cpu().numpy()
            batch_embeddings = batch_embeddings.to("cpu").to(torch.float32).numpy()

            # iterate through cell lines
            for i in range(batch_embeddings.shape[0]):

                # get cell line
                cell_line = cell_lines[count]
                cell_line_inputs = input_gene_ids[i]
                cell_line_embs = batch_embeddings[i]

                # iterate over genes
                for j in range(cell_line_embs.shape[0]):

                    # check if this is a real gene
                    input_id = cell_line_inputs[j]
                    if genes[input_id] not in valid_gene_ids:
                        continue

                    # fill embedding and label
                    labels.append(
                        f"{cell_line} | {scgpt_gene_mapping[genes[input_id]]}",
                    )
                    embeddings.append(cell_line_embs[j])

                # increment cell line
                count += 1
                pbar.update(1)

    # convert to arrays and normalize embeddings
    log.info("stacking and normalizing embeddings")
    labels = np.array(labels)
    embeddings = np.vstack(embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # sort by label order
    log.info("sorting embeddings")
    sort_idx = np.argsort(labels)
    labels = labels[sort_idx]
    embeddings = embeddings[sort_idx]

    # process and save contextual gene embeddings
    utils.process_contextual_gene_embs(
        base_path,
        log,
        labels,
        embeddings,
        "_".join(model_name.split("-")),
    )


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path",
        type=str,
        required=True,
        help="Path to DepMap benchmark base directory.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name (filenames are based on this).",
    )
    parser.add_argument("--model-path", type=str, help="Path to model folder.")
    args = parser.parse_args()

    # run function
    run_mosaicfm(args.base_path, args.model_path, args.model_name)
