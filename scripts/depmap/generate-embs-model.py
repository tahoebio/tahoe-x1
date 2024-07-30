# Copyright (C) Vevo Therapeutics 2024. All rights reserved.
"""Given a model, this script will create and save the cell line embeddings,
mean gene embeddings, and contextual gene embeddings needed to run the DepMap
benchmarks."""

import argparse
import json
import logging
import os
import pickle

import anndata as ad
import geneformer.perturber_utils as pu
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import utils
from geneformer import EmbExtractor
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from omegaconf import OmegaConf as om
from tqdm import tqdm, trange

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


# generate embeddings for an scGPT model
def run_scgpt(base_path, model_path, model_name):

    # create paths
    model_config_path = os.path.join(model_path, "model_config.yml")
    vocab_path = os.path.join(model_path, "vocab.json")
    collator_config_path = os.path.join(model_path, "collator_config.yml")
    model_file = os.path.join(model_path, "best-model.pt")

    # load configurations and vocabulary
    model_config = om.load(model_config_path)
    collator_config = om.load(collator_config_path)
    vocab = GeneVocab.from_file(vocab_path)
    vocab.set_default_index(vocab["<pad>"])

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

    # extract context-free embeddings
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):

        # load gene IDs and set step size
        gene2idx = vocab.get_stoi()
        all_gene_ids = np.array([list(gene2idx.values())])
        chunk_size = 30000

        # initialize empty array to hole embeddings
        num_genes = all_gene_ids.shape[1]
        gene_embeddings_ctx_free = (
            np.ones((num_genes, model_config["d_model"])) * np.nan
        )

        # iterate over genes
        for i in range(0, num_genes, chunk_size):

            # extract chunk of gene IDs
            chunk_gene_ids = all_gene_ids[:, i : i + chunk_size]
            chunk_gene_ids_tensor = torch.tensor(chunk_gene_ids, dtype=torch.long).to(
                device,
            )

            # pass through model
            token_embs = model.model.gene_encoder(chunk_gene_ids_tensor)
            flag_embs = model.model.flag_encoder(
                torch.tensor(1, device=token_embs.device),
            ).expand(chunk_gene_ids_tensor.shape[0], chunk_gene_ids_tensor.shape[1], -1)
            total_embs = token_embs + flag_embs
            chunk_embeddings = model.model.transformer_encoder(total_embs)

            # bring to CPU and assign correctly
            chunk_embeddings_cpu = chunk_embeddings.to("cpu").to(torch.float32).numpy()
            gene_embeddings_ctx_free[i : i + chunk_size] = chunk_embeddings_cpu

    # cleanup
    torch.cuda.empty_cache()
    log.info("extracted context-free embeddings")

    # load and process AnnData of CCLE counts
    input_path = os.path.join(base_path, "counts.h5ad")
    adata = sc.read_h5ad(input_path)
    log.info(f"loaded CCLE AnnData from {input_path} for mean gene embeddings")
    adata.var["id_in_vocab"] = [
        vocab.get(gene, -1) for gene in adata.var["feature_name"]
    ]
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    log.info(
        f"matched {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}",
    )

    # make sure all remaining genes are in vocabulary
    genes = adata.var["feature_name"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)
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
        max_length=8192,
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

    # handle genes with NaN embeddings
    nan_genes = np.where(np.any(np.isnan(gene_embeddings), axis=-1))[0]
    log.info(
        f"found {len(nan_genes)} genes with NaN embeddings, replacing with context-free embeddings",
    )
    gene_embeddings[nan_genes] = gene_embeddings_ctx_free[nan_genes]

    # save NPZ of gene embeddings
    outpath = os.path.join(base_path, f"gene-embs/npz/{model_name}-gene-embs.npz")
    np.savez(
        outpath,
        gene_embeddings=gene_embeddings,
        genes_not_expressed=nan_genes,
        gene_embeddings_context_free=gene_embeddings_ctx_free,
        gene_names=list(gene2idx.keys()),
        gene_ids=list(gene2idx.values()),
    )
    log.info(f"saved gene embeddings to {outpath}")

    # load genes and scores for marginal essentiality task
    genes, scores = utils.get_marginal_genes_scores(base_path)

    # reload embeddings
    embs_npz = np.load(
        os.path.join(base_path, f"gene-embs/npz/{model_name}-gene-embs.npz"),
    )
    mean_embs_all = embs_npz["gene_embeddings"]
    gene_names = embs_npz["gene_names"]
    log.info("loaded mean gene embeddings for processing")

    # get mean embeddings for each gene
    mean_embs = np.zeros((len(genes), mean_embs_all.shape[1]))
    for i, g in enumerate(tqdm(genes)):
        mean_embs[i] = mean_embs_all[np.where(gene_names == g)[0][0]]

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
        vocab.get(gene, -1) for gene in adata.var["gene_name_scgpt"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    log.info(
        f"matched {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}",
    )

    # get gene IDs
    genes = adata.var["gene_name_scgpt"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)

    # get count matrix
    count_matrix = adata.X
    count_matrix = (
        count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
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
        do_padding=collator_config.get("do_padding", True),
        pad_token_id=collator_config.pad_token_id,
        pad_value=collator_config.pad_value,
        do_mlm=False,
        do_binning=collator_config.get("do_binning", True),
        mlm_probability=collator_config.mlm_probability,
        mask_value=collator_config.mask_value,
        max_length=max_length,
        sampling=False,
        data_style="pcpt",
        num_bins=collator_config.get("num_bins", 51),
        right_binning=collator_config.get("right_binning", False),
    )

    # set up data loader
    batch_size = 8
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
    genes = vocab.get_itos()

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
                    if input_id in (60694, 60695, 60696):
                        continue

                    # fill embedding and label
                    labels.append(f"{cell_line} | {genes[input_id]}")
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


# generate embeddings for a Geneformer model
def run_gf(base_path, model_path, model_name, data_path, layer_offset=0):

    # extract cell line embeddings
    log.info("extracting cell embeddings")
    embex = EmbExtractor(
        model_type="Pretrained",
        emb_mode="cell",
        max_ncells=None,
        emb_layer=layer_offset,
        emb_label=["CCLEName"],
    )
    embs = embex.extract_embs(
        model_path,
        data_path,
        os.path.join(base_path, "geneformer"),
        output_prefix=f"{model_name}-cell-embs",
    )

    # convert to AnnData, based on PCA embeddings
    pca_embs = sc.read_h5ad(os.path.join(base_path, "cell-embs/pca.h5ad"))
    assert (pca_embs.obs.index.to_numpy() == embs["CCLEName"].to_numpy()).all()
    gf_embs_x = embs.iloc[:, :256].to_numpy()
    gf_embs_adata = ad.AnnData(
        X=gf_embs_x,
        obs=pca_embs.obs,
        var=pd.DataFrame({"dim": np.arange(gf_embs_x.shape[1])}),
    )
    outpath = os.path.join(base_path, f"cell-embs/{model_name}.h5ad")
    gf_embs_adata.write_h5ad(outpath)
    log.info(f"created AnnData of cell line embeddings at {outpath}")

    # extract mean gene embeddings
    log.info("extracting mean gene embeddings")
    embex = EmbExtractor(
        model_type="Pretrained",
        emb_mode="gene",
        max_ncells=None,
        emb_layer=layer_offset,
    )
    embs = embex.extract_embs(
        model_path,
        data_path,
        os.path.join(base_path, "geneformer"),
        output_prefix=f"{model_name}-mean-gene-embs",
    )

    # load other AnnDatas to process Geneformer embeddings
    counts_adata = sc.read_h5ad(os.path.join(base_path, "counts.h5ad"))
    pca_adata = sc.read_h5ad(
        os.path.join(base_path, "gene-embs/loading15-lt5gt70-bin.h5ad"),
    )
    ensembl_to_name = counts_adata.var.to_dict()["feature_name"]
    name_to_ensembl = {ensembl_to_name[k]: k for k in ensembl_to_name}

    # get embedding for each lt5gt70 gene that we have
    gf_embs_df = embs
    genes_to_keep = []
    embs = np.zeros((len(pca_adata), gf_embs_df.shape[1] - 1))
    for i, g in enumerate(tqdm(pca_adata.obs["gene"])):
        try:
            embs[i] = (
                gf_embs_df[gf_embs_df["Unnamed: 0"] == name_to_ensembl[g]]
                .iloc[:, 1:]
                .to_numpy()[0]
            )
            genes_to_keep.append(g)
        except KeyError:  # noqa: PERF203
            continue

    # create AnnData
    gf_embs_ad = ad.AnnData(
        X=embs,
        obs=pca_adata.obs,
        var=pd.DataFrame({"dim": np.arange(embs.shape[1])}),
    )

    # filter to genes we have and save
    gf_embs_ad = gf_embs_ad[gf_embs_ad.obs["gene"].isin(genes_to_keep)]
    outpath = os.path.join(
        base_path,
        f"gene-embs/{'_'.join(model_name.split('-'))}-mean-lt5gt70-bin.h5ad",
    )
    gf_embs_ad.write_h5ad(outpath)
    log.info(
        f"created AnnData of mean gene embeddings for marginal essentiality task at {outpath}",
    )

    # load dataset to iterate for contextual gene embeddings
    filtered_input_data = pu.load_and_filter(None, 4, data_path)
    filtered_input_data = pu.downsample_and_sort(filtered_input_data, None)
    total_batch_length = len(filtered_input_data)

    # load Geneformer's token dictionary
    with open(TOKEN_DICTIONARY_FILE, "rb") as f:
        gene_token_dict = pickle.load(f)
    token_gene_dict = {v: k for k, v in gene_token_dict.items()}
    pad_token_id = gene_token_dict.get("<pad>")

    # load dictionary of gene IDs to names
    gene_id_to_name = sc.read_h5ad(
        os.path.join(base_path, "geneformer/adata.h5ad"),
    ).var[["ensembl_id", "feature_name"]]
    gene_id_to_name = gene_id_to_name.set_index("ensembl_id").to_dict()["feature_name"]

    # load model
    model = pu.load_model("Pretrained", 0, model_path, mode="eval")
    layer_to_quant = pu.quant_layers(model) + layer_offset
    model_input_size = pu.get_model_input_size(model)
    forward_batch_size = 4

    # get contextual gene embeddings
    log.info("extracting contextual gene embeddings")
    labels = []
    embeddings = []
    for i in trange(0, total_batch_length, forward_batch_size):

        # get batch
        max_range = min(i + forward_batch_size, total_batch_length)
        minibatch = filtered_input_data.select(list(range(i, max_range)))
        max_len = int(max(minibatch["length"]))
        minibatch.set_format(type="torch")

        # get input data and make attention mask
        input_data_minibatch = minibatch["input_ids"]
        input_data_minibatch = pu.pad_tensor_list(
            input_data_minibatch,
            max_len,
            pad_token_id,
            model_input_size,
        )
        attention_mask = pu.gen_attention_mask(minibatch)

        # pass through model
        with torch.no_grad():
            outputs = model(
                input_ids=input_data_minibatch.to("cuda"),
                attention_mask=attention_mask.to("cuda"),
            )

        # get embeddings from specified layer
        batch_embeddings = (
            outputs.hidden_states[layer_to_quant].to("cpu").to(torch.float32).numpy()
        )

        # bring inputs back to CPU
        input_data_minibatch = input_data_minibatch.cpu().numpy()

        # iterate through cell lines
        for j, cell_line in enumerate(minibatch["ModelID"]):

            # get cell line data
            cell_line_inputs = input_data_minibatch[j]
            cell_line_embs = batch_embeddings[j]

            # iterate over genes
            for k in range(cell_line_embs.shape[0]):

                # check if this is a real gene
                input_id = cell_line_inputs[k]
                if input_id in (0, 1):
                    continue

                # fill embedding and label
                labels.append(
                    f"{cell_line} | {gene_id_to_name[token_gene_dict[input_id]]}",
                )
                embeddings.append(cell_line_embs[k])

        # clean up
        del outputs
        del minibatch
        del input_data_minibatch
        torch.cuda.empty_cache()

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


# create AnnDatas for an NVIDIA Geneformer model (embeddings already computed)
def run_nvidia_gf(base_path, model_name, raw_prefix):

    # load inference results
    with open(
        os.path.join(base_path, f"nvidia-gf-preds/{raw_prefix}_depmap.pkl"),
        "rb",
    ) as f:
        inference_results = pickle.load(f)
        cell_embeddings = np.stack(
            [inference_results[i]["embeddings"] for i in range(len(inference_results))],
        )
        gene_embeddings = np.stack(
            [inference_results[i]["hiddens"] for i in range(len(inference_results))],
        )

    # load vocabulary and create dictionaries
    with open(os.path.join(base_path, f"nvidia-gf-preds/{raw_prefix}_vocab.json")) as f:
        to_deserialize = json.load(f)
        vocab = to_deserialize["vocab"]
        gene_to_ens = to_deserialize["gene_to_ens"]
        ens_to_gene = {v: k for k, v in gene_to_ens.items()}
        vocab_itos = {v: k for k, v in vocab.items()}

    # create arrays of gene IDs and names
    gene_ids = np.stack(
        [np.array([vocab_itos[i] for i in row["text"]]) for row in inference_results],
    )
    gene_names = np.stack(
        [
            np.array([ens_to_gene.get(vocab_itos[i], None) for i in row["text"]])
            for row in inference_results
        ],
    )

    # save NPZ for future use
    np.savez_compressed(
        os.path.join(base_path, f"nvidia-gf-preds/{raw_prefix}.npz"),
        gene_embeddings=gene_embeddings,
        cell_embeddings=cell_embeddings,
        gene_names=gene_names,
        gene_ids=gene_ids,
    )

    # create and save cell line embeddings
    template = sc.read_h5ad(os.path.join(base_path, "cell-embs/pca.h5ad"))
    adata = ad.AnnData(
        X=cell_embeddings,
        obs=template.obs,
        var=pd.DataFrame({"dim": np.arange(cell_embeddings.shape[1])}).set_index("dim"),
    )
    outpath = os.path.join(base_path, f"cell-embs/{model_name}.h5ad")
    adata.write_h5ad(outpath)
    log.info(f"saved cell line embeddings to {outpath}")

    # get contextual gene embeddings
    log.info("processing contextual gene embeddings")
    counts = sc.read_h5ad(os.path.join(base_path, "counts.h5ad"))
    cell_lines = counts.obs["ModelID"].tolist()
    labels = []
    embeddings = []
    for i in tqdm(range(gene_embeddings.shape[0])):
        cell_line = cell_lines[i]
        cl_embs = gene_embeddings[i]
        for j in range(gene_embeddings.shape[1]):
            gene_name = gene_names[i, j]
            if gene_name is None:
                continue
            labels.append(f"{cell_line} | {gene_name}")
            embeddings.append(cl_embs[j])

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
    model_name_underscore = "_".join(model_name.split("-"))
    utils.process_contextual_gene_embs(
        base_path,
        log,
        labels,
        embeddings,
        model_name_underscore,
    )

    # get available genes for marginal essentiality task
    embs = sc.read_h5ad(
        os.path.join(base_path, f"gene-embs/{model_name_underscore}.h5ad"),
    )
    all_genes, all_scores = utils.get_marginal_genes_scores(base_path)
    genes, scores = [], []
    avail_genes = embs.obs["gene"].unique().tolist()
    for i in range(len(all_genes)):
        if all_genes[i] in avail_genes:
            genes.append(all_genes[i])
            scores.append(all_scores[i])

    # take means for available genes
    log.info("computing mean gene embeddings for marginal essentiality task")
    mean_embs = np.zeros((len(genes), embs.X.shape[1]))
    for i, g in enumerate(tqdm(genes)):
        mean_embs[i] = embs[embs.obs["gene"] == g].X.mean(axis=0)

    # create AnnData
    mean_embs_ad = ad.AnnData(
        X=mean_embs,
        obs=pd.DataFrame({"gene": genes, "score": scores}),
        var=pd.DataFrame({"dim": np.arange(mean_embs.shape[1])}),
    )

    # write to disk
    outpath = os.path.join(
        base_path,
        f"gene-embs/{model_name_underscore}-mean-lt5gt70-bin.h5ad",
    )
    mean_embs_ad.write_h5ad(outpath)
    log.info(f"saved AnnData for marginal essentiality task to {outpath}")


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
        "--model-type",
        type=str,
        required=True,
        help="Model type: scgpt, gf, or nvidia-gf.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name (filenames are based on this).",
    )
    parser.add_argument("--model-path", type=str, help="Path to model folder.")
    parser.add_argument(
        "--gf-data-path",
        type=str,
        help="Path to dataset for use with Geneformer.",
    )
    parser.add_argument(
        "--nvidia-gf-data-prefix",
        type=str,
        help="Prefix to find embeddings in [base-path]/nvidia-gf-preds directory.",
    )
    args = parser.parse_args()

    # run correct function
    if args.model_type == "scgpt":
        run_scgpt(args.base_path, args.model_path, args.model_name)
    elif args.model_type == "gf":
        run_gf(args.base_path, args.model_path, args.model_name, args.gf_data_path)
    elif args.model_type == "nvidia-gf":
        run_nvidia_gf(args.base_path, args.model_name, args.nvidia_gf_data_prefix)
    else:
        print("Model type not recognized.")
