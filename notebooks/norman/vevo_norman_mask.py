# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
# %%
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import wandb
from scipy.sparse import issparse
from torch.utils.data import DataLoader
from torchtext._torchtext import Vocab as VocabPybind
from torchtext.vocab import Vocab

sys.path.insert(0, "../")
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import load_pretrained, set_seed

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")

# %%
hyperparameter_defaults = {
    "seed": 42,
    "dataset_name": "norman",
    "do_train": True,
    "load_model": "/scratch/ssd004/scratch/chloexq/scGPT_models/scGPT_human_model",
    "model_name": "best_model.pt",
    "GEPC": True,
    "ecs_thres": 0.8,
    "dab_weight": 1.0,
    "mask_ratio": 0.4,
    "epochs": 15,
    "n_bins": 51,
    "lr": 1e-4,
    "batch_size": 64,
    "layer_size": 128,
    "nlayers": 4,
    "nhead": 4,
    "dropout": 0.2,
    "schedule_ratio": 0.9,
    "save_eval_interval": 5,
    "log_interval": 100,
    "fast_transformer": True,
    "pre_norm": False,
    "amp": True,
}

run = wandb.init(
    config=hyperparameter_defaults,
    project="scGPT",
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
)
config = wandb.config
print(config)

set_seed(config.seed)

# %%
# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = -1
pad_value = -2
n_input_bins = config.n_bins

n_hvg = 1200  # number of highly variable genes
max_seq_len = n_hvg + 1
per_seq_batch_sample = True
DSBN = False  # Domain-spec batchnorm
explicit_zero_prob = True  # whether explicit bernoulli for zeros

dataset_name = config.dataset_name
save_dir = Path(
    f"/scratch/ssd004/scratch/ahz/perturb/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/",
)
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")

# %%
from pathlib import Path

# %% [markdown]
# ## Load and preprocess dataset
#
# ####  ✅ Note
# Perturbation datasets can be found in this path: `/scratch/ssd004/scratch/chloexq/perturb_analysis/{dataset_name}`

# %%
data_dir = Path("/scratch/ssd004/scratch/chloexq/perturb_analysis")
adata = sc.read(data_dir / "norman/perturb_processed.h5ad")

# %%
adata.var.index = pd.Index(adata.var["gene_name"])

# %%
np.unique(adata.obs.condition.values)

# %%
len(np.unique(adata.obs.condition.values))

# %%
single_gene_filter = [
    i
    for i in np.unique(adata.obs.condition.values)
    if not ("+" in i and "ctrl" not in i)
]
print(single_gene_filter, len(single_gene_filter))

# %%
adata = adata[adata.obs.condition.isin(single_gene_filter)].copy()

# %%
ori_batch_col = "control"
adata.obs["celltype"] = adata.obs["condition"].astype("category")
adata.obs["str_batch"] = adata.obs["control"].astype(str)
data_is_raw = False

# %%
if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / config.model_name
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    print(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}.",
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    print(
        f"Resume model from {model_file}, the model args will be overriden by the "
        f"config {model_config_file}.",
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
else:
    embsize = config.layer_size
    nhead = config.nhead
    nlayers = config.nlayers
    d_hid = config.layer_size

# %%
gene_names_set = [i + "+ctrl" for i in adata.var.gene_name.values]
gene_names_set = [*gene_names_set, "ctrl"]

# %% [markdown]
# ####  ✅ Note
# This experiment is computationally expensive, so we select 1000 cells per perturbation condition.

# %%
# Cap all conditions to 1000 cells
sampled_df = (
    adata.obs[adata.obs["condition"].isin(gene_names_set)]
    .groupby("condition", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), 1000), random_state=42))
)
adata = adata[sampled_df.index].copy()
adata.obs.groupby("condition").count()

# %%
# 5 conditions are capped, including ctrl
condition_counts = adata.obs.groupby("condition").count()

# %%
condition_names = set(adata.obs.condition.tolist())

# %%
condition_names.remove("ctrl")

# %%
condition_names_gene = [i.split("+")[0] for i in list(condition_names)]

# %%
condition_names_gene.sort()

# %% [markdown]
# ####  ✅ Note
# HVGs selection will filter out some perturbed genes. We manually add them back in the experiment.

# %%
# Do filtering
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=3,  # step 1
    filter_cell_by_counts=None,  # step 2
    normalize_total=None,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=False,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=None,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
)
preprocessor(adata, batch_key=None)

# %%
sc.pp.highly_variable_genes(
    adata,
    layer=None,
    n_top_genes=1200,
    flavor="seurat_v3" if data_is_raw else "cell_ranger",
    subset=False,
)

# %%
add_counter = 0
for g in condition_names_gene:
    if not adata.var.loc[
        adata.var[adata.var.gene_name == g].index,
        "highly_variable",
    ].values[0]:
        adata.var.loc[adata.var[adata.var.gene_name == g].index, "highly_variable"] = (
            True
        )
        add_counter += 1

# %%
print(
    "Manually add conditions: {}, {}".format(
        add_counter,
        add_counter / len(condition_names_gene),
    ),
)

# %%
# This step for binning
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=0,  # step 1
    filter_cell_by_counts=None,  # step 2
    normalize_total=None,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=False,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=None,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=config.n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)
preprocessor(adata, batch_key=None)

# %%
adata = adata[:, adata.var["highly_variable"]].copy()
print(adata)

# %% [markdown]
# #### 🔵 Optional
# Create another randomly shuffled list of `condition_names_gene_match` as control, if running the control experiment.
# Note that there are many ways to construct the control list, either from perturbation targets or random from all genes.

# %%
# Here is an example of randomly shuffle perturbation targets
import random

random.seed(42)
condition_names_gene_match = condition_names_gene.copy()
random.shuffle(condition_names_gene_match)

# %%
# Here is an example of using non-targets
# This is the most recent version
genes = adata.var["gene_name"].tolist()
non_targets = list(set(genes).difference(set(condition_names_gene)))
non_targets.sort()
random.seed(42)
random.shuffle(non_targets)
non_targets
condition_names_gene_match = non_targets[: len(condition_names_gene)]

# %%
print(condition_names_gene)

# %%
print(condition_names_gene_match)

# %% [markdown]
# ## Prepare model input

# %%
max_len = adata.shape[1] + 1
max_len

# %%
if config.load_model is None:
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None),
    )  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)
adata.obs["batch_id"] = adata.obs["condition"].copy()
batch_ids = adata.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
input_layer_key = "X_binned"

# %% [markdown]
# ## Load the pre-trained scGPT model

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    vocab=vocab,
    dropout=config.dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=config.GEPC,
    do_dab=False,
    use_batch_labels=False,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=DSBN,
    n_input_bins=n_input_bins,
    ecs_threshold=config.ecs_thres,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=config.fast_transformer,
    use_generative_training=True,
    pre_norm=config.pre_norm,
)
if config.load_model is not None:
    load_pretrained(model, torch.load(model_file), verbose=False)

model.to(device)
wandb.watch(model)

# %%
model.eval()
adata_t = adata.copy()

# %%
all_counts = (
    adata_t.layers[input_layer_key].A
    if issparse(adata_t.layers[input_layer_key])
    else adata_t.layers[input_layer_key]
)
celltypes_labels = adata_t.obs["celltype"].tolist()
celltypes_labels = np.array(celltypes_labels)

batch_ids = adata_t.obs["batch_id"].tolist()
batch_ids = np.array(batch_ids)

tokenized_all = tokenize_and_pad_batch(
    all_counts,
    gene_ids,
    max_len=max_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=True,
)
all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])

# %% [markdown]
# ##  Get gene embeddings (with Value Masking), Calculate Cosine Distance & Rank, and Save Results


# %%
def expand_cell(tokenized_all, key, k, select_gene_id):
    cell_k = tokenized_all[key][k]
    # Repeat
    cell_k_expand = cell_k.repeat(n_genes).view(n_genes, n_genes)
    new_column = torch.full((n_genes, 1), vocab([pad_token])[0])
    cell_k_expand = torch.cat((cell_k_expand, new_column), dim=1)
    mask = torch.eye(n_genes).bool()
    new_column_mask = torch.full((n_genes, 1), False)
    mask = torch.cat((mask, new_column_mask), dim=1)
    mask[:, select_gene_id] = True
    mask[select_gene_id, n_genes] = True
    mask_select_expand = cell_k_expand[mask]
    select_ids_gen = mask_select_expand.view(n_genes, 2)
    select_ids_pcpt = cell_k_expand[~mask].view(n_genes, n_genes - 1)
    return select_ids_gen, select_ids_pcpt


from tqdm import tqdm


def collate_cell_by_key(tokenized_all, key, select_gene_id):
    print(key)
    select_ids_gen_list = []
    select_ids_pcpt_list = []
    for k in tqdm(range(n_cells)):
        select_ids_gen, select_ids_pcpt = expand_cell(
            tokenized_all,
            key,
            k,
            select_gene_id,
        )
        select_ids_gen_list.append(select_ids_gen)
        select_ids_pcpt_list.append(select_ids_pcpt)
    select_ids_gen = torch.cat(select_ids_gen_list, dim=0)
    select_ids_pcpt = torch.cat(select_ids_pcpt_list, dim=0)
    print(select_ids_gen.shape, select_ids_pcpt.shape)
    return select_ids_gen, select_ids_pcpt


from sklearn.metrics.pairwise import cosine_distances

# %%
from torch.utils.data import TensorDataset

# %%
# %%
select_gene_list = condition_names_gene

for select_gene in select_gene_list:
    adata_t = adata[adata.obs["condition"].isin([select_gene + "+ctrl", "ctrl"])].copy()
    print(adata_t.obs["condition"])
    select_gene_id = genes.index(select_gene) + 1
    print(select_gene_id)
    all_counts = (
        adata_t.layers[input_layer_key].A
        if issparse(adata_t.layers[input_layer_key])
        else adata_t.layers[input_layer_key]
    )
    celltypes_labels = adata_t.obs["celltype"].tolist()
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata_t.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    tokenized_all = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=max_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=True,
    )
    all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
    src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])
    print(tokenized_all["genes"].shape, tokenized_all["values"].shape)
    n_cells = tokenized_all["genes"].shape[0]
    n_genes = tokenized_all["genes"].shape[1]

    collate_genes_gen, collate_genes_pcpt = collate_cell_by_key(
        tokenized_all,
        "genes",
        select_gene_id,
    )
    _, collate_values_pcpt = collate_cell_by_key(
        tokenized_all,
        "values",
        select_gene_id,
    )

    tokenized_all_expand = {
        "genes_pcpt": collate_genes_pcpt,
        "genes_gen": collate_genes_gen,
        "values_pcpt": collate_values_pcpt,
    }
    print(tokenized_all_expand)
    query_id = tokenized_all["genes"][0].repeat(n_cells)

    cell_counter = torch.arange(0, n_cells)
    cell_counter = cell_counter.repeat(n_genes).view(n_genes, n_cells).t().flatten()
    gene_counter = torch.arange(0, n_genes).repeat(n_cells)

    dataloader = DataLoader(
        TensorDataset(
            tokenized_all_expand["genes_pcpt"],
            tokenized_all_expand["genes_gen"],
            tokenized_all_expand["values_pcpt"],
            query_id,
            cell_counter,
            gene_counter,
        ),
        batch_size=512,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gene_embeddings = np.zeros((n_cells, n_genes, 512))

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
        for batch_idx, batch_data in enumerate(tqdm(dataloader)):
            pcpt_genes = batch_data[0].to(device)
            gen_genes = batch_data[1].to(device)
            pcpt_values = batch_data[2].to(device)
            query_id_select = batch_data[3].to(device)
            cell_counter_batch = batch_data[4].to(device)
            gene_counter_batch = batch_data[5].to(device)
            pcpt_key_padding_mask = pcpt_genes.eq(vocab[pad_token]).to(device)
            gen_key_padding_mask = gen_genes.eq(vocab[pad_token]).to(device)
            _, gen_output = model.transformer_generate(
                pcpt_genes=pcpt_genes,
                pcpt_values=pcpt_values,
                pcpt_key_padding_mask=pcpt_key_padding_mask,
                gen_genes=gen_genes,
                gen_key_padding_mask=gen_key_padding_mask,
            )
            select_mask = (gen_genes == query_id_select.unsqueeze(1)).long()
            selected_output = gen_output[
                torch.arange(gen_output.shape[0]),
                select_mask.argmax(dim=1),
                :,
            ]
            selected_output_np = selected_output.detach().cpu().numpy()
            gene_embeddings[
                cell_counter_batch.detach().cpu().numpy(),
                gene_counter_batch.detach().cpu().numpy(),
                :,
            ] = selected_output_np

    conditions = adata_t.obs["condition"].values

    dict_sum_condition_mean = {}
    for c in np.unique(conditions):
        dict_sum_condition_mean[c] = gene_embeddings[np.where(conditions == c)[0]].mean(
            0,
        )

    print(dict_sum_condition_mean)

    celltype_0 = select_gene + "+ctrl"
    celltype_1 = "ctrl"
    gene_emb_celltype_0 = np.expand_dims(dict_sum_condition_mean[celltype_0][1:, 1:], 0)
    gene_emb_celltype_1 = np.expand_dims(dict_sum_condition_mean[celltype_1][1:, 1:], 0)
    gene_dist_dict = {}
    for i, g in tqdm(enumerate(genes)):
        gene_dist_dict[g] = cosine_distances(
            gene_emb_celltype_0[:, i, :],
            gene_emb_celltype_1[:, i, :],
        ).mean()
    df_gene_emb_dist = pd.DataFrame.from_dict(
        gene_dist_dict,
        orient="index",
        columns=["cos_dist"],
    )
    df_deg = df_gene_emb_dist.sort_values(by="cos_dist", ascending=False)
    rank_celltype_0 = np.where(df_deg.index == celltype_0.split("+")[0])[0][0]
    print(celltype_0, rank_celltype_0)
    np.savez(
        str(save_dir) + "/mean_gene_emb_{}_{}.npz".format(select_gene, rank_celltype_0),
        **dict_sum_condition_mean,
    )
    print(
        f'Saved:\n{str(save_dir)+"/mean_gene_emb_{}_{}.npz".format(select_gene, rank_celltype_0)}',
    )
