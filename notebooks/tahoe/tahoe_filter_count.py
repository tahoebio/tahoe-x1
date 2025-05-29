# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
# %%
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import wandb
from scipy.sparse import issparse

# %%
data_dir = Path("/datasets/Tahoe-100M/Tahoe/vevo_filter/plate2_demo_final.h5ad")
adata = sc.read(data_dir)
adata

# %%
adata.n_vars

# %%
data_is_raw = True
sc.pp.filter_genes(adata, min_cells=1)


# %%
adata

# %%
drug_df = pd.read_csv(
    "/datasets/Tahoe-100M/Tahoe/vevo_filter/plate2_demo_final_targets.csv",
)
drug_target_map = dict(zip(drug_df["compound"], drug_df["targets"]))

perturb_conditions = list(drug_target_map.values())
condition_names_gene = perturb_conditions

drug_target_map["ctrl"] = "ctrl"

# %%
adata.obs["gene_target"] = adata.obs["condition"].map(drug_target_map)


# %%
def sample_cells_per_label(adata, column, n=75, random_state=None):
    """Randomly sample `n` cells per unique label in `adata.obs[column]`.

    Parameters:
        adata (AnnData): The AnnData object.
        column (str): Column in `adata.obs` to sample from.
        n (int): Number of cells to sample per label.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        AnnData: A new AnnData object with the sampled cells.
    """
    sampled_indices = (
        adata.obs.groupby(column)
        .apply(lambda x: x.sample(n=min(n, len(x)), random_state=random_state))
        .index.get_level_values(1)
    )

    return adata[sampled_indices].copy()


# %%
adata.obs["gene_target"]

# %%
adata

import pandas as pd

# %%
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

# %%
max_len = adata.shape[1] + 1
max_len

# %%
adata.X[:5, :5].toarray()

# %%
sc.pp.log1p(adata)

# %% [markdown]
# # Baseline Wilcoxon

# %%
# Wilcoxon rank
sc.tl.rank_genes_groups(
    adata,
    "gene_target",
    method="wilcoxon",
    key_added="wilcoxon",
    n_genes=max_len - 1,
    reference="ctrl",
)
adata.uns["wilcoxon"]["names"]

# %%
adata.uns["wilcoxon"]

# %%
import pickle

# Save to a pickle file
with open("wilcoxon_filtered.pkl", "wb") as f:
    pickle.dump(adata.uns["wilcoxon"], f)

# %%
adata.uns["wilcoxon"]["names"]

# %%
baseline_rank = []

for c in perturb_conditions:
    hvg_list = adata.uns["wilcoxon"]["names"][c]
    p_val = adata.uns["wilcoxon"]["pvals_adj"][c]
    df_gene_emb_dist = pd.DataFrame()
    df_gene_emb_dist["gene"] = hvg_list
    df_gene_emb_dist["p_val"] = p_val
    df_gene_emb_dist = df_gene_emb_dist.sort_values(by="p_val")
    print(c, np.where(df_gene_emb_dist.gene.values == c.split("+")[0])[0][0])
    baseline_rank.append(
        np.where(df_gene_emb_dist.gene.values == c.split("+")[0])[0][0],
    )

# %%
baseline_rank

# %%
with open("baseline_filtered.pkl", "wb") as f:
    pickle.dump(baseline_rank, f)

# %%
np.mean(baseline_rank)

# %%
import seaborn as sns

sns.boxplot(y=baseline_rank)

# Add title and labels
plt.title("Boxplot of Wilcoxon Rank")
plt.ylabel("Rank Value")

# %% [markdown]
# # scGPT Embedding

# %%
from pathlib import Path

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
    "dataset_name": "fibro",  # Dataset name
    "do_train": True,  # Flag to indicate whether to update model parameters during training
    "load_model": "/scratch/ssd004/scratch/chloexq/scGPT_models/scGPT_human_model",
    "model_name": "best_model.pt",
    "GEPC": True,  # Gene expression modelling for cell objective
    "ecs_thres": 0.8,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    "dab_weight": 1.0,  # DAR objective weight for batch correction
    "mask_ratio": 0.4,  # Default mask ratio
    "epochs": 15,  # Default number of epochs for fine-tuning
    "n_bins": 51,  # Default number of bins for value binning in data pre-processing
    "lr": 1e-4,  # Default learning rate for fine-tuning
    "batch_size": 64,  # Default batch size for fine-tuning
    "layer_size": 128,
    "nlayers": 4,
    "nhead": 4,  # If loading model, batch_size, layer_size, nlayers, nhead will be ignored
    "dropout": 0.2,  # Default dropout rate during model fine-tuning
    "schedule_ratio": 0.9,  # Default rate for learning rate decay
    "save_eval_interval": 5,  # Default model evaluation interval
    "log_interval": 100,  # Default log interval
    "fast_transformer": True,  # Default setting
    "pre_norm": False,  # Default setting
    "amp": True,  # Default setting: Automatic Mixed Precision
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
max_seq_len = adata.n_vars + 1
per_seq_batch_sample = True
DSBN = True  # Domain-spec batchnorm
explicit_zero_prob = True  # whether explicit bernoulli for zeros

dataset_name = config.dataset_name
save_dir = Path(
    f"/scratch/ssd004/scratch/ahz/perturb/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/",
)
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")

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

    adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in adata.var.index]
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
torch.cuda.empty_cache()

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
genes = adata.var.index.tolist()
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
    do_dab=True,
    use_batch_labels=False,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=False,
    n_input_bins=n_input_bins,
    ecs_threshold=config.ecs_thres,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=config.fast_transformer,
    pre_norm=config.pre_norm,
)
if config.load_model is not None:
    load_pretrained(model, torch.load(model_file), verbose=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# %%
model.eval()
adata_t = adata.copy()

# %%
adata

# %%
all_counts = (
    adata_t.layers[input_layer_key].A
    if issparse(adata_t.layers[input_layer_key])
    else adata_t.layers[input_layer_key]
)
celltypes_labels = adata_t.obs["cell_line"].tolist()
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

# %%
gc.collect()

# %%
condition_ids = np.array(adata_t.obs["condition"].tolist())

# Initialize accumulators for running sums, counts, means, and the rank list
dict_sum_condition = {}
dict_count_condition = {}
dict_sum_condition_mean = {}
rank_list = []  # To store computed ranks from analysis iterations

sub_batch_size = 512  # Use batch_size=16
num_samples = len(all_gene_ids)

# Create a random permutation of indices
random_indices = np.random.permutation(num_samples)

# Process the dataset in random sub-batches
for i in tqdm(range(0, num_samples, sub_batch_size), desc="Random partitions"):
    gc.collect()
    # Get a batch of random indices
    batch_indices = random_indices[i : i + sub_batch_size]

    # Index the arrays using the random batch indices
    batch_gene_ids = all_gene_ids[batch_indices]
    batch_values = all_values[batch_indices].float()  # assuming all_values is a tensor
    batch_padding_mask = (
        src_key_padding_mask[batch_indices]
        if src_key_padding_mask is not None
        else None
    )

    # Compute embeddings for the current random sub-batch
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
        gene_embeddings_batch = model.encode_batch(
            batch_gene_ids,
            batch_values,
            src_key_padding_mask=batch_padding_mask,
            batch_size=16,  # Explicitly use batch_size=16 here
            batch_labels=None,
            return_np=True,
        )

    # Update cumulative sums and counts, then recalc the running mean per condition
    # Note: We index condition_ids with the original index from batch_indices.
    for idx, sample_idx in enumerate(batch_indices):
        condition = condition_ids[sample_idx]
        if condition in dict_sum_condition:
            dict_sum_condition[condition] += gene_embeddings_batch[idx, :, :]
            dict_count_condition[condition] += 1
        else:
            dict_sum_condition[condition] = gene_embeddings_batch[idx, :, :]
            dict_count_condition[condition] = 1

        # Update the running mean for this condition
        dict_sum_condition_mean[condition] = (
            dict_sum_condition[condition] / dict_count_condition[condition]
        )

    # ----- Analysis for the batches processed so far -----
    # Map conditions to target genes via drug_target_map.
    dict_sum_target_gene_mean = {
        drug_target_map[drug]: dict_sum_condition_mean[drug]
        for drug in dict_sum_condition_mean
        if drug in drug_target_map
    }

    # (Optional) Get the gene vocabulary index from the first element
    gene_vocab_idx = all_gene_ids[0].clone().detach().cpu().numpy()

    # Create a list of perturbation targets, excluding control ('ctrl')
    perturb_targets = list(dict_sum_target_gene_mean.keys())
    if "ctrl" in perturb_targets:
        perturb_targets.remove("ctrl")
    assert "ctrl" not in perturb_targets

    # For each perturbation target, compute cosine distances and determine a ranking.
    if "ctrl" in dict_sum_target_gene_mean:
        for t in perturb_targets:
            celltype_0 = t
            celltype_1 = "ctrl"
            # Expand dims so that cosine_distances receives 2D arrays.
            gene_emb_celltype_0 = np.expand_dims(
                dict_sum_target_gene_mean[celltype_0][1:, :],
                axis=0,
            )
            gene_emb_celltype_1 = np.expand_dims(
                dict_sum_target_gene_mean[celltype_1][1:, :],
                axis=0,
            )
            gene_dist_dict = {}

            for j, g in tqdm(
                enumerate(genes),
                total=len(genes),
                desc=f"Analyzing {t}",
                disable=True,
            ):
                gene_dist = cosine_distances(
                    gene_emb_celltype_0[:, j, :],
                    gene_emb_celltype_1[:, j, :],
                ).mean()
                gene_dist_dict[g] = gene_dist

            df_gene_emb_dist = pd.DataFrame.from_dict(
                gene_dist_dict,
                orient="index",
                columns=["cos_dist"],
            )
            df_deg = df_gene_emb_dist.sort_values(by="cos_dist", ascending=False)
            rank = np.where(df_deg.index == t)[0][0]
            print(f"Target {t} rank: {rank}")
            rank_list.append(rank)
        # ----- End Analysis -----

        # Save a box and whisker plot of rank_list as a PNG file.
        if rank_list:
            rank_mean = np.mean(rank_list)
            plt.figure()
            plt.boxplot(rank_list)
            plt.savefig(
                f"boxplot_{i + sub_batch_size}_samples_mean_{rank_mean:.2f}.png",
            )
            plt.close()

    # Free memory used by the current sub-batch
    del gene_embeddings_batch
    gc.collect()


# %%
dict_sum_condition_mean = {}
groups = adata_t.obs.groupby("condition").groups
for i in groups:
    dict_sum_condition_mean[i] = dict_sum_condition[i] / len(groups[i])

# %%
dict_sum_target_gene_mean = {
    drug_target_map[drug]: dict_sum_condition_mean[drug]
    for drug in dict_sum_condition_mean
}
gene_vocab_idx = all_gene_ids[0].clone().detach().cpu().numpy()
perturb_targets = list(dict_sum_target_gene_mean.keys())
perturb_targets.remove("ctrl")
assert "ctrl" not in perturb_targets
rank_list = []

for t in perturb_targets:
    celltype_0 = t
    celltype_1 = "ctrl"
    gene_emb_celltype_0 = np.expand_dims(
        dict_sum_target_gene_mean[celltype_0][1:, :],
        0,
    )
    gene_emb_celltype_1 = np.expand_dims(
        dict_sum_target_gene_mean[celltype_1][1:, :],
        0,
    )
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
    print(t, np.where(df_deg.index == t)[0][0])
    rank_list.append(np.where(df_deg.index == t)[0][0])


# %%
def is_one_to_one_mapping(d):
    """Checks if a dictionary represents a one-to-one mapping.

    Returns True if it is, otherwise returns False and a dictionary of colliding
    values with their associated keys.
    """
    value_to_keys = {}
    collisions = {}

    for key, value in d.items():
        if value in value_to_keys:
            value_to_keys[value].append(key)
            collisions[value] = value_to_keys[value]
        else:
            value_to_keys[value] = [key]

    if collisions:
        return False, collisions
    return True, None


# %%
is_one_to_one_mapping(drug_target_map)

# %% [markdown]
# ## Save Results & Analysis

# %%
adata.uns["wilcoxon"]["names"]

# %%
perturb_targets

# %%
# rerun wilcoxon rank for gene targets instead

baseline_rank_t = []

for t in perturb_targets:
    hvg_list = adata.uns["wilcoxon"]["names"][t]
    p_val = adata.uns["wilcoxon"]["pvals_adj"][t]
    df_gene_emb_dist = pd.DataFrame()
    df_gene_emb_dist["gene"] = hvg_list
    df_gene_emb_dist["p_val"] = p_val
    df_gene_emb_dist = df_gene_emb_dist.sort_values(by="p_val")
    print(t, np.where(df_gene_emb_dist.gene.values == t)[0][0])
    baseline_rank_t.append(np.where(df_gene_emb_dist.gene.values == t)[0][0])


# %%
df_results = df = pd.DataFrame(
    {
        "conditions": perturb_targets,
        "wilcoxon": baseline_rank_t,
        "scGPT_rank": rank_list,
    },
)

# %%
df_results

# %%
df_results.mean()

# %%
df_results.to_csv("/scratch/ssd004/scratch/ahz/perturb/vevo_tahoe_ranks_Mar13.csv")

# %%
import seaborn as sns

plt.figure(figsize=(10, 10))
sns.boxplot(data=df_results)

# Add titles and labels
plt.title("Comparison of Methods Across Conditions")
plt.xlabel("Method")
plt.ylabel("Rank")
plt.xticks(rotation=-45)
plt.tight_layout()

plt.legend([], [], frameon=False)

# %%
sns.scatterplot(data=df_results, x="wilcoxon", y="scGPT_rank")

# %%
from scipy.stats import pearsonr

corr, p_value = pearsonr(df_results["wilcoxon"], df_results["scGPT_rank"])

print(f"Pearson Correlation: {corr}, p-value: {p_value}")
