# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
import io
import json

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from composer import State
from composer.core.callback import Callback
from composer.loggers import Logger
from composer.utils import dist
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import kneighbors_graph
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from mosaicfm.utils import download_file_from_s3_url


# Custom Callback to run the cell classification after training
class CellClassification(Callback):
    def __init__(
        self,
        cfg: dict,
    ):

        super().__init__()

        self.dataset_registry = cfg.get("datasets")
        self.logistic_cfg = cfg.get("logistic")
        self.batch_size = cfg.get("batch_size", 50)
        self.seq_len = cfg.get("seq_len", 2048)

        # load gene_to_id mapping
        assert (
            "ensemble_to_gene_path" in cfg
        ), "ensemble_to_gene_path not found in config and should be provided!"
        ensemble_to_gene_path = cfg.get("ensemble_to_gene_path")

        if dist.get_local_rank() == 0:
            download_file_from_s3_url(
                s3_url=ensemble_to_gene_path["remote"],
                local_file_path=ensemble_to_gene_path["local"],
            )
        with dist.local_rank_zero_download_and_wait(
            ensemble_to_gene_path["local"],
        ):
            dist.barrier()

        with open(ensemble_to_gene_path["local"]) as f:
            id_to_gene = json.load(f)
        self.gene_to_id = dict(zip(id_to_gene.values(), id_to_gene.keys()))

    def fit_end(self, state: State, logger: Logger):

        self.model = state.model
        self.model.eval()
        self.model_config = self.model.model_config
        self.collator_config = self.model.collator_config
        self.vocab = state.train_dataloader.collate_fn.vocab
        self.run_name = state.run_name

        # cell classification both for zheng and Segerstolpe datasets
        for datast_name, dataset_cfg in self.dataset_registry.items():

            # download dataset splits
            for split in dataset_cfg:
                download_file_from_s3_url(
                    s3_url=dataset_cfg[split]["remote"],
                    local_file_path=dataset_cfg[split]["local"],
                )

            self.cell_classfication(datast_name, logger)

    def cell_classfication(self, dataset: str, logger: Logger):
        # step 1: load data train, test
        class_idx_to_name = np.load(
            self.dataset_registry[dataset]["class_names"]["local"],
        )
        adata_train, gene_ids_train, labels_train, _ = (
            self.prepare_cell_annotation_data(
                self.dataset_registry[dataset]["train"]["local"],
                class_idx_to_name,
            )
        )
        adata_test, gene_ids_test, labels_test, _ = self.prepare_cell_annotation_data(
            self.dataset_registry[dataset]["test"]["local"],
            class_idx_to_name,
        )

        # step 2: extract mosaicfm embeddings
        from mosaicfm.tasks import get_batch_embeddings

        with FSDP.summon_full_params(self.model.model):
            cell_embeddings_train = get_batch_embeddings(
                adata=adata_train,
                model=self.model.model,
                vocab=self.vocab,
                gene_ids=gene_ids_train,
                model_cfg=self.model_config,
                collator_cfg=self.collator_config,
                batch_size=self.batch_size,
                max_length=self.seq_len,
                return_gene_embeddings=False,
            )
            cell_embeddings_test = get_batch_embeddings(
                adata=adata_test,
                model=self.model.model,
                vocab=self.vocab,
                gene_ids=gene_ids_test,
                model_cfg=self.model_config,
                collator_cfg=self.collator_config,
                batch_size=self.batch_size,
                max_length=self.seq_len,
                return_gene_embeddings=False,
            )

        # step 3: train classifier
        clf = LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=42,
        )
        clf.fit(cell_embeddings_train, labels_train)

        # step 4: calculate and log metrics
        from sklearn.metrics import f1_score

        labels_pred = clf.predict(cell_embeddings_test)
        f1 = f1_score(labels_test, labels_pred, average="macro")
        logger.log_metrics({f"macro_f1_{dataset}": f1})

        # step 5: compute lisi
        lisi_score = self.compute_lisi_scores(
            cell_embeddings_train,
            adata_train.obs["cell_type_label"].values,
            20,
        )
        logger.log_metrics({f"LISI {dataset}": lisi_score})

        # step 6: UMAP visualization and logging
        adata_train.obsm[dataset] = cell_embeddings_train
        sc.pp.neighbors(adata_train, use_rep=dataset)
        sc.tl.umap(adata_train)
        fig = sc.pl.umap(
            adata_train,
            color=["cell_type_names"],
            frameon=False,
            title=[f"{self.run_name} LISI:{lisi_score:.2f} \n {dataset} Dataset"],
            return_fig=True,
        )

        # convert fig to ndarray
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img = np.array(plt.imread(buf))
        logger.log_images(img, name=f"clustering_{dataset}", channels_last=True)

    def compute_lisi_scores(self, emb: np.ndarray, labels: np.ndarray, k: int):
        nng = kneighbors_graph(emb, n_neighbors=k).tocoo()
        labels = np.unique(labels, return_inverse=True)[1]
        self_id = labels[nng.row]
        ne_id = labels[nng.col]

        _, c = np.unique(labels, return_counts=True)
        theoretic_score = ((c / c.sum()) ** 2).sum()
        return (self_id == ne_id).mean() / theoretic_score

    def prepare_cell_annotation_data(
        self,
        data_path: str,
        class_idx_to_name: np.ndarray,
    ):

        vocab = self.vocab
        adata = sc.read_h5ad(data_path)

        gene_name_key = "gene_symbols"
        gene_col = "gene_id"
        cell_type_key = "cell_type_label"

        adata.var[gene_col] = adata.var[gene_name_key].apply(
            lambda x: self.gene_to_id.get(x, "na"),
        )

        # filter the cell with NaN values in the cell_type_key
        adata = adata[~adata.obs[cell_type_key].isna(), :]
        adata.obs["cell_type_names"] = [
            class_idx_to_name[int(id)] for id in adata.obs[cell_type_key]
        ]
        adata.var["id_in_vocab"] = [vocab[gene] for gene in adata.var[gene_col]]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        print(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}.",
        )
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
        vocab.default_index = vocab["<pad>"]
        genes = adata.var[gene_col].tolist()
        gene_ids = np.array([vocab[gene] for gene in genes], dtype=int)

        # Extract labels from the AnnData object
        labels = adata.obs[cell_type_key].values
        unique_labels = np.unique(np.array(labels[~np.isnan(np.array(labels))]))

        # Convert labels to numeric if they are not already
        if labels.dtype.kind in "OU":  # Object or Unicode, meaning strings
            label_names = labels
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            labels = np.array([label_to_idx[label] for label in labels])
        else:
            labels = labels.astype(np.int64)
            label_names = np.array([class_idx_to_name[label] for label in labels])

        adata = adata.copy()
        adata.X = adata.X.todense()

        return adata, gene_ids, labels, label_names
