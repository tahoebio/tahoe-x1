# Copyright (C) Vevo Therapeutics 2025. All rights reserved.

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from composer import State
from composer.core.callback import Callback
from composer.loggers import Logger
from composer.utils import model_eval_mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)

from mosaicfm.utils import download_file_from_s3_url


class MarginalEssentiality(Callback):
    def __init__(
        self,
        cfg: dict,
    ):

        super().__init__()
        self.batch_size = cfg.get("batch_size", 32)
        self.seq_len = cfg.get("seq_len", 8192)
        self.adata_cfg = cfg.get("adata")
        self.labels_cfg = cfg.get("labels")
        self.classifier_cfg = cfg.get("classifier")

    def fit_end(self, state: State, logger: Logger):

        # get variables from state
        self.model = state.model

        self.model_config = self.model.model_config
        self.collator_config = self.model.collator_config
        self.vocab = state.train_dataloader.collate_fn.vocab
        self.run_name = state.run_name

        # download task data from S3
        download_file_from_s3_url(
            s3_url=self.adata_cfg["remote"],
            local_file_path=self.adata_cfg["local"],
        )
        download_file_from_s3_url(
            s3_url=self.labels_cfg["remote"],
            local_file_path=self.labels_cfg["local"],
        )

        # load and process AnnData of CCLE counts
        vocab = self.vocab
        adata = sc.read_h5ad(self.adata_cfg["local"])
        adata.var["id_in_vocab"] = [
            vocab[gene] if gene in vocab else -1
            for gene in adata.var[self.adata_cfg["gene_column"]]
        ]
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        genes = adata.var[self.adata_cfg["gene_column"]].tolist()
        gene_ids = np.array([vocab[gene] for gene in genes], dtype=int)
        print(
            f"matched {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}",
        )

        # get gene embeddings
        from mosaicfm.tasks import get_batch_embeddings

        with model_eval_mode(
            self.model.model,
        ), torch.no_grad(), FSDP.summon_full_params(self.model.model, writeback=False):
            _, gene_embeddings = get_batch_embeddings(
                adata=adata,
                model=self.model.model,
                vocab=self.vocab,
                gene_ids=gene_ids,
                model_cfg=self.model_config,
                collator_cfg=self.collator_config,
                batch_size=self.batch_size,
                max_length=self.seq_len,
                return_gene_embeddings=True,
            )

        # load task DataFrame
        gene2idx = vocab.get_stoi()
        gene_names = np.array(list(gene2idx.keys()))
        task_df = pd.read_csv(self.labels_cfg["local"])
        task_df = task_df[task_df[self.labels_cfg["gene_column"]].isin(genes)]
        task_df = task_df[task_df[self.labels_cfg["gene_column"]].isin(gene_names)]
        genes = task_df[self.labels_cfg["gene_column"]].to_numpy()
        labels = task_df[self.labels_cfg["label_column"]].to_numpy()

        # get mean embeddings for each gene
        mean_embs = np.zeros((len(genes), gene_embeddings.shape[1]))
        for i, g in enumerate(genes):
            mean_embs[i] = gene_embeddings[np.where(gene_names == g)[0][0]]

        # split into training and testing sets
        emb_train, emb_test, labels_train, labels_test = train_test_split(
            mean_embs,
            labels,
            test_size=self.classifier_cfg["test_size"],
            random_state=self.classifier_cfg["random_state"],
        )

        # train classifer and report auROC on test set
        rf = RandomForestClassifier(n_jobs=self.classifier_cfg["n_jobs"])
        rf.fit(emb_train, labels_train)
        test_probas = rf.predict_proba(emb_test)
        auroc = float(roc_auc_score(labels_test, test_probas[:, 1]))
        logger.log_metrics({"marginal gene essentiality auROC": auroc})
