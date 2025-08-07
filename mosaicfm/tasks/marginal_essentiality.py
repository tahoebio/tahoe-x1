# Copyright (C) Vevo Therapeutics 2025. All rights reserved.

from typing import Any, Dict

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
    """Callback for evaluating marginal gene essentiality prediction using gene
    embeddings.

    This callback extracts gene embeddings from a trained MosaicFM model and
    uses them to train a Random Forest classifier to predict marginal gene
    essentiality. The evaluation is performed on CCLE (Cancer Cell Line
    Encyclopedia) data to assess how well the learned gene representations
    capture functional relationships related to cellular fitness.

    The marginal essentiality task evaluates whether gene embeddings can
    distinguish between genes that are essential for cell survival versus those
    that are not, based on CRISPR knockout screening data from the CCLE.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
    ) -> None:
        """Initialize the MarginalEssentiality callback.

        Args:
            cfg: Configuration dictionary containing:
                - batch_size (int, optional): Batch size for embedding extraction. Defaults to 32.
                - seq_len (int, optional): Maximum sequence length for model input. Defaults to 8192.
                - adata: Configuration for AnnData file containing gene expression data:
                    - remote (str): S3 URL for the AnnData file
                    - local (str): Local path to save the downloaded file
                    - gene_column (str): Column name in adata.var containing gene identifiers
                - labels: Configuration for labels file containing essentiality annotations:
                    - remote (str): S3 URL for the labels file
                    - local (str): Local path to save the downloaded file
                    - gene_column (str): Column name containing gene identifiers
                    - label_column (str): Column name containing essentiality labels
                - classifier: Configuration for the Random Forest classifier:
                    - test_size (float): Fraction of data to use for testing
                    - random_state (int): Random seed for reproducible splits
                    - n_jobs (int): Number of parallel jobs for Random Forest
        """

        super().__init__()
        self.batch_size = cfg.get("batch_size", 32)
        self.seq_len = cfg.get("seq_len", 8192)
        self.adata_cfg = cfg.get("adata")
        self.labels_cfg = cfg.get("labels")
        self.classifier_cfg = cfg.get("classifier")

    def fit_end(self, state: State, logger: Logger) -> None:
        """Execute marginal essentiality evaluation at the end of training.

        This method:
        1. Downloads CCLE gene expression data and essentiality labels from S3
        2. Processes the AnnData object to match genes with the model vocabulary
        3. Extracts gene embeddings using the trained model
        4. Trains a Random Forest classifier on gene embeddings to predict essentiality
        5. Evaluates classifier performance using area under ROC curve (auROC)
        6. Logs the auROC metric for downstream analysis

        Args:
            state: Composer training state containing the trained model and data loaders
            logger: Composer logger for recording metrics and results
        """

        # get variables from state
        self.model = state.model

        self.model_config = self.model.model_config
        self.collator_config = self.model.collator_config
        self.vocab = state.train_dataloader.collate_fn.vocab
        self.run_name = state.run_name

        # Validate configuration
        if (
            self.adata_cfg is None
            or self.labels_cfg is None
            or self.classifier_cfg is None
        ):
            raise ValueError(
                "adata_cfg, labels_cfg, and classifier_cfg must all be provided",
            )

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
