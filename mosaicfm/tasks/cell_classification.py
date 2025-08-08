# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
import io
from typing import Any, Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from composer import State
from composer.core.callback import Callback
from composer.loggers import Logger
from composer.utils import model_eval_mode
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from mosaicfm.utils import download_file_from_s3_url


class CellClassification(Callback):
    """Callback for evaluating cell type classification using cell embeddings.

    This callback extracts cell embeddings from a trained MosaicFM model and
    evaluates their quality for cell type classification tasks. It supports
    multiple datasets with train/test splits, computes classification metrics
    (F1 score), calculates LISI (Local Inverse Simpson Index) scores for
    embedding quality assessment, and generates UMAP visualizations of the
    learned cell representations.

    The cell classification task is a key benchmark for single-cell foundation
    models, evaluating whether the learned representations capture meaningful
    biological cell type distinctions that can be used for downstream
    classification tasks.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
    ) -> None:
        """Initialize the CellClassification callback.

        Args:
            cfg: Configuration dictionary containing:
                - datasets (Dict[str, Dict]): Registry of datasets to evaluate, where each
                  dataset entry contains:
                    - train: Configuration for training data:
                        - remote (str): S3 URL for training data
                        - local (str): Local path to save training data
                    - test (Dict, optional): Configuration for test data (same structure as train)
                    - cell_type_key (str, optional): Column name for cell type labels.
                      Defaults to "cell_type_label"
                    - gene_id_key (str, optional): Column name for gene identifiers.
                      Defaults to "ensembl_id"
                    - batch_size (int, optional): Dataset-specific batch size
                    - seq_len (int, optional): Dataset-specific sequence length
                - classifier_config (Dict): Configuration for logistic regression classifier:
                    - max_iter (int, optional): Maximum iterations. Defaults to 5000
                    - solver (str, optional): Solver algorithm. Defaults to "lbfgs"
                    - multi_class (str, optional): Multi-class strategy. Defaults to "multinomial"
                    - random_state (int, optional): Random seed. Defaults to 42
                - batch_size (int, optional): Default batch size for embedding extraction. Defaults to 50
                - seq_len (int, optional): Default sequence length. Defaults to 2048
        """

        super().__init__()

        self.dataset_registry = cfg.get("datasets")
        self.classifier_config = cfg.get("classifier_config")
        self.batch_size = cfg.get("batch_size", 50)
        self.seq_len = cfg.get("seq_len", 2048)

    def fit_end(self, state: State, logger: Logger) -> None:
        """Execute cell classification evaluation at the end of training.

        This method iterates through all registered datasets, downloads the required
        data files from S3, and performs cell classification evaluation for each dataset.

        Args:
            state: Composer training state containing the trained model and data loaders
            logger: Composer logger for recording metrics and visualizations
        """

        self.model = state.model
        self.model_config = self.model.model_config
        self.collator_config = self.model.collator_config
        self.vocab = state.train_dataloader.collate_fn.vocab
        self.run_name = state.run_name

        if self.dataset_registry is None:
            raise ValueError(
                "dataset_registry is None but required for cell classification evaluation",
            )

        for dataset_name, dataset_cfg in self.dataset_registry.items():
            for split in ["train", "test"]:
                if split in dataset_cfg:
                    download_file_from_s3_url(
                        s3_url=dataset_cfg[split]["remote"],
                        local_file_path=dataset_cfg[split]["local"],
                    )
            self.cell_classification(dataset_name, logger)

    def cell_classification(self, dataset: str, logger: Logger) -> None:
        """Perform cell classification evaluation for a specific dataset.

        This method:
        1. Loads and prepares training (and optionally test) data
        2. Extracts cell embeddings using the trained model
        3. Trains a logistic regression classifier (if test data available)
        4. Computes LISI scores to assess embedding quality
        5. Generates UMAP visualizations colored by cell type
        6. Logs all metrics and visualizations

        Args:
            dataset: Name of the dataset to evaluate (key in dataset registry)
            logger: Composer logger for recording results
        """
        assert self.dataset_registry is not None, "dataset_registry should not be None"

        cell_type_key = self.dataset_registry[dataset].get(
            "cell_type_key",
            "cell_type_label",
        )
        gene_id_key = self.dataset_registry[dataset].get(
            "gene_id_key",
            "ensembl_id",
        )
        adata_train, gene_ids_train = self.prepare_cell_annotation_data(
            self.dataset_registry[dataset]["train"]["local"],
            cell_type_key=cell_type_key,
            gene_id_key=gene_id_key,
        )
        use_test_split = False
        adata_test = None
        gene_ids_test = None
        cell_embeddings_test = None

        if "test" in self.dataset_registry[dataset]:
            adata_test, gene_ids_test = self.prepare_cell_annotation_data(
                self.dataset_registry[dataset]["test"]["local"],
                cell_type_key=cell_type_key,
                gene_id_key=gene_id_key,
            )
            use_test_split = True

        # step 2: extract mosaicfm embeddings
        from mosaicfm.tasks import get_batch_embeddings

        dataset_batch_size = self.dataset_registry[dataset].get(
            "batch_size",
            self.batch_size,
        )
        dataset_seq_len = self.dataset_registry[dataset].get("seq_len", self.seq_len)
        with model_eval_mode(
            self.model.model,
        ), torch.no_grad(), FSDP.summon_full_params(self.model.model, writeback=False):

            cell_embeddings_train = get_batch_embeddings(
                adata=adata_train,
                model=self.model.model,
                vocab=self.vocab,
                gene_ids=gene_ids_train,
                model_cfg=self.model_config,
                collator_cfg=self.collator_config,
                batch_size=dataset_batch_size,
                max_length=dataset_seq_len,
                return_gene_embeddings=False,
            )
            assert isinstance(
                cell_embeddings_train,
                np.ndarray,
            ), "Expected np.ndarray when return_gene_embeddings=False"
            if use_test_split:
                assert (
                    adata_test is not None and gene_ids_test is not None
                ), "Test data should be available when use_test_split is True"
                cell_embeddings_test = get_batch_embeddings(
                    adata=adata_test,
                    model=self.model.model,
                    vocab=self.vocab,
                    gene_ids=gene_ids_test,
                    model_cfg=self.model_config,
                    collator_cfg=self.collator_config,
                    batch_size=dataset_batch_size,
                    max_length=dataset_seq_len,
                    return_gene_embeddings=False,
                )

        if use_test_split:
            # step 3: train classifier if test split is available
            assert (
                self.classifier_config is not None
            ), "classifier_config should not be None"
            assert (
                cell_embeddings_test is not None and adata_test is not None
            ), "Test data should be available"

            clf = LogisticRegression(
                max_iter=self.classifier_config.get("max_iter", 5000),
                solver=self.classifier_config.get("solver", "lbfgs"),
                multi_class=self.classifier_config.get("multi_class", "multinomial"),
                random_state=self.classifier_config.get("random_state", 42),
            )
            clf.fit(cell_embeddings_train, adata_train.obs[cell_type_key].values)

            # step 4: calculate and log metrics
            labels_pred = clf.predict(cell_embeddings_test)
            f1 = f1_score(
                adata_test.obs[cell_type_key].values,
                labels_pred,
                average="macro",
            )
            logger.log_metrics({f"macro_f1_{dataset}": f1})

        # Step 5: compute LISI score for train split
        lisi_score = self.compute_lisi_scores(
            cell_embeddings_train,
            adata_train.obs[cell_type_key].values.to_numpy(dtype="str"),
            20,
        )
        logger.log_metrics({f"LISI {dataset}": lisi_score})

        # step 6: UMAP visualization and logging
        adata_train.obsm[dataset] = cell_embeddings_train
        sc.pp.neighbors(adata_train, use_rep=dataset)
        sc.tl.umap(adata_train)
        fig = sc.pl.umap(
            adata_train,
            color=[cell_type_key],
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

    def prepare_cell_annotation_data(
        self,
        data_path: str,
        gene_id_key: str,
        cell_type_key: str,
    ) -> Tuple[AnnData, np.ndarray]:
        """Prepare cell annotation data for classification evaluation.

        This method loads an AnnData object, filters out cells with missing cell type
        annotations, matches genes with the model vocabulary, and returns the processed
        data along with corresponding gene IDs.

        Args:
            data_path: Path to the h5ad file containing single-cell data
            gene_id_key: Column name in adata.var containing gene identifiers
            cell_type_key: Column name in adata.obs containing cell type labels

        Returns:
            Tuple containing:
                - adata: Processed AnnData object with matched genes and filtered cells
                - gene_ids: NumPy array of gene vocabulary IDs corresponding to the genes
        """

        vocab = self.vocab
        adata = sc.read_h5ad(data_path)
        adata = adata[~adata.obs[cell_type_key].isna(), :]

        adata.var["id_in_vocab"] = [
            vocab[gene] if gene in vocab else -1 for gene in adata.var[gene_id_key]
        ]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        print(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            + f"in vocabulary of size {len(vocab)}.",
        )
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
        genes = adata.var[gene_id_key].tolist()
        gene_ids = np.array([vocab[gene] for gene in genes], dtype=int)

        return adata, gene_ids

    @staticmethod
    def compute_lisi_scores(
        emb: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        k: int,
    ) -> float:
        """Computes a LISI score. Accepts numpy arrays or torch tensors.

        Args:
            emb (Union[np.ndarray, torch.Tensor]): (n_samples, n_features) embedding matrix.
            labels (Union[np.ndarray, torch.Tensor]): (n_samples,) label vector, can be strings or ints.
            k (int): Number of neighbors.

        Returns:
            float: The LISI score.
        """
        # Convert to tensors/arrays as needed
        emb = torch.as_tensor(emb, dtype=torch.float32)
        labels_np = np.asarray(labels)
        _, inverse_labels = np.unique(labels_np, return_inverse=True)
        labels = torch.from_numpy(inverse_labels).long()

        # Compute pairwise distances
        distances = torch.cdist(emb, emb, p=2)

        # Get k nearest neighbors for each point (excluding itself)
        _, knn_indices = torch.topk(distances, k + 1, largest=False)
        knn_indices = knn_indices[:, 1:]  # exclude self

        # Self vs neighbor labels
        self_labels = labels.unsqueeze(1).expand(-1, k)
        neighbor_labels = labels[knn_indices]

        # Compute label agreement
        same_label = (self_labels == neighbor_labels).float().mean()

        # Theoretical LISI normalization
        label_counts = torch.bincount(labels)
        theoretic_score = ((label_counts / label_counts.sum()) ** 2).sum()

        return (same_label / theoretic_score).item()
