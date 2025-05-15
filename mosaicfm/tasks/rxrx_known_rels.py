# Copyright (C) Vevo Therapeutics 2025. All rights reserved.

import pickle

import numpy as np
import pandas as pd
from composer import State
from composer.core.callback import Callback
from composer.loggers import Logger
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import Bunch

from mosaicfm.utils import download_file_from_s3_url

##### the following code is adapted from https://github.com/recursionpharma/EFAAR_benchmarking #####


def compute_recall(
    null_distribution: np.ndarray,
    query_distribution: np.ndarray,
    recall_threshold_pairs: list,
) -> dict:
    """Compute recall at given percentage thresholds for a query distribution
    with respect to a null distribution. Each recall threshold is a pair of
    floats (left, right) where left and right are floats between 0 and 1.

    Args:
        null_distribution (np.ndarray): The null distribution to compare against
        query_distribution (np.ndarray): The query distribution
        recall_threshold_pairs (list) A list of pairs of floats (left, right) that represent different recall threshold
            pairs, where left and right are floats between 0 and 1.

    Returns:
        dict: A dictionary of metrics with the following keys:
            - null_distribution_size: the size of the null distribution
            - query_distribution_size: the size of the query distribution
            - recall_{left_threshold}_{right_threshold}: recall at the given percentage threshold pair(s)
    """

    metrics = {}
    metrics["null_distribution_size"] = null_distribution.shape[0]
    metrics["query_distribution_size"] = query_distribution.shape[0]

    sorted_null_distribution = np.sort(null_distribution)
    query_percentage_ranks_left = np.searchsorted(
        sorted_null_distribution,
        query_distribution,
        side="left",
    ) / len(
        sorted_null_distribution,
    )
    query_percentage_ranks_right = np.searchsorted(
        sorted_null_distribution,
        query_distribution,
        side="right",
    ) / len(
        sorted_null_distribution,
    )
    for threshold_pair in recall_threshold_pairs:
        left_threshold, right_threshold = np.min(threshold_pair), np.max(threshold_pair)
        metrics[f"recall_{left_threshold}_{right_threshold}"] = sum(
            (query_percentage_ranks_right <= left_threshold)
            | (query_percentage_ranks_left >= right_threshold),
        ) / len(query_distribution)
    return metrics


def convert_metrics_to_df(metrics: dict, source: str) -> pd.DataFrame:
    """Convert metrics dictionary to dataframe to be used in summary.

    Args:
        metrics (dict): metrics dictionary
        source (str): benchmark source name

    Returns:
        pd.DataFrame: a dataframe with metrics
    """
    metrics_dict_with_list = {key: [value] for key, value in metrics.items()}
    metrics_dict_with_list["source"] = [source]
    return pd.DataFrame.from_dict(metrics_dict_with_list)


def known_relationship_benchmark(
    map_data: Bunch,
    pert_col: str,
    known_rels: dict,
    benchmark_sources=["CORUM", "HuMAP", "Reactome", "SIGNOR", "StringDB"],
    recall_thr_pairs=[(0.05, 0.95)],
    min_req_entity_cnt=20,
    log_stats=False,
    compute_recall_fn=compute_recall,
    convert_metrics_to_df_fn=convert_metrics_to_df,
):

    md = map_data.metadata
    features = map_data.features.set_index(md[pert_col]).rename_axis(index=None)
    del map_data

    if not len(features) >= min_req_entity_cnt:
        raise ValueError("Not enough entities in the map for benchmarking.")
    if log_stats:
        print(len(features), "perturbations exist in the map.")

    metrics_lst = []
    cossim_matrix = pd.DataFrame(
        cosine_similarity(features, features),
        index=features.index,
        columns=features.index,
    )
    cossim_values = cossim_matrix.values[np.triu_indices(cossim_matrix.shape[0], k=1)]

    for s in benchmark_sources:
        rels = known_rels[s]
        rels = rels[
            rels.entity1.isin(features.index) & rels.entity2.isin(features.index)
        ]
        query_cossim = np.array(
            [cossim_matrix.loc[e1, e2] for e1, e2 in rels.itertuples(index=False)],
        )
        if log_stats:
            print(
                len(query_cossim),
                "relationships are used from the benchmark source",
                s,
            )
        if len(query_cossim) > 0:
            metrics = compute_recall_fn(cossim_values, query_cossim, recall_thr_pairs)
            metrics_df = convert_metrics_to_df_fn(metrics=metrics, source=s)
            metrics_lst.append(metrics_df)

    return pd.concat(metrics_lst, ignore_index=True)


##### end of adapted code #####


class RxRxKnownRels(Callback):
    def __init__(
        self,
        cfg: dict,
    ):

        super().__init__()
        self.known_rels_cfg = cfg["known_rels"]
        self.gene_metadata_cfg = cfg["gene_metadata"]

    def fit_end(self, state: State, logger: Logger):

        # get variables from state
        self.model = state.model
        self.model.eval()
        self.run_name = state.run_name

        # download task data from S3
        download_file_from_s3_url(
            s3_url=self.known_rels_cfg["remote"],
            local_file_path=self.known_rels_cfg["local"],
        )
        download_file_from_s3_url(
            s3_url=self.gene_metadata_cfg["remote"],
            local_file_path=self.gene_metadata_cfg["local"],
        )

        # open known relationships
        with open(self.known_rels_cfg["local"], "rb") as f:
            known_rels = pickle.load(f)

        # load gene metadata
        gene_metadata = pd.read_csv(self.gene_metadata_cfg["local"])

        # extract gene emebddings from model and prepare for benchmark
        gene_embs = (
            self.model.model.gene_encoder.embedding.weight.detach().cpu().numpy()
        )
        gene_embs = gene_embs[gene_metadata["token_id"].to_numpy()]
        features = pd.DataFrame(gene_embs)

        # run benchmark
        results = known_relationship_benchmark(
            Bunch(metadata=gene_metadata, features=features),
            pert_col="gene_symbol",
            known_rels=known_rels,
            log_stats=True,
        )

        # log results for each source
        sources = results["source"].unique().tolist()
        for s in sources:
            logger.log_metrics(
                {
                    f"known relationships recall ({s})": results[
                        results["source"] == s
                    ]["recall_0.05_0.95"].item(),
                },
            )
