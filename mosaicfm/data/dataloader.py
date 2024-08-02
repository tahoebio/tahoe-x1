# Copyright (C) Vevo Therapeutics 2024. All rights reserved.
from collections.abc import MutableSequence
from typing import Dict

import numpy as np
import torch
from composer.core.data_spec import DataSpec
from omegaconf import DictConfig
from streaming import StreamingDataLoader, StreamingDataset

from mosaicfm.data import DataCollator


def build_dataloader(
    loader_cfg: DictConfig,
    collator_cfg: DictConfig,
    device_batch_size: int,
) -> DataSpec:
    """Builds a dataloader from a config.

    Args:
        loader_cfg (DictConfig): An omegaconf dictionary used to configure the loader.
        device_batch_size (int): The size of the batches (number of examples)
            that the dataloader will produce.
    """
    dataset_cfg = loader_cfg.dataset
    # Build Dataset
    dataset = StreamingDataset(
        remote=dataset_cfg.remote,
        local=dataset_cfg.local,
        download_timeout=dataset_cfg.get("download_timeout", 300),
        allow_unsafe_types=dataset_cfg.get("allow_unsafe_types", True),
        shuffle=dataset_cfg.shuffle,
        predownload=dataset_cfg.get("predownload", None),
        shuffle_seed=dataset_cfg.get("shuffle_seed", None),
        num_canonical_nodes=dataset_cfg.get("num_canonical_nodes", 2),
    )
    if isinstance(collator_cfg.mlm_probability, MutableSequence):
        mlm_probability = list(collator_cfg.mlm_probability)
    else:
        mlm_probability = collator_cfg.mlm_probability
    collate_fn = DataCollator(
        do_padding=collator_cfg.get("do_padding", True),
        pad_token_id=collator_cfg.pad_token_id,
        pad_value=collator_cfg.pad_value,
        do_mlm=collator_cfg.get("do_mlm", True),
        do_binning=collator_cfg.get("do_binning", True),
        mlm_probability=mlm_probability,
        mask_value=collator_cfg.mask_value,
        max_length=collator_cfg.max_length,
        sampling=collator_cfg.sampling,
        data_style=collator_cfg.data_style,
        num_bins=collator_cfg.get("num_bins", 51),
        right_binning=collator_cfg.get("right_binning", False),
    )

    data_loader = StreamingDataLoader(
        dataset,
        batch_size=device_batch_size,
        collate_fn=collate_fn,
        drop_last=loader_cfg.get("drop_last", False),
        num_workers=loader_cfg.get("num_workers", 8),
        pin_memory=loader_cfg.get("pin_memory", True),
        prefetch_factor=loader_cfg.get("prefetch_factor", 48),
        persistent_workers=loader_cfg.get("persistent_workers", True),
    )
    return DataSpec(dataloader=data_loader)


class CountDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        count_matrix: np.ndarray,
        gene_ids: np.ndarray,
        cls_token_id: int,
        pad_value: float,
    ):
        """
        Args:
            count_matrix (np.ndarray): A 2D expression count array of shape (n_cells, n_genes)
            gene_ids (np.ndarray): Integer Gene IDs corresponding to gene names in the count matrix (n_genes,)
            cls_token_id (int): The id of the <cls> token
            pad_value (float): The expression value used for PAD tokens
        """
        self.count_matrix = count_matrix
        self.gene_ids = gene_ids
        self.cls_token_id = cls_token_id
        self.pad_value = pad_value

    def __len__(self) -> int:
        return len(self.count_matrix)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.count_matrix[idx]
        nonzero_idx = np.nonzero(row)[0]
        values = row[nonzero_idx]
        genes = self.gene_ids[nonzero_idx]
        # append <cls> token at the beginning
        genes = np.insert(genes, 0, self.cls_token_id)
        values = np.insert(values, 0, self.pad_value)
        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values).float()
        output = {
            "id": idx,
            "genes": genes,
            "expressions": values,
        }
        return output
