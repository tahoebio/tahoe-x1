from scgpt.data import DataCollator
from scgpt.tokenizer import GeneVocab
from streaming import StreamingDataset, StreamingDataLoader
from composer.core.data_spec import DataSpec
from omegaconf import DictConfig



def build_dataloader(loader_cfg: DictConfig,
                     vocab: GeneVocab,
                     device_batch_size: int) -> DataSpec:
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
    )

    # Build Collator
    collator_cfg = loader_cfg.collator
    pad_token_id = vocab["<pad>"]
    collate_fn = DataCollator(
        do_padding=collator_cfg.get("do_padding", True),
        pad_token_id=pad_token_id,
        pad_value=collator_cfg.pad_value,
        do_mlm=collator_cfg.get("do_mlm", True),
        do_binning=collator_cfg.get("do_binning", True),
        mlm_probability=collator_cfg.mlm_probability,
        mask_value=collator_cfg.mask_value,
        max_length=collator_cfg.max_length,
        sampling=collator_cfg.sampling,
        data_style=collator_cfg.data_style,
    )

    data_loader = StreamingDataLoader(
        dataset,
        batch_size=device_batch_size,
        collate_fn=collate_fn,
        drop_last = loader_cfg.get("drop_last", False),
        num_workers = loader_cfg.get("num_workers", 8),
        pin_memory = loader_cfg.get("pin_memory", True),
        prefetch_factor = loader_cfg.get("prefetch_factor", 48),
        persistent_workers = loader_cfg.get("persistent_workers", True)
    )
    return DataSpec(dataloader=data_loader)
