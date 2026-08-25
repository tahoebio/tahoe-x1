# Copyright (C) Tahoe Therapeutics 2025-2026. All rights reserved.
"""Extract CLS embeddings for a dataset and write them out as Parquet shards.

Usage:
    Single GPU / debug (world_size=1, no torch.distributed involved):
        python scripts/inference/save_embeddings.py scripts/inference/tahoe_100m.yaml

    Multi-GPU / multi-node:
        composer -n <NUM_GPUS> scripts/inference/save_embeddings.py scripts/inference/tahoe_100m.yaml
    or equivalently:
        torchrun --standalone --nproc_per_node=<NUM_GPUS> \
            scripts/inference/save_embeddings.py scripts/inference/tahoe_100m.yaml

    Each rank writes its own shard(s) to `{output_dir}/{output.prefix}_rank{rank}_{shard:03d}.parquet`,
    so outputs never collide across ranks. `paths.output_dir` may be a local path or an `s3://` URI.

    The metadata columns written alongside each embedding are driven by config, not
    hardcoded, so this script works for any dataset (not only Tahoe-100M's schema):
      - `data.reserve_keys` lists which columns the collator passes through untouched.
      - `dataset.num_cells` (optional) caps the total number of rows embedded across
        all ranks -- use it with `streaming: True` for a bounded smoke run.
      - `output.metadata_fields` (optional) selects and shapes which of those reserve
        keys get written to Parquet: each entry is `{key, name?, dtype?}`, where `name`
        renames the output column (default: same as `key`) and `dtype` is `"string"` or
        `"dict_string"` (default: dictionary-encoded string). If omitted, every reserve
        key is written as-is with dictionary-encoded string type.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Optional, Tuple, Union

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import streaming
import torch
import torch.distributed as dist
from datasets import load_dataset
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch.utils.data import DistributedSampler
from tqdm.auto import tqdm

from tahoe_x1.data import DataCollator
from tahoe_x1.model import ComposerTX
from tahoe_x1.tokenizer import GeneVocab

log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)


def setup() -> int:
    """Read LOCAL_RANK and set the CUDA device for this process, if any."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def get_rank_world() -> Tuple[int, int]:
    """Read RANK/WORLD_SIZE, defaulting to a single-process run."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def get_output_filesystem_and_path(
    output_dir: str,
) -> Tuple[Union[pafs.S3FileSystem, pafs.LocalFileSystem], str]:
    """Get filesystem and path for output directory."""
    if output_dir.startswith("s3://"):
        fs = pafs.S3FileSystem()
        path = output_dir[5:]  # strip s3://
    else:
        fs = pafs.LocalFileSystem()
        path = output_dir
    return fs, path


def resolve_metadata_fields(cfg: DictConfig) -> list:
    """Resolve (batch_key, output_name, arrow_type) triples for metadata
    columns.

    Defaults to one dictionary-encoded string column per `data.reserve_keys` entry.
    Override via `output.metadata_fields` (a list of `{key, name?, dtype?}`) to drop
    reserve keys from the output, rename a column, or switch a column to a plain
    `pa.string()` (e.g. for high-cardinality columns like a barcode).
    """
    dtype_map = {
        "string": pa.string(),
        "dict_string": pa.dictionary(pa.int32(), pa.string()),
    }
    specs = cfg.output.get("metadata_fields", None)
    if specs is None:
        specs = [{"key": key} for key in cfg.data.reserve_keys]
    fields = []
    for spec in specs:
        key = spec["key"]
        name = spec.get("name", key)
        dtype = dtype_map[spec.get("dtype", "dict_string")]
        fields.append((key, name, dtype))
    return fields


def get_parquet_writer(
    fs: Union[pafs.S3FileSystem, pafs.LocalFileSystem],
    path_prefix: str,
    output_prefix: str,
    rank: int,
    shard_idx: int,
    schema: pa.Schema,
) -> Tuple[pq.ParquetWriter, Any]:
    """Create a rank-specific parquet writer."""
    file_path = f"{path_prefix}/{output_prefix}_rank{rank}_{shard_idx:03d}.parquet"
    sink = fs.open_output_stream(file_path)
    writer = pq.ParquetWriter(sink, schema, use_dictionary=True)
    return writer, sink


def main(cfg: DictConfig) -> None:
    """
    Main entrypoint: load model, dataset, compute embeddings, and write chunked Parquet shards.
    """
    local_rank = setup()
    rank, world_size = get_rank_world()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if world_size > 1 and not (dist.is_available() and dist.is_initialized()):
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
    log.info(
        f"Rank {rank}/{world_size} starting on device {device} "
        f"(hostname={os.uname().nodename}, pid={os.getpid()}, time={datetime.now().isoformat()})",
    )

    hf_repo_id = cfg.paths.get("hf_repo_id", None)
    if hf_repo_id is not None:
        log.info(
            f"Loading model from Hugging Face repo {hf_repo_id} ({cfg.paths.hf_model_size})...",
        )
        model, vocab, model_cfg, coll_cfg = ComposerTX.from_hf(
            hf_repo_id,
            cfg.paths.hf_model_size,
            use_chem_inf=cfg.paths.get("use_chem_inf", True),
            attn_impl=cfg.paths.get("attn_impl", None),
        )
        model.to(device).eval()
    else:
        log.info("Loading vocabulary, collator, and model configuration...")
        vocab = GeneVocab.from_file(cfg.paths.vocab_file)
        coll_cfg = om.load(cfg.paths.collator_config_path)
        model_cfg = om.load(cfg.paths.model_config_path)
        model_cfg["attn_config"]["attn_impl"] = cfg.model.attn_impl
        model_cfg["attn_config"]["use_attn_mask"] = cfg.model.use_attn_mask

        model = ComposerTX(model_config=model_cfg, collator_config=coll_cfg)
        torch.cuda.empty_cache()

        state = torch.load(cfg.paths.model_file, map_location=device)["state"]["model"]
        model.load_state_dict(state, strict=True)
        model.to(device).eval()

    transformer_generate = model.model.transformer_generate

    drug_to_id_path = coll_cfg.get("drug_to_id_path", None)
    if drug_to_id_path is not None:
        # Bundle the drug->id mapping with the script so the pipeline never depends on
        # the private S3 bucket referenced in the checkpoint's collator_config. The
        # collator still attempts that bucket first (a no-op when unreachable) and
        # loads this bundled copy when the fetch fails.
        drug_to_id_path = dict(drug_to_id_path)
        drug_to_id_path["local"] = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "drug_to_id_pad.json",
        )

    collator = DataCollator(
        vocab=vocab,
        drug_to_id_path=drug_to_id_path,
        use_chem_token=coll_cfg.get("use_chem_token", False),
        do_padding=coll_cfg.get("do_padding", True),
        unexp_padding=False,
        pad_token_id=coll_cfg.pad_token_id,
        pad_value=coll_cfg.pad_value,
        do_mlm=False,
        do_binning=coll_cfg.get("do_binning", True),
        log_transform=coll_cfg.get("log_transform", False),
        target_sum=coll_cfg.get("target_sum"),
        mlm_probability=coll_cfg.mlm_probability,
        mask_value=coll_cfg.mask_value,
        max_length=cfg.data.max_length,
        sampling=coll_cfg.sampling,
        num_bins=coll_cfg.get("num_bins", 51),
        right_binning=coll_cfg.get("right_binning", False),
        keep_first_n_tokens=coll_cfg.get("keep_first_n_tokens", 1),
        reserve_keys=cfg.data.reserve_keys,
    )

    log.info("Loading dataset and preparing DataLoader...")
    ds = load_dataset(
        cfg.dataset.name,
        split=cfg.dataset.split,
        streaming=cfg.dataset.streaming,
        cache_dir=cfg.dataset.get("cache_dir", None),
        data_files=cfg.dataset.get("data_files", None),
    )
    ds = ds.with_format("torch")

    # Optional cap on how many rows to embed, applied before sharding so the number is
    # the total across all ranks. With `streaming: True` this is the only practical way
    # to bound a run: a non-streaming `split: "train[:N]"` slice downloads every shard
    # file of the source dataset before it can slice (Tahoe-100M is spread over thousands
    # of them -- 4419 as of writing -- i.e. a few hundred GB) no matter how small N is.
    # Leave unset for a full pass.
    num_cells = cfg.dataset.get("num_cells", None)
    if num_cells is not None:
        ds = (
            ds.take(num_cells) if cfg.dataset.streaming else ds.select(range(num_cells))
        )

    if cfg.dataset.streaming:
        ds = ds.shard(num_shards=world_size, index=rank, contiguous=True)
        sampler = None
    else:
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

    loader = streaming.StreamingDataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        collate_fn=collator,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        prefetch_factor=cfg.data.prefetch_factor,
        persistent_workers=False,
    )

    # Rows assigned to this rank, for the progress bar. A capped run divides evenly
    # across ranks; an uncapped streaming run has no known length without a full pass,
    # so the bar stays indeterminate.
    if sampler is not None:
        total_rows: Optional[int] = len(sampler)
    elif num_cells is not None:
        total_rows = -(-num_cells // world_size)  # ceil, so rank 0 isn't short-changed
    else:
        total_rows = None
    pbar = tqdm(
        total=total_rows,
        desc=f"Rank {rank} embedding & writing",
        disable=(rank != 0),
    )

    metadata_fields = resolve_metadata_fields(cfg)
    schema = pa.schema(
        [pa.field(name, dtype) for _, name, dtype in metadata_fields]
        + [pa.field(cfg.output.prefix, pa.list_(pa.float32(), model_cfg["d_model"]))],
    )

    fs, output_path = get_output_filesystem_and_path(cfg.paths.output_dir)
    if not str(cfg.paths.output_dir).startswith("s3://"):
        os.makedirs(cfg.paths.output_dir, exist_ok=True)

    precision = {
        "fp32": torch.float32,
        "amp_bf16": torch.bfloat16,
        "amp_fp16": torch.float16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[model_cfg["precision"]]

    use_autocast = (
        device.type == "cuda" and precision in (torch.float16, torch.bfloat16)
    ) or (device.type == "cpu" and precision is torch.bfloat16)

    row_count: int = 0
    shard_idx: int = 0
    writer: Optional[pq.ParquetWriter] = None
    sink: Optional[Any] = None

    try:
        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type,
            dtype=precision,
            enabled=use_autocast,
        ):
            for batch in loader:
                bs = batch["gene"].shape[0]

                # Rotate to a new ParquetWriter if starting a shard
                if writer is None:
                    writer, sink = get_parquet_writer(
                        fs,
                        output_path,
                        cfg.output.prefix,
                        rank,
                        shard_idx,
                        schema,
                    )

                # Extract metadata
                metadata = {name: batch[key] for key, name, _ in metadata_fields}

                # Compute CLS embeddings
                ids = batch["gene"].to(device)
                expr = batch["expr"].to(device)
                gen_masks = batch["gen_mask"].to(device)
                key_padding_mask = ~ids.eq(coll_cfg.pad_token_id)
                drug_ids = batch["drug_ids"].to(device) if "drug_ids" in batch else None
                embs = transformer_generate(
                    ids,
                    expr,
                    gen_masks,
                    key_padding_mask,
                    drug_ids=drug_ids,
                )
                cls_np = embs[:, 0, :].cpu().numpy()

                # Build and write Arrow Table
                table = pa.Table.from_pydict(
                    {**metadata, cfg.output.prefix: [list(r) for r in cls_np]},
                    schema=schema,
                )
                writer.write_table(table)

                row_count += bs
                pbar.update(bs)

                # If chunk size reached, close and advance shard
                if row_count >= cfg.parquet.chunk_size:
                    writer.close()
                    sink.close()
                    writer = None
                    sink = None
                    row_count = 0
                    shard_idx += 1
    finally:
        if writer:
            writer.close()
        if sink:
            sink.close()
        pbar.close()
        if rank == 0:
            log.info(f"Finished writing embeddings to: {cfg.paths.output_dir}")
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    yaml_path: str = sys.argv[1]
    log.info(f"Loading configuration from {yaml_path}...")
    cfg: DictConfig = om.load(yaml_path)
    om.resolve(cfg)
    main(cfg)
    log.info("Script execution completed.")
