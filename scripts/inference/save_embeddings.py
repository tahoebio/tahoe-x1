# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
import logging
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import streaming
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from tqdm.auto import tqdm

from tahoe_x1.data import DataCollator
from tahoe_x1.model import ComposerTX
from tahoe_x1.tokenizer import GeneVocab

log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)


def main(cfg: DictConfig) -> None:
    """
    Main entrypoint: load model, dataset, compute embeddings, and write chunked Parquet shards.
    """
    log.info("Loading vocabulary and collator configuration...")
    vocab = GeneVocab.from_file(cfg.paths.vocab_file)
    coll_cfg = om.load(cfg.paths.collator_config_path)
    collator = DataCollator(
        vocab=vocab,
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
        data_style="pcpt",
        num_bins=coll_cfg.get("num_bins", 51),
        right_binning=coll_cfg.get("right_binning", False),
        reserve_keys=cfg.data.reserve_keys,
    )

    log.info("Loading model checkpoint and configuration...")
    model_cfg = om.load(cfg.paths.model_config_path)
    model_cfg["attn_config"]["attn_impl"] = cfg.model.attn_impl
    model_cfg["attn_config"]["use_attn_mask"] = cfg.model.use_attn_mask

    model = ComposerTX(model_config=model_cfg, collator_config=coll_cfg)
    state = torch.load(cfg.paths.model_file)["state"]["model"]
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    log.info("Loading dataset and preparing DataLoader...")
    ds = load_dataset(
        cfg.dataset.name,
        split=cfg.dataset.split,
        streaming=cfg.dataset.streaming,
    )
    ds = ds.with_format("torch")
    loader = streaming.StreamingDataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        collate_fn=collator,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        prefetch_factor=cfg.data.prefetch_factor,
        persistent_workers=True,
    )

    schema = pa.schema(
        [
            pa.field("drug", pa.dictionary(pa.int32(), pa.string())),
            pa.field("sample", pa.dictionary(pa.int32(), pa.string())),
            pa.field("cell_line", pa.dictionary(pa.int32(), pa.string())),
            pa.field("BARCODE_SUB_LIB_ID", pa.string()),
            pa.field("tx-70m-merged", pa.list_(pa.float32(), 512)),
        ],
    )

    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    total_rows = len(ds)
    row_count = 0
    shard_idx = 0
    writer = None
    pbar = tqdm(total=total_rows, desc="Embedding & writing")

    precision = {
        "fp32": torch.float32,
        "amp_bf16": torch.bfloat16,
        "amp_fp16": torch.float16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[model_cfg["precision"]]

    with (
        torch.no_grad(),
        torch.amp.autocast(
            enabled=True,
            dtype=precision,
            device_type=device.type,
        ),
    ):
        for batch in loader:
            bs = batch["gene"].shape[0]

            # Rotate to a new ParquetWriter if starting a shard
            if writer is None:
                shard_path = os.path.join(
                    cfg.paths.output_dir,
                    f"{cfg.output.prefix}_{shard_idx:03d}.parquet",
                )
                writer = pq.ParquetWriter(shard_path, schema, use_dictionary=True)

            # Extract metadata
            drugs = batch["drug"]
            samples = batch["sample"]
            cells = batch["cell_line_id"]
            barcodes = batch["BARCODE_SUB_LIB_ID"]

            # Compute CLS embeddings
            ids = batch["gene"].to(device)
            expr = batch["expr"].to(device)
            mask = ~ids.eq(coll_cfg.pad_token_id)
            embs = model.model._encode(ids, expr, src_key_padding_mask=mask)
            cls_np = embs[:, 0, :].cpu().numpy()

            # Build and write Arrow Table
            table = pa.Table.from_pydict(
                {
                    "drug": drugs,
                    "sample": samples,
                    "cell_line": cells,
                    "BARCODE_SUB_LIB_ID": barcodes,
                    "tx-70m-merged": [list(r) for r in cls_np],
                },
                schema=schema,
            )
            writer.write_table(table)

            row_count += bs
            pbar.update(bs)

            # If chunk size reached, close and advance shard
            if row_count >= cfg.parquet.chunk_size:
                writer.close()
                writer = None
                row_count = 0
                shard_idx += 1

    # Final close
    if writer:
        writer.close()
    pbar.close()

    log.info(f"Finished writing embeddings to: {cfg.paths.output_dir}")


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    log.info(f"Loading configuration from {yaml_path}...")
    cfg = om.load(yaml_path)
    om.resolve(cfg)
    main(cfg)
    log.info("Script execution completed.")
