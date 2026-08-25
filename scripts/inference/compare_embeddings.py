# Copyright (C) Tahoe Therapeutics 2025-2026. All rights reserved.
"""Validate the inference pipeline against the precomputed reference dataset.

Computes fresh CLS embeddings for a slice of Tahoe-100M and compares them, per cell,
against `tahoebio/Tahoe-x1-embeddings` -- the existing public embeddings dataset --
instead of re-embedding the full 100M-cell corpus. Cells are matched by
`BARCODE_SUB_LIB_ID` (not row position), since the reference dataset's own README
gives no guarantee the two datasets are ordered identically.

The reference dataset's parquet files declare a "List" feature type that this repo's
installed `datasets` version doesn't recognize (`ValueError: Feature type 'List' not
found`), so it's read directly with pyarrow instead of `datasets.load_dataset`.

Usage:
    python scripts/inference/compare_embeddings.py scripts/inference/compare_embeddings.yaml
"""

import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tahoe_x1.data import DataCollator
from tahoe_x1.model import ComposerTX

log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)


def compute_embeddings(
    cfg: DictConfig,
    device: torch.device,
) -> Tuple[List[str], np.ndarray]:
    """Run the (optimized) inference pipeline on cfg.dataset.split, returning
    (barcodes, embeddings) with matching row order."""
    model, vocab, model_cfg, coll_cfg = ComposerTX.from_hf(
        cfg.paths.hf_repo_id,
        cfg.paths.hf_model_size,
        use_chem_inf=cfg.paths.get("use_chem_inf", True),
        attn_impl=cfg.paths.get("attn_impl", None),
    )
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

    join_key = cfg.reference.join_key
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
        reserve_keys=[join_key],
    )

    # Non-streaming load_dataset(..., split=f"{split}[:N]") downloads every one of
    # Tahoe-100M's thousands of underlying files before it can slice, regardless of N --
    # fine for a full-corpus run, useless for a quick N-cell check.
    # Streaming + take(N) reads incrementally instead.
    log.info(
        f"Streaming {cfg.dataset.name} split={cfg.dataset.split!r}, first {cfg.dataset.num_cells} cells...",
    )
    ds = load_dataset(cfg.dataset.name, split=cfg.dataset.split, streaming=True)
    ds = ds.take(cfg.dataset.num_cells)
    ds = ds.with_format("torch")

    num_workers = cfg.data.get("num_workers", 0)
    loader = DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=(
            cfg.data.get("prefetch_factor", None) if num_workers > 0 else None
        ),
    )

    precision = {
        "fp32": torch.float32,
        "amp_bf16": torch.bfloat16,
        "amp_fp16": torch.float16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[model_cfg["precision"]]
    use_autocast = device.type == "cuda" and precision in (
        torch.float16,
        torch.bfloat16,
    )

    barcodes: List[str] = []
    embeddings: List[np.ndarray] = []

    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type,
        dtype=precision,
        enabled=use_autocast,
    ):
        for batch in tqdm(loader, desc="Computing embeddings"):
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
            embeddings.append(embs[:, 0, :].float().cpu().numpy())
            barcodes.extend(batch[join_key])

    return barcodes, np.concatenate(embeddings, axis=0)


def fetch_reference_embeddings(
    cfg: DictConfig,
    needed_barcodes: List[str],
) -> Dict[str, np.ndarray]:
    """Stream reference parquet shards (raw pyarrow, bypassing `datasets`) until
    every needed barcode is found or `max_shards_to_scan` is hit."""
    api = HfApi()
    files = sorted(
        f
        for f in api.list_repo_files(cfg.reference.repo_id, repo_type="dataset")
        if f.endswith(".parquet")
    )
    max_shards = cfg.reference.get("max_shards_to_scan", 10)

    still_needed = set(needed_barcodes)
    found: Dict[str, np.ndarray] = {}

    for shard_path in files[:max_shards]:
        if not still_needed:
            break
        log.info(
            f"Scanning {shard_path} ({len(still_needed)} barcodes still needed)...",
        )
        local_path = hf_hub_download(
            repo_id=cfg.reference.repo_id,
            repo_type="dataset",
            filename=shard_path,
        )
        tbl = pq.read_table(
            local_path,
            columns=[cfg.reference.join_key, cfg.reference.embedding_column],
        )
        mask = pc.is_in(
            tbl.column(cfg.reference.join_key),
            value_set=pa.array(list(still_needed)),
        )
        matched = tbl.filter(mask)
        for bc, emb in zip(
            matched.column(cfg.reference.join_key).to_pylist(),
            matched.column(cfg.reference.embedding_column).to_pylist(),
        ):
            if bc in still_needed:
                found[bc] = np.asarray(emb, dtype=np.float32)
                still_needed.discard(bc)

    if still_needed:
        log.warning(
            f"{len(still_needed)}/{len(needed_barcodes)} barcodes not found after "
            f"scanning {min(len(files), max_shards)} reference shards.",
        )

    return found


def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    barcodes, computed = compute_embeddings(cfg, device)
    log.info(f"Computed {len(barcodes)} embeddings ({computed.shape[1]}-dim).")

    reference = fetch_reference_embeddings(cfg, barcodes)

    matched_computed, matched_reference, matched_barcodes = [], [], []
    for bc, emb in zip(barcodes, computed):
        ref = reference.get(bc)
        if ref is not None:
            matched_computed.append(emb)
            matched_reference.append(ref)
            matched_barcodes.append(bc)

    unmatched = len(barcodes) - len(matched_computed)
    if not matched_computed:
        log.error("No barcodes matched the reference dataset -- nothing to compare.")
        return

    a = np.stack(matched_computed)
    b = np.stack(matched_reference)
    cos_sim = (a * b).sum(axis=1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    )

    log.info(
        f"Matched {len(matched_computed)}/{len(barcodes)} cells ({unmatched} unmatched).",
    )
    p1, p5, p95, p99 = np.percentile(cos_sim, [1, 5, 95, 99])
    log.info(
        "Cosine similarity vs. reference -- "
        f"mean={cos_sim.mean():.6f} median={np.median(cos_sim):.6f} std={cos_sim.std():.6f} "
        f"min={cos_sim.min():.6f} p1={p1:.6f} p5={p5:.6f} p95={p95:.6f} p99={p99:.6f}",
    )
    n = len(cos_sim)
    for threshold in (0.95, 0.90, 0.50):
        below = int((cos_sim < threshold).sum())
        log.info(f"  below {threshold:.2f}: {below} cells ({100.0 * below / n:.3f}%)")

    save_path = cfg.output.get("save_path", None)
    if save_path:
        import pandas as pd

        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        pd.DataFrame(
            {"barcode": matched_barcodes, "cosine_similarity": cos_sim},
        ).to_parquet(save_path)
        log.info(f"Per-cell cosine similarities written to {save_path}")


if __name__ == "__main__":
    yaml_path: str = sys.argv[1]
    log.info(f"Loading configuration from {yaml_path}...")
    cfg: DictConfig = om.load(yaml_path)
    om.resolve(cfg)
    main(cfg)
    log.info("Script execution completed.")
