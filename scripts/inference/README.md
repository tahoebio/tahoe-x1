# Inference scripts

Flash-attn-4 / B200-ready embedding inference for the released Tahoe-x1 checkpoints, plus a
validation script that checks output against the published `tahoebio/Tahoe-x1-embeddings`
reference. Full technical write-up: [`OPTIMIZATION_REPORT.md`](OPTIMIZATION_REPORT.md).

## Core files

| File | Role |
|---|---|
| `save_embeddings.py` | Compute CLS embeddings and write Parquet shards (local or `s3://`). |
| `save_embeddings.py` configs | `tahoe_100m_3b.yaml` (3B from HF), `tahoe_100m.yaml` (70M, pre-existing). |
| `compare_embeddings.py` | Validate freshly computed embeddings against the reference dataset. |
| `compare_embeddings.yaml` | Config for the above (1M cells, join on `BARCODE_SUB_LIB_ID`). |
| `drug_to_id_pad.json` | Drug->id mapping, bundled so inference never needs the private S3 bucket referenced in the checkpoint's collator config. |
| `OPTIMIZATION_REPORT.md` | The FA4/B200 optimization and validation report. |
| `report_assets/` | Charts for the report. |

## What changed in this work

- **Real FA4 kernels on Blackwell.** `tahoe_x1/_flash_attn_compat.py` (wired in at package
  import time by `tahoe_x1/__init__.py`) patches only what flash-attn-4 lacks, then bridges
  llm-foundry's `flash_attn_varlen_func` call onto FA4's `flash_attn.cute` kernel, so
  `attn_impl: "flash"` runs real B200 attention instead of the `"torch"` fallback --
  measured at ~2.7x throughput and about half the peak memory on real Tahoe-100M
  batches. No-op when classic flash-attn v1/v2 is installed, and safe when flash-attn
  is absent entirely.
- **Config-driven metadata columns.** Which reserve keys become Parquet columns, their names, and
  dtypes are set in `output.metadata_fields`, not hardcoded.
- **Direct-from-HF model loading.** `ComposerTX.from_hf(...)` loads weights + configs off the
  Hub; `from_hf(..., attn_impl="torch")` forces non-kernel attention. Note `"torch"`
  does not mask padding (see `OPTIMIZATION_REPORT.md` §2.1) — use it for throughput
  baselines and unpadded batches, not as a correctness reference.

## Quick start

```bash
# Smoke slice (5k cells) with the 3B model:
python scripts/inference/save_embeddings.py scripts/inference/tahoe_100m_3b.yaml

# Full 1M-cell validation against the reference:
python scripts/inference/compare_embeddings.py scripts/inference/compare_embeddings.yaml
```

Notes:

- `tahoe_100m_3b.yaml` ships a bounded smoke run: `streaming: True` with
  `num_cells: 5000`. Delete `num_cells` for the full ~100M-cell pass. Always prefer
  streaming over a `split: "train[:N]"` slice -- a non-streaming slice downloads all
  Tahoe-100M shard files (thousands of them) before it can slice, however small N is.
- `drug_to_id_pad.json` is only needed when `use_chem_inf: true`; set it to `false` for
  chem-agnostic embeddings and the mapping is unused.
- `attn_impl` need not be set in the 3B config; the checkpoint's native `"flash"` is what the
  compatibility shim accelerates.
