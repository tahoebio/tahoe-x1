# Docker Image Changelog

## state-tahoe v1.0.0

**Base Image:** `mosaicml/llm-foundry:2.7.0_cu128-latest`

### Packages Installed

| Package | Version | Purpose |
|---------|---------|---------|
| AWS CLI v2 | 2.33.x | S3 data access authentication |
| scanpy | >=1.9.0,<2.0 | Single-cell data analysis |
| wandb | latest | Experiment tracking |
| llm-foundry | >=0.17.1,<1.0 | MosaicML LLM training framework |
| geomloss | v0.2.5 | Optimal transport for embedding comparisons |
| importlib_metadata | <8.0 | Compatibility fix (see below) |

### Design Decisions

#### 1. AWS CLI v2 via apt (not pip)

AWS CLI v2 is not available via pip. It must be installed using the official installer.
The installation downloads and extracts the CLI, then cleans up to minimize image size.

#### 2. importlib_metadata<8.0 Constraint

**Problem:** After installing scanpy, importing `llmfoundry` fails with:
```
ImportError: cannot import name 'Distribution' from 'importlib_metadata'
```

**Root Cause:** The error chain is `llmfoundry` -> `mlflow` -> `opentelemetry` -> `importlib_metadata`.
`opentelemetry` versions in the base image use the `importlib_metadata` backport package,
but scanpy's dependencies can upgrade it to v8.0+ which has breaking API changes.

**Solution:** Pin `importlib_metadata<8.0` before installing scanpy to lock the compatible version.

#### 3. llm-foundry with --no-deps

**Problem:** The base image includes all llm-foundry dependencies but NOT the llm-foundry package itself.
Installing `llm-foundry[gpu]` triggers pip dependency resolution that tries to rebuild flash-attn from source.

**Solution:** Install `llm-foundry` with `--no-deps` since all dependencies are already in the base image.

#### 4. cellxgene-census Excluded

**Problem:** `cellxgene-census` requires `s3fs` which requires a specific `fsspec` version.
The base image has `fsspec==2023.6.0`, but no `s3fs` version is compatible with this version.
This causes `ResolutionImpossible` errors during pip install.

**Decision:** After reviewing the codebase, `cellxgene-census` is only used in
`scripts/data_prep/download_cellxgene.py` for downloading raw data from CellxGene Census.
It is NOT required for training or inference.

**Solution:** Exclude from Docker image. Users can install it separately in a data-prep environment:
```bash
pip install cellxgene-census
```

#### 5. geomloss with --no-deps

Installed with `--no-deps` to avoid pulling in conflicting versions of PyTorch or other dependencies.
The base image already has all required dependencies (PyTorch, numpy).

#### 6. Build-time Verification

The Dockerfile includes verification steps to catch import errors at build time:
```dockerfile
RUN python -c "import geomloss; print('geomloss OK')" && \
    python -c "import scanpy; print('scanpy OK')" && \
    python -c "import torch; print(f'torch {torch.__version__} OK')"
```

Note: `llmfoundry` is NOT tested at build time because its import chain
(`megablocks` -> `stk` -> `triton`) requires a GPU to initialize.

### Verified Working

Tested on linux/amd64:
- geomloss: imports successfully
- scanpy: 1.12
- torch: 2.7.0+cu128
- anndata: 0.12.8
- AWS CLI: v2.33.8
