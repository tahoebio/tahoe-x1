<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN -->
<p align="center">
  <a href="https://github.com/tahoebio/tahoe-x1">
    <picture>
      <img alt="tahoe-therapeutics" src="./assets/tahoe-navy-logo.png" width="95%">
    </picture>
  </a>
</p>
<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END -->

<p align="center">
<a href="https://github.com/astral-sh/ruff"><img alt="Linter: Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
<a href="https://github.com/tahoebio/tahoe-x1/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg">
    </a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>
<br />

# TahoeX1: Foundation Models for Single-Cell Genomics

**TahoeX1** is a series of transformer-based foundation models for single-cell RNA-seq data developed by Tahoe Therapeutics. These models are trained on millions of single-cell profiles and can be used for various downstream tasks including cell type classification, gene essentiality prediction, pathway analysis, and cellular state transitions.

📄 **Preprint**: Coming soon - [Internal Technical Report](https://drive.google.com/drive/u/1/folders/1KeAXZ9zNYh4uHbLL5XUMmreAkHXW4yXo)

## Table of Contents
- [Repository Structure](#repository-structure)
- [Hardware and Software Requirements](#hardware-and-software-requirements)
- [Installation](#installation)
- [Datasets](#datasets)
- [Data Processing](#data-processing)
- [Pre-trained Models](#pre-trained-models)
- [Training and Fine-tuning](#training-and-fine-tuning)
- [Generating Cell and Gene Embeddings](#generating-cell-and-gene-embeddings)
- [Tutorials and Documentation](#tutorials-and-documentation)
- [Results and Benchmarks](#results-and-benchmarks)
- [Developer Guidelines](#developer-guidelines)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Repository Structure

This repository follows a similar structure to [llm-foundry](https://github.com/mosaicml/llm-foundry/tree/main) and imports several utility functions from it.

```
tahoe-x1/
├── tahoex/                    # Core TahoeX library
│   ├── model/
│   │   ├── blocks/           # Building block modules used across models
│   │   └── model/            # Full architectures subclassed from ComposerModel
│   ├── tasks/                # Helper functions for downstream applications
│   ├── tokenizer/            # Vocabulary building and tokenization functions
│   ├── data/                 # Data loaders and collators
│   └── utils/                # Miscellaneous utility functions (S3 downloads, etc.)
├── scripts/                   # Training, evaluation, and data preparation scripts
│   ├── train.py              # Main training script (accepts YAML or CLI arguments)
│   ├── prepare_for_inference.py  # Package models with vocabulary for inference
│   ├── clustering_tutorial.ipynb  # Cell clustering and embedding tutorial
│   ├── depmap/               # DepMap benchmark scripts
│   ├── msigdb/               # MSigDB pathway benchmark scripts
│   ├── state transition/     # State transition prediction scripts
│   ├── data_prep/            # Dataset preparation scripts
│   └── inference/            # Inference utilities
├── mcli/                      # MosaicML platform configuration files
├── runai/                     # RunAI configuration files
└── debug_notebooks/           # Development and debugging notebooks
```

## Hardware and Software Requirements

### Hardware
- **GPU**: NVIDIA A100 or H100 GPUs
- **CUDA**: Version 12.1 or compatible

### Software
- **Python**: ≥3.10
- **PyTorch**: 2.5.*
- **llm-foundry**: v0.17.1 (restricted to ≤v0.6.0 for triton flash-attention support)
- **Docker**: Recommended for consistent environment

### Supported Platforms
- MosaicML platform (recommended for large-scale training)
- RunAI (for local GPU clusters)
- Local machines with compatible GPUs

### Docker Images

We provide pre-built Docker images for ease of use:

| Image Name | Base Image | Description |
|------------|------------|-------------|
| [`vevotx/mosaicfm:1.1.0`](https://hub.docker.com/repository/docker/vevotx/mosaicfm/) | `mosaicml/llm-foundry:2.2.1_cu121_flash2-813d596` | Current release image for TahoeX1 |

## Installation

### Option 1: With Docker (Recommended)
```bash
# Pull the pre-built Docker image
docker pull vevotx/mosaicfm:1.1.0

# Clone the repository
git clone https://github.com/tahoebio/tahoe-x1.git
cd tahoe-x1

# Install the package
pip install -e .
```

### Option 2: Without Docker
```bash
# Clone the repository
git clone https://github.com/tahoebio/tahoe-x1.git
cd tahoe-x1

# Create conda environment
mamba env create -f envs/mosaicfm_env.yml
mamba activate tahoe-x1

# Install the package
pip install -e . --no-deps
```

> **Note**: If you are on an H100 GPU, you may see `'sm_90' is not a recognized processor for this target (ignoring processor)`. This warning is expected and safe to ignore.

## Datasets

TahoeX1 models are trained on large-scale single-cell RNA-seq datasets. The following datasets are used for training and evaluation:

| Dataset | Description | Location |
|---------|-------------|----------|
| **CellxGene 2024-04** | ~45M cells from Apr 2024 CellxGene release + Vevo dataset 35 | `s3://vevo-ml-datasets/vevo-scgpt/datasets/cellxgene_primary_2024-04-29_MDS/` |
| **CellxGene 2023-12** | ~34M cells from Dec 2023 CellxGene release | `s3://vevo-ml-datasets/vevo-scgpt/datasets/cellxgene_primary_2023-12-15_MDS_v2/` |
| **DepMap** | Cancer cell line dependency data | `s3://vevo-ml-datasets/umair/scgpt-depmap/` |
| **MSigDB** | Pathway signature data | `s3://vevo-drives/drive_3/ANALYSIS/analysis_107/` |

Public access to datasets: `s3://tahoe-hackathon-data/MFM/benchmarks/`

For detailed information on dataset preparation, see [scripts/data_prep/README.md](scripts/data_prep/README.md).

## Data Processing

TahoeX1 uses a multi-step pipeline to convert raw single-cell data into a format suitable for training. The data processing workflow converts AnnData objects (`.h5ad` files) into the MDS (MosaicML Streaming Dataset) format for efficient streaming from cloud storage during training.

### Processing Pipeline Overview

The standard data processing workflow consists of these steps:

1. **Download/Prepare Raw Data** - Obtain single-cell data from various sources
2. **Tokenization** - Map gene IDs to vocabulary indices
3. **Dataset Conversion** - Convert to HuggingFace Arrow format
4. **Train/Test Splitting** - Create appropriate data splits
5. **MDS Generation** - Convert to compressed MDS shards for streaming

### TahoeX v1 Dataset Processing

For the original TahoeX release, see [scripts/data_prep/README.md](scripts/data_prep/README.md).

#### CellxGene Data (v1)

```bash
# Step 1: Download data from CellxGene
python scripts/data_prep/download_cellxgene.py yamls/cellxgene_apr_29_2024.yml

# Step 2: Convert h5ad chunks to HuggingFace format
python scripts/data_prep/make_dataset.py \
  --adata_dir <ADATA_PATH> \
  --vocab_path <VOCAB_PATH> \
  --output_dir <OUTPUT_PATH>

# Step 3: Merge dataset chunks
python scripts/data_prep/concatenate_datasets.py \
  --path <CHUNK_DIR> \
  --dataset_name <DATASET_NAME>

# Step 4: Convert to MDS format
python scripts/data_prep/generate_mds.py \
  --out_root <OUTPUT_PATH> \
  --train_dataset_path <PATH_TO_TRAIN.DATASET> \
  --valid_dataset_path <PATH_TO_VALID.DATASET>
```

#### Perturbation Datasets (v1)

Process perturbation screens (Adamson, Norman, Replogle, etc.):

```bash
# Process perturbation data
python scripts/data_prep/process_perturbseq.py yamls/perturbseq_adamson.yml

# Split dataset by perturbation
python scripts/data_prep/split_dataset.py yamls/perturbseq_adamson.yml

# Generate MDS format
python scripts/data_prep/generate_mds_perturbseq.py yamls/perturbseq_adamson.yml
```

#### Drug Resistance Dataset (v1)

```bash
python scripts/data_prep/process_mosaic_sensitivity.py yamls/mosaic_resistance_is_futile.yml
```

### TahoeX v2 Dataset Processing

The v2 dataset pipeline includes improvements such as using Ensembl IDs instead of gene names for better cross-dataset compatibility. See [scripts/data_prep/dataset_v2/README.md](scripts/data_prep/dataset_v2/README.md).

**v2 Datasets include:**
- CellxGene (Jan 2025 release): ~60M cells
- scBasecamp: ~115M cells
- Tahoe-100M: ~96M cells

```bash
# Step 1: Update vocabulary based on Tahoe data
python scripts/data_prep/dataset_v2/update_vocabulary.py cellxgene_2025_01_21.yaml

# Step 2: Download datasets
python scripts/data_prep/dataset_v2/download_cellxgene.py cellxgene_2025_01_21.yaml

# Step 3: Convert to HuggingFace Arrow format
HF_HOME=<PATH_ON_PVC> python scripts/data_prep/dataset_v2/make_hf_dataset.py <DATASET_YAML>

# Step 4: Convert to MDS format
python scripts/data_prep/dataset_v2/generate_mds.py <DATASET_YAML>
```

### Processed Datasets

All processed datasets in MDS format are available on S3:

| Dataset | Format | Description | Path |
|---------|--------|-------------|------|
| Resistance-is-Futile v1 | MDS | Drug resistance data split by drug | `s3://vevo-ml-datasets/vevo-scgpt/datasets/resistance_is_futile_35_MDS_v1/` |
| Adamson v1 | MDS | Perturb-seq split by perturbation | `s3://vevo-ml-datasets/vevo-scgpt/datasets/adamson_v1/` |
| Norman v1 | MDS | Perturb-seq split by perturbation | `s3://vevo-ml-datasets/vevo-scgpt/datasets/norman_v1/` |
| Replogle RPE1 v1 | MDS | Perturb-seq split by perturbation | `s3://vevo-ml-datasets/vevo-scgpt/datasets/replogle_rpe1_v1/` |
| Replogle K562 v1 | MDS | Perturb-seq split by perturbation | `s3://vevo-ml-datasets/vevo-scgpt/datasets/replogle_k562_v1/` |
| CellxGene 2025-01 | MDS | v2 dataset with 99:1 train/val split | `s3://vevo-ml-datasets/mosaicfm_v2/datasets/cellxgene_2025_01_21_merged_MDS/` |
| scBasecamp 2025-02 | MDS | v2 dataset from arc-virtual-cell atlas | `s3://vevo-ml-datasets/mosaicfm_v2/datasets/scbasecamp_2025_02_25_MDS_v2/` |
| Tahoe-100M | MDS | v2 Tahoe proprietary dataset | `s3://vevo-ml-datasets/mosaicfm_v2/datasets/tahoe_100m_MDS_v2/` |

### Data Format

Each MDS dataset contains:
- **`metadata.json`**: Dataset splits and median library size information
- **`mean_ctrl_log1p.json`**: Mean log1p expression values for control cells (perturbation datasets)
- **Compressed shards**: Data stored as compressed MDS shards with configurable compression (zstd)

### Custom Data Processing

To process your own single-cell data:

1. Prepare data in AnnData format with gene IDs as Ensembl IDs
2. Create a YAML configuration file specifying:
   - Input data paths
   - Vocabulary path
   - Output directories
   - Preprocessing parameters (gene filters, normalization, etc.)
3. Run the processing pipeline scripts with your configuration

See the example YAML files in `scripts/data_prep/yamls/` for configuration templates.

## Pre-trained Models

We provide pre-trained TahoeX1 models of various sizes:

| Model Name | Parameters | Context Length | Run Name | Checkpoint Path | WandB ID |
|------------|-----------|----------------|----------|-----------------|----------|
| **TX1-1.3B** | 1.3B | 2048 | scgpt-1_3b-2048-prod | `s3://vevo-scgpt/models/scgpt-1_3b-2048-prod/` | lv6jl8kl |
| **TX1-70M** | 70M | 1024 | scgpt-70m-1024-fix-norm-apr24-data | `s3://vevo-scgpt/models/scgpt-70m-1024-fix-norm-apr24-data/` | 55n5wvdm |
| **TX1-25M** | 25M | 1024 | scgpt-25m-1024-fix-norm-apr24-data | `s3://vevo-scgpt/models/scgpt-25m-1024-fix-norm-apr24-data/` | bt4a1luo |
| **TX1-9M** | 9M | 1024 | scgpt-test-9m-full-data | `s3://vevo-scgpt/models/scgpt-test-9m-full-data/` | di7kyyf1 |

Models are also available on HuggingFace: `tahoebio/TahoeX1`

## Training and Fine-tuning

### Training from Scratch

Use the main training script with a YAML configuration file:

```bash
python scripts/train.py -f configs/your_config.yaml
```

Or with command-line arguments:

```bash
python scripts/train.py \
  --model_name tahoex \
  --data_path /path/to/data \
  --max_seq_len 2048 \
  --batch_size 32
```

### Fine-tuning

To fine-tune a pre-trained model on your own data:

1. Download a pre-trained checkpoint from S3
2. Modify the training configuration to load from checkpoint
3. Prepare your dataset in the MDS format (see [scripts/data_prep/README.md](scripts/data_prep/README.md))
4. Launch training with the `--load_path` argument

```bash
python scripts/train.py \
  -f configs/finetune_config.yaml \
  --load_path s3://path/to/checkpoint
```

### Launching on MosaicML Platform

```bash
mcli run -f mcli/train_config.yaml
```

See [mcli/README.md](mcli/README.md) for detailed instructions on configuring and launching runs on MosaicML.

### Launching on RunAI

```bash
runai submit -f runai/train_config.yaml
```

## Generating Cell and Gene Embeddings

### Quick Start with Inference Script

Extract cell embeddings from an AnnData object:

```python
from omegaconf import OmegaConf as om
from scripts.inference.predict_embeddings import predict_embeddings

cfg = {
    "model_name": "Tx1-70m",
    "paths": {
        "hf_repo_id": "tahoebio/TahoeX1",
        "hf_model_size": "70m",
        "adata_input": "/path/to/your_data.h5ad",
    },
    "data": {
        "cell_type_key": "cell_type",
        "gene_id_key": "ensembl_id"
    },
    "predict": {
        "seq_len_dataset": 2048,
        "return_gene_embeddings": False,
    }
}

cfg = om.create(cfg)
adata = predict_embeddings(cfg)

# Access embeddings
cell_embeddings = adata.obsm["Tx1-70m"]
```

### Extracting Gene Embeddings

Set `return_gene_embeddings: True` in the configuration to extract gene-level representations.

### Preparing Models for Inference

Package a trained model with its vocabulary and metadata:

```bash
python scripts/prepare_for_inference.py \
  --model_path /path/to/checkpoint \
  --vocab_path /path/to/vocab.json \
  --output_path /path/to/inference_model
```

## Tutorials and Documentation

### Jupyter Notebooks
- **[scripts/clustering_tutorial.ipynb](scripts/clustering_tutorial.ipynb)**: Cell clustering and UMAP visualization tutorial
- See `debug_notebooks/` for additional examples on:
  - Cell type classification
  - Gene embedding extraction
  - Model conversion and testing

### Benchmark Documentation
- **[DepMap Benchmark](scripts/depmap/README.md)**: Gene essentiality prediction and cell line classification
- **[MSigDB Benchmark](scripts/msigdb/README.md)**: Pathway signature prediction
- **State Transition**: Scripts in `scripts/state transition/` for cellular state prediction

### Additional Resources
- **Data Preparation**: [scripts/data_prep/README.md](scripts/data_prep/README.md)
- **Platform Usage**: [mcli/README.md](mcli/README.md) and [gcloud/README.md](gcloud/README.md)

## Results and Benchmarks

TahoeX1 models have been extensively evaluated on multiple benchmarks:

### DepMap Benchmarks
Three tasks centered around cancer cell line data:
1. **Tissue of Origin Classification**: Separate cancer cell lines by tissue type
2. **Gene Essentiality Prediction**: Predict broadly essential vs. inessential genes
3. **Cell Line-Specific Essentiality**: Predict gene essentiality for specific cell lines

📊 See [scripts/depmap/README.md](scripts/depmap/README.md) for detailed results and evaluation protocols.

### MSigDB Pathway Prediction
Predict gene membership in MSigDB pathway signatures using gene embeddings.

📊 See [scripts/msigdb/README.md](scripts/msigdb/README.md) for results and benchmarking pipeline.

### Technical Report
For comprehensive results, analysis, and model comparisons, please refer to our technical report:
- [Internal Link](https://drive.google.com/drive/u/1/folders/1KeAXZ9zNYh4uHbLL5XUMmreAkHXW4yXo)
- Public preprint: Coming soon

## Developer Guidelines

### Code Style
We use **Black** for code formatting and **Ruff** for linting to maintain consistency across contributions.

### Pre-commit Hooks
Set up pre-commit hooks before contributing:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files  # Run before committing
```

### Contribution Workflow
1. **Do not push directly to `main`** - create a new branch for your changes
2. Open a pull request for review
3. Ensure all pre-commit checks pass
4. Use type annotations and Google-style docstrings for new code

### Type Checking and Documentation
We encourage the use of:
- Type annotations for functions and classes
- Google-style docstrings
- Future additions will include `pyright` and `pydocstyle` checks

### Infrastructure Access
For launching training/evaluation runs, ensure you have access to:
- AWS S3 buckets
- Weights & Biases (wandb)
- MosaicML CLI (`mcli`) or RunAI

Contact the team on `#infrastructure` for access.

### Code Structure Best Practices
- Keep models in `tahoex/model/`
- Add new tasks to `tahoex/tasks/`
- Place training scripts in `scripts/`
- Document new features in relevant READMEs

## Acknowledgements

We would like to thank the developers of the following open-source projects that made TahoeX1 possible:

- **[scGPT](https://github.com/bowang-lab/scGPT/tree/main)**: Pioneer work in single-cell foundation models
- **[llm-foundry](https://github.com/mosaicml/llm-foundry)**: Efficient training infrastructure for large language models
- **[streaming](https://github.com/mosaicml/streaming)**: Fast, efficient dataset streaming
- **[Hugging Face datasets](https://github.com/huggingface/datasets)**: Dataset handling and processing
- **[Scanpy](https://scanpy.readthedocs.io/)**: Single-cell analysis in Python
- **CellxGene**: For curating and providing access to large-scale single-cell datasets

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

**Developed by Tahoe Therapeutics**

For questions, issues, or collaboration inquiries, please open an issue on GitHub or contact the development team.
