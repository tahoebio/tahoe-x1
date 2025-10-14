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

# Tahoe-x1: A Perturbation-Trained Single-Cell Foundation Model

**Tahoe-x1** is a series of transformer-based foundation models for single-cell RNA-seq data developed by Tahoe Therapeutics. These models are trained on millions of single-cell profiles and can be used for various downstream tasks including cell type classification, gene essentiality prediction, pathway analysis, and cellular state transitions.

📄 **Preprint**: Coming soon - [Paper](https://drive.google.com/drive/u/1/folders/1KeAXZ9zNYh4uHbLL5XUMmreAkHXW4yXo)

## Table of Contents
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Training Infrastructure](#training-infrastructure)
- [Datasets](#datasets)
- [Pre-trained Models](#pre-trained-models)
- [Training and Fine-tuning](#training-and-fine-tuning)
- [Generating Cell and Gene Embeddings](#generating-cell-and-gene-embeddings)
- [Tutorials and Benchmarks](#tutorials-and-benchmarks)
- [Developer Guidelines](#developer-guidelines)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Repository Structure

This repository follows a similar structure to [llm-foundry](https://github.com/mosaicml/llm-foundry/tree/main) and imports several utility functions from it.

```
tahoe-x1/
├── tahoex/                    # Core Tahoe-x1 library
│   ├── model/
│   │   ├── blocks/           # Building block modules used across models
│   │   └── model/            # Full architecture subclassed from ComposerModel
│   ├── tasks/                # Helper functions for downstream tasks
│   ├── tokenizer/            # Vocabulary building and tokenization functions
│   ├── data/                 # Data loaders and collators
│   └── utils/                # Utility functions 
├── scripts/                   
│   ├── train.py              # Training script 
│   ├── prepare_for_inference.py  # Prepares model for inference
│   ├── clustering_tutorial.ipynb  # Cell clustering tutorial
│   ├── depmap/               # DepMap benchmark scripts
│   ├── msigdb/               # MSigDB pathway benchmark scripts
│   ├── state transition/     # State transition prediction scripts
│   ├── data_prep/            # Dataset preparation scripts
│   └── inference/            # Inference utilities
├── mcli/                      # MosaicML platform configuration files
└── runai/                     # RunAI configuration files
```

## Installation

### Option 1: With uv (Recommended)

```bash

# Clone the repo
git clone https://github.com/tahoebio/tahoe-x1.git
cd tahoe-x1

# Install uv if it doesn't exist
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create env
uv venv
source .venv/bin/activate

# Install the package
uv pip install -e . --no-build-isolation-package flash-attn
```

### Option 2: With Docker

```bash
# Pull the pre-built Docker image
docker pull ghcr.io/tahoebio/tahoe-x1:1.0.0

# Clone the repo
git clone https://github.com/tahoebio/tahoe-x1.git
cd tahoe-x1

# Install the package
pip install -e .
```





## Training Infrastructure

The model is trained and tested on:
- **GPU**: NVIDIA H100 GPUs
- **CUDA**: Version 12.1 or compatible
- **Python**: ≥3.10
- **PyTorch**: 2.5.2
- **llm-foundry**: v0.17.1 

Platforms such as MosaicML and RunAI are used for training and deployment.

### Docker Images

We provide pre-built Docker images for ease of use:

| Image Name | Base Image | Description |
|------------|------------|-------------|
| [`ghcr.io/tahoebio/tahoe-x1:1.0.0`](https://github.com/tahoebio/tahoe-x1/pkgs/container/tahoe-x1) | `mosaicml/llm-foundry:2.2.1_cu121_flash2-813d596` | Current release image for Tahoe-x1 |

## Datasets

Tahoe-x1 models are trained on large-scale single-cell RNA-seq datasets. The following datasets are used for training and evaluation:

| Dataset | Description | Usage | Location |
|---------|-------------|-------|----------|
| **CellxGene 2025-01** | ~61M  cells  from Jan 2025 CellxGene release | Tx1-3b stage 1 Pre-training  | `s3://tahoe-hackathon-data/MFM/cellxgene_2025_01_21_merged_MDS/` |
| **scBaseCamp 2025-02** | ~112M  cells from Feb 2025 scBaseCamp release | Tx1-3b stage 1 Pre-training | `s3://tahoe-hackathon-data/MFM/scbasecamp_2025_02_25_MDS_v2/` |
| **Tahoe 100M** | ~96M  cells from Tahoe-100M | Tx1-3b stage 1 Pre-training | `s3://tahoe-hackathon-data/MFM/tahoe_100m_MDS_v2/` |
| **filtered CellxGene 2025-01** | ~43M filtered cells  from Jan 2025 CellxGene release | Tx1-3b stage 2 Pre-training  | `s3://tahoe-hackathon-data/MFM/cellxgene_2025_01_21_merged_MDS_filtered/` |
| **filtered scBaseCamp 2025-02** | ~76M filtered cells from Feb 2025 scBaseCamp release | Tx1-3b stage 2 Pre-training | `s3://tahoe-hackathon-data/MFM/scbasecamp_2025_02_25_MDS_v2_filtered/` |
| **filtered Tahoe 100M** | ~34M filtered cells from Tahoe-100M | Tx1-3b stage 2 Pre-training | `s3://tahoe-hackathon-data/MFM/tahoe_100m_MDS_v2_filtered/` |
| **DepMap** | Cancer cell line dependency data | DepMap Benchmark | `s3://tahoe-hackathon-data/MFM/benchmarks/depmap/` |
| **MSigDB** | Pathway signature data | MsigDB Benchmark | `s3://tahoe-hackathon-data/MFM/benchmarks/msigdb/` |

Filtered versions of the pre-training datasets above exclude cells with very few expressed genes and are used for stage 2 pre-training of Tx1-3b.

Public access to datasets: `s3://tahoe-hackathon-data/MFM/benchmarks/`

If you require access to datasets not available in the public bucket, please open a GitHub issue or contact the team.

For more information on dataset preparation, see [scripts/data_prep/README.md](scripts/data_prep/README.md).



## Pre-trained Models

We provide pre-trained Tahoe-x1 models of various sizes:

| Model Name | Parameters | Context Length | Checkpoint Path | WandB ID |
|------------|------------|----------------|-----------------|----------|
| **TX1-3B** | 3B | 2056  | `s3://tahoe-hackathon-data/MFM/ckpts/3b/` | [mygjkq5c](https://wandb.ai/vevotx/tahoe-x1/runs/mygjkq5c) |
| **TX1-1.3B** | 1.3B | 2048 | `s3://tahoe-hackathon-data/MFM/ckpts/1b/` | [26iormxc](https://wandb.ai/vevotx/tahoe-x1/runs/26iormxc) |
| **TX1-70M** | 70M | 1024 | `s3://tahoe-hackathon-data/MFM/ckpts/70m/` | [ftb65le8](https://wandb.ai/vevotx/tahoe-x1/runs/ftb65le8) |

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

### Preparing Models for Inference

Package a trained model with its vocabulary and metadata:

```bash
python scripts/prepare_for_inference.py \
  --model_path /path/to/checkpoint \
  --vocab_path /path/to/vocab.json \
  --output_path /path/to/inference_model
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


## Tutorials and Benchmarks

### Tutorials
- **[scripts/clustering_tutorial.ipynb](scripts/clustering_tutorial.ipynb)**: Cell clustering and UMAP visualization tutorial


### Benchmarks
Tahoe-x1 models have been extensively evaluated on multiple benchmarks:

#### DepMap Benchmarks
Three tasks centered around cancer cell line data:
1. **Tissue of Origin Classification**: Separate cancer cell lines by tissue type
2. **Gene Essentiality Prediction**: Predict broadly essential vs. inessential genes
3. **Cell Line-Specific Essentiality**: Predict gene essentiality for specific cell lines

📊 See manuscript and [scripts/depmap/](scripts/depmap/README.md) for detailed results and evaluation protocols.

#### MSigDB Pathway Prediction
Predict gene membership in MSigDB pathway signatures using gene embeddings.

📊 See manuscript and [scripts/msigdb/](scripts/msigdb/README.md) for results and benchmarking pipeline.

#### State Transition 

📊 See manuscript and [scripts/state transition/](scripts/state transition/README.md) for results and post-training protocols.

### Technical Report
For comprehensive results, analysis, and model comparisons, please refer to our technical report:
- [Internal Link](https://drive.google.com/drive/u/1/folders/1KeAXZ9zNYh4uHbLL5XUMmreAkHXW4yXo)
- Public preprint: Coming soon


### Additional Resources
- **Data Preparation**: [scripts/data_prep/README.md](scripts/data_prep/README.md)
- **Platform Usage**: [mcli/README.md](mcli/README.md) and [gcloud/README.md](gcloud/README.md)




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
- MosaicML CLI (`mcli`) or RunAI or local gpu


## Acknowledgements

We would like to thank the developers of the following open-source projects:

- **[scGPT](https://github.com/bowang-lab/scGPT/tree/main)**: Pioneering work in single-cell foundation models
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
