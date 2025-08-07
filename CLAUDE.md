# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MosaicFM is a series of single-cell RNA-seq foundation models developed by Vevo Therapeutics. The repository builds on top of [llm-foundry](https://github.com/mosaicml/llm-foundry) and follows Composer/PyTorch training workflows.

## Development Setup

### Installation
```bash
# With docker (recommended)
git clone https://github.com/vevotx/mosaicfm.git
cd mosaicfm
pip install -e .

# Without docker
mamba env create -f envs/mosaicfm_env.yml
mamba activate mosaicfm
pip install -e . --no-deps
```

### Pre-commit Hooks
Set up pre-commit hooks before making changes:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files  # Run before committing
```

## Common Development Commands

### Code Quality
- **Linting**: `ruff check .` (configured in pyproject.toml)
- **Formatting**: `black .` (line length: 88)
- **Type checking**: `pyright` (configured in pyproject.toml)
- **Pre-commit**: `pre-commit run --all-files`

### Testing
- **Run tests**: `pytest` (configured in pyproject.toml)
- **GPU tests**: `pytest -m gpu` (requires GPU)
- **Skip GPU tests**: `pytest -m 'not gpu'` (default)

### Training & Evaluation
- **Train model**: `python scripts/train.py <config.yaml>`
- **Prepare for inference**: `python scripts/prepare_for_inference.py`
- **Extract embeddings**: `python scripts/get_embeddings.py`

## Code Architecture

### Core Components

**`mosaicfm/`** - Main package with model building blocks:
- `model/` - Full architectures built on ComposerModel
  - `blocks.py` - Transformer blocks, encoders, decoders 
  - `model.py` - SCGPTModel and ComposerSCGPTModel classes
- `tokenizer/` - Gene vocabulary and tokenization (`GeneVocab` class)
- `data/` - Data loaders and collators for single-cell data
- `tasks/` - Downstream task implementations (cell classification, embedding extraction)
- `loss.py` - Specialized loss functions (masked MSE, Spearman correlation)

**`scripts/`** - Training and evaluation scripts:
- `train.py` - Main training script accepting YAML configs
- `data_prep/` - Dataset processing pipeline for CellxGene, PerturbSeq data
- `depmap/` - DepMap benchmark evaluation scripts
- `perturbation/` - Perturbation analysis and fine-tuning

### Model Architecture

- Based on transformer architecture with specialized components for single-cell data
- Uses gene vocabularies for tokenization (~60K genes from Ensembl)
- Supports both generative and discriminative training modes
- Custom encoders/decoders for gene expressions, chemical compounds

### Data Pipeline

- Processes h5ad (AnnData) files into streaming MDS format
- Supports CellxGene consortium data (~45M cells)
- PerturbSeq datasets (Adamson, Norman, Replogle)
- Drug resistance screening data (Vevo internal)

## Configuration

### YAML Configs
Training configurations are in `mcli/` and `runai/` directories:
- Model architecture parameters (n_layers, d_model, vocab_size)
- Training hyperparameters (batch_size, learning_rate)
- Data loading configuration (streaming datasets)

### Environment Files
- `envs/mosaicfm_env.yml` - Main conda environment
- `envs/composer_env.yml` - Alternative Composer-based setup

## Platform Support

### Compute Platforms
- **MosaicML**: YAML configs in `mcli/`, use `mcli run -f <config>.yaml`
- **RunAI**: YAML configs in `runai/`, use shell scripts for submission
- **Local**: Direct Python execution with proper environment setup

### Docker Images
- Recommended: `vevotx/ml-scgpt:shreshth` (based on mosaicml/llm-foundry)
- Includes CUDA 12.1, Flash Attention support

## Key Datasets

Located on S3 under `s3://vevo-ml-datasets/vevo-scgpt/datasets/`:
- CellxGene primary datasets (MDS format for streaming)
- PerturbSeq processed datasets with train/val/test splits
- DepMap evaluation datasets

## Testing Strategy

- Unit tests in `tests/` directory
- GPU-specific tests marked with `@pytest.mark.gpu`
- Integration tests for tokenizer and encoder components
- Benchmark evaluation scripts for downstream tasks

## Development Workflow

1. Create feature branch (never push directly to main)
2. Run pre-commit hooks before committing
3. Ensure tests pass: `pytest -m 'not gpu'` for CPU, `pytest` for full suite
4. Run linting: `ruff check .` and formatting: `black .`
5. Open pull request for review

## File Patterns to Recognize

- `*.yaml` files in `mcli/` and `runai/` are training configurations
- `scripts/data_prep/yamls/*.yml` are data processing configurations
- Files ending in `_v1`, `_v2` indicate dataset/model versions
- `.dataset` directories are Hugging Face datasets
- `_MDS` suffixed directories are streaming MDS format datasets