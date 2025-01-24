<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN -->
<p align="center">
  <a href="https://github.com/vevotx/mosaicfm">
    <picture>
      <img alt="vevo-therapeutics" src="./assets/vevo_logo.png" width="95%">
    </picture>
  </a>
</p>
<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END -->

<p align="center">
<a href="https://github.com/astral-sh/ruff"><img alt="Linter: Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
<a href="https://github.com/vevotx/mosaicfm/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg">
    </a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>
<br />

# MosaicFM

This is the internal codebase for the **MosaicFM** series of single-cell RNA-seq foundation models 
developed by Vevo Therapeutics. Our repository follows a similar structure to [llm-foundry](https://github.com/mosaicml/llm-foundry/tree/main) 
and imports several utility functions from it. Please follow the developer guidelines if you are 
contributing to this repository. For main results and documentation, please refer to the results section. 
If you are looking to train or finetune a model on single-cell data, please refer to the training section.

The repository is organized as follows:
* `mosaicfm/` contains the building blocks for the MosaicFM models.
  * `mosaicfm/model/blocks` Building block modules that may be used across models
  * `mosaicfm/model/model` Full architectures subclassed from [ComposerModel](https://docs.mosaicml.com/projects/composer/en/latest/composer_model.html)
  * `mosaicfm/tasks/` Helper functions to use in downstream applications, such as embedding extraction.
  * `mosaicfm/tokenizer` Vocabulary building and tokenization functions.
  * `mosaicfm/data` Data loaders and collators
  * `mosaicfm/utils` Miscellaneous utility functions such to dowload files from s3 etc.
* `scripts/` contains scripts to train/evaluate models and to build datasets.
  * `scripts/train.py` Script to train a model. Accepts a yaml file or command line arguments for specifying job parameters.
  * `scripts/prepare_for_inference.py` Script to save a model for inference by packaging it with the vocabulary and saving metadata.
  * `scripts/depmap` Scripts to run the depmap benchmark.
* `mcli` yaml files to configure and launch runs on the MosaicML platform.
* `runai` yaml files to configure and launch runs on RunAI.
* `tutorials` Notebooks to demonstrate some applications of the models.

## Hardware and Software Requirements

We have tested our code on NVIDIA A100 and H100 GPUs with CUDA 12.1. 
At the moment, we are also restricted to use a version of llm-foundry no later v0.6.0, since support for the triton 
implementation of flash-attention was removed in [v0.7.0](https://github.com/mosaicml/llm-foundry/releases/tag/v0.7.0).

We support launching runs on the MosaicML platform as well as on local machines through RunAI.
The recommended method for using MosaicFM is to use the pre-built [vevotx/ml-scgpt](https://hub.docker.com/repository/docker/vevotx/ml-scgpt/) docker image.

Currently, we have the following images available:

| Image Name                     | Base Image | Description         |
|--------------------------------|-------------|-------------------------------------------------|
| [`vevotx/ml-scgpt:shreshth`](https://github.com/vevotx/vevo-docker/tree/main/ml_docker_vevo_scgpt) | docker.io/mosaicml/llm-foundry:2.2.1_cu121_flash2-813d596 | Image used for MosaicFM-1.3B (July 2024 release) |

## Installation

### With docker
```shell
git clone https://github.com/vevotx/mosaicfm.git
cd mosaicfm
pip install -e .
```
### Without docker
```shell
git clone https://github.com/vevotx/mosaicfm.git 
cd mosaicfm
mamba env create -f envs/mosaicfm_env.yml
mamba activate mosaicfm
pip install -e . --no-deps # Inside the mosaicfm directory
```
> [!NOTE]  
> If you are on an H100 GPU you may see `'sm_90' is not a recognized processor for this target (ignoring processor)`. This is expected and safe to ignore.


## Datasets

The following datasets are used for training and evaluation:

| Dataset Path                             | Description                                                                                                     |
|------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| `s3://vevo-ml-datasets/vevo-scgpt/datasets/cellxgene_primary_2024-04-29_MDS/`     | MDS dataset comprising ~45M cells from Apr 2024 release by CellxGene and Vevo dataset 35 (resistance-is-futile) |
| `s3://vevo-ml-datasets/vevo-scgpt/datasets/cellxgene_primary_2023-12-15_MDS_v2/` | MDS dataset comprising ~34M cells from Dec 2023 release by CellxGene.                                           |
 |`s3://vevo-ml-datasets/umair/scgpt-depmap/`                        | Root folder containing Depmap dataset and model predictions                                                     |
|`s3://vevo-drives/drive_3/ANALYSIS/analysis_107/`                  | Root folder MSigDB data and model predictions                                                                   |

## Pre-trained Models

| Model Name                  | Run Name                           | Path to Checkpoints                                           | WandB id |
|-----------------------------|------------------------------------|---------------------------------------------------------------|----------|
| **MosaicFM-1.3B**           | scgpt-1_3b-2048-prod               | `s3:/vevo-scgpt/models/scgpt-1_3b-2048-prod/`                 | lv6jl8kl |
| **MosaicFM-70M**            | scgpt-70m-1024-fix-norm-apr24-data | `s3:/vevo-scgpt/models/scgpt-70m-1024-fix-norm-apr24-data/`    | 55n5wvdm |
| **MosaicFM-25M**            | scgpt-25m-1024-fix-norm-apr24-data | `s3:/vevo-scgpt/models/scgpt-25m-1024-fix-norm-apr24-data/`    | bt4a1luo |
| **MosaicFM-9M**             | sscgpt-test-9m-full-data           | `s3:/vevo-scgpt/models/scgpt-test-9m-full-data/`              | di7kyyf1 |

## Results
Links to evaluations and benchmarks are provided below:
 - [Depmap](scripts/depmap/README.md)
 - [MSigDB](https://github.com/vevotx/shreshth_sandbox/tree/main/analysis/04_msigdb_benchmark/README.md)

Please refer to our  technical report  for detailed results and analysis: [Internal Link](https://drive.google.com/drive/u/1/folders/1KeAXZ9zNYh4uHbLL5XUMmreAkHXW4yXo)

## Developer Guidelines
We use the black code style and the Ruff linter to mantain consistency across contributions.
Please set-up `pre-commit` and run the repository level hooks before committing any changes.
Please do not push to master directly. Create a new branch and open a pull request for review.
To set up pre-commit hooks, run the following command:
```shell
pip install pre-commit
pre-commit install
pre-commit run --all-files # Before committing
```
We also encourage new contributions to use type annotations and docstrings for functions and classes. In the future we
will add `pyright` and `pydocstyle` checks to the pre-commit hooks. We encourage the use of Google style docstrings.

If you will be launching any training/evaluation runs, please also make sure you have access to `s3`, `wandb` 
and `mcli`/`runai` by reaching out on #infrastructure.

## Acknowledgements
We would like to thank the developers of the following open-source projects:
 - [scGPT](https://github.com/bowang-lab/scGPT/tree/main)
 - [llm-foundry](https://github.com/mosaicml/llm-foundry)
 - [streaming](https://github.com/mosaicml/streaming)
 - [datasets](https://github.com/huggingface/datasets)