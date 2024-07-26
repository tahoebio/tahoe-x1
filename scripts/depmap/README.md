# DepMap benchmarks

This folder contains the scripts required to evaluate models on three benchmarks centered around the DepMap dataset.

1. separate cancer cell lines by tissue of origin.
2. predict gene essentiality in a cell line specific manner.
3. predict whether genes are broadly essential or inessential.

This README explains how to set up and run these benchmarks. For a complete writeup on what these benchmarks involve and results from the first half of 2024, see [here](https://docs.google.com/document/d/1yBAXkhriSCzdmDWeewAEKO3rOC4RFfnw1cuJtDSvIaY/edit?usp=sharing). 

---

### Step 0: set up base folder

The scripts in this folder operate on files organized in a specific directory structure.

```
base-folder             Root directory.
  cell-embs             Contains cell line embeddings.
  gene-embs             Contains gene embeddings.
    npz                 Contains archives of cell line and mean gene embeddings.
    results             Contains results from random forests trained on gene embeddings.
  geneformer            Contains direct inputs and outputs for Geneformer model.
  misc                  Miscellaneous files for benchmarks.
  nvidia-gf-preds       Contains predictions from and files to work with NVIDIA Geneformer models.
  raw                   Raw DepMap and CCLE data.
```

All of these folders need to at least exist for the scripts to work. The following S3 URI contains this base folder, populated with all files from experiments performed in the first half of 2024 (including some unnecessary for the final versions of these benchmarks).

```
s3://vevo-ml-datasets/umair/scgpt-depmap/
```

**If you sync the entire directory, you can skip to step 4 and start evaluating new models.**

**If you sync only the raw data, you need to go through steps 1-3.**

---

### Step 1: generate prerequisite files

If you start with the raw data, there's some preprocessing to do that will create all the files needed for the downstream benchmarks.

```
python generate-prereqs.py --base-path [path]
```

This script requires the following files in the base folder.

```
raw/ccle-counts.gct
raw/depmap-gene-effects.csv
raw/depmap-metadata.csv
raw/scgpt-genes.csv
raw/depmap-gene-dependencies.csv
```

This script will create the following files.

```
counts.h5ad
misc/genes-by-mean-disc.csv
misc/split-cls.csv
misc/split-genes-lt5gt70.csv
geneformer/ccle-nonzero-medians.pkl
geneformer/adata.h5ad
geneformer/tokenized.dataset
geneformer/tokenized-new-medians.dataset
```

---

### Step 2: generate baseline PCA embeddings for all tasks

Once the preprocessing is complete, use the following script to create baseline PCA embeddings for all the benchmark tasks.

```
python generate-embs-pca.py --base-path [path]
```

Some of the downstream scripts rely on the PCA baseline files as templates, so it's important these baseline embeddings exist.

---

### Step 3: generate null model predictions for task #2 and task #3

The null model predictions for task #2 and task #3 can be generated with the following script.

```
python generate-nulls.py --base-path [path]
```

---

### Step 4: generate embeddings for the model being evaluated

When you have a new model to evaluate, use `generate-embs-model.py` to extract the embeddings needed for the DepMap benchmarks. Currently, the script can extract embeddings scGPT models developed internally at Vevo and Geneformer models that follow the Hugging Face framework. It can also operate on inference results from the NVIDIA Geneformer models (see [here](https://github.com/vevotx/vevo-eval/blob/main/wrangle/scgpt-depmap/depmap_nvidia_geneformer.ipynb) for some more details.)

Note that in all the following examples, the model name is something you can choose. File names are based this parameter, and you'll use it in steps 5 and 6. For the most seamless operation, specify the model name with nothing but hyphens. In step 5, and for task #2 and task #3 in step 6, the hyphens will be replaced with underscores.

```
model name: scgpt-70m-verify-script
downstream: scgpt_70m_verify_script
```

To run the script with an scGPT model, you'll need a directory containing `best-model.pt`, `collator_config.yml`, `model_config.yml`, and `vocab.json`. Pass the location of this directory to the script using the `--model-path` argument.

```
python generate-embs-model.py --base-path [path] --model-type scgpt --model-name [name] --model-path [path]
```

To run the script with a Geneformer model, you'll need a directory containing `config.json`, `pytorch_model.bin`, and `training_args.bin`. Pass the location of this directory to the script using the `--model-path` argument. Additionally, you'll need to pass in a path to a dataset processed by Geneformer's `TranscriptomeTokenizer` class. (You should have at least two tokenized datasets in the `geneformer` subdirectory in the base folder by this point.)

```
python generate-embs-model.py --base-path [path] --model-type gf --model-name [name] --model-path [path] --gf-data-path [path]
```

To operate on inference results from the NVIDIA Geneformer models, you'll need to place `{prefix}_depmap.pkl` and `{prefix}_vocab.json` in the `nvidia-gf-preds` subdirectory in the base folder for the model you want. (If you synced everything from the S3 URI in Step 0, these already exist.)

```
python generate-embs-model.py --base-path [path] --model-type nvidia-gf --model-name [name] --nvidia-gf-data-prefix [prefix]
```

---

### Step 5: train random forests for task #2 and task #3

Once you've generated embeddings for the model you want to evaluate, you'll need to train random forest models, specifically for task #2 and task #3. The script `rf.py` will train a single random forest (either regressor or classifier) for a given fold of the given task and save the results. For convenience, you can use shell scripts for the task you want, which will run `rf.py` with the appropriate parameters multiple times to complete the full task.

Use `rf-marginal-task.sh` for task #2.

```
bash rf-marginal-task.sh [base folder path] [embedding name, which will usually be model_name-mean-lt5gt70-bin] [number of cores to use, 8 is usually good]
```

Use `rf-cl-specific-task-[1,2,3]of3.sh` for task #3. The three scripts are intended to be run in parallel to go through all strata.

```
bash rf-cl-specific-task-[1,2,3]of3.sh [base folder path] [embedding name, usually model name with underscores] [number of cores to use, 16-32 is usually good]
```

---

### Step 6: evaluate and compare models

Once the random forests have been trained, you can go through the `evaluate-models.ipynb` notebook to compute and plot results for each of the three tasks. See the notebook for more details.