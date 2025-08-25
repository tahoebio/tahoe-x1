# DepMap benchmarks

This folder contains the scripts required to evaluate MosaicFM on three benchmarks centered around the DepMap dataset.

1. separate cancer cell lines by tissue of origin.
2. predict whether genes are broadly essential or inessential.
3. predict gene essentiality in a cell line specific manner.

This README explains how to set up and run these benchmarks. 

---

### Step 0: set up base folder

The scripts in this folder operate on files organized in a specific directory structure.

```
base-folder             Root directory.
  cell-embs             Contains cell line embeddings.
  gene-embs             Contains gene embeddings.
    npz                 Contains archives of cell line and mean gene embeddings.
    results             Contains results from random forests trained on gene embeddings.
  misc                  Miscellaneous files for benchmarks.
  raw                   Raw DepMap and CCLE data.
```

All of these folders need to at least exist for the scripts to work. The following S3 URI contains this base folder.

```
s3://tahoe-hackathon-data/MFM/benchmarks/depmap/
```

This S3 bucket is populated with embeddings and results from the following models.

- PCA (baseline)
- Geneformer (Theodoris Lab, trained on Genecorpus-103M, ~95M parameters)
- Geneformer (NVIDIA BioNeMo, trained on CZ CELLxGENE, ~10M parameters)
- Geneformer (NVIDIA BioNeMo, trained on CZ CELLxGENE, ~100M parameters)
- scGPT
- UCE (4 layer) *cell embeddings only*
- UCE (33 layer) *cell embeddings only*
- TranscriptFormer (Sapiens)
- TranscriptFormer (Exemplar)
- TranscriptFormer (Metazoa)
- STATE
- MosaicFM (~70M parameters)
- MosaicFM (~1.3B parameters)
- MosaicFM (~3B parameters)
- MosaicFM (~3B parameters, training continued with alternate gene encoder)

For more details on how we retrieved embeddings from non-MosaicFM models, please see `other-models.md` in this repository.

**If you sync the entire directory (415GB), you can skip to step 4 and start evaluating new models.**

**If you sync only the raw data (919MB), you need to go through steps 1-3.**

---

### Step 1: generate prerequisite files

If you start with the raw data, there's some preprocessing to do that will create all the files needed for the downstream benchmarks.

```
python generate-prereqs.py --base-path [path]
```

This script requires the following files in the base folder.

```
raw/ccle-counts.gct
raw/depmap-gene-dependencies.csv
raw/depmap-gene-effects.csv
raw/depmap-metadata.csv
raw/scgpt-genes.csv
```

This script will create the following files.

```
counts.h5ad
misc/genes-by-mean-disc.csv
misc/split-cls.csv
misc/split-genes-lt5gt70.csv
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

When you have a new MosaicFM model to evaluate, use `generate-embs-model.py` to extract the embeddings needed for the DepMap benchmarks. Note that in all the following examples, the model name is something you can choose. File names are based this parameter, and you'll use it in steps 5 and 6. For the most seamless operation, specify the model name with nothing but hyphens. In step 5, and for task #2 and task #3 in step 6, the hyphens will be replaced with underscores.

```
model name: mosaicfm-70m-verify-script
downstream: mosaicfm_70m_verify_script
```

To run the script, you'll need a directory containing `best-model.pt`, `collator_config.yml`, `model_config.yml`, and `vocab.json`. Pass the location of this directory to the script using the `--model-path` argument.

```
python generate-embs-model.py --base-path [path] --model-name [name] --model-path [path]
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