# Data preparation

> [!NOTE]
> These instructions outline the steps followed for the 2024 release of MosaicFM.
> For the datasets processing pipeline for MosaicFM-v2 please refer to the **datasets_v2** folder

This folder contains scripts for converting single-cell 
data in the adata format from multiple sources into the MDS format used by our training workflow.

## Dataset List


| Dataset                             | Description                                                                                                                  | s3 path                                                               |
|-------------------------------------|------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| Resistance is Futile v1 (MDS)       | Training data tokenized using the vocab for MosaicFM-1.3B. Contains a train and eval split segregated by drug                | s3://vevo-ml-datasets/vevo-scgpt/datasets/resistance_is_futile_35_MDS_v1/ |
| Adamson v1 (MDS)                    | Adamson data tokenized using the MosaicFM-1.3B vocab. Containes train, validation and test splits segregated by perturbation | s3://vevo-ml-datasets/vevo-scgpt/datasets/adamson_v1 |
| Norman v1 (MDS)                     | Norman data tokenized using the MosaicFM-1.3B vocab. Containes train, validation and test splits segregated by perturbation  | s3://vevo-ml-datasets/vevo-scgpt/datasets/norman_v1  |
| Replogle RPE1 v1 (MDS)              | Replogle RPE1 data tokenized using the MosaicFM-1.3B vocab. Containes train, validation and test splits segregated by perturbation | s3://vevo-ml-datasets/vevo-scgpt/datasets/replogle_rpe1_v1 |
| Replogle K562 v1 (MDS)              | Replogle K562 data tokenized using the MosaicFM-1.3B vocab. Containes train, validation and test splits segregated by perturbation | s3://vevo-ml-datasets/vevo-scgpt/datasets/replogle_k562_v1 |
| Resistace is Futile v1 (HF dataset) | Resistance is futile data tokenized using vocab for MosaicFM-1.3B. Not split into train/test                                 | s3://vevo-ml-datasets/vevo-scgpt/datasets/resistance_is_futile_full.dataset |
| Adamson (HF dataset)                | Adamson dataset mapped using MosaicFM 1.3B vocab in the Huggingface datasets format. Not split into train/test               | s3://vevo-ml-datasets/perturbseq/vevo-processed/aidan_filtered/adamson.dataset/ |
| Norman  (HF dataset)                | Norman dataset mapped using MosaicFM 1.3B vocab in the Huggingface datasets format. Not split into train/test                | s3://vevo-ml-datasets/perturbseq/vevo-processed/aidan_filtered/norman.dataset/ |
| Replogle RPE1 (HF dataset)          | Replogle RPE1 dataset mapped using MosaicFM 1.3B vocab in the Huggingface datasets format. Not split into train/test         | s3://vevo-ml-datasets/perturbseq/vevo-processed/aidan_filtered/replogle_rpe1.dataset/ |
| Replogle K562 (HF dataset)          | Replogle K562 dataset mapped using MosaicFM 1.3B vocab in the Huggingface datasets format. Not split into train/test         | s3://vevo-ml-datasets/perturbseq/vevo-processed/aidan_filtered/replogle_k562.dataset/ |

The MDS folders also contain
 - `metadata.json` : Contains the splits and median_library_size for each dataset.
 - `mean_ctrl_log1p.json`: Dictionary mapping each vocab ID to its corresponding mean log1p value in the control cells.
This can also be computed from the dataset quickly as follows:
```python
target_sum = metadata["median_library_size"]
expressions_raw = dataset["expressions_ctrl_raw"]
expressions_normed = torch.log1p((expressions_raw / expressions_raw.sum(axis=1, keepdims=True))*target_sum)
mean_perturbed_log1p = torch.mean(expressions_normed, axis=0)
mean_ctrl_dict = {
    int(gene_id):float(log1p) for (gene_id, log1p) in zip(hf[0]["genes"], mean_perturbed_log1p)
}
 ```



## CellXGene Dataset

Step 1: Download data from CellXGene release into chunks
```shell
python download_cellxgene.py yamls/cellxgene_apr_29_2024.yml
```

(Optionally) Any additional dataset chunks can be added to the same output folder as above 
and they will be merged into the training dataset as long as they have the same columns.

Step 2: Convert h5ad chunks into huggingface dataset format
```shell
python make_dataset.py --adata_dir <ADATA_PATH> --vocab_path <VOCAB_PATH> --output_dir <OUTPUT_PATH>
````
The `--mem-size` flag can be used to limit cache size if running into 
ephemeral storage issues.

Step 3: Merge huggingface dataset chunks into a single dataset
```shell
python concatenate_datasets.py --path <CHUNK_DIR> --dataset_name <DATASET_NAME>
```
The train-test split ratio is hard-coded to 1% in the script. This may be modified if needed.

Step 4: Convert huggingface dataset into MDS format for streaming from S3
```shell
python generate_mds.py --out_root <OUTPUT_PATH> --train_dataset_path <PATH_TO_TRAIN.DATASET> --valid_dataset_path <PATH_TO_VALID.DATASET>
```
The compression parameters and data-field are hard-coded in this script but may be modified if needed.
The output root folder may now be uploaded to S3 and used as a path in the training workflow.

## Vevo Tx - Drug Resistance Dataset

For this task we have considered the drugs screened in the Resistance-if-Futile screen (dataset ID: 35) that 
have a well characterized set of target genes.

The data is processed as follows:
- Preprocessing:
  - Select genes that are present in the MosaicFM vocabulary (matched using Ensemble gene ID, not gene symbol)
  - Select the 5000 most variable genes using Seurat v3 (Note: Seurat v3 expects un-normalized raw counts)
  - Add the 
- Iterate over cell-lines
   - Retrieve cells with drug = "DMSO_TF" as the control group
   - Iterate over drugs that have data in the cell-line (except control)
     - Get the sensitivity from the sensitivity table (growth_rate, growth_rate_bin, growth_rate_mdn)
     - Get the drug-targets from the drug-target table (derived using manual curation)
     - Enumerate cells in that combination (drug, dose, cell-line). Only the highest dose for each drug is selected. 
     - Sample `num_ctrl_samples_to_pair` control cells from the matched control set for the cell-line
     - Store raw counts for the perturbed and control cells for all genes (n_genes=5008)
     - Also record sensitivity, drug-targets, and cell-line information



The heatmap below shows the distribution of drugs and cell-lines in the dataset.

![img](assets/drug_cell_line_heatmap.png)

To reproduce the dataset, run the following command:
```shell
python process_mosaic_sensitivity.py yamls/mosaic_resistance_is_futile.yml
```

An example record in the dataset is show below:

```python
{'growth_rate': [0.0259861653678554],
 'growth_rate_mdn': [0.132619704816021],
 'growth_rate_bin': [1],
 'expressions_ctrl_raw': [array([0., 0., 0., ..., 0., 0., 0.], dtype=float32)],
 'expressions_perturbed_raw': [array([0., 0., 0., ..., 0., 0., 0.], dtype=float32)],
 'perturbation_target_genes': [[8304, 8615]],
 'genes': [array([ 9393, 30891,  7312, ..., 50821, 21581, 50771])],
 'cell_line': ['c_38'],
 'drug': ['Lapatinib ditosylate'],
 'cell_key': ['04_06_31-lib_316'],
 'cell_key_ctrl': ['86_32_33-lib_310']}
```

The dataset is then split by the drug-target. We used Galunisertib and LY3200882 as the test-set since they 
both target TGFBR1. This results in a roughly 80-20 split between the train and test sets.

## PerturbSeq - Adamson et al Dataset

The data is processed as follows:

```shell
python process_perturbseq.py yamls/perturbseq_adamson.yml
```

Example record:
```python
{'depmap_dependency': 1.0,
 'perturbation_edist': 62.28814212111882,
 'perturbation_target_genes': [20254],
 'expressions_ctrl_raw': array([  0.,   0.,   2., ..., 185.,  12., 150.], dtype=float32),
 'expressions_perturbed_raw': array([ 0.,  1.,  1., ..., 56.,  0., 33.], dtype=float32),
 'genes': array([30607, 10954, 12241, ..., 17076, 17079, 17072], dtype=int32),
 'cell_line': 'K562',
 'perturbation_name': 'SLMO2+ctrl'}
```

The DepMap dependency score is used as a proxy for sensitivity to the perturbation.
The e-distance is a measure of the degree of change induced by the perturbation and 
may be seen as a measure of intensity of the signal present in the data.

The median counts per cell before any filtering is performed is *15239* for this dataset.

If the training script requires log1p data the records may be transformed as follows:
```python
target_sum = 15239
expressions_normed = torch.log1p((expressions_raw / expressions_raw.sum(axis=1, keepdims=True))*target_sum)
```

## PerturbSeq - Norman et al Dataset

The data is processed as follows:

```shell
python process_perturbseq.py yamls/perturbseq_norman.yml
```

Example record:
```python
{'depmap_dependency': [0.0941306056201959],
 'perturbation_edist': 58.21759247789265,
 'perturbation_target_genes': [5338],
 'expressions_ctrl_raw': array([ 0.,  1.,  0., ..., 12.,  0.,  0.], dtype=float32),
 'expressions_perturbed_raw': array([ 0.,  0.,  0., ..., 11.,  0.,  2.], dtype=float32),
 'genes': array([24905, 47418, 29728, ..., 17079, 48327, 48331], dtype=int32),
 'cell_line': 'K562',
 'perturbation_name': 'COL2A1+ctrl'}
```

The median counts per cell before any filtering is performed is *13855* for this dataset. Similar to the Adamson dataset, 
transforming the data to log1p may be done as described above.

## PerturbSeq - Replogle RPE1 Dataset

The data is processed as follows:

```shell
python process_perturbseq.py yamls/perturbseq_replogle_rpe1.yml
```

Example record:

```python
{'perturbation_edist': 14.8090,
 'perturbation_target_genes': [5384],
 'expressions_ctrl_raw': array([  1.,   1.,   1.,  ...,  48.,   6., 176.], dtype=float32),
 'expressions_perturbed_raw': array([ 0.,  1.,  0.,  ..., 14.,  0., 46.], dtype=float32),
 'genes': array([19766, 10954, 12241,  ..., 17078, 17079, 17072], dtype=int32),
 'cell_line': 'RPE1',
 'perturbation_name': 'COMTD1+ctrl'}
```

Note that the RPE dataset does not contain a depmap dependency score, since we couldn't map RPE1 to a depmap cell-line.
The median counts per cell before any filtering is performed is *11899* for this dataset. 

## PerturbSeq - Replogle K562 Dataset

The data is processed as follows:

```shell
python process_perturbseq.py yamls/perturbseq_replogle_rpe1.yml
```

Example record:

```python
{'perturbation_edist': 1.6231,
 'perturbation_target_genes': [31002],
 'expressions_ctrl_raw': array([1., 0., 1.,  ..., 3., 0., 0.], dtype=float32),
 'expressions_perturbed_raw': array([0., 0., 0.,  ..., 0., 0., 0.], dtype=float32),
 'genes': array([14169, 12851, 10954,  ..., 48326, 48331, 48328], dtype=int32),
 'cell_line': 'K562',
 'perturbation_name': 'SET+ctrl',
 'depmap_dependency': [0.0144]}
```

The Replogle K562 dataset has a median count of *11306* before any filtering is performed. It is the largest of the
PerturbSeq datasets with a total of around 7M cells (~1.4M unique). This dataset includes a mix of both essential and inessential genes.
The magnitude of the effect (e-distance) is larger for the essential genes. For training the perturbations are not filtered.

# Splits and MDS Generation

To split the HF dataset according to a column, modify the Dataset splits section in the yaml. 
The following parameters are used for example in the Adamson dataset:

```yaml
# Dataset splits
split_column: "perturbation_name"
train_proportion: 0.8
val_proportion: 0.1
test_proportion: 0.1
random_seed: 42
split_save_path: "/vevo/scgpt/datasets/perturbseq/vevo-processed/aidan_filtered/adamson_splits"
```

The splits are then generated using the following command:

```shell
python split_dataset.py yamls/perturbseq_adamson.yml
```
The MDS parameters need to be specified in the MDS section in the yaml. An encoding type 
for each column needs to be specified. Note that pytorch tensors are stored as pkls (any general object). In order to 
load them in a StreamingDataset the argument `allow_unsafe_types=True` needs to be set.

```yaml
mds:
    out_root: "/vevo/scgpt/datasets/perturbseq/vevo-processed/aidan_filtered/MDS/adamson_v1"
    root_dir: ${split_save_path}
    splits:
      - "train"
      - "validation"
      - "test"
    columns:
      depmap_dependency: "pkl"
      perturbation_edist: "float32"
      perturbation_target_genes: "pkl"
      expressions_ctrl_raw: "pkl"
      expressions_perturbed_raw: "pkl"
      genes: "pkl"
      cell_line: "str"
      perturbation_name: "str"
    compression: zstd
    hashes:
      - "sha1"
      - "xxh64"
```
The MDS dataset is then generated using the following command:

```shell
python generate_mds_perturbseq.py yamls/perturbseq_adamson.yml
```


