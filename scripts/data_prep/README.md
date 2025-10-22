# v2 Dataset Preparation Scripts

This folder contains scripts for the first major update to the pretraining dataset for Tahoe-x1.
This release includes data from CellXGene (~60M cells), scBasecamp (~115M cells) as well as Vevo's Tahoe-100M dataset (~96M).

| Dataset                             | Description                                                                                                               | s3 path                                                               |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| cellxgene_2025_01_21                | Jan 21 release of single cell data from cellxgene tokenized using vevo_v2_vocab. Includes a random 99:1 train:valid split | s3://tahoe-hackathon-data/MFM/cellxgene_2025_01_21_merged_MDS/ |
| scbasecamp (human)                  | scbasecamp data from the arc-virtual-cell atlas (2025-02-25 release)                                                      |s3://tahoe-hackathon-data/MFM/scbasecamp_2025_02_25_MDS_v2/ |
| Tahoe-100M                          | Tahoe-100M data passing full filter, includes metadata for drugs and sample IDs.                                          |s3://tahoe-hackathon-data/MFM/tahoe_100m_MDS_v2/ |

# Step 1: Update Vocab based on Tahoe data
```bash
python update_vocabulary.py cellxgene_2025_01_21.yaml
```
Note that for the new release, the vocabulary is keyed on ensembl ID instead of gene name.
We found that using the gene-names reported by cellxgene led to large mismatches when applied to other datasets, 
whereas the gene-IDs were more reliable.
For this release we use the Tahoe-100M dataset as the base and restrict cellxgene genes to the ones also included 
in Tahoe (which is almost all of them when keyed using gene-IDs).

# Step 2: Download and Prepare Datasets
## Step 2.1: Download CellXGene Data
```bash
python download_cellxgene.py cellxgene_2025_01_21.yaml
```
The January 21 update of the CellXGene dataset contains 59.8M cells across 3 datasets.

## Step 2.3: Download and process Tahoe-100M data
For this release we used the portion of the Tahoe-100M dataset that passes "full" filters. 
For v1 of the dataset, we do not store any additional columns such as cell-line, plate or treatment information. 
These could be added in a future release if needed for model training. Furthermore, we do not aggregate the data based on 
any information about replication structure (eg: plate, batch ).

## Step 2.4: Download and process scBasecamp data
Follow the guide on the [arc-virtual-cell-atlas](https://github.com/ArcInstitute/arc-virtual-cell-atlas/blob/main/scBaseCamp/tutorial-py.ipynb) to 
download the h5ad files from the arc-ctc-scbasecamp bucket. See `scbasecamp.yaml` for processing settings used for this dataset, 
we filtered out cells with fewer than 20 genes and add metadata such as the srx_accession ID.

## Step 3: Convert datasets to HuggingFace Arrow format

```bash
HF_HOME=<PATH ON PVC> python make_hf_dataset.py <PATH TO DATASET YAML>
```

Specifying the HF_HOME variable to be a path on PVC (such as "/vevo/cache") is necessary to ensure that the temporary 
cache doesn't blow up ephemeral storage when using a pod-based environment such as RunAI. 
Keep in mind that the memory usage of this script will keep growing up to 1TB and then stabilize around there.

The HF dataset format allows for quickly loading data into memory for training. 
While this can be used directly when training locally, for cloud training we perform one additional step to convert the 
dataset to compressed MDS shards.

## Step 4: Convert datasets to MDS format

```bash
python generate_mds.py <PATH TO DATASET YAML>
```

After this step the MDS file can be uploaded to S3. 
