# v2 Dataset Preparation Scripts

This folder contains scripts for the first major update to the pretraining dataset for MosaicFM.
This release includes data from CellXGene (~60M cells) as well as Vevo's Tahoe-100M dataset.


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

## Step 2.2: Download and process scPerturb Data

[scPerturb](https://www.nature.com/articles/s41592-023-02144-y) is a collection of 44 single-cell perturbation datasets.
We first downloaded the collection of adata files from the scperturb Zenodo repository [here](https://zenodo.org/records/7041849) 
and then subsetted the data to only include genes that are in the vocabulary generated in Step 1. 
Since different datasets use different keys for indexing genes (e.g. gene names, ensembl IDs, etc.) 
or have data from non-human cells some of them are filtered out. 
We use a minimum filtering criteria that at least 60% of the genes in the dataset should be indexable to our vocabulary.
```bash
python process_scperturb.py scperturb.yaml
```
After filtering, we get 37 datasets with a total of 6.48M cells. 
The top 10 datasets by cell count are:

| Dataset Name | Number of Cells | Number of Genes |
|--------------|-----------------|-----------------|
| ReplogleWeissman2022_K562_gwps | 1989578 | 8242 |
| JoungZhang2023_atlas | 1145823 | 22971 |
| ReplogleWeissman2022_K562_essential | 310385 | 8555 |
| TianKampmann2019_iPSC | 275708 | 32839 |
| NadigOConner2024_jurkat | 262956 | 8875 |
| ReplogleWeissman2022_rpe1 | 247914 | 8739 |
| FrangiehIzar2021_RNA | 218331 | 17453 |
| GasperiniShendure2019_atscale | 207324 | 12786 |
| McFarlandTsherniak2020 | 182875 | 30867 |
| TianKampmann2019_day7neuron | 182790 | 32839 |


## Step 2.3: Download and process Vevo Data
For this release we used the portion of the Tahoe-100M dataset that passes "full" filters. 
For v1 of the dataset, we do not store any additional columns such as cell-line, plate or treatment information. 
These could be added in a future release if needed for model training. Furthermore, we do not aggregate the data based on 
any information about replication structure (eg: plate, batch ).

## Step 2.4: Convert datasets to HuggingFace Arrow format

```bash
HF_HOME=<PATH ON PVC> python make_hf_dataset.py <PATH TO DATASET YAML>
```

Specifying the HF_HOME variable to be a path on PVC (such as "/vevo/cache") is necessary to ensure that the temporary 
cache doesn't blow up ephemeral storage when using a pod-based environment such as RunAI. 
Keep in mind that the memory usage of this script will keep growing up to 1TB and then stabilize around there.

The HF dataset format allows for quickly loading data into memory for training. 
While this can be used directly when training locally, for cloud training we perform one additional step to convert the 
dataset to compressed MDS shards.

## Step 2.5: Convert datasets to MDS format

```bash
python generate_mds.py <PATH TO DATASET YAML>
```
> [!NOTE]
> Sometimes inside Docker multiprocessing doesn't work correctly. In that case, copy over the 
script to a jupyter notebook and try again.

After this step the MDS file can be uploaded to S3. 
