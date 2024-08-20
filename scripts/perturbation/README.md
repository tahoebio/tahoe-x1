### Steps to fine-tune

1. Prepare data first by running

```
python scripts/perturbation/prepare_data.py --data_path "/vevo/datasets/perturbation_datasets/" --dataset_name "adamson" --vocab_path "/vevo/scgpt/checkpoints/release/scgpt-70m-1024-fix-norm-apr24-data/vocab.json" --gene_info_path "/vevo/datasets/cellxgene/cellxgene_primary_2024-04-29_MDS/gene_info_2024-04-29.json"
```

2. Change the config and run finetune.py with composer

```
composer scripts/perturbation/finetune.py runai/config_perts/runai_finetune_70m_norman_full_length.yaml 
```


### Path to files
File Names | s3 path
--- | --- 
Norman dataset | s3://vevo-ml-datasets/perturbseq/far-Haotian_processed/norman/
Adamson dataset | s3://vevo-ml-datasets/perturbseq/far-Haotian_processed/adamson/
70m-1024-fix-norm-apr24-data model's vocab.json | s3://vevo-ml-datasets/vevo-scgpt/models/release/scgpt-70m-1024-fix-norm-apr24-data/vocab.json
cellxgene_primary_2024-04-29_MDS gene_info.json | s3://vevo-ml-datasets/vevo-scgpt/datasets/cellxgene_primary_2024-04-29_MDS/gene_info_2024-04-29.json
