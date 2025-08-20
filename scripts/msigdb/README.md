MsigDB benchmark

# Steps to reproduce:

1. Download the sigs folder from s3 to your pvs:
s3://vevo-drives/drive_3/ANALYSIS/analysis_107/sigs/

2. You also need the gene embeddings in your pvc.
You can find gene embeddings for some of the models in here:
    - s3://vevo-drives/drive_3/ANALYSIS/analysis_107/gene_embs_v2
    - s3://vevo-drives/drive_3/ANALYSIS/analysis_107/gene_embs_v3

    - If the gene embeddings for your desired model doesn't exist you need to generate the mebeddings yourself using either by https://github.com/vevotx/vevo-scgpt-private/blob/main/scripts/get_embeddings.py or https://github.com/vevotx/vevo-scgpt-private/blob/main/scripts/get_msigdb_embs.py. 

    - To generate the embeddings for your own model you first need to prepare your model for inference by https://github.com/vevotx/vevo-scgpt-private/blob/main/scripts/prepare_for_inference.py

    - Example:
    ``` 
        python scripts/prepare_for_inference.py --model_name scgpt-70m-1024-fix-norm-apr24-data --wandb_id 55n5wvdm --save_dir /vevo/scgpt/checkpoints/release/
    ```

    - Remember to copy the best-mpdel.pt to your release foler and upload everything to s3 under releases/.

    - get_embeddings.py generate the context free embeddings (GE and TE)
    
    - Example:
    ```
    python scripts/get_msigdb_embs.py --releases_path /vevo/scgpt/checkpoints/release/ --save_path /vevo/datasets/msigdb_gene_emb_subset/gene_embeds_new/      
    ```
    - get_msigdb_embs.py generates the transformer-based context free and expression aware embeddings (TE and EA). You need to provide one subset of cellxgene as the --input_path. You can download such subset from s3 (s3://vevo-drives/drive_3/ANALYSIS/analysis_107/cellxgene_data/ds_data/rep_1.h5ad.gz).

    - Example:
    ``` 
    python scripts/get_embeddings.py --model_name scgpt-1_3b-2048 --input_path /vevo/datasets/msigdb_gene_emb_subset/cellxgene_subset/rep_1.h5ad.gz 
    ```
    - After creating the embeddings you then need to parse them using parse_embeddings.py
    Note that for all the steps above scgpt-dev environment was used. 
    - Example:
    ``` 
    python analysis/04_msigdb_benchmark/parse_embeddings.py "/vevo/datasets/msigdb_gene_emb_subset/gene_embeddings_new/" ./analysis/04_msigdb_benchmark/gene_embs_v2/
    ```

3. Now generate the environment for running MsigDB task using the environment.yml file
4. Run generate_anndata.py. 
 This code read sigs and embs and create an anndata object which is a specific data structure for single cell genomics data. It reads gene expression embeddings and gene set signatures, integrates them into an AnnData object, optionally filters the data, and saves the results.

    - Example:
    ``` 
    python analysis/04_msigdb_benchmark/generate_anndata.py ./analysis/04_msigdb_benchmark/gene_embs_v2/ ./analysis/04_msigdb_benchmark/sigs/
    ```

    - Note that if you want the results per strata you can provide the sigs for that gene set (e.g. sigs/hallmarks/)

 5.  Run run_benchmark_reps.sh 
    - The script runs benchmark_mlp.py for 4 times and saves the results. benchmark_mlp.py trains a neural network sigpredictor  based on the given data.

    -  Example: 
    ```
    bash run_benchmark_reps.sh gene_embs_v3/embs_adata.h5ad.gz v1
    ```

    - You can similarly run the run_benchmark_reps_per_gene_set.sh by providing the partitioned sigs folder to get msigdb results per gene set.

6. Modify preds_directory in mlp_benchmark/viz_preds.ipynb and run it to compute the AUPRCs per replicate
7. Modify csv_paths in plot_reps.ipynb to refer to the results of viz_preds.ipynb. It will plot the AUPRCs across reps.

- Remember to run scripts on single gpu as pytorch lightning does weird stuff with dist training. 
