## 🧬 Generating Cell and Gene Embeddings

There are two main ways to extract embeddings from the Tahoe-x1 model:

### Method A:  `get_batch_embeddings` function
It provides a direct API for embedding extraction without cloning the repo.

```python
import numpy as np
import scanpy as sc
from tahoe_x1.tasks import get_batch_embeddings
from tahoe_x1.utils.util import load_model


# --- Load model ---
composer_model, vocab, model_cfg, coll_cfg = load_model(model_dir="/path/to/model_dir", device)
model = composer_model.model

# --- Load data ---
adata = sc.read_h5ad(adata_path="/path/to/data.h5ad")
adata.var["id_in_vocab"] = [vocab.get(g, -1) for g in adata.var[gene_col]]
valid_genes = adata.var["id_in_vocab"] >= 0
adata = adata[:, valid_genes]
gene_ids = adata.var["id_in_vocab"].astype(int)

# --- Extract embeddings ---
cell_embs, gene_embs = get_batch_embeddings(
    adata=adata,
    model=model,
    vocab=vocab,
    gene_ids=gene_ids,
    model_cfg=model_cfg,
    collator_cfg=coll_cfg,
    batch_size=32, #configure
    max_length=2048, #configure
    return_gene_embeddings=True, #set to False if you only want cell emeddings
)
```


### Method B): `predict_embeddings.py` script
A higher-level script via Composer Trainer.predict() which requires cloning the repository.

1. You can configure the `inference/configs/predict.yaml` and run:

```python inference/predict_embeddings.py inference/configs/predict.yaml```

2. Alternatively, you can use this sample code:

```python
from omegaconf import OmegaConf as om
from inference.predict_embeddings import predict_embeddings

cfg = {
    "model_name": "Tx1-70m",
    "paths": {
        "hf_repo_id": "tahoebio/TahoeX1",
        "hf_model_size": "70m",
        "adata_input": "/path/to/data.h5ad",
    },
    "data": {
        "cell_type_key": "cell_type",
        "gene_id_key": "ensembl_id"
    },
    "predict": {
        "seq_len_dataset": 2048,
        "return_gene_embeddings": False, #set to True if you want gene embeddings
    }
}

cfg = om.create(cfg)
adata = predict_embeddings(cfg)

cell_embs = adata.obsm["Tx1-70m"]
```
