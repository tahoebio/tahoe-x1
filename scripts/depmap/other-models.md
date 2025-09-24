### Geneformer

We prepared the dataset by passing `counts.h5ad` through the [`TranscriptomeTokenizer`](https://geneformer.readthedocs.io/en/latest/geneformer.tokenizer.html) class from the Geneformer package.

To extract cell line embeddings and mean gene embeddings, we passed the prepared dataset through the [`EmbExtractor`](https://geneformer.readthedocs.io/en/latest/geneformer.emb_extractor.html) class from the Geneformer package with `emb_mode` respectively set to "cls" and "gene".

To extract contextual gene embeddings, we manually iterated over minibatches from the prepared dataset after loading using [`load_and_filter`](https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/perturber_utils.py#L26). We prepared each minibatch using [`pad_tensor_list`](https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/perturber_utils.py#L700) and [`gen_attention_mask`](https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/perturber_utils.py#L736) and then passed it through the Geneformer model, which we loaded with [`load_model`](https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/perturber_utils.py#L112). For each minibatch, we iterated over the last layer of `hidden_states` from the model output, indexed with [`quant_layers`](https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/perturber_utils.py#L221), and saved each vector as the contextual gene embedding for the corresponding input gene (if the `input_id` was not a special token).

---

### scGPT

To extract cell line embeddings, we passed `counts.h5ad` through the `embed_data` function from the [`scgpt.tasks.cell_emb`](https://scgpt.readthedocs.io/en/latest/scgpt.tasks.html#scgpt.tasks.cell_emb.embed_data) module.

To extract contextual gene embeddings, we manually loaded the scGPT model with [`load_pretrained`](https://scgpt.readthedocs.io/en/latest/scgpt.utils.html#scgpt.utils.util.load_pretrained) and iterated over the CCLE dataset using scGPT's [`DataCollator`](https://scgpt.readthedocs.io/en/latest/scgpt.html#module-scgpt.data_collator). We passed each batch through the scGPT model using `model._encode` and saved each vector from the resulting embeddings as the contextual gene embedding for the corresponding input gene (if the `input_id` was not a special token).

We mean-pooled the available contextual embeddings for each gene to create mean gene embeddings.

---

### UCE

We prepared `counts.h5ad` by setting `var_names` to correspond to gene names and scaling counts with `sc.pp.normalize_total` so that every cell line had 10,000 counts. We then used [`eval_single_anndata.py`](https://github.com/snap-stanford/UCE) to generate UCE embeddings for the prepared AnnData with the 4 layer and 33 layer models.

We did not attempt to extract gene embeddings from UCE.

---

### TranscriptFormer

We used the `transcriptformer inference` [CLI](https://github.com/czi-ai/transcriptformer) to generate embeddings as follows.

```bash
# cell line embeddings
transcriptformer inference \
    --checkpoint-path ./checkpoints/{tf_sapiens, tf_exemplar, tf_metazoa} \
    --data-file <path to CCLE AnnData> \
    --output-path <path to output directory> \
    --emb-type cell \
    --config-override model.data_config.normalize_to_scale=10000

# contextual gene embeddings
transcriptformer inference \
    --checkpoint-path ./checkpoints/{tf_sapiens, tf_exemplar, tf_metazoa} \
    --data-file <path to CCLE AnnData> \
    --output-path <path to output directory> \
    --emb-type cge \
    --config-override model.data_config.normalize_to_scale=10000
```

We mean-pooled the available contextual embeddings for each gene to create mean gene embeddings.

---

### STATE

We patched the [STATE code](https://github.com/ArcInstitute/state) as follows.

```
##### src/state/emb/data/loader.py #####
@@ -6,6 +6,7 @@ import torch.nn.functional as F
 import functools
 import numpy as np
 
+import pickle
 from typing import Dict
 
 from torch.utils.data import DataLoader
@@ -195,6 +196,11 @@ class FilteredGenesCounts(H5adSentenceDataset):
             # compute its embedding‐index vector
             esm_data = self.protein_embeds or torch.load(emb_cfg.all_embeddings, weights_only=False)
             valid_genes_list = list(esm_data.keys())
+
+            # save the valid gene list
+            with open("<output_dir>/valid_genes_list.pkl", "wb") as f:
+                pickle.dump(valid_genes_list, f)
+
             # make a gene→global‐index lookup
             global_pos = {g: i for i, g in enumerate(valid_genes_list)}
 
@@ -296,6 +302,9 @@ class VCIDatasetSentenceCollator(object):
 
         print(len(self.global_to_local))
 
+        # keep track of batch number
+        self.batch_num = 0
+
     def __call__(self, batch):
         num_aug = getattr(self.cfg.model, "num_downsample", 1)
         if num_aug > 1 and self.training:
@@ -411,6 +420,11 @@ class VCIDatasetSentenceCollator(object):
                 batch_sentences_counts[i, :] = cell_sentence_counts
             i += 1
 
+        # save cell and gene indices
+        np.save(f"<output_dir>/batch_{self.batch_num}_cell_idxs.npy", idxs.numpy())
+        np.save(f"<output_dir>/batch_{self.batch_num}_gene_idxs.npy", batch_sentences[:, 1:].numpy())
+        self.batch_num += 1
+
         return (
             batch_sentences[:, :max_len],
             Xs,

##### src/state/emb/nn/model.py #####
@@ -210,6 +210,9 @@ class StateEmbeddingModel(L.LightningModule):
         else:
             self.dataset_token = None
 
+        # keep track of batch number processed
+        self.batch_num = 0
+
     def _compute_embedding_for_batch(self, batch):
         batch_sentences = batch[0].to(self.device)
         X = batch[1].to(self.device)
@@ -401,6 +404,12 @@ class StateEmbeddingModel(L.LightningModule):
         embedding = gene_output[:, 0, :]  # select only the CLS token.
         embedding = nn.functional.normalize(embedding, dim=1)  # Normalize.
 
+        # save the per-token embeddings
+        token_embs = gene_output[:, 1:, :]
+        token_embs = nn.functional.normalize(token_embs, dim=2)
+        np.save(f"<output_dir>/batch_{self.batch_num}_embs.npy", token_embs.cpu().numpy())
+        self.batch_num += 1
+
         # we must be in train mode to use dataset correction
         dataset_emb = None
         if self.dataset_token is not None:
```

We then used the `state emb transform` CLI to run the CCLE AnnData through [SE-600M](https://huggingface.co/arcinstitute/SE-600M) as follows.

```bash
state emb transform \
  --model-folder <path to model folder \
  --checkpoint <path to model checkpoint> \
  --input <path to CCLE AnnData> \
  --output <path to output>
```

This saved the cell line embeddings to the path specified in the command, and the contextual gene embeddings and related data to the paths specified in the patched code. We then processed the contextual gene embeddings using the saved `valid_gene_list.pkl` to map each vector in `_embs.npy` to the correct cell line and gene with `_cell_idxs.npy` and `_gene_idxs.npy`. We mean-pooled the available contextual embeddings for each gene to create mean gene embeddings.