Notebooks stored in directories split according to dataset.

Experiments under `tahoe` directory:
- `vevo_tahoe_wilcoxon_filter_count.ipynb`: filter by genes expressed by n number of cells; use embedding-based method.
- `tahoe_filter_count.py`: python script of `vevo_tahoe_wilcoxon_filter_count.ipynb` notebook.
- `vevo_tahoe_wilcoxon_filter_protein_coding.ipynb`: filter by protein-coding genes; use embedding-based method.
- `vevo_tahoe_wilcoxon_hvg.ipynb`: filter by HVG; use embedding based method.
- `vevo_tahoe_wilcoxon_filter_hvg_attn.ipynb`: filter by HVG; use attention-based method.

`vevo_tahoe_wilcoxon_filter_count.ipynb` and `vevo_tahoe_wilcoxon_filter_hvg_attn.ipynb` calculates results cumulatively by randomly partitioning the adata and iterates through every partition. A png containing a box-and-whisker plot will be saved for every iteration once that iteration contains control profiles. The cumulative partition size and the scGPT rank mean for that iteration will be included in the filename of the png.

Experiments under `norman` directory:
- `vevo_norman_gene_emb.ipynb`: embedding-based method.
- `vevo_norman_gene_mask.ipynb`: embedding-based method with value masking.
- `vevo_norman_mask.py`: python script of `vevo_norman_gene_mask.ipynb` notebook.

Experiments under `adamson` directory (early experiments):
- `vevo_adamson_attn.ipynb`: attention-based method.
- `vevo_adamson_gene_emb.ipynb`: embedding-based method.
- `vevo_adamson_mask.py`: python script of `vevo_adamson_mask.ipynb` notebook (see [commit](https://github.com/tahoebio/mosaicfm/blob/dev_attn_rank/notebooks/adamson/vevo_adamson_mask.ipynb)).
- `vevo_adamson_mask_null.py`: variation of `vevo_adamson_mask.py`, but report ranks of random genes.

scGPT module is required. For embedding and attention-based methods without value masking, please use scGPT perceptual model from [main](https://github.com/bowang-lab/scGPT) branch. For value masking, use scGPT generative model from [dev-temp](https://github.com/bowang-lab/scGPT/tree/dev-temp) branch. Refer to previous [commit](https://github.com/tahoebio/mosaicfm/tree/3a3915ca38e2a1b5167633fd666a3af0ce4ac442) for model used.
