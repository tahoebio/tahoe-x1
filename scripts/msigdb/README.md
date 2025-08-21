# MSigDB Benchmark Pipeline

This folder provides a YAML-driven workflow for generating gene embeddings, building an AnnData object, running the MLP benchmark and visualizing results. Existing scripts are kept for backward compatibility but the recommended entrypoints are the new ones below.

## 1. Configure paths
Edit `config.yaml` to point to your input data and desired outputs:

```yaml
model_name: MFM-10m
input_path: path/to/input.h5ad
output_dir: path/to/output
embeddings_path: path/to/embeddings
signatures_path: path/to/signatures
aadata_path: path/to/adata.h5ad
benchmark:
  seeds: [0]
  reps: [GE, TE, EA]
  output_dir: path/to/benchmark_output
visualization:
  pred_dir: path/to/benchmark_output
  output: path/to/plot.png
```

## 2. Generate embeddings
Create TE (transformer context-free), GE (gene encoder) and EA (expression aware) embeddings:

```bash
python scripts/msigdb/generate_embeddings.py scripts/msigdb/config.yaml --modes GE TE EA
```

## 3. Build AnnData
Combine embeddings with MSigDB signatures into a single AnnData object:

```bash
python scripts/msigdb/build_anndata.py scripts/msigdb/config.yaml
```

## 4. Run benchmark
Train the MLP predictor for each embedding representation and seed defined in the config:

```bash
python scripts/msigdb/run_benchmark.py scripts/msigdb/config.yaml
```

## 5. Visualize results
Aggregate prediction CSVs and create simple bar plots:

```bash
python scripts/msigdb/visualize_results.py scripts/msigdb/config.yaml
```

Legacy notebooks (`viz_preds.ipynb`, `plot_reps.ipynb`) and helper scripts remain available but may be removed in the future.
