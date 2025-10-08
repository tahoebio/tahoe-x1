# MSigDB Benchmark Pipeline

This folder provides a YAML-driven workflow for generating gene embeddings, building an AnnData object, running the MLP benchmark and visualizing results.

## Setup

### Environment Requirements
- **For embedding generation** (step 2): Use the main TahoeX environment
- **For benchmark and visualization** (steps 3-5): Install the MSigDB-specific environment:

```bash
conda env create -f scripts/msigdb/environment.yml
conda activate msigdb
```

## Scripts Overview
- `get_msigdb_embs.py` - Generate gene embeddings (GE, TE, EA modes)
- `generate_anndata.py` - Build AnnData object from embeddings and signatures  
- `run_benchmark.py` - Train MLP predictor and evaluate performance
- `visualize_results.py` - Create plots from benchmark results
- `sig_predictor.py` - Core MLP predictor implementation

## Usage

All scripts support command-line overrides: `python msigdb/*.py config.yaml --key=value --nested.key=value`

Example: `python scripts/msigdb/get_msigdb_embs.py config.yaml --embeddings_path=./custom_embs --batch_size=16`

### 1. Configure paths
Edit `config.yaml` to point to your input data and desired outputs.

### 2. Generate embeddings
Create TE (transformer context-free), GE (gene encoder) and EA (expression aware) embeddings:

```bash
python scripts/msigdb/get_msigdb_embs.py scripts/msigdb/config.yaml --embeddings_path="./new"
```

### 3. Build AnnData
Combine embeddings with MSigDB signatures into a single AnnData object:

```bash
python scripts/msigdb/generate_anndata.py scripts/msigdb/config.yaml
```

### 4. Run benchmark
Train the MLP predictor for each embedding representation and seed defined in the config:

```bash
python scripts/msigdb/run_benchmark.py scripts/msigdb/config.yaml
```

### 5. Visualize results
Aggregate prediction CSVs and create plots with error bars across replicates:

```bash
python scripts/msigdb/visualize_results.py scripts/msigdb/config.yaml
```

## Configuration Tips
- Ensure visualization prediction directory matches benchmark output directory
- Use consistent embedding modes between generation and benchmarking  
- Results are saved per seed in `{output_dir}/seed_{N}/` subdirectories
