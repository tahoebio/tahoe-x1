# STATE TRAnSITION post-training

This folder contains training scripts and data splits for post-training with STATE TRANSITION.

## Data formatting and organization

Training data needs to be converted to AnnData format.
The AnnData requires a field `.obsm['X_hvg']` which contains a HVG subset of genes in dense and pre-processed units.
For training on Tahoe-100M, it is crucial for successful training that data is split into individual `.h5ad` per plate.

TOML files defining the splits need to point to the directory containing the files.

The individual `.sh` run scripts should point to the TOML files which in turn point to the data.

For adaptation scripts, a previously trained checkpoint need to be provided and the path to it need to be specified in the training run settings in the run script.

## Launching training runs

Once data is organized, training can be launched by running the shell script. For example:
```bash
chmod +x train_tahoe_state_tx_20250812_mfm.sh
./train_tahoe_state_tx_20250812_mfm.sh
```
