#!/bin/bash

# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
# Train State Transition model with TahoeX embeddings on Tahoe data
# This script implements a fewshot scenario with drug-cell (DC) splits

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/tahoe_5_holdout/generalization_converted_cell_lines_3b.toml"
OUTPUT_DIR="/tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/experiments"
EXPERIMENT_NAME="tahoe_state_tx_$(date +%Y%m%d_%H%M%S)_MFM3B_hvg_full"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting State Transition training..."
echo "Config file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR/$EXPERIMENT_NAME"
echo "Timestamp: $(date)"

# Navigate to state directory for training
cd /tahoe/drive_3/ANALYSIS/analysis_190/Code/state

# Train the State Transition model
state tx train \
    data.kwargs.toml_config_path="$CONFIG_FILE" \
    data.kwargs.embed_key="tx-70m-merged" \
    data.kwargs.output_space="gene" \
    data.kwargs.num_workers=12 \
    data.kwargs.pert_col="drugname_drugconc" \
    data.kwargs.cell_type_key="cell_line_id" \
    data.kwargs.control_pert="DMSO_TF" \
    data.kwargs.batch_col="plate" \
    training.wandb_track=true \
    training.batch_size=64 \
    training.lr=1e-4 \
    training.max_steps=300000 \
    training.val_freq=4000 \
    training.ckpt_every_n_steps=50000 \
    model.kwargs.cell_set_len=256 \
    model.kwargs.residual_decoder=false \
    model=tahoe_llama_212693232 \
    wandb.tags="[tahoe,fewshot,drug_cell_split,tahoex_3b_merged]" \
    wandb.project="state_tx_tahoe" \
    +wandb.name="$EXPERIMENT_NAME" \
    ++wandb.entity="vevotx" \
    output_dir="$OUTPUT_DIR" \
    name="$EXPERIMENT_NAME"

echo "Training completed!"
echo "Model saved in: $OUTPUT_DIR/$EXPERIMENT_NAME"
