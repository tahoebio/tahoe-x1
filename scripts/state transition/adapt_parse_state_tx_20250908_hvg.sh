#!/bin/bash

# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="/tahoe/drive_3/ANALYSIS/analysis_190/Code/state-tahoe-fork/scripts/train_state_tx/parse_pbmc_holdout/donor_p.toml"
OUTPUT_DIR="/tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/experiments"
EXPERIMENT_NAME="parse_state_adaptation_tx_$(date +%Y%m%d_%H%M%S)_HVG_hvg_full"

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

# Train the State Transition model
state tx train \
    data.kwargs.toml_config_path="$CONFIG_FILE" \
    data.kwargs.embed_key="X_hvg" \
    data.kwargs.output_space="gene" \
    data.kwargs.num_workers=12 \
    data.kwargs.pert_col="cytokine" \
    data.kwargs.cell_type_key="donor" \
    data.kwargs.control_pert="PBS" \
    data.kwargs.batch_col="cell_type" \
    training.wandb_track=true \
    training.batch_size=32 \
    training.devices=1 \
    +training.accelerator="gpu" \
    training.lr=1e-4 \
    training.max_steps=150000 \
    training.val_freq=4000 \
    training.ckpt_every_n_steps=50000 \
    model.kwargs.cell_set_len=512 \
    model.kwargs.batch_encoder=true \
    model.kwargs.residual_decoder=false \
    model.kwargs.init_from=/tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/experiments/tahoe_state_tx_20250812_181503_HVG_hvg_full/checkpoints/final.ckpt \
    model=tahoe_llama_212693232 \
    wandb.tags="[parse,fewshot,adaptation,donor,hvg]" \
    wandb.project="state_tx_tahoe" \
    +wandb.name="$EXPERIMENT_NAME" \
    ++wandb.entity="vevotx" \
    output_dir="$OUTPUT_DIR" \
    name="$EXPERIMENT_NAME"

echo "Training completed!"
echo "Model saved in: $OUTPUT_DIR/$EXPERIMENT_NAME"
