#!/bin/sh

NPROC=1
QUERY_NAME="pancreas"
DATASET="/home/shreshth/vevo-scGPT/data/cellxgene/pancreas/pancreas_scb/all_counts/"
JOB_NAME="cellxgene_census_${QUERY_NAME}"
LOG_INTERVAL=2000
VALID_SIZE_OR_RATIO=0.03
MAX_LENGTH=1200
per_proc_batch_size=32
LAYERS=12
MODEL_SCALE=8
VOCAB_PATH="/home/shreshth/vevo-scGPT/scgpt/tokenizer/default_census_vocab.json"

torchrun \
    --nproc_per_node=$NPROC \
    pretrain.py \
    --data-source $DATASET \
    --save-dir ./save/$JOB_NAME-$(date +%b%d-%H-%M-%Y) \
    --vocab-path ${VOCAB_PATH} \
    --valid-size-or-ratio $VALID_SIZE_OR_RATIO \
    --max-seq-len $MAX_LENGTH \
    --batch-size $per_proc_batch_size \
    --eval-batch-size $(($per_proc_batch_size * 2)) \
    --nlayers $LAYERS \
    --nheads 8 \
    --embsize $((MODEL_SCALE * 64)) \
    --d-hid $((MODEL_SCALE * 64)) \
    --grad-accu-steps 1 \
    --epochs 6 \
    --lr 0.0001 \
    --warmup-ratio-or-step 10000 \
    --log-interval $LOG_INTERVAL \
    --save-interval $(($LOG_INTERVAL * 3)) \
    --trunc-by-sample \
    --no-cls \
    --no-cce \
    --fp16
