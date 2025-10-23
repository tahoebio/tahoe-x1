# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
import os

import wandb
from omegaconf import OmegaConf as om

from tahoex.tokenizer import GeneVocab
from tahoex.utils import download_file_from_s3_url

# ============================================
# Configuration - Update these for your model
# ============================================
model_name = "<your_model_name>"  # e.g., "tx-3b-prod"
wandb_id = "<your_wandb_run_id>"  # e.g., "mygjkq5c" - find this in your WandB run URL
wandb_project = "<your_wandb_project>"  # e.g., "vevotx/tahoex"
save_dir = "<your_output_path>"  # e.g., "/tahoe/tahoex/checkpoints/"
default_vocab_url = "s3://tahoe-hackathon-data/MFM/vevo_v2_vocab.json"  
# ============================================

api = wandb.Api()
run = api.run(f"{wandb_project}/{wandb_id}")
yaml_path = run.file("config.yaml").download(replace=True)

with open("config.yaml") as f:
    yaml_cfg = om.load(f)
om.resolve(yaml_cfg)
model_config = yaml_cfg.pop("model", None)["value"]
collator_config = yaml_cfg.pop("collator", None)["value"]
vocab_config = yaml_cfg.pop("vocabulary", None)["value"]
if vocab_config is None:
    vocab_remote_url = default_vocab_url
else:
    vocab_remote_url = vocab_config["remote"]


download_file_from_s3_url(
    vocab_remote_url,
    local_file_path="vocab.json",
)

# Step 1 - Add special tokens to the vocab
vocab = GeneVocab.from_file("vocab.json")
special_tokens = ["<pad>", "<cls>", "<eoc>"]

for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)
if collator_config.get("use_junk_tokens", False):
    # Based on Karpathy's observation that 64 is a good number for performance
    # https://x.com/karpathy/status/1621578354024677377?s=20
    original_vocab_size = len(vocab)
    remainder = original_vocab_size % 64
    if remainder > 0:
        junk_tokens_needed = 64 - remainder
        for i in range(junk_tokens_needed):
            junk_token = f"<junk{i}>"
            vocab.append_token(junk_token)

save_dir = f"{save_dir}/{model_name}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
vocab.save_json(f"{save_dir}/vocab.json")

## Step 2: Store PAD Token ID in collator config
collator_config.pad_token_id = vocab["<pad>"]
## Step 3: Update model config with Vocab Size
model_config.vocab_size = len(vocab)
## Step 4: Set generate_training=False for inference
model_config.use_generative_training = False

## Step 5: Add precision and wandb ID to config
model_config["precision"] = yaml_cfg["precision"]["value"]
model_config["wandb_id"] = f"{wandb_project}/{wandb_id}"

om.save(config=model_config, f=f"{save_dir}/model_config.yml")
om.save(config=collator_config, f=f"{save_dir}/collator_config.yml")
