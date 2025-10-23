# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
import os

import wandb
from omegaconf import OmegaConf as om

from tahoe_x1.tokenizer import GeneVocab
from tahoe_x1.utils import download_file_from_s3_url

model_name = "tx-3b-prod"
wandb_id = "mygjkq5c"
api = wandb.Api()
run = api.run(f"vevotx/tahoe-x1/{wandb_id}")
yaml_path = run.file("config.yaml").download(replace=True)

with open("config.yaml") as f:
    yaml_cfg = om.load(f)
om.resolve(yaml_cfg)
model_config = yaml_cfg.pop("model", None)["value"]
collator_config = yaml_cfg.pop("collator", None)["value"]
vocab_config = yaml_cfg.pop("vocabulary", None)["value"]
if vocab_config is None:
    vocab_remote_url = "s3://tahoe-hackathon-data/MFM/vevo_v2_vocab.json"
else:
    vocab_remote_url = vocab_config["remote"]


download_file_from_s3_url(
    vocab_remote_url,
    local_file_path="vocab.json",
)

save_dir = f"/tahoe/tahoe_x1/checkpoints/release/{model_name}"  # Change this to the path where you want to save the model

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
model_config["wandb_id"] = f"vevotx/tahoe-x1/{wandb_id}"

om.save(config=model_config, f=f"{save_dir}/model_config.yml")
om.save(config=collator_config, f=f"{save_dir}/collator_config.yml")
