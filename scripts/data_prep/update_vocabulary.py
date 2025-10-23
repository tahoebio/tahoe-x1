# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
import json
import logging
import os

import scanpy as sc
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from tahoe_x1.tokenizer import GeneVocab

# Logging setup
log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)


def main(cfg: DictConfig):
    adata_reference = sc.read_h5ad(cfg.vocab.reference_adata)
    vocab_col = cfg.vocab.use_col
    id_to_gene = dict(zip(adata_reference.var[vocab_col], adata_reference.var.index))
    id_to_gene_path = os.path.join(
        cfg.vocab.output_root,
        cfg.vocab.id_to_gene_output_file,
    )
    with open(id_to_gene_path, "w") as f:
        json.dump(id_to_gene, f, indent=2)
    log.info(f"Gene to ID mapping saved to {id_to_gene_path}")
    new_vocab = GeneVocab(
        gene_list_or_vocab=list(id_to_gene.keys()),
        specials=cfg.vocab.special_tokens,
    )
    original_vocab_size = len(new_vocab)
    if cfg.vocab.add_junk_tokens:
        remainder = original_vocab_size % 64
        if remainder > 0:
            junk_tokens_needed = 64 - remainder
            for i in range(junk_tokens_needed):
                junk_token = f"<junk{i}>"
                new_vocab.append_token(junk_token)
    log.info(f"Vocabulary size: {len(new_vocab)}")
    output_root = cfg.vocab.output_root
    vocab_save_path = os.path.join(output_root, cfg.vocab.output_file)
    new_vocab.save_json(vocab_save_path)
    log.info(f"Saved new vocab to {vocab_save_path}")
    log.info("Script completed succesfully.")


if __name__ == "__main__":
    import sys

    yaml_path = sys.argv[1]
    log.info(f"Loading configuration from {yaml_path}...")

    with open(yaml_path) as f:
        cfg = om.load(f)
    om.resolve(cfg)

    main(cfg)
