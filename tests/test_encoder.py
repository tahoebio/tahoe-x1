# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
import importlib.util
from pathlib import Path

import torch

_blocks_path = Path(__file__).resolve().parent.parent / "mosaicfm" / "model" / "blocks.py"
spec = importlib.util.spec_from_file_location("blocks", _blocks_path)
blocks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blocks)
SCGPTBlock = blocks.SCGPTBlock
SCGPTEncoder = blocks.SCGPTEncoder
blocks.gen_flash_attn_padding_info = lambda **kwargs: None


def _build_encoder(embed_dim: int, n_heads: int, device: torch.device) -> SCGPTEncoder:
    layer = SCGPTBlock(d_model=embed_dim, n_heads=n_heads, expansion_ratio=1).to(device)
    encoder = SCGPTEncoder(encoder_layer=layer, num_layers=3).to(device)
    encoder.eval()
    return encoder


def test_encoder_mask_equivalence_pcpt_only():
    n_cells = 2
    n_heads = 4
    embed_dim = 32 * n_heads
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    encoder = _build_encoder(embed_dim, n_heads, device)

    pcpt_total_embs = torch.rand(n_cells, n_genes, embed_dim).to(device)
    gen_total_embs = torch.rand(n_cells, n_genes * 2, embed_dim).to(device)

    total_embs = torch.cat([pcpt_total_embs, gen_total_embs], dim=1)
    gen_mask = torch.zeros(n_cells, total_embs.shape[1], dtype=torch.bool).to(device)
    gen_mask[:, n_genes:] = True

    # test forward, run with torch amp
    with torch.cuda.amp.autocast():
        output = flash_gpt_generator(total_embs, gen_mask=gen_mask)
    output1, output2 = output[:, :n_genes, :], output[:, n_genes:, :]
    assert output1.shape == (n_cells, n_genes, embed_dim)
    assert output2.shape == (n_cells, n_genes * 2, embed_dim)

    # test only pcpt_total_embs
    pcpt_mask = torch.zeros(n_cells, n_genes, dtype=torch.bool).to(device)
    with torch.cuda.amp.autocast():
        output1 = flash_gpt_generator(pcpt_total_embs, gen_mask=pcpt_mask)
    assert output1.shape == (n_cells, n_genes, embed_dim)
