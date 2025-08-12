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

    pcpt_real = torch.rand(n_cells, 3, embed_dim, device=device)
    with torch.cuda.amp.autocast():
        old_out = encoder(pcpt_real)

    pad = torch.zeros(n_cells, 2, embed_dim, device=device)
    pcpt_padded = torch.cat([pcpt_real, pad], dim=1)
    pcpt_mask = torch.tensor([[1, 1, 1, 0, 0]] * n_cells, dtype=torch.bool, device=device)

    with torch.cuda.amp.autocast():
        new_out = encoder(pcpt_padded, pcpt_key_padding_mask=pcpt_mask)

    assert torch.allclose(new_out[:, :3, :], old_out, atol=1e-6)


def test_encoder_mask_equivalence_mixed():
    n_cells = 2
    n_heads = 4
    embed_dim = 32 * n_heads
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    encoder = _build_encoder(embed_dim, n_heads, device)

    pcpt_real = torch.rand(n_cells, 2, embed_dim, device=device)
    gen_real = torch.rand(n_cells, 2, embed_dim, device=device)
    with torch.cuda.amp.autocast():
        old_pcpt, old_gen = encoder(pcpt_real, gen_real)

    pcpt_pad = torch.zeros(n_cells, 1, embed_dim, device=device)
    pcpt_padded = torch.cat([pcpt_real, pcpt_pad], dim=1)
    gen_pad = torch.zeros(n_cells, 1, embed_dim, device=device)
    gen_padded = torch.cat([gen_real, gen_pad], dim=1)
    pcpt_mask = torch.tensor([[1, 1, 0]] * n_cells, dtype=torch.bool, device=device)
    gen_mask = torch.tensor([[1, 1, 0]] * n_cells, dtype=torch.bool, device=device)

    with torch.cuda.amp.autocast():
        new_pcpt, new_gen = encoder(
            pcpt_padded,
            gen_padded,
            pcpt_key_padding_mask=pcpt_mask,
            gen_key_padding_mask=gen_mask,
        )

    assert torch.allclose(new_pcpt[:, :2, :], old_pcpt, atol=1e-6)
    assert torch.allclose(new_gen[:, :2, :], old_gen, atol=1e-6)

