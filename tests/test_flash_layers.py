from pathlib import Path
import tempfile
import sys

import numpy as np
import torch
import pytest
from scipy.sparse import csr_matrix
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scgpt.model.flash_layers import FlashscGPTMHA

# get the available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# test FlashscGPTMHA
def test_FlashscGPTMHA():
    n_cells = 10
    n_genes = 5
    n_heads = 4
    embed_dim = 32 * n_heads

    total_embs = np.random.rand(n_cells, n_genes, embed_dim)
    total_embs = torch.tensor(total_embs, dtype=torch.float16).to(device)

    flashscGPTMHA = FlashscGPTMHA(
        embed_dim=embed_dim,
        num_heads=n_heads,
        device=device,
    )

    # test forward, run with torch amp
    with torch.cuda.amp.autocast():
        output = flashscGPTMHA(total_embs)
    assert output.shape == (n_cells, n_genes, embed_dim)


# test FullFlashscGPTMHA forward with only pcpt input
