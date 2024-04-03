from pathlib import Path
import sys
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scgpt.model.blocks import SCGPTEncoder, SCGPTBlock

# get the available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_encoder():
    n_cells = 10
    n_genes = 5
    n_heads = 4
    embed_dim = 32 * n_heads

    flash_gpt_layer = SCGPTBlock(
        d_model=embed_dim, n_heads=n_heads, expansion_ratio=1
    )
    flash_gpt_generator = SCGPTEncoder(
        encoder_layer=flash_gpt_layer,
        num_layers=3,
    ).to(device)

    pcpt_total_embs = torch.rand(n_cells, n_genes, embed_dim).to(device)
    gen_total_embs = torch.rand(n_cells, n_genes * 2, embed_dim).to(device)

    # test forward, run with torch amp
    with torch.cuda.amp.autocast():
        output1, output2 = flash_gpt_generator(pcpt_total_embs, gen_total_embs)
    assert output1.shape == (n_cells, n_genes, embed_dim)
    assert output2.shape == (n_cells, n_genes * 2, embed_dim)

    # test only pcpt_total_embs
    with torch.cuda.amp.autocast():
        output1 = flash_gpt_generator(pcpt_total_embs)
    assert output1.shape == (n_cells, n_genes, embed_dim)

if __name__ == "__main__":
    test_encoder()
    print("All tests passed!")