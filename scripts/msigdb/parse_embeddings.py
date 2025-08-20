import numpy as np
import os
from tqdm.auto import tqdm

def main(inpath, outpath):
    for emb_path in tqdm(os.listdir(inpath)):
        if not emb_path.endswith(".npz"):
            continue
        out_gene_encoder = os.path.join(outpath, emb_path.replace(".npz", "_GE.npz").replace("gene_embeddings_", ""))
        out_transformer = os.path.join(outpath, emb_path.replace(".npz", "_TE.npz").replace("gene_embeddings_", ""))
        emb = np.load(os.path.join(inpath, emb_path), allow_pickle=True)
        np.savez_compressed(
        out_gene_encoder,
        gene_embeddings=emb["gene_embeddings_context_free_old"],
        gene_names=emb["gene_names"],
        gene_ids=emb["gene_ids"],
        )
        
        np.savez_compressed(
        out_transformer,
        gene_embeddings=emb["gene_embeddings_context_free"],
        gene_names=emb["gene_names"],
        gene_ids=emb["gene_ids"],
        )
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("inpath", type=str, help="path to the raw embeddings.")
    parser.add_argument("outpath", type=str, help="where to save the parsed embeddings.")

    args = parser.parse_args()
    main(args.inpath, args.outpath)