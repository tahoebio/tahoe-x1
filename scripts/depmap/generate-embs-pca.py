"""

Creates principal component and gene loading baseline comparison AnnDatas
for cell line embeddings, mean gene embeddings, and contextual gene
embeddings.

"""

# imports
import os
import argparse
import logging
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import utils
from tqdm import tqdm
from sklearn.decomposition import PCA

# set up logging
log = logging.getLogger(__name__)
logging.basicConfig(format=f"%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s")
logging.getLogger(__name__).setLevel("INFO")

# main function
def main(base_path):

    # run PCA on raw counts, save as cell line embeddings
    raw = sc.read_h5ad(os.path.join(base_path, "counts.h5ad"))
    sc.pp.normalize_total(raw)
    sc.pp.log1p(raw)
    sc.tl.pca(raw, n_comps=15)
    pca = ad.AnnData(X=raw.obsm["X_pca"], obs=raw.obs, var=pd.DataFrame({"dim": [str(i) for i in range(15)]}).set_index("dim"))
    outpath = os.path.join(base_path, "cell-embs/pca.h5ad")
    pca.write_h5ad(outpath)
    log.info(f"computed PCA and saved cell line embeddings to {outpath}")

    # load genes and scores for marginal essentiality task
    genes, scores = utils.get_marginal_genes_scores(base_path)

    # compute gene loadings for all genes
    log.info("computing gene loadings for all genes")
    raw = sc.read_h5ad(os.path.join(base_path, "counts.h5ad"))
    pca = PCA(n_components=15)
    pca.fit(raw.X.toarray())
    all_loadings_X = pca.components_.T * np.sqrt(pca.explained_variance_)
    all_loadings = ad.AnnData(
        X=all_loadings_X,
        obs=pd.DataFrame({"gene": raw.var["feature_name"]}),
        var=pd.DataFrame({"dim": np.arange(all_loadings_X.shape[1])})
    )

    # extract loadings for required genes
    log.info("selecting loadings for required genes")
    loadings_X = np.zeros((len(genes), all_loadings_X.shape[1]))
    for i, g in enumerate(tqdm(genes)):
        loadings_X[i] = all_loadings[all_loadings.obs["gene"] == g].X[0]

    # create AnnData
    loadings = ad.AnnData(
        X=loadings_X,
        obs=pd.DataFrame({"gene": genes, "score": scores}),
        var=pd.DataFrame({"dim": np.arange(loadings_X.shape[1])})
    )

    # save AnnData
    outpath = os.path.join(base_path, "gene-embs/loading15-lt5gt70-bin.h5ad")
    loadings.write_h5ad(outpath)
    log.info(f"saved gene loading AnnData to {outpath}")

    # load counts and cell line embeddings, build gene dictionary
    counts = sc.read_h5ad(os.path.join(base_path, "counts.h5ad"))
    pca_cell_embs = sc.read_h5ad(os.path.join(base_path, "cell-embs/pca.h5ad"))
    gene_dict = {gene: i for i, gene in enumerate(sorted(counts.var["feature_name"].unique().tolist()))}

    # get lists for iterating
    cell_lines = sorted(counts.obs["ModelID"].tolist())
    genes = sorted(list(gene_dict.keys()))

    # iterate over cell lines
    log.info("creating gene_idx+pca15 embeddings")
    labels = []
    embeddings = []
    for cl in tqdm(cell_lines):
        cl_emb = pca_cell_embs[pca_cell_embs.obs["ModelID"] == cl].X[0]
        for g in genes:
            ctx_gene_emb = np.zeros(cl_emb.shape[0] + 1)
            ctx_gene_emb[0] = gene_dict[g]
            ctx_gene_emb[1:] = cl_emb
            labels.append(f"{cl} | {g}")
            embeddings.append(ctx_gene_emb)

    # convert to arrays
    log.info("stacking embeddings")
    labels = np.array(labels)
    embeddings = np.vstack(embeddings)

    # sort by label order
    log.info("sorting embeddings")
    sort_idx = np.argsort(labels)
    labels = labels[sort_idx]
    embeddings = embeddings[sort_idx]

    # process and save contextual gene embeddings
    utils.process_contextual_gene_embs(base_path, log, labels, embeddings, "gene_idx+pca15")

if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str, required=True, help="Path to DepMap benchmark base directory.")
    args = parser.parse_args()

    # run main function
    main(args.base_path)