from math import inf
import os
import logging
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
from scipy import sparse
import scib
from matplotlib import pyplot as plt
import itertools
import psutil


# rpy2 for running R code
import rpy2.rinterface_lib.callbacks
import logging
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR) # Ignore R warning messages
import rpy2.robjects as ro
import anndata2ri

def load_azimuth_ref(url):
    anndata2ri.activate()
    ro.r('.libPaths("/exports/humgen/cnovellarausell/conda_envs/atlas_work/lib/R/library")')
    ro.r('library(Seurat)')
    ro.r(f'url_user <- "{url}"')
    ro.globalenv['ref'] = ro.r('readRDS(url(url_user,"rb"))')
    adata_ref = ro.r('as.SingleCellExperiment(ref)')
    return adata_ref
    

def find_conserved_markers(adata, cluster_column, cluster, confounder):
    anndata2ri.activate()
    ro.r('.libPaths("/exports/humgen/cnovellarausell/conda_envs/atlas_work/lib/R/library")')
    ro.r('library(Seurat)')
    ro.r('library(dplyr)')
    ro.r('library(tibble)')
    ro.r(f'cluster <- "{cluster}"')
    ro.r(f'confounder <- "{confounder}"')
    ro.r(f'cluster_column <- "{cluster_column}"')
    ro.globalenv['Data_Ann'] = adata
    ro.globalenv['Adata'] = ro.r('as.Seurat(Data_Ann, counts="counts", data="Seurat_LogNormalize")')
    ro.r(f'Idents(Adata) <- "{cluster_column}"')
    df = ro.r("""
    FindConservedMarkers(Adata, ident.1 = cluster, grouping.var = confounder, only.pos = TRUE, assay="originalexp", verbose=FALSE, min.cells.group=0) %>%
    rownames_to_column(var = "gene")
    """)
    return df

def clean_adata(adata, obs_tokeep, return_metadata=True):
    tmp_col = dict()
    tmp_var = dict()
    tmp_obsm = dict()
    for (col, var, obsm) in itertools.zip_longest(adata.obs.columns.to_list(), adata.var.columns.to_list(), list(adata.obsm)):
        tmp_col[col] = adata.obs[col]
        if var:
            tmp_var[var] = adata.var[var]
        if obsm:
            tmp_obsm[obsm] = adata.obsm[obsm]
    
    adata.obs.drop(columns= adata.obs.columns[~adata.obs.columns.isin(obs_tokeep)], inplace=True) # remove everything except obs_tokeep
    adata.var.drop(columns= adata.var.columns.to_list(), inplace=True) # remove everything
    del adata.obsm
    
    if return_metadata:
        return tmp_col, tmp_var, tmp_obsm
    
def clean_seurat(adata, tmp_col, tmp_var, obs_tokeep):
    del adata.obsm
    del adata.layers
    adata.obs.drop(columns=adata.obs.columns[~adata.obs.columns.isin(list(tmp_col.keys()) + obs_tokeep)], inplace=True) ## Drop columns introduced by seurat (i.e not present in original adata)
    adata.var.drop(columns=adata.var.columns[~adata.var.columns.isin(tmp_var.keys())], inplace=True) ## Drop columns introduced by seurat (i.e not present in original adata)
    

    
def transfer_anchors(adata_reference, adata_query, metadata):
    """
    Seurat's PCA anchors transfer
    adata_reference: valid anndata object to use as reference
    adata_query: valid anndata object to transfer the labels to
    
    This function uses all genes and all cells by default.
    """
    anndata2ri.activate()
    ro.r('.libPaths("/exports/humgen/cnovellarausell/conda_envs/atlas_work/lib/R/library")')
    ro.r('library(Seurat)')
    ro.r('library(Azimuth)')
    
    # pass anndata to R 
    ro.globalenv['Data_SCE_ref'] = adata_reference
    ro.globalenv['Reference'] = ro.r('as.Seurat(Data_SCE_ref, counts = "counts", data=NULL)')
    ro.globalenv['Data_SCE_query'] = adata_query
    ro.globalenv['Query'] = ro.r('as.Seurat(Data_SCE_query, counts = "counts", data=NULL)')
    print("SCT Transform Query...")
    ro.globalenv['Query'] = ro.r("""
                                query <- SCTransform(
                                object = Query,
                                assay = "originalexp",
                                new.assay.name = "SCT",
                                method = 'glmGamPoi',
                                n_genes = NULL,
                                residual.features=NULL,
                                do.correct.umi = FALSE,
                                do.scale = FALSE,
                                do.center = TRUE)
                                """)
    print("SCTransform Reference...")
    ro.globalenv['Reference'] = ro.r("""
                                query <- SCTransform(
                                object = Reference,
                                assay = "originalexp",
                                new.assay.name = "SCT",
                                method = 'glmGamPoi',
                                n_genes = NULL,
                                residual.features=NULL,
                                do.correct.umi = FALSE,
                                do.scale = TRUE,
                                do.center = TRUE)
                                """)
    print("Run PCA Reference...")
    ro.globalenv['Reference'] = ro.r('RunPCA(Reference, npcs = 26)')
    print("Finding anchors...")
    ro.globalenv['anchors'] = ro.r("""
                                    FindTransferAnchors(
                                    reference = Reference,
                                    reduction="pcaproject",
                                    reference.reduction="pca",
                                    k.filter = NA,
                                    query = Query,
                                    features = intersect(rownames(x = Reference), rownames(x = Query)),
                                    normalization.method = "SCT",
                                    dims = 1:26,
                                    n.trees = 20,
                                    mapping.score.k = 100)
                                    """)
    print("Transfering data...")
    ro.globalenv['Query'] = ro.r('MapQuery(anchorset = anchors,query = Query,reference = Reference,refdata =  Reference$Celltype_finest,reference.reduction = "pca")')
    print("Adding Metadata...")
    ro.globalenv['Query'] = ro.r("""
                                AddMetaData(object = Query,
                                metadata = MappingScore(anchors = anchors, ndim=26),
                                col.name = "mapping.score")
                                """)
          
    print("Returning query to python...")
    adata_query = ro.r('as.SingleCellExperiment(Query)')
    return adata_query


def merge_adata_p(adata_list): #modified version of merge_adata()
    """
    merge adatas from list and remove duplicated obs and var columns
    """
    
    if len(adata_list) == 1:
        return adata_list[0]
    
    adata = adata_list[0].concatenate(*adata_list[1:], index_unique=None, batch_key='tmp')
    del adata.obs['tmp']

    return adata

normalize_scran = scib.preprocessing.normalize

def scran_size_factors(adata, precluster=True, sparsify=True):
    anndata2ri.activate()
    ro.r('library("scran")')

    is_sparse = False
    X = adata.X.T
    # convert to CSC if possible. See https://github.com/MarioniLab/scran/issues/70
    if sparse.issparse(X):
        is_sparse = True

        if X.nnz > 2 ** 31 - 1:
            X = X.tocoo()
        else:
            X = X.tocsc()

    ro.globalenv['data_mat'] = X

    if precluster:
        # Preliminary clustering for differentiated normalisation
        adata_pp = adata.copy()
        sc.pp.normalize_per_cell(adata_pp, counts_per_cell_after=1e6)
        sc.pp.log1p(adata_pp)
        sc.pp.pca(adata_pp, n_comps=15, svd_solver='arpack')
        sc.pp.neighbors(adata_pp)
        sc.tl.louvain(adata_pp, key_added='groups', resolution=0.5)

        ro.globalenv['input_groups'] = adata_pp.obs['groups']
        size_factors = ro.r('sizeFactors(computeSumFactors(SingleCellExperiment('
                            'list(counts=data_mat)), clusters = input_groups,'
                            f' min.mean = {min_mean}))')

        del adata_pp

    else:
        size_factors = ro.r('sizeFactors(computeSumFactors(SingleCellExperiment('
                            f'list(counts=data_mat)), min.mean = {min_mean}))')

    # modify adata
    adata.obs['size_factors'] = size_factors
    if is_sparse:
        # convert to sparse, bc operation always converts to dense
        adata.X = sparse.csr_matrix(adata.X)

    # Free memory in R
    ro.r('rm(list=ls())')
    ro.r('gc()')

    anndata2ri.deactivate()
    
    

    
    

    
def categorize_clusters(adata, clusters=None, obs_column=None):
    """
    categorizes specified cluster(s) so they can be plotted with scanpy.pl{embbedings}
    adata: valid anndata object
    clusters: list of strings specifying the categories of obs_column we want to categorize
    obs_column: string specifying the column in adata.obs to use 
    """

    if type(clusters) is not list:
        raise TypeError('Input is not a valid list')
    else:
        for cluster in clusters:
            if cluster not in adata.obs:
                adata.obs[cluster] = adata.obs[obs_column] == cluster
            adata.obs[cluster+'_cat'] = adata.obs[cluster].astype(str).astype('category')
            adata.obs[cluster+'_cat'] = adata.obs[cluster+'_cat'].replace('True', cluster)
            adata.obs[cluster+'_cat'] = adata.obs[cluster+'_cat'].replace('False', np.nan)
            
                                                  
def reduce_adata(adata):
    sc.pp.scale(adata)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=10)
    sc.tl.umap(adata) 

def plot_clusters(adata, clusters=None, combine=True):
    """
    plots categorized clusters 
    adata: valid anndata object
    clusters: list of strings specifying the categories of obs_column we want to plot
    combine: boolean specifying whether to combine the plots or not  
    """     
    if type(clusters) is not list:
        raise TypeError('Input is not a valid list')
    else:
        for cluster in clusters:
            # Check that cluster is categorized
            if not adata.obs[cluster].dtypes.name == 'category':
                raise TypeError('Cluster' + cluster + 'is not categorical')
        adata.obs["+".join(clusters)] = adata.obs[clusters].apply(lambda x: x.str.cat(sep=''), axis=1)
        barcodes = adata.obs.index.tolist()
        clusters_barcodes = adata.obs["+".join(clusters)].index.tolist()
        sizes = [5 if i in clusters_barcodes else 0.5 for i in barcodes]
        sc.pl.umap(adata, color=["+".join(clusters)], save="+".join(clusters), size = sizes)


def build_subplots(n):
    """
    Build subplots grid
    n: number of subplots
    """
    nrow = int(np.sqrt(n))
    ncol = int(np.ceil(n / nrow))
    fig, axs = plt.subplots(nrow, ncol, dpi=100, figsize=(ncol*5, nrow*5))

    return fig, axs, nrow, ncol

def groupby_mean(adata, groupby):
    grouped = adata.obs.groupby(groupby)
    results = np.zeros((grouped.ngroups, adata.n_vars), dtype=np.float64)

    for idx, indices in enumerate(grouped.indices.values()):
        results[idx] = np.ravel(adata.X[indices].mean(axis=0))

    return pd.DataFrame(results, columns=adata.var_names, index=grouped.groups.keys())
