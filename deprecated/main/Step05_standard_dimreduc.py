"""
Step05_standard_dimreduc.py
Author: John Hsiung
Update date: 2025-08-29
Description:
    - 将聚合、去批次效应后的数据进行标准的降维聚类流程
Notes:
    - 依赖环境: conda activate scvpy10
    - 依赖内部函数位于 src.ScanpyTools.Scanpy_Plot，及 src.ScanpyTools.ScanpyTools
"""

####################################
import igraph  # avoid ImportError: dlopen
import gc
import os
import anndata
import numpy as np
import pandas as pd
import scanpy as sc

####################################
sc.settings.verbosity = 0
sc.settings.set_figure_params(
    dpi=80,
    facecolor="white",
    frameon=False,
)
####################################
import sys
sys.path.append("/data/HeLab/bio/IBD_analysis/")

from src.ScanpyTools.Scanpy_Plot import ScanpyPlotWrapper
umap_plot = ScanpyPlotWrapper(func = sc.pl.umap)

from src.ScanpyTools.ScanpyTools import subcluster
from src.ScanpyTools.ScanpyTools import easy_DEG

####################################
# 标准流程
####################################
os.chdir("/data/HeLab/bio/IBD_analysis/output/Step05")
adata = anndata.read_h5ad("/data/HeLab/bio/IBD_analysis/output/Step04/Step04_Combined(umap).h5ad")

sc.tl.tsne(adata, use_rep="X_scVI")
sc.tl.leiden(adata,
             neighbors_key = "neighbors", #If not specified, leiden looks at .obsp[‘connectivities’]
             key_added="leiden_res0_25", resolution=0.25)

resolutions_list = [0.5,1.0]
adata = subcluster(adata,
                   n_neighbors=20, n_pcs= min(adata.obsm["X_scVI"].shape[1], 50),
                   resolutions=resolutions_list,
                   use_rep="X_scVI")
adata.write_h5ad("Step05_DR_clustered.h5ad")
#
adata = anndata.read_h5ad("Step05_DR_clustered.h5ad")
adata.obs_names_make_unique()

resolutions_list=[1.0]
for res in resolutions_list:
    groupby_key = f"leiden_res{res}"
    adata = easy_DEG(adata, save_addr="/data/HeLab/bio/IBD_analysis/output/Step05/DEG",
                     filename="Primary_Cluster",obs_key=groupby_key,
                     save_plot=True, plot_gene_num=5,
                     downsample=False,method="wilcoxon")
adata.write_h5ad("Step05_DR_clustered_DEG.h5ad")


umap_plot(save_addr = "/data/HeLab/bio/IBD_analysis/output/Step05/DEG",
          filename = "Umap by Cluster",
          adata = adata,
          color=["leiden_res0.5","leiden_res1.0"], ncols = 2,
          frameon=False)




