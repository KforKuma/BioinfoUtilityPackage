"""
Step04b_scvi_debatch.py
Author: John Hsiung
Update date: 2025-08-29
Description:
    - 利用 scvi 包（single-cell variational inference tools）进行批次效应的去除

Notes:
    - 依赖环境: conda activate scvpy10
    - 依赖内部函数位于 src.SCVITools.ScviTools，包含对接 scvi 相关功能所需要的函数
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
import sys
sys.path.append("/data/HeLab/bio/IBD_analysis/")
from src.ScanpyTools.Scanpy_Plot import ScanpyPlotWrapper
umap_plot = ScanpyPlotWrapper(func = sc.pl.umap)
####################################
# 读取数据
####################################
    # model 文件：os.path.join(save_addr, prefix)
    # adata 文件：os.path.join(save_addr, f"Step04_Combined.h5ad")

save_addr = "/data/HeLab/bio/IBD_analysis/output/Step04/"
prefix = "Combined"
adata_save = os.path.join(save_addr, f"Step04_Combined.h5ad")
model_save = os.path.join(save_addr, prefix)

adata = anndata.read_h5ad(adata_save)

# 查看数据
adata.obsm["X_scVI"] #
adata.layers["scvi_normalized"] # 标准化的表达矩阵，在[-1, +1]之间
####################################
# 对比降维制图
####################################
plt_addr = "/data/HeLab/bio/IBD_analysis/output/Step04/plot"
# 1）非批次整合
sc.tl.pca(adata)
neighbor_key = "n_wo_bc"
sc.pp.neighbors(adata, n_pcs=30, n_neighbors=20, use_rep='X_pca',
                key_added=neighbor_key)
    # use_rep：n_vars小于n_pcs（50）时使用adata.X，否则优先用adata.X_pca
sc.tl.umap(adata, min_dist=0.3,
           neighbors_key=neighbor_key)
umap_plot(save_addr = plt_addr,
          filename = "Umap by Sample Composition",
          adata = adata,neighbors_key = f"{neighbor_key}_connectivities",
          color=["orig.project","Patient"], ncols = 2,
          frameon=False)

# 2）批次整合，并做聚类处理
sc.pp.neighbors(adata, n_neighbors=20, use_rep="X_scVI")
sc.tl.umap(adata, min_dist=0.3,
           neighbors_key='neighbors')
umap_plot(save_addr = plt_addr,
          filename = "Umap by Sample Composition with Batch Correction",
          adata = adata,
          # neighbors_key = f"{neighbor_key}_connectivities",
          color=["orig.project","Patient"], ncols = 2,
          frameon=False)


adata.write_h5ad("Step04_Combined(umap).h5ad")

# # 发现有些行空了，检查一下
# na_rows = adata.obs[pd.isna(adata.obs["orig.project"])]
# print(na_rows)
# print(adata.obs[adata.obs["orig.ident"]=="SRR17062837"])

