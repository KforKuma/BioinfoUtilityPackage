"""
Step04a_merge_for_debatch.py
Author: John Hsiung
Update date: 2025-08-29
Description:
    - 将单独的 h5ad 数据整合为一个，包含表达量矩阵的拼接（concat）和meta数据的对齐两个部分
    - 为下一步借助 scvi 包（single-cell variational inference tools）进行批次效应（batch-effect）的去除做准备
Notes:
    - 依赖环境: conda activate scvi-tools
    - 依赖内部函数位于 src.SCVITools.ScviTools，包含对接 scvi 相关功能所需要的函数
"""
import igraph  # avoid ImportError: dlopen
import torch  # avoid ImportError: dlopen
import scanpy as sc
import scvi
from rich import print
from scib_metrics.benchmark import Benchmarker
import pymde
import anndata
import os, gc
import time
import numpy as np
import scipy.sparse as sp
####################################
# 读取整合 adata
####################################
flst = os.listdir(read_addr)
adt_list = []
for adt in filter(lambda x: "h5ad" in x, flst):
    print(adt)
    read_path = os.path.join(read_addr, adt)
    adata = anndata.read_h5ad(read_path)
    adt_list.append(adata)
    del adata

for i, adata in enumerate(adt_list):
    print(f"--- AnnData object {i+1} ---")
    print(f"Cell count (adata.shape[1]): {adata.shape[0]}")
    print(f"Gene count (adata.shape[1]): {adata.shape[1]}")
    print(f"obs columns: {list(adata.obs.columns)}")
    print(f"Available layers: {list(adata.layers.keys())}")
    print()

for adt in adt_list:
    adt.obs = adt.obs[['orig.ident', 'percent.mt', 'percent.ribo', 'percent.hb']]

adata = anndata.concat(adt_list, merge="same")

del adt_list;gc.collect()
####################################
# 重整meta data
####################################
# 1. 读取 Excel 表格
import pandas as pd
meta_df = pd.read_excel("/data/HeLab/bio/IBD_analysis/assets/Sample_grouping_new.xlsx")

# 2. 设置索引为 orig.ident（确保唯一性）
meta_df = meta_df.set_index("orig.ident")

# 3. 你希望加入的列
meta_cols = ["orig.project", "Patient", "disease", "tissue-type", "tissue-origin", "presorted"]

# 4. 合并元信息
adata.obs = adata.obs[['orig.ident', 'percent.mt', 'percent.ribo', 'percent.hb']]
merged = adata.obs.join(meta_df[meta_cols], on="orig.ident")
adata.obs = merged
#__________________
print(adata.obs["Patient"].value_counts())
print(adata.obs["orig.ident"].value_counts())
adata.write_h5ad("/data/HeLab/bio/IBD_analysis/output/Step04/Step04_Combined.h5ad")
adata_test = anndata.read_h5ad("/data/HeLab/bio/IBD_analysis/output/Step04/Step04_Combined.h5ad")

save_addr = "/data/HeLab/bio/IBD_analysis/output/Step04/"
read_addr = "/data/HeLab/bio/IBD_analysis/output/Step03_Toh5ad/"

####################################
# 进行数据处理 + 自动保存
####################################
from src.SCVITools.ScviTools import process_adata

adata = process_adata(adata, prefix="Combined", save_addr=save_addr,
                      max_epochs=400,batch_size=512,span=0.8)
# 保存结果储存在：
# model 文件：os.path.join(save_addr, prefix)
# adata 文件：os.path.join(save_addr, f"Step04_Combined.h5ad")


