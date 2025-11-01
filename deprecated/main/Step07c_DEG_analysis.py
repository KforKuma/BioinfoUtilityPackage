"""
Step07c_DEG_analysis.py
Author: John Hsiung
Update date: 2025-08-29
Description:
    - 这是一个更复杂的尝试，除基本的 DEG 外，希望通过
        跨疾病类型的差异基因分析，寻找出某种差异基因存在的模式
        可以做单调变化基因的统计，计数本身可以做 barplot；
        基因可以做跨细胞类群统计；做聚类或GO分析；
        本项目先只按照疾病分群，不拆分疾病x炎症进行探索，如果成功的话再进一步精细化，以避免处理缺省值问题太多
Notes:
    - 依赖环境: conda activate scvpy10
"""
####################################
import igraph  # avoid ImportError: dlopen
import gc, os
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
####################################
import matplotlib
matplotlib.use('Agg')
sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=80,facecolor="white",frameon=False,)
####################################
import sys
sys.path.append("/data/HeLab/bio/IBD_analysis/")
from src.ScanpyTools.deg_tool_box import (split_and_DEG, load_and_filter_hvg, winsorize_df,
                                          clustermap_with_custom_cluster, load_merge_and_filter,
                                          run_pca_and_deg_for_celltype)
# from src.ScanpyTools.Scanpy_Plot import ScanpyPlotWrapper
# umap_plot = ScanpyPlotWrapper(func = sc.pl.umap)
####################################
import yaml
with open('/data/HeLab/bio/IBD_analysis/assets/io_config.yaml', 'r') as f:
    config = yaml.safe_load(f) # 读取的格式为一系列字典的列表
    
os.chdir("/data/HeLab/bio/IBD_analysis/output/Step07")
output_dir = "/data/HeLab/bio/IBD_analysis/output/Step07/Step07c_DEG_analysis"
###############################################################################################################
# 图1的计算部分：
#   1） 对每个细胞亚群按疾病分组、储存、筛选
#   2） 进行 HVG 富集
###############################################################################################################

adata = anndata.read_h5ad("Step07_finalversion_5.h5ad")

celllist = adata.obs["Subset_Identity"].unique().tolist()
split_and_DEG(subset_list=celllist,subset_key="Subset_Identity", split_by_key="disease", output_dir=output_dir)

# 按照疾病
celllist = adata.obs["Subset_Identity"].unique().tolist()
split_and_DEG(subset_list=celllist,subset_key="Subset_Identity", split_by_key="disease", output_dir=output_dir)

# 按照疾病+组织类型
adata.obs["disease_type"] = (adata.obs["disease"].astype(str) + "_" + adata.obs["tissue-type"].astype(str))
celllist = adata.obs["Subset_Identity"].unique().tolist()
split_and_DEG(subset_list=celllist,subset_key="Subset_Identity", split_by_key="disease_type", output_dir=output_dir)

###############################################################################################################
# 图1：按照每个亚群的HVG文件，在子文件夹里做热图
    # 这个作图效果不是特别理想，但是足以有个基本的概念，并且我们在这里对基因进行了初步的筛选，也可以返回保存一下
    # 横轴为疾病分型，纵轴为基因，对纵轴做聚类，基因不显示（或选择性显示）
###############################################################################################################
fig1_dir = f"{output_dir}/fig1_heatmap_by_hvg"
os.makedirs(fig1_dir,exist_ok=True)

# 对 disease
for Subset in celllist:
    readin_addr = Path(output_dir) / f"_{Subset}"
    HVG_file = readin_addr / f"{Subset}_HVG_wilcoxon_disease.xlsx"
    if HVG_file.exists():
        print(f"File found: {HVG_file}")
        # desired_order = ["HC_normal", "Colitis", "BD", "CD", "UC"]
        # 读取保存的 HVG数据，进行整形和筛选
        pivot_df = load_and_filter_hvg(HVG_file)
        # 直接保存
        pivot_df.to_csv(readin_addr / "Filtered_pivot_df.csv")
        # 对数据进行 winsorize（缩尾处理）
        pivot_df = winsorize_df(pivot_df, lower_q=0.1, upper_q=0.9)
        # 制图
        clustermap_with_custom_cluster(pivot_df,save_dir=fig1_dir,figname=f"{Subset}_autoheatmap")
    else:
        print(f"File not found: {HVG_file}")

# 对 disease_type：在命名上区分处理一下
for Subset in celllist:
    readin_addr = Path(output_dir) / f"_{Subset}"
    HVG_file = readin_addr / f"{Subset}_HVG_wilcoxon_disease_type.xlsx"
    if HVG_file.exists():
        print(f"File found: {HVG_file}")
        desired_order = ['HC_normal', 'Colitis_if', 'BD_if', 'CD_nif', 'UC_nif', 'CD_if', 'UC_if']
        # 读取保存的 HVG数据，进行整形和筛选
        pivot_df = load_and_filter_hvg(HVG_file, load_only = False, desired_order = desired_order)
        # 直接保存
        pivot_df.to_csv(readin_addr / "Filtered_pivot_df(disease_type).csv")
        # 对数据进行 winsorize（缩尾处理）
        pivot_df = winsorize_df(pivot_df, lower_q=0.1, upper_q=0.9)
        # 制图
        clustermap_with_custom_cluster(pivot_df,save_dir=fig1_dir,figname=f"{Subset}_autoheatmap(disease_type)")
    else:
        print(f"File not found: {HVG_file}")

###############################################################################################################
# 图2：Filtered_pivot_df 筛选过的 DEG 做 PCA，及使用 PCA 结果做 DEG
###############################################################################################################
# from src.ScanpyTools.Scanpy_statistics import count_element_list_occurrence
# from src.ScanpyTools.ScanpyTools import easy_DEG

save_dir=f"{output_dir}/fig2_pca_by_hvg"
os.makedirs(save_dir,exist_ok=True)

#____________________________________________________________________________________
#____________________________________________________________________________________
# 批量读取，整理为 PCA 使用的表
merged_df_filtered, HVG_list = load_merge_and_filter(celllist, output_dir)

# 检查免疫细胞整体 pca， 并做 DEG
celltype = ["B Cell","T Cell","Myeloid","Plasma"]
# 新增一列，以对应 HVG_df["cluster"]
adata.obs["tmp"] = (adata.obs["disease"].astype(str) + "_" + adata.obs["Subset_Identity"].astype(str))

run_pca_and_deg_for_celltype(celltype = celltype, merged_df_filtered = merged_df_filtered,
                             adata = adata, save_dir = save_dir,
                             pca_fig_prefix = "among_disease(All_imm)", DEG_file_suffix = "by_PCA_cluster")

# 对每个细胞大类分别做 PCA 聚类
for celltype in ["B Cell","T Cell","Myeloid","Plasma","Endo","Epi","Fibroblast"]:
    run_pca_and_deg_for_celltype(celltype, merged_df_filtered, adata, save_dir,
                                 pca_fig_prefix = "among_disease", DEG_file_suffix = "by_PCA_cluster")
#____________________________________________________________________________________
# 批量读取，整理为 PCA 使用的表
merged_df_filtered, HVG_list = load_merge_and_filter(celllist, output_dir, suffix = "disease_type")
# 保存
merged_df_filtered.to_pickle("merged_df_filtered(disease_type).pkl")
# 读取
merged_df_filtered = pd.read_pickle("merged_df_filtered.pkl")

# 检查免疫细胞整体 pca， 并做 DEG
celltype = ["B Cell","T Cell","Myeloid","Plasma"]
# 新增一列，以对应 HVG_df["cluster"]
adata.obs["tmp"] = (adata.obs["disease_type"].astype(str) + "_" + adata.obs["Subset_Identity"].astype(str))

run_pca_and_deg_for_celltype(celltype = celltype, merged_df_filtered = merged_df_filtered,
                             adata = adata, save_dir = save_dir,figsize=(16,10),
                             pca_fig_prefix = "among_disease_type(All_imm)")

# 对每个细胞大类分别做 PCA 聚类
for celltype in ["B Cell","T Cell","Myeloid","Plasma","Endo","Epi","Fibroblast"]:
    run_pca_and_deg_for_celltype(celltype, merged_df_filtered, adata, save_dir,
                                 pca_fig_prefix = "among_disease_type")


###############################################################################################################
# 图3： 单调基因（mDEG）的运算
###############################################################################################################








