"""
Step10b_xgboost_lodo_cv.py
Author: John Hsiung
Update date: 2025-09-08
Description:
    - 进行 lodo-cv，作为对 xgboost 方法的检验
Notes:
    - 使用环境：conda activate sc_xgb
    # 如果不能运行，在控制台输入export LD_PRELOAD=$CONDA_PREFIX/lib/libgomp.so.1
"""
###################################################
import xgboost as xgb
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 将图输出到文件而不是屏幕
import matplotlib.pyplot as plt

import leidenalg
import sklearn
import scanpy as sc
import anndata
import pandas as pd
import os, gc, sys

os.chdir("/data/HeLab/bio/IBD_analysis/output/Step05")
import sys
sys.path.append("/data/HeLab/bio/IBD_analysis/")
from src.EasyInterface.XgboostTools import (xgb_data_prepare_lodo, xgboost_process_lodo,
                                            xgb_outcome_analyze_lodo, plot_lodo_boxplots,
                                            plot_confusion_stats, compute_donor_similarity_matrix,
                                            bootstrap_consensus_dendrogram, plot_stability_dendrogram,
                                            plot_stability_clustermap, plot_consensus_dendrogram
                                            )
from src.EasyInterface.XgboostTools import donor_vectors_from_proba
###################################################
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
###################################################
adata = anndata.read_h5ad("/data/HeLab/bio/IBD_analysis/output/Step07/Step07_finalversion_5.h5ad")
###################################################
# 找一个分布均匀的
print(adata[adata.obs["Subset_Identity"]=="T Cell_CD8.Trm.KLRC2+"].obs["disease"].value_counts())
sample_id = "Epi_Col.BEST4+"

###################################################
# 试验/debug：对比可解读性方面是否有价值，以及预测准确度方面是否有价值
# 试验：增加一个混淆矩阵聚类功能

sample_id="Epi_Stem.LGR5"
save_path="/data/HeLab/bio/IBD_analysis/output/Step10_xgboost/singlecell_lodo_check/Epi_Stem.LGR5"

xgb_data_prepare_lodo(adata,obs_select=sample_id,
                      save_path=save_path,
                      method="combined",
                      verbose=False)

xgboost_process_lodo(save_path,method_suffix="combined",
                     eval_metric="mlogloss",objective="multi:softmax",
                     max_depth=4,colsample_bytree=0.8,
                     min_split_loss=0,min_child_weight=2,reg_lambda=2,
                     verbose=False)

results = xgb_outcome_analyze_lodo(save_path,"combined")

plot_lodo_boxplots(results, save_path=f"{save_path}/output/combined_boxplot.png")

plot_confusion_stats(results, save_path=f"{save_path}/output/average_confusion_matrix.png")

# 映射一个疾病标签用
donor_labels = results["donor"]
label_mapping = dict(adata.obs.groupby("Patient")["disease_type"].first())
disease_label = [label_mapping.get(lab, lab) for lab in donor_labels]

# 旧代码，目前 *compute_donor_similarity* 已经废弃；co_matrix 即新代码中的 sim

# cluster_labels_list = []
# for i in range(0, len(results["y_pred"])):
#     # 这里每个 donor 都得到一个 label 向量
#     labels = results["y_pred"][i]  # 或者 y_proba.argmax(axis=1)
#     cluster_labels_list.append(labels)

# co_matrix = compute_donor_similarity(y_pred_list=cluster_labels_list, n_classes=len(results["mapping"][0]))

# donor_mat = donor_vectors_from_preds(results["y_pred"], n_classes=len(results["mapping"][0]), label_shift=0)
donor_mat = donor_vectors_from_proba(results["y_proba"])
sim = compute_donor_similarity_matrix(donor_mat, metric='cosine')


plot_stability_dendrogram(sim, disease_label, save_path=f"{save_path}/output/stability_dendro.png")

plot_stability_clustermap(sim, disease_label, save_path=f"{save_path}/output/stability_cluster.png")

consensus_tree, branch_supports = bootstrap_consensus_dendrogram(sim_matrix=sim, n_bootstrap=100,
                                                                 method="average", support_threshold=0.5)

np.fill_diagonal(sim, 0)
D_condensed = squareform(sim)  # 对角设为0，把方阵转换成一维距离向量
Z_real = linkage(D_condensed, method='average')  # 或 'complete', 'ward' 等


plot_consensus_dendrogram(Z_real, disease_label, branch_supports,
                          save_path=f"{save_path}/output/consensual_tree.png")

###################################################
# 正式开始循环
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 单独样本循环
list2run = adata.obs["Subset_Identity"].unique().tolist()

ltmp = ["T Cell_CD4.Th17","T Cell_ILC3","T Cell_CD8.mem","Epi_Stem.OLFM4"]
list2run = [item for item in list2run if item not in ltmp]
# list2run.index("Plasma_Plasma.mitotic")

for cell_subtype in list2run[7:]:
    save_path = f"/data/HeLab/bio/IBD_analysis/output/Step10_xgboost/singlecell_lodo_check/{cell_subtype}"
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n>>> Starting  cell subtype: {cell_subtype}")
    
    
    print("   ... Starting xgb_data_prepare ...")
    try:
        xgb_data_prepare_lodo(adata, obs_select=cell_subtype,
                              save_path=save_path,
                              method="combined",
                              verbose=False)
    except SkipThis:
        continue
    
    xgboost_process_lodo(save_path, method_suffix="combined",
                         eval_metric="mlogloss", objective="multi:softmax",
                         max_depth=4, colsample_bytree=0.8,
                         min_split_loss=0, min_child_weight=2, reg_lambda=2,
                         verbose=False)
    results = xgb_outcome_analyze_lodo(save_path, "combined")
    
    print("   ... Starting plot_lodo_boxplots ...")
    plot_lodo_boxplots(results, save_path=f"{save_path}/output/combined_boxplot.png")
    
    print("   ... Starting plot_confusion_stats ...")
    plot_confusion_stats(results, save_path=f"{save_path}/output/average_confusion_matrix.png")
    
    # 映射一个疾病标签用
    donor_labels = results["donor"]
    label_mapping = dict(adata.obs.groupby("Patient")["disease_type"].first())
    disease_label = [label_mapping.get(lab, lab) for lab in donor_labels]
    
    print("   ... Starting plot_stability_dendrogram ...")
    donor_mat = donor_vectors_from_proba(results["y_proba"])
    sim = compute_donor_similarity_matrix(donor_mat, metric='cosine')
    plot_stability_dendrogram(sim, disease_label, save_path=f"{save_path}/output/stability_dendro.png")
    
    print("   ... Starting plot_stability_clustermap ...")
    plot_stability_clustermap(sim, disease_label, save_path=f"{save_path}/output/stability_cluster.png")
    
    print("   ... Starting plot consensual tree ...")
    consensus_tree, branch_supports = bootstrap_consensus_dendrogram(sim_matrix=sim, n_bootstrap=100,
                                                                     method="average", support_threshold=0.5)
    
    np.fill_diagonal(sim, 0)  # 直接修改 sim
    D_condensed = squareform(sim)  # 对角设为0，把方阵转换成一维距离向量
    Z_real = linkage(D_condensed, method='average')  # 或 'complete', 'ward' 等
    plot_consensus_dendrogram(Z_real, disease_label, branch_supports,
                              save_path=f"{save_path}/output/consensual_tree.png")
    
    
    print(f">>> Finished cell subtype: {cell_subtype}\n")


