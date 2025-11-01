"""
Step10a_xgboost_model_test.py
Author: John Hsiung
Update date: 2025-08-29
Description:
    - 进行 xgboost 的建模
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
from src.EasyInterface.XgboostTools import xgboost_process, xgb_outcome_analyze, xgb_data_prepare, plot_xgb_prediction_umap, SkipThis, plot_taxonomy
###################################################
adata = anndata.read_h5ad("/data/HeLab/bio/IBD_analysis/output/Step07/Step07_finalversion_5.h5ad")

# 脚本工作时打开
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print(adata.obs["Subset_Identity"].unique().tolist())
print(adata.obs["Subset_Identity"].value_counts())

###################################################
# 找一个分布均匀的
print(adata[adata.obs["Subset_Identity"]=="T Cell_CD8.Trm.KLRC2+"].obs["disease"].value_counts())
sample_id = "Epi_Col.BEST4+"


# 试验/debug：对比可解读性方面是否有价值，以及预测准确度方面是否有价值
# 试验：增加一个混淆矩阵聚类功能

sample_id="T Cell_CD4.Th17"
save_path="/data/HeLab/bio/IBD_analysis/output/Step10_xgboost/example/cm_cluster"
xgb_data_prepare(adata,obs_select=sample_id,
                 save_path=save_path,
                 method="scvi")
xgboost_process(save_path,method_suffix="combined",
                eval_metric="mlogloss",objective="multi:softmax",
                max_depth=4,colsample_bytree=0.8,
                min_split_loss=0,min_child_weight=2,eta=0.1,reg_lambda=2)

xgb_outcome_analyze(save_path,"combined")

adata_pred = plot_xgb_prediction_umap(
    adata=adata,
    save_path=save_path,
    method_suffix="combined"
)
plot_taxonomy(adata=adata,save_path=save_path,method_suffix="combined")

# ###################################################
# 试验：多个细胞亚群组合是否可行
celltype_dict = (
    adata.obs.groupby("Celltype")["Subset_Identity"]
    .apply(set)
    .apply(list) # 必须要list化，不然prepare的类型判断会出错
    .to_dict()
)

xgb_data_prepare(adata,celltype_dict["T Cell"],
                 save_path="/data/HeLab/bio/IBD_analysis/output/Step10_xgboost/example",
                 method="scvi")

xgboost_process(save_path,"scvi")


xgb_outcome_analyze(save_path,"scvi")

adata_pred = plot_xgb_prediction_umap(
    adata=adata,
    save_path="/data/HeLab/bio/IBD_analysis/output/Step10_xgboost/example",
    method_suffix="scvi"
)

###################################################
# 正式开始循环
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 单独样本循环
list2run = adata.obs["Subset_Identity"].unique().tolist()

list2run.index("Plasma_Plasma.mitotic")

for cell_subtype in list2run[64:]:
    save_path = f"/data/HeLab/bio/IBD_analysis/output/Step10_xgboost/single_type_combined/{cell_subtype}"
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n>>> Starting  cell subtype: {cell_subtype}")
    
    method = "combined"
    print(f"   ---> Method: {method}")
    
    print("   ... Starting xgb_data_prepare ...")
    try:
        xgb_data_prepare(
            adata,
            cell_subtype,
            save_path=save_path,
            method=method
        )
    except SkipThis:
        continue
    
    print("   ... Starting xgboost_process ...")
    xgboost_process(save_path, method_suffix=method,
                    eval_metric="mlogloss", objective="multi:softmax",
                    max_depth=4, colsample_bytree=0.8,
                    min_split_loss=0, min_child_weight=2, eta=0.1, reg_lambda=2)
    
    print("    ... Starting xgb_outcome_analyze ...")
    xgb_outcome_analyze(save_path, method)
    
    print("   ... Starting plot_xgb_prediction_umap ...")
    # data = np.load(os.path.join(save_path, f"dataset_{method}.npz"), allow_pickle=True)
    adata_pred = plot_xgb_prediction_umap(
        adata=adata,
        save_path=save_path,
        method_suffix=method
    )
    
    plot_taxonomy(adata=adata, save_path=save_path, method_suffix=method)
    
    out_file = f"{save_path}/adata_pred.h5ad"
    adata_pred.write_h5ad(out_file)
    print(f"      --> adata_pred is saved at {out_file}")
    
    print(f">>> Finished cell subtype: {cell_subtype}\n")


##############################################
# 多个样本循环
celltype_dict = (
        adata.obs.groupby("Celltype")["Subset_Identity"]
        .apply(set)
        .apply(list)  # 必须要list化，不然prepare的类型判断会出错
        .to_dict()
    )
list2run = adata.obs["Celltype"].unique().tolist()


for cell_type in list2run:
    if len(celltype_dict[cell_type]) == 1:
        print(f"Skip {cell_type} because only 1 Subset_Identity")
        continue
    
    print(f"\n>>> Starting  cell subtype: {cell_type}")
    
    save_path = f"/data/HeLab/bio/IBD_analysis/output/Step10_xgboost/combined_type/{cell_type}"
    os.makedirs(save_path, exist_ok=True)
    
    method="combined"
    print(f"   ---> Method: {method}")
    
    print("   ... Starting xgb_data_prepare ...")
    try:
        xgb_data_prepare(
            adata,
            celltype_dict[cell_type],
            save_path=save_path,
            method=method
        )
    except SkipThis:
        continue
    print("   ... Starting xgboost_process ...")
    xgboost_process(save_path, method_suffix=method,
                    eval_metric="mlogloss", objective="multi:softmax",
                    max_depth=4, colsample_bytree=0.8,
                    min_split_loss=0, min_child_weight=2, eta=0.1, reg_lambda=2)
    
    print("    ... Starting xgb_outcome_analyze ...")
    xgb_outcome_analyze(save_path, method)
    
    print("   ... Starting plot_xgb_prediction_umap ...")
    adata_pred = plot_xgb_prediction_umap(
        adata=adata,
        save_path=save_path,
        method_suffix=method
    )
    plot_taxonomy(adata=adata, save_path=save_path, method_suffix=method)
    
    out_file = f"{save_path}/adata_pred.h5ad"
    adata_pred.write_h5ad(f"{out_file}.h5ad")



    

