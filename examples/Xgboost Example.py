# 系统库
import os, gc, sys

# 第三方库
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 将图输出到文件而不是屏幕
import matplotlib.pyplot as plt

import leidenalg
import sklearn
import scanpy as sc
import anndata
import pandas as pd

# 自定义函数
from src.core.base_anndata_ops import subcluster
from src.external_adaptor.xgboost import xgb_analysis,xgb_vis,xgb_compute
###################################################
# 为了清洗获得信息流，建议进行如下设置
# 1） 确保刷新信息流
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
# 2） 过滤一些反复打印的 warnings
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

###################################################
adata = anndata.read_h5ad("/data/HeLab/bio/IBD_analysis/output/Step07/Step07_finalversion_5.h5ad")

print(adata.obs["Subset_Identity"].unique().tolist())
print(adata.obs["Subset_Identity"].value_counts())
###################################################
# 进行标准的 xgboost 模型训练
sample_id="T Cell_CD4.Th17"
save_path="/data/HeLab/bio/IBD_analysis/output/Step10_xgboost/example/standard"
# 1） 数据准备
xgb_compute.xgb_data_prepare(adata,obs_select=sample_id,
                             save_path=save_path,
                             method="combined")

# 2） 进行建模拟合
xgb_compute.xgboost_process(save_path, eval_metric="mlogloss")

# 3） 输出结果
xgb_analysis.xgb_outcome_analyze(save_path)

# 4） 补充绘图
# 4.1） 绘制预测结果
# 只画图，一图两面，分别展示 "disease_type", "predicted_label_name"
xgb_vis.plot_xgb_prediction_umap(adata=adata,save_path=save_path,do_return=False)

# 如果这一步希望保持 UMAP 图的一致性，则应该先行亚群和降维，确保 adata_sub.obsm['X_umap']
adata_sub = adata[adata.obs["Subset_Identity"] == sample_id]
adata_sub = subcluster(adata_sub, n_neighbors=20, n_pcs=50)
# 返回结果为 adata_pred.obs["predicted_label"], adata_pred.obs["predicted_label_name"] 两列
# 且应当注意，adata_pred 是 adata_sub 的真子集，因为测试集的索引是总体数据随机的一部分。
adata_pred = xgb_vis.plot_xgb_prediction_umap(adata=adata,save_path=save_path,do_return=True,skip_subcluster=True)

# 4.2）绘制三种分类结果
xgb_vis.plot_taxonomy(adata=adata,save_path=save_path)

# 5） 数据的保存
adata_pred.write_h5ad(f"{save_path}/adata_pred.h5ad")
###################################################
# 进行验证鲁棒性的模型训练
sample_id="T Cell_CD4.Th17"
save_path="/data/HeLab/bio/IBD_analysis/output/Step10_xgboost/example/lodo"

# 1） 数据准备
xgb_compute.xgb_data_prepare_lodo(adata,save_path=save_path,
                                  obs_select=sample_id,obs_key="Subset_Identity",
                                  method="combined")

# 2） 进行建模拟合
xgb_compute.xgboost_process_lodo(save_path,eval_metric="mlogloss")

# 3） 数据分析
xgb_analysis.xgb_outcome_analyze_lodo(save_path)





