# 在协和医院高算上测试
# conda activate sc-min

# 在更新 sccoda 之后，想要使用则必须 conda activate sccoda-2025
##################################
import os, gc, sys
import numpy as np
import pandas as pd
import inspect
import anndata

sys.path.append('/public/home/xiongyuehan/data/BioinfoUtilityPackage')

from src.stats import *
####################################
# 重新加载
# import importlib
# importlib.reload(sys.modules['src.core.utils.geneset_editor'])

# 删除模块缓存
for module_name in list(sys.modules.keys()):
    if module_name.startswith('src.core'):
        del sys.modules[module_name]

# 重新读入
from src.core.plot.umap import plot_hierarchical_umap
####################################
# 路径初始化
save_addr = "/public/home/xiongyuehan/data/IBD_analysis/output/Step08_Diff_Abundance"
save_fig_addr = f"{save_addr}/fig"
os.makedirs(save_fig_addr, exist_ok=True)
####################################
# 读取和准备数据
adata = anndata.read_h5ad(
    "/public/home/xiongyuehan/data/IBD_analysis/output/Step07_Summary/Step07_DR_clustered_clean_20260210.h5ad")
adata_obs = adata.obs
del adata; gc.collect()
###########################################################
# 微调信息
adata_obs[adata_obs["tissue-origin"] == "rectum"]["tissue-origin"] = "colon"
adata_obs = adata_obs[adata_obs["tissue-origin"] != "blood"]
adata_obs["tissue-origin"] = adata_obs["tissue-origin"].tolist()
adata_obs["disease_group"] = (adata_obs["disease"].astype(str) + "_" + adata_obs["tissue-type"].astype(str))
###########################################################
# 预处理
adata_obs.presorted[adata_obs.presorted == "CD45+CD3+"] = "CD3+CD19-"
adata_obs["tissue-type"][adata_obs["tissue-type"] == "mixed"] = "if"
adata_obs["tissue-type"][adata_obs["tissue-type"] == "normal"] = "nif"


test = adata_obs.groupby("orig.ident").size()
mask = test.index[test > 500]
adata_obs = adata_obs[adata_obs["orig.ident"].isin(mask)]

ct_stratified_dict = {"CD3+CD19-": ['CD4 Tnaive', 'CD4 Tmem', 'CD4 Tmem GZMK+', 'CD4 Tfh', 'CD4 Treg', 'CD4 Th17',
                                     'CD8 Tnaive', 'CD8 Tmem', 'CD8 Tmem GZMK+', 'CD8 Trm', 'CD8 Trm GZMA+',
                                     'CD8 NKT FCGR3A+', 'CD8aa IEL',
                                     'gdTnaive', 'g9d2T cytotoxic', 'gdTrm',
                                     'MAIT TRAV1-2+',
                                     'Mitotic T cell'],
                       "CD45+": ['B cell IL6+', 'B cell kappa', 'B cell lambda', 'Germinal center B cell',
                                 'Plasma IgA+', 'Plasma IgG+', 'Mitotic plasma cell',
                                 'Natural killer cell FCGR3A+', 'Natural killer cell NCAM1+', 'ILC1', 'ILC3',
                                 'Classical monocyte CD14+', 'Nonclassical monocyte CD16A+', 'cDC1 CLEC9A+',
                                 'cDC2 CD1C+', 'pDC GZMB+',
                                 'Macrophage', 'Macrophage M1', 'Macrophage M2', 'Neutrophil CD16B+',
                                 'Mast cell'],
                       "CD45-": ['Intestinal stem cell OLFM4+LGR5+',
                                 'pre-TA cell', 'Transit amplifying cell', 'Regenerative colonocyte LEFTY1+',
                                 'Antigen-presenting colonocyte MHC-II+',
                                 'Goblet', 'Paneth cell', 'Tuft cell', 'Enteroendocrine',
                                 'Ion-sensing colonocyte BEST4+', 'Microfold cell',
                                 'Absorptive colonocyte PPARs+', 'Absorptive colonocyte',
                                 'Absorptive colonocyte Guanylins+',
                                 'Endothelium', 'Fibroblast', 'Fibroblast ADAMDEC1+'],
                       }
# 将分层信息保留
import json
with open(f"{save_addr}/stratified_config.json", "w") as f:
    json.dump(ct_stratified_dict, f, indent=2)

###########################################################
# # 分层过滤异常值
adata_obs_l1 = adata_obs[adata_obs["presorted"].isin(['CD3+CD19-', 'CD45+', 'intact'])];
adata_obs_l1 = adata_obs[adata_obs["Subset_Identity"].isin(ct_stratified_dict['CD3+CD19-'])]

adata_obs_l2 = adata_obs[adata_obs["presorted"].isin(['CD45+', 'intact'])];
adata_obs_l2 = adata_obs[adata_obs["Subset_Identity"].isin(ct_stratified_dict['CD45+'])]

adata_obs_l3 = adata_obs[adata_obs["presorted"].isin(['CD45-', 'intact'])];
adata_obs_l3 = adata_obs[adata_obs["Subset_Identity"].isin(ct_stratified_dict['CD45-'])]

adata_obs_ls = [adata_obs_l1, adata_obs_l2, adata_obs_l3]

from scipy.stats import median_abs_deviation

# 用于存储统计结果的列表
qc_stats = []

for i, adata_obs_sub in enumerate(adata_obs_ls):
    # --- 0. 初始统计 ---
    n_samples_before = adata_obs_sub["orig.ident"].nunique()
    n_cells_before = len(adata_obs_sub)
    
    # --- 1. 计算指标 ---
    freq = (
        adata_obs_sub
        .groupby(["orig.ident", "Subset_Identity"])
        .size()
        .unstack(fill_value=0)
    )
    freq_prop = freq.div(freq.sum(axis=1), axis=0)
    
    # Shannon entropy & Dominant fraction
    entropy = - (freq_prop * np.log(freq_prop + 1e-9)).sum(axis=1)
    dominant_frac = freq_prop.max(axis=1)
    
    # --- 2. 联合判定 ---
    # 稍微调低了 MAD 倍数（建议 3-5），6 可能太严苛
    mad_val = median_abs_deviation(entropy)
    low_entropy = entropy < (np.median(entropy) - 3 * mad_val)
    high_dominance = dominant_frac > 0.75
    
    outlier_samples = entropy.index[low_entropy & high_dominance].tolist()
    
    # --- 3. 执行剔除 ---
    filtered_obs = adata_obs_sub[~adata_obs_sub["orig.ident"].isin(outlier_samples)].copy()
    adata_obs_ls[i] = filtered_obs
    
    # --- 4. 记录变化 ---
    n_samples_after = filtered_obs["orig.ident"].nunique()
    n_cells_after = len(filtered_obs)
    
    qc_stats.append({
        "Layer": i,
        "Samples_Before": n_samples_before,
        "Samples_After": n_samples_after,
        "Samples_Removed": n_samples_before - n_samples_after,
        "Cells_Removed": n_cells_before - n_cells_after,
        "Outliers": ", ".join(outlier_samples) if outlier_samples else "None"
    })

# --- 5. 汇总展示 ---
stats_df = pd.DataFrame(qc_stats)
print("\n=== QC Filtering Summary ===")
print(stats_df.to_string(index=False))


adata_obs = pd.concat(adata_obs_ls)
adata_obs.to_csv(f"{save_addr}/Celltype_meta(stratified_clean).csv")
gc.collect()
###########################################################
# 后续的 readin，默认格式为这个 count_df
adata_obs = pd.read_csv(f"{save_addr}/Celltype_meta(stratified_clean).csv")
count_df = make_input(adata_obs)

count_df["tissue"] = count_df["tissue"].cat.remove_unused_categories()
print(count_df.tissue)

count_df["presort"].unique()

count_df1 = count_df[count_df["presort"].isin(['CD3+CD19-', 'CD45+', 'intact'])];
count_df1 = count_df1[count_df1["cell_type"].isin(ct_stratified_dict['CD3+CD19-'])]

count_df2 = count_df[count_df["presort"].isin(['CD45+', 'intact'])];
count_df2 = count_df2[count_df2["cell_type"].isin(ct_stratified_dict['CD45+'])]

count_df3 = count_df[count_df["presort"].isin(['CD45-', 'intact'])];
count_df3 = count_df3[count_df3["cell_type"].isin(ct_stratified_dict['CD45-'])]

count_df_sep_ls = [count_df1, count_df2, count_df3]

###########################################################
# 柱状图可视化
for i,df in enumerate(count_df_sep_ls):
    plot_stacked_barplot(df,
                         save_addr=save_fig_addr,
                         filename=f"Stacked_barplot(layer{i+1})")
