# conda activate sccoda-2025
##################################
import os, gc, sys
import numpy as np
import pandas as pd
import anndata

sys.path.append('/public/home/xiongyuehan/data/BioinfoUtilityPackage')


from src.stats import *
# from src.stats.engine.sccoda import run_scCODA

####################################
# 重新加载
# import importlib
# importlib.reload(sys.modules['src.core.utils.geneset_editor'])

# 删除模块缓存
for module_name in list(sys.modules.keys()):
    if module_name.startswith('src.stats'):
        del sys.modules[module_name]

# 重新读入
from src.stats.engine import *
####################################
# 路径初始化
save_addr = "/public/home/xiongyuehan/data/IBD_analysis/output/Step08_Diff_Abundance"
save_fig_addr = f"{save_addr}/fig_0228"
os.makedirs(save_fig_addr, exist_ok=True)

##############################################################
# 聚类降维
##############################################################
from src.stats.plot.post_analysis import *
results_df = pd.read_csv(f"{save_addr}/0306_Realdata(separated)_Output(no_filter).csv", index_col=0)

date = "0308"
# 额外分析一：Graphical Lasso
## 数据准备
results_df['weighted_beta'] = np.sign(results_df['Coef.']) * -np.log10(results_df['P>|z|'] + 1e-10)
beta_matrix = results_df.reset_index().groupby(['cell_type', 'other'])['weighted_beta'].mean().unstack(fill_value=0)

partial_corr, gl_model = plot_glasso_partial_corr(beta_matrix,save_fig_addr,filename=f"{date}_Lasso_Part_correlation_heatmap(disease)")

## 将 beta 矩阵转置，计算细胞之间的关系
partial_corr, gl_model = plot_glasso_partial_corr_celltype(beta_matrix,save_fig_addr,filename=f"{date}_Lasso_Part_correlation_heatmap(celltype)")


## 剔除与其他细胞全不相关的亚群
partial_corr_clustered, filtered_celltypes, Z = plot_glasso_partial_corr_celltype_filtered(partial_corr,
                                                                                           beta_matrix,
                                                                                           save_fig_addr,
                                                                                           filename=f"{date}_Lasso_Part_correlation_heatmap(celltype-filtered)")


# 绘制 PCA
pcs_df, loading_df, pca = plot_pca_celltype_and_loading(
    beta_matrix,
    save_fig_addr,
    f"{date}_PCA",
    f"{date}_PCA_loading(disease)"
)

# FA, NMF, and ICA
fa_df, nmf_df, ica_df = plot_celltype_decomposition(
    beta_matrix,
    save_fig_addr,
    f"{date}_FA(celltype)",
    f"{date}_NMF(celltype)",
    f"{date}_ICA(celltype)"
)



###########################################################
# 细胞比例检测

df_all = count_df_sep_ls[0]

clr_lmm_result = run_CLR_LMM(df_all=df_all,
            cell_type=("CD8 Tmem GZMK+","CD8 Trm"),
            formula="disease + C(tissue, Treatment(reference='nif'))",
            main_variable="disease",
            ref_label= "HC",
            alpha=0.05,
            group_label="sample_id")
print(clr_lmm_result["contrast_table"])


df_ratio = compute_ratio_df(df_all,disease_col="disease",celltype_pair=("CD8 Tmem GZMK+","CD8 Trm"))

plot_ratio_scatter(df_ratio,save_fig_addr,filename=f"{date}_CLR_Ratio_CD8Trm_CD8TrmGZMK",
                   cell_pair=("CD8 Tmem GZMK+","CD8 Trm"),disease_col="disease",
                   clr_lmm_result=clr_lmm_result)


df_ratio = compute_ratio_df(df_all,celltype_pair=("CD8 Tmem GZMK+","CD8 Trm"),disease_col="tissue")

plot_ratio_scatter(df_ratio,save_fig_addr,filename=f"{date}_CLR_Ratio_CD8Trm_CD8TrmGZMK(tissue)",
                   cell_pair=("CD8 Tmem GZMK+","CD8 Trm"),disease_col="tissue",
                   clr_lmm_result=clr_lmm_result)


##########################
df_all = count_df_sep_ls[2]

df_all_test = df_all[df_all["disease"]!="Colitis"]



clr_lmm_result = run_CLR_LMM(df_all=df_all,
            cell_type=("Absorptive colonocyte Guanylins+","Absorptive colonocyte"),
            formula="disease + C(tissue, Treatment(reference='nif'))",
            main_variable="disease",
            ref_label= "HC",
            alpha=0.05,
            group_label="sample_id")

print(clr_lmm_result["contrast_table"])

df_ratio = compute_ratio_df(df_all,disease_col="disease",
                            celltype_pair=("Absorptive colonocyte Guanylins+","Absorptive colonocyte"))

plot_ratio_scatter(df_ratio,save_fig_addr,filename=f"{date}_CLR_Ratio_AbsColon_AbsColonGuanylins",
                   cell_pair=("Absorptive colonocyte Guanylins+","Absorptive colonocyte"),disease_col="disease",
                   clr_lmm_result=clr_lmm_result)

df_ratio = compute_ratio_df(df_all,disease_col="tissue",
                            celltype_pair=("Absorptive colonocyte Guanylins+","Absorptive colonocyte"))

plot_ratio_scatter(df_ratio,save_fig_addr,filename=f"{date}_CLR_Ratio_AbsColon_AbsColonGuanylins(tissue)",
                   cell_pair=("Absorptive colonocyte Guanylins+","Absorptive colonocyte"),disease_col="tissue",
                   clr_lmm_result=clr_lmm_result)

####
# hack 获得 early absorp + absorp / absorp guanylins+
merge_types = ["Absorptive colonocyte PPARs+", "Absorptive colonocyte"]
new_name = "Absorp colonocyte-all"

df = df_all_test.copy()

# 1. 改名字
df.loc[df["cell_type"].isin(merge_types), "cell_type"] = new_name

# 2. 重新汇总
df = (
    df.groupby(
        ["sample_id", "donor_id", "disease", "tissue", "presort", "cell_type"],
        as_index=False
    )
    .agg(
        count=("count", "sum"),
        total_count=("total_count", "first")
    )
)

# 3. 重新算 prop
df["prop"] = df["count"] / df["total_count"]

clr_lmm_result = run_CLR_LMM(df_all=df,
            cell_type=("Absorptive colonocyte Guanylins+","Absorp colonocyte-all"),
            formula="disease + C(tissue, Treatment(reference='nif'))",
            main_variable="disease",
            ref_label= "HC",
            alpha=0.05,
            group_label="sample_id")

print(clr_lmm_result["contrast_table"])

df_ratio = compute_ratio_df(df,disease_col="disease",
                            celltype_pair=("Absorptive colonocyte Guanylins+","Absorp colonocyte-all"))

plot_ratio_scatter(df_ratio,save_fig_addr,filename=f"{date}_CLR_Ratio_AbsColonAll_AbsColonGuanylins",
                   cell_pair=("Absorptive colonocyte Guanylins+","Absorp colonocyte-all"),disease_col="disease",
                   clr_lmm_result=clr_lmm_result)

df_ratio = compute_ratio_df(df,disease_col="tissue",
                            celltype_pair=("Absorptive colonocyte Guanylins+","Absorp colonocyte-all"))

plot_ratio_scatter(df_ratio,save_fig_addr,filename=f"{date}_CLR_Ratio_AbsColonAll_AbsColonGuanylins(tissue)",
                   cell_pair=("Absorptive colonocyte Guanylins+","Absorp colonocyte-all"),disease_col="tissue",
                   clr_lmm_result=clr_lmm_result)
