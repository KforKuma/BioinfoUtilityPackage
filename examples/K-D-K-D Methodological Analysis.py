# 在协和医院高算上测试
# conda activate sc-min
##################################
# 读取 anndata 环境依赖项
# import anndata
import os, gc, sys
import numpy as np
import pandas as pd

sys.path.append('/public/home/xiongyuehan/data/BioinfoUtilityPackage')


from src.core.kdk_methodology import *

save_addr = "/public/home/xiongyuehan/data/IBD_analysis/output/Step07/Step07c_KDKD_methodology"
###########################################################

# 重新载入
import importlib
# 删除模块缓存
for module_name in list(sys.modules.keys()):
    if module_name.startswith('src.core.kdk'):
        del sys.modules[module_name]

sys.path.append('/public/home/xiongyuehan/data/BioinfoUtilityPackage')
# importlib.reload(sys.modules['src.core.kdk_methodology'])

from src.core.kdk_methodology import *

###########################################################
# 读取和准备数据
# adata = anndata.read_h5ad("/public/home/xiongyuehan/data/IBD_analysis/output/Step07/Step07_DR_clustered_clean.h5ad")
# adata_obs = adata.obs
# del adata; gc.collect()
###########################################################
# 微调
# adata_obs[adata_obs["tissue-origin"]=="rectum"]["tissue-origin"] = "colon"
# adata_obs = adata_obs[adata_obs["tissue-origin"]!="blood"]
# adata_obs["tissue-origin"] = adata_obs["tissue-origin"].tolist()
# adata_obs["disease_group"] = (adata_obs["disease"].astype(str) + "_" + adata_obs["tissue-type"].astype(str))
# adata_obs.to_csv(f"{save_addr}/Celltype_meta.csv")
###########################################################
# 后续的 readin
adata_obs = pd.read_csv(f"{save_addr}/Celltype_meta.csv")
count_df = make_input(adata_obs)

# 对数据进行一定处理，已经去除了血液来源
count_df.presort[count_df.presort=="CD45+CD3+"] = "CD3+CD19-"
count_df.tissue[count_df.tissue=="mixed"] = "if"
count_df.tissue[count_df.tissue=="normal"] = "nif"

unique_combinations = count_df[['disease', 'tissue', 'presort']].drop_duplicates()

###########################################################
# 测试函数
celltype_test = count_df.cell_type[10]

# DKD_res = run_DKD(count_df, celltype_test)
DKD_res = run_DKD(count_df, celltype_test,
                  formula="disease + C(tissue, Treatment(reference=\"nif\"))",
                  main_variable = "disease")
DKD_res["extra"]["contrast_table"]

# LMM_res = run_LMM(count_df, celltype_test)
LMM_res = run_LMM(count_df, celltype_test,
                  formula = "disease + C(tissue, Treatment(reference=\"nif\"))",
                  main_variable = "disease")
LMM_res["extra"]["contrast_table"]

# CLR_LMM_res = run_CLR_LMM(count_df, celltype_test)
CLR_LMM_res = run_CLR_LMM(count_df, celltype_test,
                          formula="disease + C(tissue, Treatment(reference=\"nif\"))",
                          main_variable="disease")
CLR_LMM_res["extra"]["contrast_table"]


# PermMix_res = run_Perm_Mixed(count_df, celltype_test)
PermMix_res = run_Perm_Mixed(count_df, celltype_test,
                             formula="disease + C(tissue, Treatment(reference=\"nif\"))",
                             main_variable="disease")
PermMix_res["extra"]["contrast_table"]

PermMix_res = run_Perm_Mixed(count_df, celltype_test,
                             formula="disease + C(tissue, Treatment(reference=\"nif\"))",
                             main_variable="disease",
                             pairwise_level="sample_id")
PermMix_res["extra"]["contrast_table"]


# Dir_res = run_Dirichlet_Wald(count_df, celltype_test)
Dir_res = run_Dirichlet_Wald(count_df, celltype_test,
                             formula="disease + C(tissue, Treatment(reference=\"nif\"))")
# Dir_res["extra"]["fixed_effect"]
Dir_res["extra"]["contrast_table"]


anova_res = run_ANOVA_naive(count_df, celltype_test, formula="prop ~ disease + C(tissue, Treatment(reference=\"nif\"))")
# anova_res["extra"]["anova_table"]
anova_res["extra"]["contrast_table"]

anova_t_res = run_ANOVA_transformed(count_df, celltype_test,"prop ~ disease + C(tissue, Treatment(reference=\"nif\"))")
# anova_t_res["extra"]["anova_table"]
anova_t_res["extra"]["contrast_table"]

results_full = {
    "DKD": DKD_res["extra"]["contrast_table"],
    "LMM": LMM_res["extra"]["contrast_table"],
    "CLR_LMM": CLR_LMM_res["extra"]["contrast_table"],
    "PermMix": PermMix_res["extra"]["contrast_table"],
    "Dirichlet": Dir_res["extra"]["contrast_table"],
    "ANOVA_naive": anova_res["extra"]["contrast_table"],
    "ANOVA_transformed": anova_t_res["extra"]["contrast_table"],
}


def merge_contrast_tables(tables_dict):
    """Merge multiple contrast_tables into one readable DataFrame."""
    
    merged = None
    for method, df in tables_dict.items():
        df_copy = df.copy()
        # 保留关键信息
        keep_cols = ["ref", "other", "mean_ref", "mean_other", "prop_diff",
                     "Coef", "p_adj", "significant", "direction"]
        for col in df_copy.columns:
            if col not in keep_cols:
                df_copy = df_copy.drop(columns=col)
        
        # 为列加方法前缀
        df_copy = df_copy.rename(columns={c: f"{method}_{c}" for c in df_copy.columns if c not in ["ref", "other"]})
        
        if merged is None:
            merged = df_copy
        else:
            merged = pd.merge(merged, df_copy, on=["ref", "other"], how="outer")
    
    return merged


readable_df = merge_contrast_tables(results_full)
readable_df.to_csv("/public/home/xiongyuehan/data/IBD_analysis/output/Step07/Step07c_KDKD_methodology/test_output.csv")
###########################################################
# 对原始数据进行拆分之后运行
unique_combinations = count_df[['disease', 'tissue', 'presort']].drop_duplicates()

cell_type_inclusion = {"CD3+CD19-":['CD4 Th17', 'CD4 Tmem', 'CD4 Tfh', 'CD4 Tmem GZMK+', 'CD4 Tnaive', 'CD4 Treg',
                                    'CD8 NKT FCGR3A+', 'CD8 Tmem', 'CD8 Tmem GZMK+', 'CD8 Tnaive',
                                    'CD8 Trm', 'CD8 Trm GZMA+', 'CD8aa IEL',
                                    'g9d2T cytotoxic', 'gdTnaive', 'gdTrm','MAIT TRAV1-2+','ILC1'],
                       "CD45+":['B cell IL6+', 'B cell kappa', 'B cell lambda','Germinal center B cell','Plasma IgA+','Plasma IgG+',
                                'Natural killer cell FCGR3A+','Natural killer cell NCAM1+','ILC3',
                                'Classical monocyte CD14+','Nonclassical monocyte CD16A+','cDC1 CLEC9A+','cDC2 CD1C+', 'pDC GZMB+',
                                'Macrophage', 'Macrophage M1', 'Macrophage M2','Neutrophil CD16B+',
                                'Mast cell'],
                       "CD45-":['Absorptive colonocyte', 'Absorptive colonocyte Guanylins+','Enteroendocrine',
                                'Epithelial stem cell OLFM4+LGR5+', 'Goblet', 'Ion-sensing colonocyte BEST4+',
                                'Stressed epithelium', 'Tuft cell',
                                'Ion-transport colonocyte CFTR+', 'Paneth cell','Microfold cell','Mitotic epithelial stem cell',
                                'Endothelium', 'Fibroblast','Fibroblast ADAMDEC1+']
}


###########################################################
# 模拟输入数据
df_sim,df_true_effect = simulate_DM_data(
    n_donors=8,
    n_samples_per_donor=4,
    disease_levels=["HC", "CD","UC"],
    sampling_bias_strength=2
)

df_sim.head()
df_true_effect[df_true_effect.True_Significant==True]

# 测试
Dir_res = run_Dirichlet_Wald(df_sim, "CT1",
                             formula="disease + C(tissue, Treatment(reference=\"nif\"))")
# Dir_res["extra"]["fixed_effect"]
Dir_res["extra"]["contrast_table"]





df_sim,df_true_effect = simulate_LogisticNormal_hierarchical(
    N_samples=12,
    N_cell_types=50,
    # disease_levels=("HC", "CD","UC","BD")
)
df_sim.head()
df_true_effect[df_true_effect.True_Significant==True]
# 测试
Dir_res = run_Dirichlet_Wald(df_sim, "CT1",
                             formula="disease + C(tissue, Treatment(reference=\"nif\"))")
# Dir_res["extra"]["fixed_effect"]
Dir_res["extra"]["contrast_table"]


df_sim,df_true_effect = simulate_CLR_resample_data(count_df,
                                    disease_levels=["HC", "CD","UC"])
df_sim.head()
df_true_effect[df_true_effect.True_Significant==True]

# 测试
Dir_res = run_Dirichlet_Wald(df_sim, "CT1",
                             formula="disease + C(tissue, Treatment(reference=\"nif\"))")
# Dir_res["extra"]["fixed_effect"]
Dir_res["extra"]["contrast_table"]

