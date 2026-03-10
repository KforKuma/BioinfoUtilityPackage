import pandas as pd
import anndata
import os
import gc
import scanpy as sc
import sys

sys.stdout.reconfigure(encoding='utf-8')
####################################
sys.path.append('/public/home/xiongyuehan/data/BioinfoUtilityPackage')

from src.external_adaptor.cellphonedb.cellphone_inspector import *
from src.external_adaptor.cellphonedb.plot import *
from src.external_adaptor.cellphonedb.toolkit import *

####################################
parent_dir = "/public/home/xiongyuehan/data/IBD_analysis/output"

save_fig_dir = f"{parent_dir}/Step12_Custom_Vis/SpearCorr"

os.makedirs(save_fig_dir, exist_ok=True)

adata = anndata.read_h5ad(
    "/public/home/xiongyuehan/data/IBD_analysis/output/Step07_Summary/Step07_DR_clustered_clean_20260210.h5ad")
# adata = adata[(adata.obs["disease"] == "HC") |
#               ((adata.obs["disease"] != "HC") & (adata.obs["tissue-type"] == "if"))]


c3a_c5a_downstream_genes = {
    "PI3K_AKT": [  # upstream components (optional anchors)
        "PIK3CG", "PIK3CD", "PIK3R1",
        "AKT1", "PDPK1",
        
        # mTOR axis
        "MTOR", "RPS6KB1", "EIF4EBP1",
        
        # metabolism / growth
        "MYC", "HIF1A", "SREBF1",
        "HK2", "PFKFB3", "LDHA",
        
        # survival / anti-apoptosis
        "BCL2L1", "MCL1", "XIAP",
        
        # proliferation
        "CCND1", "CDK4",
        
        # angiogenesis / inflammation context
        "VEGFA",
        
        # FOXO suppression readout (often inverse but informative)
        "FOXO1", "FOXO3"],
    "MAPK_ERK": [
        
        # core anchors
        "MAPK1", "MAPK3",
        "MAP2K1",
        
        # transcription factors (IEG)
        "FOS", "FOSB",
        "JUN", "JUNB",
        "EGR1", "EGR2",
        "ATF3",
        "NR4A1", "NR4A2",
        
        # feedback inhibitors
        "DUSP1", "DUSP2",
        "DUSP5", "DUSP6"
    ],
    "NFkB": [
        # core
        "NFKB1", "NFKB2", "RELA", "RELB", "REL",
        # activation
        "IKBKB", "IKBKG", "CHUK",
        # feedback
        "NFKBIA", "TNFAIP3",
        # outputs
        "ICAM1", "VCAM1",
        "TNF", "IL1B", "IL6", "CCL2", "CCL3"
    ],
    "Rho_GTPase": [
        
        # upstream GTPases (anchors)
        "RHOA", "RAC1", "CDC42",
        
        # effectors
        "ROCK1", "ROCK2",
        "PAK1", "PAK2",
        "LIMK1",
        
        # cytoskeleton remodeling
        "ACTN1",
        "VCL",
        "TLN1",
        "PXN",
        
        # migration / contractility
        "MYL9",
        "MYLK",
        
        # adhesion / motility
        "ITGB1",
        "ITGA5",
        
        # mechanotransduction
        "YAP1",
        "WWTR1",
        
        # invasion related
        "MMP2",
        "MMP9"
    ],
    "C3_C5_Signaling": ["C3AR1", "C5AR1",
                        "GNAI2", "GNB1",
                        "ITPR2", "MAPK3"
                        ]
}

adata_sub = adata[adata.obs["Celltype"].isin(["Myeloid Cell"])]

for k, v in c3a_c5a_downstream_genes.items():
    sc.tl.score_genes(adata_sub, gene_list=v, score_name=f"{k}_score", use_raw=False)
###############
from src.core.plot.regplot import *

print(adata_sub.obs["Subset_Identity"].value_counts())
plot_significant_regression_by_disease(adata_sub, subset_cells=["Macrophage M1", "Macrophage M2", "Macrophage"],
                                       save_addr=save_fig_dir, filename="M1M2M")
plot_significant_regression_by_disease(adata_sub, subset_cells=["Macrophage M1", "Macrophage M2", "Macrophage",
                                                                "cDC2 CD1C+"],
                                       save_addr=save_fig_dir, filename="M1M2MC2")
plot_significant_regression_by_disease(adata_sub, subset_cells=["Macrophage M1", "Macrophage M2", "Macrophage",
                                                                "cDC1 CLEC9A+", "cDC2 CD1C+", "pDC GZMB+"],
                                       alpha=0.001,
                                       save_addr=save_fig_dir, filename="M1M2MC1C2pDC(a=0.001)")
plot_significant_regression_by_disease(adata_sub, subset_cells=["Macrophage M1", "Macrophage M2", "Macrophage",
                                                                "Neutrophil CD16B+"
                                                                "cDC1 CLEC9A+", "cDC2 CD1C+", "pDC GZMB+"],
                                       alpha=0.001,
                                       save_addr=save_fig_dir, filename="M1M2MC1C2pDCNeut(a=0.001)")

# plot_significant_regression_by_disease(adata_sub,subset_cells=["Macrophage M1",
#                                                                "Macrophage M2",
#                                                                "Classical monocyte CD14+",
#                                                                "Neutrophil CD16B+"],
#                                        save_addr=save_fig_dir,filename="MacroMonoNeutro(inflam)")

#################################################
# 相关性研究
from scipy.stats import spearmanr, pearsonr

scores = ["PI3K_AKT_score", "MAPK_ERK_score", "NFkB_score", "Rho_GTPase_score"]

adata_ss = adata_sub[adata_sub.obs["Subset_Identity"].isin(["Macrophage M1", "Macrophage M2", "Neutrophil CD16B+",
                                                            ])]

cells_df = adata_sub.obs[
    ["disease", "C3_C5_Signaling_score"] + scores
    ]

results = []

for disease, df in cells_df.groupby("disease"):
    for score in scores:
        rho, p = spearmanr(
            df["C3_C5_Signaling_score"],
            df[score],
            nan_policy="omit"
        )
        
        results.append({
            "disease": disease,
            "pathway": score,
            "rho": rho,
            "p": p
        })

results_df = pd.DataFrame(results)

genes = ["C3AR1", "C5AR1", "GNAI2", "GNB1",
         "PLCB2", "PLCB3", "PLCG1", "PIK3CG", "ITPR2", "AKT1", "MAPK3"]
###############
# 取出表达矩阵
expr = adata_ss[:, genes].to_df()

# 加上细胞分类
expr["disease"] = adata_ss.obs["disease"].values

# 按 disease 分组计算平均表达
table = expr.groupby("disease")[genes].mean()

print(table)

####################
import statsmodels.api as sm
from scipy.stats import mannwhitneyu


def verify_necessity_via_residuals(obs_df, receptor_col, pathway_col, disease_col, target_group="BD"):
    # 1. 准备数据
    data = obs_df[[receptor_col, pathway_col, disease_col]].dropna()
    X = sm.add_constant(data[receptor_col])
    y = data[pathway_col]
    
    # 2. 线性回归提取残差 (Residuals)
    model = sm.OLS(y, X).fit()
    data['residual'] = model.resid
    
    # 3. 统计检验：BD vs 其他组
    group_target = data[data[disease_col] == target_group]
    group_others = data[data[disease_col] != target_group]
    
    # 原始评分差异
    u_raw, p_raw = mannwhitneyu(group_target[pathway_col], group_others[pathway_col])
    # 剔除受体后的残差差异
    u_res, p_res = mannwhitneyu(group_target['residual'], group_others['residual'])
    
    print(f"--- 验证通路: {pathway_col} ---")
    print(f"原始差异 P-value: {p_raw:.2e}")
    print(f"剔除受体后残差差异 P-value: {p_res:.2e}")
    
    # 计算效应值改善比例 (简单用均值差衡量)
    raw_diff = group_target[pathway_col].mean() - group_others[pathway_col].mean()
    res_diff = group_target['residual'].mean() - group_others['residual'].mean()
    reduction = (1 - res_diff / raw_diff) * 100
    print(f"受体对组间差异的解释贡献度: {reduction:.2f}%")


# 使用示例
adata_ss = adata_sub[adata_sub.obs["Subset_Identity"].isin(["Macrophage M1", "Macrophage M2", "Macrophage",
                                                            "Neutrophil CD16B+",
                                                            "cDC1 CLEC9A+", "cDC2 CD1C+", "pDC GZMB+"])]
print(adata_ss.obs.columns)
verify_necessity_via_residuals(adata_ss.obs, 'C3_C5_Signaling_score', 'NFkB_score', 'disease')
verify_necessity_via_residuals(adata_ss.obs, 'C3_C5_Signaling_score', 'PI3K_AKT_score', 'disease')

g = sns.lmplot(
    x='C3_C5_Signaling_score',
    y='NFkB_score',
    hue='disease',
    data=adata_ss.obs
)

g.savefig(f"{save_fig_dir}/lmplot.png", dpi=300, bbox_inches="tight")

import statsmodels.formula.api as smf

# 建立交互模型：探究疾病背景是否改变了受体对通路的带动能力
import statsmodels.formula.api as smf

model = smf.ols(formula="NFkB_score ~ C3_C5_Signaling_score * C(disease, Treatment('HC'))", data=adata_ss.obs).fit()

print(model.summary())

model = smf.ols(formula="NFkB_score ~ PI3K_AKT_score * C(disease, Treatment('HC'))", data=adata_ss.obs).fit()

print(model.summary())

################################
# 虚拟敲除
import seaborn as sns
import matplotlib.pyplot as plt


def virtual_knockout_test(adata, save_addr, filename,
                          receptor_col, pathway_col, disease_col, target_group="BD", control_group="HC"):
    # 1. 提取目标组和对照组数据
    sub = adata.obs[adata.obs[disease_col].isin([target_group, control_group])].copy()
    
    # 2. 定义“虚拟敲除”状态：受体评分最低的 25% 细胞定义为 Low
    # 或者直接定义受体评分 == 0 的细胞（如果零值多的话）
    low_thresh = sub[receptor_col].quantile(0.25)
    high_thresh = sub[receptor_col].quantile(0.75)
    
    sub['receptor_status'] = 'Middle'
    sub.loc[sub[receptor_col] <= low_thresh, 'receptor_status'] = 'Low'
    sub.loc[sub[receptor_col] >= high_thresh, 'receptor_status'] = 'High'
    
    # 3. 绘图对比：观察在 Low 状态下，BD 还是否保持高炎症
    
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=sub[sub['receptor_status'].isin(['Low', 'High'])],
                x='receptor_status', y=pathway_col, hue=disease_col,
                order=['Low', 'High'], palette='Set1')
    
    plt.title("Necessity Check: NFkB levels in Receptor-Low vs High cells")
    abs_file_path = os.path.join(save_addr, filename)
    plt.savefig(f'{abs_file_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{abs_file_path}.pdf', bbox_inches='tight')

# 调用
adata_ss = adata_sub[adata_sub.obs["Subset_Identity"].isin(["Macrophage M1", "Macrophage M2", "Macrophage",
                                                            "Neutrophil CD16B+",
                                                            "cDC1 CLEC9A+", "cDC2 CD1C+", "pDC GZMB+"])]

adata_ss.obs["group"]
virtual_knockout_test(adata_ss, save_fig_dir,"Virtual_Knockout",
                      receptor_col='C3_C5_Signaling_score', pathway_col=['NFkB_score','PI3K_AKT_score'],
                      disease_col='disease')
