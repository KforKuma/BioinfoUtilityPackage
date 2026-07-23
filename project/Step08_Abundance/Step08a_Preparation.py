"""Step08a: prepare a canonical abundance cohort from the existing local CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from project.Step08_Abundance.phase4_shared import (
    LOCAL_INPUT,
    LOCAL_WORK_ROOT,
    Step08PreparationSpec,
    prepare_step08_csv,
)


# %% Optional upstream preparation from H5AD
# Original HPC path:
# /public/home/xiongyuehan/data/IBD_analysis/output/Step07_Summary/
# Step07_DR_clustered_clean_20260210.h5ad
# The exact historical script is retained below as inert source text. It is disabled in
# Phase 4 because Celltype_meta(stratified_clean).csv already exists locally.
LEGACY_HPC_WORKFLOW = r'''
# 在协和医院高算上测试
# conda activate sc-min

# 在更新 sccoda 之后，想要使用则必须 conda activate sccoda-2025
##################################
import os, gc, sys
import numpy as np
import pandas as pd
import inspect
import anndata
import warnings

warnings.warn(
    "Step08a_Preparation.py remains a legacy cohort-preparation script and is not part "
    "of the formal differential-abundance pipeline. Migrate cohort rules to configuration.",
    DeprecationWarning,
    stacklevel=2,
)

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
save_fig_addr = f"{save_addr}/fig_preparation"
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
'''


# %% Configuration
INPUT_PATH = LOCAL_INPUT
OUTPUT_ROOT = LOCAL_WORK_ROOT
PREPARATION_RUN_ID = "phase4-step08-preparation"
ANALYSIS_ID = "phase4-step08-layer1"
COHORT_ID = "layer1_tcell"
GROUP_1 = "CD_if"
GROUP_2 = "HC_normal"
REFERENCE_CELL_TYPE = "CD4 Tnaive"


# %% Reusable functions
def build_preparation_spec(
    *,
    run_id: str = PREPARATION_RUN_ID,
    analysis_id: str = ANALYSIS_ID,
    cohort_id: str = COHORT_ID,
    group_1: str = GROUP_1,
    group_2: str = GROUP_2,
    reference_cell_type: str = REFERENCE_CELL_TYPE,
    samples_per_group: int | None = None,
) -> Step08PreparationSpec:
    return Step08PreparationSpec(
        run_id=run_id,
        analysis_id=analysis_id,
        cohort_id=cohort_id,
        group_1=group_1,
        group_2=group_2,
        reference_cell_type=reference_cell_type,
        samples_per_group=samples_per_group,
    )


def run_preparation(
    input_path: str | Path = INPUT_PATH,
    output_root: str | Path = OUTPUT_ROOT,
    *,
    spec: Step08PreparationSpec | None = None,
) -> Path:
    """Create the immutable Step08a manifest consumed by Step08c and Step08d."""
    return prepare_step08_csv(
        input_path,
        output_root,
        spec or build_preparation_spec(),
    )


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-id", default=PREPARATION_RUN_ID)
    parser.add_argument("--analysis-id", default=ANALYSIS_ID)
    parser.add_argument("--cohort-id", default=COHORT_ID)
    parser.add_argument("--group-1", default=GROUP_1)
    parser.add_argument("--group-2", default=GROUP_2)
    parser.add_argument("--reference-cell-type", default=REFERENCE_CELL_TYPE)
    parser.add_argument("--samples-per-group", type=int)
    args = parser.parse_args(argv)
    manifest = run_preparation(
        args.input,
        args.output_root,
        spec=build_preparation_spec(
            run_id=args.run_id,
            analysis_id=args.analysis_id,
            cohort_id=args.cohort_id,
            group_1=args.group_1,
            group_2=args.group_2,
            reference_cell_type=args.reference_cell_type,
            samples_per_group=args.samples_per_group,
        ),
    )
    print(manifest)
    return manifest


# %% Interactive execution
result: Path | None = None

if __name__ == "__main__":
    result = main()
