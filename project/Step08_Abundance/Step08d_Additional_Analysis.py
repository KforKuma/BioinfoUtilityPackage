"""Step08d: reproducible descriptive analyses over Step08a/Step08c artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from project.Step08_Abundance.phase4_shared import (
    LOCAL_WORK_ROOT,
    _file_records,
    _sha256,
    load_prepared_input,
)
from src.stats.validation import parse_boolean_series


# %% Retained legacy downstream analysis
# Original HPC output root:
# /public/home/xiongyuehan/data/IBD_analysis/output/Step08_Diff_Abundance
# The exact historical interactive source is retained below. Direct CLR_LMM calls and
# session-only count_df_sep_ls are not executed by the Phase-4 formal workflow.
LEGACY_INTERACTIVE_ANALYSIS = r'''
# conda activate sccoda-2025
##################################
import os, gc, sys
import numpy as np
import pandas as pd
import anndata
import warnings

warnings.warn(
    "Step08d_Additional_Analysis.py is optional legacy post-analysis and is outside the "
    "phase-2 canonical abundance pipeline.",
    DeprecationWarning,
    stacklevel=2,
)

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
save_fig_addr = f"{save_addr}/fig_addition"
os.makedirs(save_fig_addr, exist_ok=True)

##############################################################
# 聚类降维
##############################################################
from src.stats.plot.post_analysis import *
results_df = pd.read_csv(f"{save_addr}/0306_Realdata(separated)_Output(no_filter).csv", index_col=0)

date = "0311"
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

###########
gzmk_mask = df_all["cell_type"].isin(["CD8 Tmem GZMK+","CD4 Tmem GZMK+"])
df_all_modified = df_all.copy()
df_all_modified.loc[gzmk_mask,"cell_type"] = "GZMK+ T cell"


clr_lmm_result = run_CLR_LMM(df_all=df_all_modified,
            cell_type=("GZMK+ T cell","CD8 Trm"),
            formula="disease + C(tissue, Treatment(reference='nif'))",
            main_variable="disease",
            ref_label= "HC",
            alpha=0.05,
            group_label="sample_id")
print(clr_lmm_result["contrast_table"])


df_ratio = compute_ratio_df(df_all_modified,celltype_pair=("GZMK+ T cell","CD8 Trm"),disease_col="tissue")

plot_ratio_scatter(df_ratio,save_fig_addr,filename=f"{date}_CLR_Ratio_CD8Trm_All_GZMK(tissue)",
                   cell_pair=("GZMK+ T cell","CD8 Trm"),disease_col="tissue",
                   clr_lmm_result=clr_lmm_result)

df_ratio = compute_ratio_df(df_all_modified,celltype_pair=("GZMK+ T cell","CD8 Trm"),disease_col="disease")

plot_ratio_scatter(df_ratio,save_fig_addr,filename=f"{date}_CLR_Ratio_CD8Trm_All_GZMK(disease)",
                   cell_pair=("GZMK+ T cell","CD8 Trm"),disease_col="disease",
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
'''


# %% Configuration
OUTPUT_ROOT = LOCAL_WORK_ROOT
ADDITIONAL_RUN_ID = "phase4-step08-additional"
REAL_RUN_MANIFEST: Path | None = None
PREPARATION_MANIFEST: Path | None = None


# %% Reusable functions
def _verify_run_manifest(path: str | Path) -> tuple[Path, dict]:
    manifest_path = Path(path).resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "success" or manifest.get("mode") != "real_data":
        raise ValueError("Step08d requires a successful Step08c real-data run manifest.")
    run_root = manifest_path.parent
    for record in manifest.get("files", []):
        artifact = run_root / record["path"]
        if artifact.stat().st_size != int(record["bytes"]) or _sha256(artifact) != record["sha256"]:
            raise ValueError(f"Step08c artifact integrity failure: {record['path']}")
    return run_root, manifest


def _pca_tables(proportions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = proportions.to_numpy(dtype=float)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    components = min(2, len(singular), matrix.shape[1])
    scores = u[:, :components] * singular[:components]
    loadings = vt[:components].T
    score_frame = pd.DataFrame(
        scores,
        index=proportions.index,
        columns=[f"PC{index + 1}" for index in range(components)],
    ).reset_index()
    loading_frame = pd.DataFrame(
        loadings,
        index=proportions.columns,
        columns=[f"PC{index + 1}" for index in range(components)],
    ).reset_index(names="cell_type")
    return score_frame, loading_frame


def run_additional_analysis(
    real_run_manifest: str | Path,
    preparation_manifest: str | Path,
    output_root: str | Path = OUTPUT_ROOT,
    *,
    run_id: str = ADDITIONAL_RUN_ID,
) -> Path:
    """Read fixed upstream artifacts; never rerun DA or redefine decisions."""
    real_root, real_manifest = _verify_run_manifest(real_run_manifest)
    canonical, preparation = load_prepared_input(preparation_manifest)
    target = Path(output_root).resolve() / "runs" / run_id
    target.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    try:
        summaries = target / "summaries"
        figures = target / "figures"
        manifests = target / "manifests"
        summaries.mkdir(parents=True)
        figures.mkdir()
        manifests.mkdir()

        public = pd.read_csv(real_root / "canonical" / "contrast_public.csv")
        public["primary_decision"] = parse_boolean_series(public["primary_decision"])
        direction_score = public["effect_direction"].map({
            "group_1_higher": 1,
            "group_2_higher": -1,
            "no_effect": 0,
            "undetermined": 0,
            "not_applicable": 0,
        }).fillna(0)
        public["discovery_direction_score"] = np.where(
            public["primary_decision"].fillna(False), direction_score, 0
        )
        direction_matrix = public.pivot_table(
            index="cell_type",
            columns="method",
            values="discovery_direction_score",
            aggfunc="first",
            fill_value=0,
        ).reset_index()
        direction_matrix.to_csv(summaries / "method_direction_matrix.csv", index=False)

        proportions = canonical.abundance_long.pivot(
            index="sample_id", columns="cell_type", values="proportion"
        )
        proportions.corr().to_csv(summaries / "composition_correlation.csv")
        pca_scores, pca_loadings = _pca_tables(proportions)
        pca_scores.to_csv(summaries / "pca_scores.csv", index=False)
        pca_loadings.to_csv(summaries / "pca_loadings.csv", index=False)

        agreement = pd.read_csv(real_root / "summaries" / "method_agreement.csv")
        agreement.to_csv(summaries / "method_agreement.csv", index=False)
        status = pd.DataFrame([
            {"analysis": "composition_correlation", "status": "executed", "missing_dependency": pd.NA},
            {"analysis": "PCA", "status": "executed", "missing_dependency": pd.NA},
            {
                "analysis": "FA_NMF_ICA", "status": "retained_but_not_executable",
                "missing_dependency": "Phase-4 decomposition component/selection specification",
            },
            {
                "analysis": "ratio_analysis", "status": "retained_but_not_executable",
                "missing_dependency": "preregistered cell-type ratio pairs for the selected cohort",
            },
            {
                "analysis": "legacy_supplementary_figures", "status": "retained_but_not_executable",
                "missing_dependency": "legacy figure-specific design and labeling specification",
            },
        ])
        status.to_csv(summaries / "analysis_status.csv", index=False)

        os.environ.setdefault(
            "MPLCONFIGDIR", str(Path(output_root).resolve() / ".runtime_cache" / "matplotlib")
        )
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(6, 5))
        x = pca_scores.get("PC1", pd.Series(0.0, index=pca_scores.index))
        y = pca_scores.get("PC2", pd.Series(0.0, index=pca_scores.index))
        axis.scatter(x, y, s=30, alpha=0.8)
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.set_title("Step08 cohort composition PCA")
        figure.tight_layout()
        figure.savefig(figures / "composition_pca.png", dpi=150)
        plt.close(figure)

        handoff = {
            "run_id": run_id,
            "real_data_run_id": real_manifest["run_id"],
            "preparation_run_id": preparation["run_id"],
            "real_run_manifest": str(Path(real_run_manifest).resolve()),
            "preparation_manifest": str(Path(preparation_manifest).resolve()),
        }
        (manifests / "upstream_handoff.json").write_text(
            json.dumps(handoff, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        run_manifest = {
            **handoff,
            "mode": "additional_analysis",
            "status": "success",
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "files": _file_records(target),
        }
        (target / "run_manifest.json").write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target / "run_manifest.json"
    except Exception as exc:
        failure = {
            "run_id": run_id,
            "mode": "additional_analysis",
            "status": "failed",
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "files": _file_records(target),
        }
        (target / "run_manifest.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-run-manifest", type=Path, required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-id", default=ADDITIONAL_RUN_ID)
    args = parser.parse_args(argv)
    manifest = run_additional_analysis(
        args.real_run_manifest,
        args.preparation_manifest,
        args.output_root,
        run_id=args.run_id,
    )
    print(manifest)
    return manifest


# %% Interactive execution
result: Path | None = None

if __name__ == "__main__":
    result = main()
