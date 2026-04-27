import logging
import os
from typing import Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.sparse import issparse
from scipy.stats import mannwhitneyu

from src.core.plot.utils import matplotlib_savefig
from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


def _to_1d_array(matrix_like) -> np.ndarray:
    """将稀疏或稠密矩阵安全转为一维数组。"""
    if issparse(matrix_like):
        return np.asarray(matrix_like.toarray()).ravel()
    return np.asarray(matrix_like).ravel()


@logged
def verify_necessity_via_residuals(
    obs_df: pd.DataFrame,
    receptor_col: str,
    pathway_col: str,
    disease_col: str,
    target_group: str = "BD",
):
    """用残差分析评估 receptor 对 pathway 差异的解释贡献。

    Args:
        obs_df: 至少包含 receptor、pathway 与 disease 三列的 DataFrame。
        receptor_col: receptor 评分列名。
        pathway_col: pathway 评分列名。
        disease_col: disease 分组列名。
        target_group: 需要重点比较的目标分组。

    Returns:
        包含原始检验、残差检验和贡献比例的结果字典。

    Example:
        results = verify_necessity_via_residuals(
            obs_df=adata.obs,
            receptor_col="C3AR1_score",
            pathway_col="NFkB_score",
            disease_col="disease",
            target_group="BD",
        )
        # 输出中会同时打印原始差异和去除 receptor 影响后的残差差异
    """
    if not isinstance(obs_df, pd.DataFrame):
        raise TypeError("Argument `obs_df` must be a pandas DataFrame.")

    required_cols = [receptor_col, pathway_col, disease_col]
    missing_cols = [col for col in required_cols if col not in obs_df.columns]
    if missing_cols:
        raise KeyError(f"Required columns are missing in `obs_df`: {missing_cols}.")

    data = obs_df[required_cols].dropna().copy()
    if data.empty:
        raise ValueError("No valid rows remain after dropping missing values.")

    group_target = data[data[disease_col] == target_group]
    group_others = data[data[disease_col] != target_group]
    if group_target.empty or group_others.empty:
        raise ValueError(
            f"Both target and non-target groups must contain data for `target_group`: '{target_group}'."
        )

    X = sm.add_constant(data[receptor_col])
    y = data[pathway_col]
    model = sm.OLS(y, X).fit()
    data["residual"] = model.resid

    _, p_raw = mannwhitneyu(group_target[pathway_col], group_others[pathway_col])
    residual_target = data.loc[data[disease_col] == target_group, "residual"]
    residual_others = data.loc[data[disease_col] != target_group, "residual"]
    _, p_res = mannwhitneyu(residual_target, residual_others)

    raw_diff = group_target[pathway_col].mean() - group_others[pathway_col].mean()
    res_diff = residual_target.mean() - residual_others.mean()
    reduction = np.nan if raw_diff == 0 else (1 - res_diff / raw_diff) * 100

    print(f"[verify_necessity_via_residuals] Pathway: '{pathway_col}'")
    print(f"[verify_necessity_via_residuals] Raw group difference p-value: {p_raw:.2e}")
    print(f"[verify_necessity_via_residuals] Residual group difference p-value: {p_res:.2e}")
    if np.isnan(reduction):
        print("[verify_necessity_via_residuals] Warning! Raw difference is 0, reduction percentage is undefined.")
    else:
        print(f"[verify_necessity_via_residuals] Estimated contribution explained by receptor: {reduction:.2f}%")

    return {
        "model": model,
        "raw_p_value": p_raw,
        "residual_p_value": p_res,
        "raw_difference": raw_diff,
        "residual_difference": res_diff,
        "reduction_percent": reduction,
    }


@logged
def virtual_knockout_test(
    adata,
    save_addr,
    filename,
    receptor_col: str,
    pathway_col: str,
    disease_col: str,
    target_group: str = "BD",
    control_group: str = "HC",
):
    """通过低/高 receptor 分层模拟“虚拟敲除”效应。

    Args:
        adata: 输入 AnnData 对象。
        save_addr: 输出目录。
        filename: 输出文件名。
        receptor_col: receptor 评分列名。
        pathway_col: pathway 评分列名。
        disease_col: disease 分组列名。
        target_group: 目标 disease 分组。
        control_group: 对照 disease 分组。

    Returns:
        用于绘图的数据框。

    Example:
        plot_df = virtual_knockout_test(
            adata=adata,
            save_addr=save_addr,
            filename="virtual_knockout",
            receptor_col="C3AR1_score",
            pathway_col="NFkB_score",
            disease_col="disease",
            target_group="BD",
            control_group="HC",
        )
        # 结果图会比较 receptor-low 和 receptor-high 细胞中的 pathway score 分布
    """
    required_cols = [receptor_col, pathway_col, disease_col]
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise KeyError(f"Required columns are missing in `adata.obs`: {missing_cols}.")

    sub = adata.obs[adata.obs[disease_col].isin([target_group, control_group])].copy()
    if sub.empty:
        raise ValueError("No cells remain after filtering the requested disease groups.")

    low_thresh = sub[receptor_col].quantile(0.25)
    high_thresh = sub[receptor_col].quantile(0.75)
    sub["receptor_status"] = "Middle"
    sub.loc[sub[receptor_col] <= low_thresh, "receptor_status"] = "Low"
    sub.loc[sub[receptor_col] >= high_thresh, "receptor_status"] = "High"

    plot_data = sub[sub["receptor_status"].isin(["Low", "High"])].copy()
    if plot_data.empty:
        raise ValueError("No cells remain after defining `Low` and `High` receptor groups.")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(
        data=plot_data,
        x="receptor_status",
        y=pathway_col,
        hue=disease_col,
        order=["Low", "High"],
        palette="Set1",
        ax=ax,
    )
    ax.set_title("Necessity Check: receptor-low vs receptor-high cells")
    ax.set_xlabel("Receptor Status")
    ax.set_ylabel(pathway_col)

    abs_file_path = os.path.join(save_addr, filename)
    matplotlib_savefig(fig, abs_file_path, close_after=False)
    plt.close(fig)
    return plot_data


@logged
def necessity_statistical_test(
    adata,
    save_addr,
    filename,
    receptor_col: str,
    pathway_col: str,
    disease_col: str,
    target_group: str = "BD",
    control_group: str = "HC",
):
    """对“虚拟敲除”场景进行统计学检验并绘图。

    Args:
        adata: 输入 AnnData 对象。
        save_addr: 输出目录。
        filename: 输出文件名。
        receptor_col: receptor 评分列名。
        pathway_col: pathway 评分列名。
        disease_col: disease 分组列名。
        target_group: 目标 disease 分组。
        control_group: 对照 disease 分组。

    Returns:
        包含交互项检验和分组差异的结果字典。

    Example:
        stats_result = necessity_statistical_test(
            adata=adata,
            save_addr=save_addr,
            filename="necessity_test",
            receptor_col="C3AR1_score",
            pathway_col="NFkB_score",
            disease_col="disease",
            target_group="BD",
            control_group="HC",
        )
        # 图中会展示 Low/High receptor 状态下，不同 disease 组的 pathway score 差异
    """
    required_cols = [receptor_col, pathway_col, disease_col]
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise KeyError(f"Required columns are missing in `adata.obs`: {missing_cols}.")

    plot_df = adata.obs[adata.obs[disease_col].isin([target_group, control_group])].copy()
    if plot_df.empty:
        raise ValueError("No cells remain after filtering the requested disease groups.")

    low_cutoff = plot_df[receptor_col].quantile(0.20)
    high_cutoff = plot_df[receptor_col].quantile(0.80)
    plot_df["receptor_status"] = "Middle"
    plot_df.loc[plot_df[receptor_col] <= low_cutoff, "receptor_status"] = "Low"
    plot_df.loc[plot_df[receptor_col] >= high_cutoff, "receptor_status"] = "High"

    test_df = plot_df[plot_df["receptor_status"].isin(["Low", "High"])].copy()
    if test_df.empty:
        raise ValueError("No cells remain after defining `Low` and `High` receptor groups.")

    try:
        from statannotations.Annotator import Annotator
    except Exception as exc:
        raise ImportError(
            "Package `statannotations` is required for `necessity_statistical_test`."
        ) from exc

    test_df["receptor_status"] = pd.Categorical(test_df["receptor_status"], categories=["Low", "High"])
    model = smf.ols(f"{pathway_col} ~ receptor_status * {disease_col}", data=test_df).fit()
    interaction_p = model.pvalues.iloc[-1]

    pairs = [
        (("Low", target_group), ("Low", control_group)),
        (("High", target_group), ("High", control_group)),
    ]

    fig, ax = plt.subplots(figsize=(5, 6))
    sns.boxplot(
        data=test_df,
        x="receptor_status",
        y=pathway_col,
        hue=disease_col,
        palette="Set1",
        width=0.6,
        showfliers=False,
        ax=ax,
    )
    annotator = Annotator(ax, pairs, data=test_df, x="receptor_status", y=pathway_col, hue=disease_col)
    annotator.configure(test="Mann-Whitney", text_format="star", loc="outside")
    annotator.apply_and_annotate()

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.2)
    ax.set_title(f"Necessity Check: {pathway_col}\nInteraction p-value: {interaction_p:.2e}")
    ax.set_ylabel(f"{pathway_col} Score")

    abs_file_path = os.path.join(save_addr, filename)
    matplotlib_savefig(fig, abs_file_path, close_after=False)
    plt.close(fig)

    mean_diffs = {}
    for status in ["Low", "High"]:
        sub = test_df[test_df["receptor_status"] == status]
        diff = (
            sub[sub[disease_col] == target_group][pathway_col].mean()
            - sub[sub[disease_col] == control_group][pathway_col].mean()
        )
        mean_diffs[status] = diff
        print(f"[necessity_statistical_test] Mean difference in '{status}' receptor cells ({target_group} - {control_group}): {diff:.4f}")

    return {
        "model": model,
        "interaction_p_value": interaction_p,
        "mean_differences": mean_diffs,
    }


@logged
def analyze_trail_decoy_effect(
    adata,
    save_addr,
    filename,
    target_cells_mask,
    gene_dict: Mapping[str, list],
    color_by: str = "RB_Cluster",
):
    """分析 DcR2 优势比与下游信号的关联。

    Args:
        adata: 输入 AnnData 对象。
        save_addr: 输出目录。
        filename: 输出文件名。
        target_cells_mask: 用于筛选目标细胞的布尔索引。
        gene_dict: 需要打分的基因集字典。
        color_by: 散点着色所依据的 `obs` 列名。

    Returns:
        打分并补充 `RB_Score` 后的子集 AnnData 对象。

    Example:
        subset = analyze_trail_decoy_effect(
            adata=adata,
            save_addr=save_addr,
            filename="trail_decoy_effect",
            target_cells_mask=adata.obs["Subset_Identity"].isin(["Mono", "Macro"]),
            gene_dict={"NFkB": ["NFKB1", "RELA"], "Apoptosis": ["BAX", "CASP3"]},
            color_by="RB_Cluster",
        )
        # 返回的 subset.obs 中会新增 `RB_Score` 与各 pathway score
    """
    if not isinstance(gene_dict, Mapping) or len(gene_dict) == 0:
        raise ValueError("Argument `gene_dict` must be a non-empty dictionary.")

    subset = adata[target_cells_mask].copy()
    if subset.n_obs == 0:
        raise ValueError("No cells were selected by `target_cells_mask`.")

    dcr2_val = (
        _to_1d_array(subset[:, "TNFRSF10D"].X)
        if "TNFRSF10D" in subset.var_names else np.zeros(subset.n_obs, dtype=float)
    )
    dr_genes = [gene for gene in ["TNFRSF10A", "TNFRSF10B"] if gene in subset.var_names]
    dr_sum = (
        np.asarray(subset[:, dr_genes].X.sum(axis=1)).ravel()
        if dr_genes else np.zeros(subset.n_obs, dtype=float)
    )
    subset.obs["RB_Score"] = np.log1p(dcr2_val / (dr_sum + 1))

    for pathway_name, genes in gene_dict.items():
        valid_genes = [gene for gene in genes if gene in subset.var_names]
        if not valid_genes:
            logger.warning(
                f"[analyze_trail_decoy_effect] Warning! No valid genes remain for pathway: '{pathway_name}'."
            )
            continue
        sc.tl.score_genes(subset, gene_list=valid_genes, score_name=f"{pathway_name}_Score")

    plot_df = subset.obs.copy()
    scatter_colors = None
    if color_by in plot_df.columns:
        labels = plot_df[color_by].astype(str)
        unique_cats = np.unique(labels)
        palette = sns.color_palette("husl", len(unique_cats))
        color_map = dict(zip(unique_cats, palette))
        scatter_colors = [color_map[label] for label in labels]
    else:
        logger.warning(
            f"[analyze_trail_decoy_effect] Warning! Column `{color_by}` was not found in `adata.obs`. Fallback to default scatter color."
        )

    num_plots = len(gene_dict)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    for index, (pathway_name, _) in enumerate(gene_dict.items()):
        score_col = f"{pathway_name}_Score"
        if score_col not in plot_df.columns:
            logger.warning(
                f"[analyze_trail_decoy_effect] Warning! Column `{score_col}` was not generated. Skip this pathway."
            )
            continue

        sns.regplot(
            data=plot_df,
            x="RB_Score",
            y=score_col,
            ax=axes[index],
            scatter_kws={"alpha": 0.4, "s": 15, "color": None},
            line_kws={"color": "red", "lw": 2},
        )
        if scatter_colors is not None and len(axes[index].collections) > 0:
            axes[index].collections[0].set_facecolor(scatter_colors)
            axes[index].collections[0].set_edgecolor(scatter_colors)

        correlation = plot_df["RB_Score"].corr(plot_df[score_col])
        axes[index].set_title(f"DcR2 Dominance vs {pathway_name}\n(Pearson r = {correlation:.2f})")
        axes[index].set_xlabel("Log1p(DcR2 / (DRs + 1))")
        axes[index].set_ylabel(f"{pathway_name} Pathway Score")

    plt.tight_layout()
    abs_file_path = os.path.join(save_addr, filename)
    matplotlib_savefig(fig, abs_file_path, close_after=False)
    plt.close(fig)
    return subset


@logged
def plot_kde_comparison(
    df: pd.DataFrame,
    save_addr,
    filename,
    x: str = "RB_Score",
    y: str = "NFkB_Score",
    hue: str = "RB_Cluster",
    title_prefix: str = "",
):
    """绘制二维 KDE 与散点叠加图。

    Args:
        df: 输入数据框。
        save_addr: 输出目录。
        filename: 输出文件名。
        x: 横轴列名。
        y: 纵轴列名。
        hue: 着色分组列名。
        title_prefix: 图标题前缀。

    Returns:
        生成的 Matplotlib figure 对象。

    Example:
        fig = plot_kde_comparison(
            df=subset.obs,
            save_addr=save_addr,
            filename="rbscore_kde",
            x="RB_Score",
            y="NFkB_Score",
            hue="RB_Cluster",
            title_prefix="Myeloid",
        )
        # 适合快速查看 receptor 优势比与 pathway activity 的联合分布
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Argument `df` must be a pandas DataFrame.")
    for column in [x, y, hue]:
        if column not in df.columns:
            raise KeyError(f"Column `{column}` was not found in `df`.")

    fig, ax = plt.subplots(figsize=(8, 6))
    current_palette = "viridis" if hue == "RB_Cluster" else "Set2"

    sns.kdeplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        fill=True,
        alpha=0.5,
        levels=5,
        palette=current_palette,
        ax=ax,
    )
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        s=5,
        alpha=0.2,
        legend=False,
        palette=current_palette,
        ax=ax,
    )

    ax.set_title(f"{title_prefix} Relation between {x} and {y} (Hue: {hue})")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    abs_file_path = os.path.join(save_addr, filename)
    matplotlib_savefig(fig, abs_file_path, close_after=False)
    plt.close(fig)
    return fig
