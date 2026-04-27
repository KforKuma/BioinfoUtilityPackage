import logging
import os
from typing import Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.core.plot.utils import matplotlib_savefig
from src.utils.hier_logger import logged

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


@logged
def plot_pca_results(
    save_addr,
    filename,
    pca_df,
    exp_var,
    ax=None,
    figsize=(10, 7),
    title="PCA Analysis",
    show_labels=True,
    color_by="cluster",
):
    """绘制 PCA 二维散点图。

    Args:
        save_addr: 输出目录。
        filename: 输出文件名。
        pca_df: `run_pca_analysis` 返回的坐标表，至少包含 `PC1` 与 `PC2`。
        exp_var: PCA 各主成分的解释方差比例。
        ax: 可选的 Matplotlib axes；若为空则内部创建新图。
        figsize: 当 `ax` 为空时使用的图像尺寸。
        title: 图标题。
        show_labels: 是否在点旁标注 index 名称。
        color_by: 用于着色的列名；为空或列不存在时将退回单色绘图。

    Returns:
        生成的 Matplotlib figure 对象。

    Example:
        fig = plot_pca_results(
            save_addr=save_addr,
            filename="immune_pca",
            pca_df=pca_df,
            exp_var=exp_var,
            title="Immune cell subtype PCA",
            color_by="cluster",
            show_labels=True,
        )
        # 如果 pca_df.index 是 cell subtype 名称，标签会直接显示在散点旁
    """
    if not isinstance(pca_df, pd.DataFrame):
        raise TypeError("Argument `pca_df` must be a pandas DataFrame.")
    if "PC1" not in pca_df.columns or "PC2" not in pca_df.columns:
        raise KeyError("Columns `PC1` and `PC2` must exist in `pca_df`.")
    if len(exp_var) < 2:
        raise ValueError("Argument `exp_var` must contain at least 2 principal components.")

    owns_figure = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    pc1_var = float(exp_var[0]) * 100
    pc2_var = float(exp_var[1]) * 100
    hue_param = color_by if (color_by is not None and color_by in pca_df.columns) else None
    if color_by is not None and hue_param is None:
        logger.warning(
            f"[plot_pca_results] Warning! Column `{color_by}` was not found in `pca_df`. Fallback to single-color scatter."
        )

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue=hue_param,
        palette="viridis" if hue_param else None,
        s=120,
        edgecolor="w",
        alpha=0.8,
        ax=ax,
    )

    if show_labels:
        for index in range(len(pca_df)):
            ax.annotate(
                pca_df.index[index],
                (pca_df["PC1"].iloc[index], pca_df["PC2"].iloc[index]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                alpha=0.8,
            )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"PC1 ({pc1_var:.2f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pc2_var:.2f}%)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    if hue_param:
        ax.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc="upper left")

    abs_path = os.path.join(save_addr, filename)
    matplotlib_savefig(fig, abs_path, close_after=False)
    if owns_figure:
        plt.close(fig)
    return fig


@logged
def plot_cluster_expression_heatmap(
    merged_df,
    result_df,
    save_addr,
    filename,
    top_n_genes: int = 40,
    figsize=(12, 10),
    title="Cluster Specific Expression Pattern",
):
    """按 cluster 展示基因表达模式热图。

    Args:
        merged_df: 包含至少 `cell_type`、`names`、`scores` 三列的数据框。
        result_df: `run_pca_and_clustering` 返回的数据框，需包含 cluster 信息。
        save_addr: 输出目录。
        filename: 输出文件名。
        top_n_genes: 选择方差最高的前多少个 genes 进行展示。
        figsize: 热图尺寸。
        title: 图标题。

    Returns:
        过滤后的 cluster-by-gene 矩阵。

    Example:
        cluster_matrix = plot_cluster_expression_heatmap(
            merged_df=merged_df,
            result_df=result_df,
            save_addr=save_addr,
            filename="cluster_heatmap",
            top_n_genes=50,
            title="Cluster specific expression pattern",
        )
        # 返回值 cluster_matrix 可用于继续检查哪些基因主导不同 cluster 的差异
    """
    if not isinstance(merged_df, pd.DataFrame) or not isinstance(result_df, pd.DataFrame):
        raise TypeError("Arguments `merged_df` and `result_df` must be pandas DataFrames.")
    if top_n_genes <= 0:
        raise ValueError("Argument `top_n_genes` must be greater than 0.")

    required_merged_cols = {"cell_type", "names", "scores"}
    missing_merged_cols = required_merged_cols - set(merged_df.columns)
    if missing_merged_cols:
        raise KeyError(f"Required columns are missing in `merged_df`: {sorted(missing_merged_cols)}.")

    result_reset = result_df.reset_index()
    if "cluster" not in result_reset.columns:
        raise KeyError("Column `cluster` was not found in `result_df`.")

    if "cell_type" not in result_reset.columns:
        first_column = result_reset.columns[0]
        result_reset = result_reset.rename(columns={first_column: "cell_type"})
        logger.warning(
            f"[plot_cluster_expression_heatmap] Warning! Column `cell_type` was not found in `result_df`. "
            f"Fallback to first column: '{first_column}'."
        )

    cluster_map = result_reset.set_index("cell_type")["cluster"].to_dict()
    plot_df = merged_df.copy()
    plot_df["cluster"] = plot_df["cell_type"].map(cluster_map)
    plot_df = plot_df.dropna(subset=["cluster"])
    if plot_df.empty:
        raise ValueError("No rows remain after mapping `cell_type` to `cluster`.")

    cluster_matrix = plot_df.pivot_table(
        index="names",
        columns="cluster",
        values="scores",
        aggfunc="mean",
    ).fillna(0)
    if cluster_matrix.empty:
        raise ValueError("The cluster expression matrix is empty after pivoting.")

    gene_variance = cluster_matrix.var(axis=1).sort_values(ascending=False)
    top_genes = gene_variance.head(top_n_genes).index
    cluster_matrix_filtered = cluster_matrix.loc[top_genes]

    grid = sns.clustermap(
        cluster_matrix_filtered,
        cmap="RdYlBu_r",
        standard_scale=0,
        figsize=figsize,
        annot=False,
        cbar_kws={"label": "Relative Expression (Row Scaled)"},
    )
    grid.fig.suptitle(title, y=1.02, fontsize=16)

    abs_path = os.path.join(save_addr, filename)
    matplotlib_savefig(grid.fig, abs_path, close_after=False)
    plt.close(grid.fig)
    return cluster_matrix_filtered
