"""CellRank 相关绘图函数。"""

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.core.plot.utils import matplotlib_savefig
from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


def _ensure_save_dir(save_addr: str) -> str:
    """检查并创建输出目录。"""
    if not isinstance(save_addr, str) or save_addr.strip() == "":
        raise ValueError("Argument `save_addr` must be a non-empty string.")
    save_addr = save_addr.strip()
    os.makedirs(save_addr, exist_ok=True)
    return save_addr


@logged
def plot_phase_diff_heatmap(save_addr, filename, df_plot, figsize_ratio=0.4, cmap="RdBu_r", center=0):
    """绘制 source 与 target velocity 差异热图。

    Args:
        save_addr: 图像输出目录。
        filename: 输出文件名主体，不带扩展名。
        df_plot: 行为基因、列为阶段或比较项的绘图矩阵。
        figsize_ratio: 行数与图高的比例系数。
        cmap: 热图配色。
        center: 热图中心值。

    Returns:
        `None`。

    Example:
        plot_phase_diff_heatmap(
            save_addr=save_addr,
            filename="Stem_to_Enterocyte_phase_driver",
            df_plot=df_plot,
            figsize_ratio=0.35,
        )
    """
    if not isinstance(df_plot, pd.DataFrame):
        raise TypeError("Argument `df_plot` must be a pandas DataFrame.")
    if df_plot.empty:
        raise ValueError("Argument `df_plot` must not be empty.")
    if not isinstance(filename, str) or filename.strip() == "":
        raise ValueError("Argument `filename` must be a non-empty string.")

    save_addr = _ensure_save_dir(save_addr)
    fig, ax = plt.subplots(figsize=(6, figsize_ratio * df_plot.shape[0] + 2))
    sns.heatmap(
        df_plot,
        cmap=cmap,
        center=center,
        linewidths=0.5,
        cbar_kws={"label": "Velocity"},
        ax=ax,
    )
    ax.set_title("Source vs Target Velocity (Phase Drivers)")
    ax.set_ylabel("Gene")
    ax.set_xlabel("")
    fig.tight_layout()

    abs_path = os.path.join(save_addr, filename.strip())
    matplotlib_savefig(fig, abs_path)
    logger.info(f"[plot_phase_diff_heatmap] Figure was saved with base filename: '{filename.strip()}'.")


@logged
def plot_driver_gene_corr_heatmap(
    save_addr,
    filename,
    merged_df,
    genes_of_interest=None,
    corr_suffix="_corr",
    min_top_genes=200,
    figsize=(10, 12),
    cmap="RdBu_r",
):
    """从相关性结果中筛选基因并绘制 driver-gene correlation heatmap。

    Args:
        save_addr: 图像输出目录。
        filename: 输出文件名主体，不带扩展名。
        merged_df: 行为基因、列包含 `*_corr` 的相关性结果表。
        genes_of_interest: 额外强制保留的关注基因列表。
        corr_suffix: 相关性列后缀。
        min_top_genes: 按最大相关性筛出的 Top 基因数量。
        figsize: 图像大小。
        cmap: 热图配色。

    Returns:
        `None`。

    Example:
        plot_driver_gene_corr_heatmap(
            save_addr=save_addr,
            filename="driver_corr_heatmap",
            merged_df=merged_df,
            genes_of_interest=["CFTR", "MUC2", "EPCAM"],
            min_top_genes=150,
        )
    """
    if not isinstance(merged_df, pd.DataFrame):
        raise TypeError("Argument `merged_df` must be a pandas DataFrame.")
    if merged_df.empty:
        raise ValueError("Argument `merged_df` must not be empty.")
    if not isinstance(filename, str) or filename.strip() == "":
        raise ValueError("Argument `filename` must be a non-empty string.")

    save_addr = _ensure_save_dir(save_addr)
    corr_cols = [column for column in merged_df.columns if column.endswith(corr_suffix)]
    if not corr_cols:
        raise ValueError(f"No columns ending with `{corr_suffix}` were found in `merged_df`.")

    df_plot = merged_df[corr_cols].copy()
    df_plot.columns = [column.replace(corr_suffix, "") for column in df_plot.columns]

    genes_of_interest = genes_of_interest or []
    genes_of_interest = list(dict.fromkeys(gene for gene in genes_of_interest if gene in df_plot.index))
    top_genes = df_plot.max(axis=1).sort_values(ascending=False).head(min_top_genes).index.tolist()
    final_genes = list(dict.fromkeys(genes_of_interest + top_genes))
    df_final = df_plot.loc[final_genes].fillna(0)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df_final,
        cmap=cmap,
        center=0,
        annot=False,
        cbar_kws={"label": "Correlation with Fate"},
        ax=ax,
    )
    ax.set_title("Driver Genes Correlation by Lineage Origin")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    abs_path = os.path.join(save_addr, filename.strip())
    matplotlib_savefig(fig, abs_path)
    logger.info(
        f"[plot_driver_gene_corr_heatmap] Figure was saved with base filename: '{filename.strip()}'."
    )
