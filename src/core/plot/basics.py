import logging
import os
from typing import Mapping, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns

from src.core.handlers.plot_wrapper import ScanpyPlotWrapper
from src.core.plot.utils import matplotlib_savefig
from src.utils.hier_logger import logged

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


@logged
def plot_stacked_bar(
    cluster_counts: pd.DataFrame,
    cluster_palette: Optional[Sequence] = None,
    xlabel_rotation: int = 0,
    plot: bool = True,
    save_addr: Optional[str] = None,
    filename_prefix: Optional[str] = None,
    save: bool = True,
):
    """绘制细胞类群组成的堆叠柱状图。

    该函数常与 `get_cluster_counts` 或 `get_cluster_proportions` 配合使用，
    用于比较不同 sample、disease 或其他分组下的 cell subtype/subpopulation
    组成差异。

    Args:
        cluster_counts: 行为分组、列为细胞类群的 DataFrame。
        cluster_palette: 自定义颜色列表或颜色映射。
        xlabel_rotation: x 轴标签旋转角度。
        plot: 是否直接显示图像。
        save_addr: 输出目录；为空时使用当前工作目录。
        filename_prefix: 输出文件名前缀。
        save: 是否保存图像。

    Returns:
        生成的 Matplotlib figure 对象。

    Example:
        counts = get_cluster_counts(
            adata,
            obs_key="Subset_Identity",
            group_by="disease",
        )
        # 使用已有调色板比较不同 disease 下的细胞组成
        fig = plot_stacked_bar(
            cluster_counts=counts,
            cluster_palette=adata.uns["leiden_res1_colors"],
            filename_prefix="AllSample_Counts",
            save_addr=save_addr,
            save=True,
            plot=False,
        )
    """
    if not isinstance(cluster_counts, pd.DataFrame):
        raise TypeError("Argument `cluster_counts` must be a pandas DataFrame.")
    if cluster_counts.empty:
        raise ValueError("Argument `cluster_counts` must not be empty.")
    if not plot and not save:
        raise ValueError("At least one of `plot` or `save` must be True.")

    save_addr = save_addr or os.getcwd()
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    filename = f"{prefix}Stacked_Barplot"
    abs_fig_path = os.path.join(save_addr, filename)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    fig.patch.set_facecolor("white")

    cluster_counts.plot(kind="bar", stacked=True, ax=ax, color=cluster_palette)
    ax.legend(bbox_to_anchor=(1.01, 1), frameon=False, title="Cluster")
    sns.despine(fig=fig, ax=ax)
    ax.tick_params(axis="x", rotation=xlabel_rotation)
    ax.set_xlabel(cluster_counts.index.name.capitalize() if cluster_counts.index.name else "")
    ax.set_ylabel("Counts")
    fig.tight_layout()

    if save:
        matplotlib_savefig(fig, abs_fig_path, close_after=False)
    if plot:
        fig.show()
    else:
        plt.close(fig)
    return fig


@logged
def plot_stacked_violin(
    adata,
    output_dir,
    filename_prefix,
    save_addr,
    gene_dict,
    cell_type,
    obs_key="Subset_Identity",
    group_by="disease",
    split=False,
    **kwargs,
):
    """批量绘制 stacked violin 图。

    该函数通常用于在指定 cell subtype/subpopulation 中，比较一组 marker 或
    pathway genes 在不同 disease / sample 分组下的表达分布。

    Args:
        adata: 输入 AnnData 对象。
        output_dir: 输出目录。若为空则回退到 `save_addr`。
        filename_prefix: 输出文件名前缀。
        save_addr: 兼容旧接口保留的输出目录参数。
        gene_dict: 形如 `{gene_set_name: [genes]}` 的字典。
        cell_type: 单个 cell subtype 名称或其列表。
        obs_key: 用于筛选细胞子群的 `obs` 列名。
        group_by: stacked violin 的分组列名。
        split: 是否在文件名中标记 split 版本。
        **kwargs: 透传给 `scanpy.pl.stacked_violin` 的参数。

    Returns:
        None

    Example:
        pathway_genes = {
            "NFkB": ["NFKB1", "RELA", "TNFAIP3"],
            "MAPK": ["MAPK1", "MAPK3", "DUSP1"],
        }
        # 在 CD4 T cell 中比较不同 disease 分组下的通路相关基因表达
        plot_stacked_violin(
            adata=adata,
            output_dir=save_addr,
            save_addr=save_addr,
            filename_prefix="CD4T",
            gene_dict=pathway_genes,
            cell_type="CD4 T",
            obs_key="Subset_Identity",
            group_by="disease",
            swap_axes=False,
        )
    """
    if obs_key not in adata.obs.columns:
        raise KeyError(f"Column `{obs_key}` was not found in `adata.obs`.")
    if group_by not in adata.obs.columns:
        raise KeyError(f"Column `{group_by}` was not found in `adata.obs`.")
    if not isinstance(gene_dict, Mapping) or len(gene_dict) == 0:
        raise ValueError("Argument `gene_dict` must be a non-empty dictionary.")

    output_dir = output_dir or save_addr or os.getcwd()
    stacked_violin = ScanpyPlotWrapper(func=sc.pl.stacked_violin)

    if isinstance(cell_type, list):
        adata_subset = adata[adata.obs[obs_key].isin(cell_type)].copy()
    elif isinstance(cell_type, str):
        adata_subset = adata[adata.obs[obs_key] == cell_type].copy()
    else:
        raise TypeError("Argument `cell_type` must be a string or a list of strings.")

    if adata_subset.n_obs == 0:
        raise ValueError(f"No cells were matched for `{obs_key}` with value: '{cell_type}'.")

    use_raw = kwargs.pop("use_raw", False)
    if use_raw and getattr(adata_subset, "raw", None) is None:
        logger.warning("[plot_stacked_violin] Warning! `use_raw` is True but `adata.raw` is not available. Fallback to processed data.")
        use_raw = False

    default_params = {
        "swap_axes": False,
        "cmap": "viridis_r",
        "show": False,
        "use_raw": use_raw,
    }
    if not use_raw and "layer" not in kwargs and "log1p_norm" in adata_subset.layers:
        default_params["layer"] = "log1p_norm"
    default_params.update(kwargs)

    for gene_name, gene_list in gene_dict.items():
        if gene_list is None or len(gene_list) == 0:
            logger.warning(f"[plot_stacked_violin] Warning! Gene list is empty for gene set: '{gene_name}'. Skip plotting.")
            continue

        prefix = f"{filename_prefix}_" if filename_prefix else ""
        filename = f"{prefix}{gene_name}_Stacked_Violin{'(split)' if split else ''}.png"
        stacked_violin(
            filename=filename,
            save_addr=output_dir,
            adata=adata_subset,
            var_names=list(gene_list),
            groupby=group_by,
            **default_params,
        )


@logged
def plot_piechart(
    outer_count,
    inner_count,
    colormaplist,
    plot_title: Optional[str] = None,
    plot: bool = False,
    save: bool = True,
    save_path: Optional[str] = None,
    filename: Optional[str] = None,
):
    """绘制双层饼图。

    该图适合展示某一层级类群的整体构成，以及其内部进一步拆分后的比例。

    Args:
        outer_count: 外层计数序列，通常为更细一级的 cell subtype/subpopulation。
        inner_count: 内层计数序列，通常为更粗一级的分组。
        colormaplist: 颜色列表。
        plot_title: 图标题。
        plot: 是否直接显示图像。
        save: 是否保存图像。
        save_path: 输出目录。
        filename: 输出文件名。

    Returns:
        生成的 Matplotlib figure 对象。

    Example:
        # inner_count 可表示 major type，outer_count 可表示其下的 subset
        fig = plot_piechart(
            outer_count=subset_counts,
            inner_count=major_counts,
            colormaplist=sns.color_palette("Set3", len(subset_counts)),
            plot_title="Myeloid composition",
            save_path=save_addr,
            filename="myeloid_piechart",
        )
    """
    if outer_count is None or inner_count is None:
        raise ValueError("Arguments `outer_count` and `inner_count` must not be `None`.")
    if len(outer_count) == 0 or len(inner_count) == 0:
        raise ValueError("Arguments `outer_count` and `inner_count` must not be empty.")
    if len(colormaplist) < max(len(outer_count), len(inner_count)):
        raise ValueError("Argument `colormaplist` does not contain enough colors for the requested pie chart.")

    plot_title = plot_title or "Piechart"
    save_path = save_path or os.getcwd()
    filename = filename or "Piechart"
    abs_fig_path = os.path.join(save_path, filename)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.pie(
        x=outer_count,
        colors=colormaplist[: len(outer_count)],
        radius=0.8,
        pctdistance=0.765,
        autopct="%3.1f%%",
        labels=list(getattr(outer_count, "index", range(len(outer_count)))),
        textprops=dict(color="w"),
        wedgeprops=dict(width=0.3, edgecolor="w"),
    )
    ax.pie(
        x=inner_count,
        autopct="%3.1f%%",
        radius=1.0,
        pctdistance=0.85,
        colors=colormaplist[: len(inner_count)],
        textprops=dict(color="w"),
        labels=list(getattr(inner_count, "index", range(len(inner_count)))),
        wedgeprops=dict(width=0.3, edgecolor="w"),
    )
    ax.set_title(plot_title, fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(1, 0.5))

    if save:
        matplotlib_savefig(fig, abs_fig_path, close_after=False)
    if plot:
        plt.show()
    else:
        plt.close(fig)
    return fig
