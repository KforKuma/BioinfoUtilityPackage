import logging
import os
import warnings
from typing import Dict, Mapping, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

from src.core.adata.deg import easy_DEG
from src.core.plot.utils import matplotlib_savefig
from src.utils.hier_logger import logged

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def _thin_xticklabels(ax, label_step: int) -> None:
    """按固定步长稀疏显示 x 轴标签。"""
    if label_step <= 1:
        return

    labels = ax.get_xticklabels()
    new_labels = []
    for index, label in enumerate(labels):
        new_labels.append(label.get_text() if index % label_step == 0 else "")
    ax.set_xticklabels(new_labels, rotation=90)


def _extract_matrixplot_figure(matrixplot_result):
    """尽量稳健地从 Scanpy MatrixPlot 结果中提取 figure。"""
    if hasattr(matrixplot_result, "fig") and matrixplot_result.fig is not None:
        return matrixplot_result.fig
    if hasattr(matrixplot_result, "make_figure"):
        matrixplot_result.make_figure()
        if hasattr(matrixplot_result, "fig") and matrixplot_result.fig is not None:
            return matrixplot_result.fig
    return plt.gcf()


@logged
def plot_deg_matrixplot(
    adata,
    save_addr,
    filename,
    obs_key="Celltype",
    col_order=None,
    exclude_groups=None,
    rename_map=None,
    top_n_genes: int = 20,
    label_step: int = 5,
    save: bool = True,
    show: bool = False,
    **kwargs,
):
    """运行 DEG 后绘制 marker matrixplot。

    该函数适合在一个 cell subtype/subpopulation 列上先做 DEG，再自动为
    每个分组筛选较有代表性的 marker genes，并输出 matrixplot。

    Args:
        adata: 输入 AnnData 对象。
        save_addr: 输出目录。
        filename: 输出文件名。
        obs_key: 用于 DEG 与 matrixplot 分组的 `obs` 列名。
        col_order: 指定 cluster 展示顺序。
        exclude_groups: 需要排除的分组列表。
        rename_map: 用于展示名称替换的映射字典。
        top_n_genes: 每个分组最多保留多少个 marker genes。
        label_step: x 轴标签稀疏显示步长。
        save: 是否保存图像。
        show: 是否显示图像。
        **kwargs: 透传给 `scanpy.pl.matrixplot` 的参数。

    Returns:
        用于作图的 marker 字典。

    Example:
        marker_dict = plot_deg_matrixplot(
            adata=adata,
            save_addr=save_addr,
            filename="Tcell_matrixplot",
            obs_key="Subset_Identity",
            exclude_groups=["Proliferative Cell"],
            top_n_genes=15,
            label_step=3,
            cmap="magma",
        )
        # 返回值 marker_dict 可继续复用到 plot_matrixplot 中
    """
    if obs_key not in adata.obs.columns:
        raise KeyError(f"Column `{obs_key}` was not found in `adata.obs`.")
    if top_n_genes <= 0:
        raise ValueError("Argument `top_n_genes` must be greater than 0.")
    if label_step <= 0:
        raise ValueError("Argument `label_step` must be greater than 0.")

    exclude_groups = exclude_groups or ["Proliferative Cell"]
    if exclude_groups:
        adata = adata[~adata.obs[obs_key].isin(exclude_groups)].copy()
        logger.info(f"[plot_deg_matrixplot] Excluded groups from `{obs_key}`: {exclude_groups}.")
    if adata.n_obs == 0:
        raise ValueError("No cells remain after filtering `exclude_groups`.")

    adata = easy_DEG(
        adata,
        save_addr=save_addr,
        filename_prefix=filename,
        obs_key=obs_key,
        downsample=6000,
    )

    uns_key = f"deg_{obs_key}"
    if uns_key not in adata.uns:
        raise KeyError(f"Key `{uns_key}` was not found in `adata.uns` after DEG analysis.")

    groups = adata.uns[uns_key]["names"].dtype.names
    if groups is None:
        raise ValueError(f"No DEG groups were found under `adata.uns['{uns_key}']`.")

    target_order = list(col_order) if col_order is not None else list(groups)
    missing_groups = [group for group in target_order if group not in groups]
    if missing_groups:
        raise ValueError(f"Unknown groups were found in `col_order`: {missing_groups}.")

    df_all = pd.concat(
        [sc.get.rank_genes_groups_df(adata, group=group, key=uns_key).assign(cluster=group) for group in groups],
        ignore_index=True,
    )

    all_candidates = []
    for cluster in target_order:
        genes = df_all.query("cluster == @cluster & logfoldchanges > 1 & pvals_adj < 0.05").head(200)
        if genes.empty:
            logger.warning(
                f"[plot_deg_matrixplot] Warning! No strict DEG markers were found for cluster: '{cluster}'. "
                "Fallback to top-ranked genes without strict filtering."
            )
            genes = df_all.query("cluster == @cluster").head(max(top_n_genes * 5, top_n_genes))
        all_candidates.append(genes)

    df_candidates = pd.concat(all_candidates, ignore_index=True)
    df_unique = df_candidates.sort_values("logfoldchanges", ascending=False).drop_duplicates(
        subset=["names"],
        keep="first",
    )

    marker_dict = {}
    for cluster in target_order:
        cluster_genes = df_unique[df_unique["cluster"] == cluster].head(top_n_genes)["names"].dropna().tolist()
        if cluster_genes:
            marker_dict[cluster] = cluster_genes

    if not marker_dict:
        raise ValueError("No marker genes were collected for matrixplot generation.")

    if rename_map:
        marker_dict = {rename_map.get(key, key): value for key, value in marker_dict.items()}

    use_raw = kwargs.pop("use_raw", True)
    if use_raw and getattr(adata, "raw", None) is None:
        logger.warning("[plot_deg_matrixplot] Warning! `use_raw` is True but `adata.raw` is not available. Fallback to processed data.")
        use_raw = False

    base_params = {
        "adata": adata,
        "var_names": marker_dict,
        "groupby": obs_key,
        "use_raw": use_raw,
        "standard_scale": "var",
        "cmap": "magma",
        "figsize": (12, 6),
        "return_fig": True,
    }
    base_params.update(kwargs)

    mp = sc.pl.matrixplot(**base_params)
    fig = _extract_matrixplot_figure(mp)
    axes_dict = mp.get_axes() if hasattr(mp, "get_axes") else {}
    ax = axes_dict.get("mainplot_ax") or axes_dict.get("main_plot_ax")
    if ax is not None:
        _thin_xticklabels(ax, label_step=label_step)

    if save:
        abs_fig_path = os.path.join(save_addr, filename)
        matplotlib_savefig(fig, abs_fig_path, close_after=False)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return marker_dict


@logged
def plot_matrixplot(
    adata,
    marker_dict,
    save_addr,
    filename,
    obs_key="Celltype",
    label_step: int = 5,
    save: bool = True,
    show: bool = False,
    **kwargs,
):
    """根据给定 marker 字典直接绘制 matrixplot。

    Args:
        adata: 输入 AnnData 对象。
        marker_dict: 形如 `{cluster_name: [gene1, gene2]}` 的 marker 字典。
        save_addr: 输出目录。
        filename: 输出文件名。
        obs_key: 用于分组的 `obs` 列名。
        label_step: x 轴标签稀疏显示步长。
        save: 是否保存图像。
        show: 是否显示图像。
        **kwargs: 透传给 `scanpy.pl.matrixplot` 的参数。

    Returns:
        经过有效基因过滤后的 marker 字典。

    Example:
        marker_dict = {
            "CD4 T": ["IL7R", "LTB", "MALAT1"],
            "CD8 T": ["NKG7", "CCL5", "PRF1"],
        }
        filtered_markers = plot_matrixplot(
            adata=adata,
            marker_dict=marker_dict,
            save_addr=save_addr,
            filename="manual_matrixplot",
            obs_key="Subset_Identity",
            label_step=2,
        )
    """
    if obs_key not in adata.obs.columns:
        raise KeyError(f"Column `{obs_key}` was not found in `adata.obs`.")
    if not isinstance(marker_dict, Mapping) or len(marker_dict) == 0:
        raise ValueError("Argument `marker_dict` must be a non-empty dictionary.")
    if label_step <= 0:
        raise ValueError("Argument `label_step` must be greater than 0.")

    use_raw = kwargs.pop("use_raw", True)
    valid_genes = adata.raw.var_names if use_raw and getattr(adata, "raw", None) is not None else adata.var_names
    if use_raw and getattr(adata, "raw", None) is None:
        logger.warning("[plot_matrixplot] Warning! `use_raw` is True but `adata.raw` is not available. Fallback to processed data.")
        use_raw = False
        valid_genes = adata.var_names

    filtered_marker_dict = {}
    for cluster_name, genes in marker_dict.items():
        valid_gene_list = [gene for gene in genes if gene in valid_genes]
        missing_genes = [gene for gene in genes if gene not in valid_genes]
        if missing_genes:
            logger.info(f"[plot_matrixplot] Missing genes for cluster '{cluster_name}': {missing_genes}")
        if valid_gene_list:
            filtered_marker_dict[cluster_name] = valid_gene_list

    if not filtered_marker_dict:
        raise ValueError("No valid genes remain in `marker_dict` after filtering.")

    base_params = {
        "adata": adata,
        "var_names": filtered_marker_dict,
        "groupby": obs_key,
        "use_raw": use_raw,
        "standard_scale": "var",
        "cmap": "magma",
        "figsize": (12, 6),
        "return_fig": True,
    }
    base_params.update(kwargs)

    mp = sc.pl.matrixplot(**base_params)
    fig = _extract_matrixplot_figure(mp)
    axes_dict = mp.get_axes() if hasattr(mp, "get_axes") else {}
    ax = axes_dict.get("mainplot_ax") or axes_dict.get("main_plot_ax")
    if ax is not None:
        _thin_xticklabels(ax, label_step=label_step)

    if save:
        abs_fig_path = os.path.join(save_addr, filename)
        matplotlib_savefig(fig, abs_fig_path, close_after=False)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return filtered_marker_dict


@logged
def plot_professional_matrix(
    adata,
    gene_dict,
    groupby,
    save_addr=None,
    filename="matrixplot",
    dendrogram=False,
    cmap="RdYlBu_r",
    show_gene_groups=True,
    show_group_names=False,
    **kwargs,
):
    """绘制更适合展示和汇报的 matrixplot。

    该函数在普通 matrixplot 的基础上，增加了更稳健的 figure 提取与
    gene group 标题控制，适合用于报告或文章中的基因模块展示。

    Args:
        adata: 输入 AnnData 对象。
        gene_dict: 基因模块字典，形如 `{module_name: [genes]}`。
        groupby: 分组列名。
        save_addr: 输出目录；为空时只返回图而不保存。
        filename: 输出文件名。
        dendrogram: 是否绘制 dendrogram。
        cmap: 配色方案。
        show_gene_groups: 是否显示顶部 gene group bracket。
        show_group_names: 是否显示顶部 gene group 名称。
        **kwargs: 透传给 `scanpy.pl.matrixplot` 的参数。

    Returns:
        过滤后的 gene_dict。

    Example:
        pathway_dict = {
            "NFkB": ["NFKB1", "RELA", "TNFAIP3"],
            "MAPK": ["MAPK1", "MAPK3", "DUSP1"],
        }
        plot_professional_matrix(
            adata=adata,
            gene_dict=pathway_dict,
            groupby="Subset_Identity",
            save_addr=save_addr,
            filename="pathway_matrixplot",
            show_gene_groups=True,
            show_group_names=False,
        )
    """
    if groupby not in adata.obs.columns:
        raise KeyError(f"Column `{groupby}` was not found in `adata.obs`.")
    if not isinstance(gene_dict, Mapping) or len(gene_dict) == 0:
        raise ValueError("Argument `gene_dict` must be a non-empty dictionary.")

    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("fontTools").setLevel(logging.ERROR)

    filtered_gene_dict = {}
    for group_name, genes in gene_dict.items():
        existing_genes = [gene for gene in genes if gene in adata.var_names]
        if existing_genes:
            filtered_gene_dict[group_name] = existing_genes
        else:
            logger.warning(
                f"[plot_professional_matrix] Warning! No valid genes remain for gene group: '{group_name}'."
            )

    if not filtered_gene_dict:
        raise ValueError("No valid genes remain in `gene_dict` after filtering by `adata.var_names`.")

    var_names_input = (
        filtered_gene_dict
        if show_gene_groups
        else [gene for genes in filtered_gene_dict.values() for gene in genes]
    )

    plot_kwargs = dict(
        adata=adata,
        var_names=var_names_input,
        groupby=groupby,
        standard_scale="var",
        use_raw=False,
        cmap=cmap,
        return_fig=True,
        dendrogram=dendrogram,
    )
    plot_kwargs.update(kwargs)

    mp = sc.pl.matrixplot(**plot_kwargs)
    if hasattr(mp, "style"):
        mp.style(cmap=cmap, edge_color="white")

    fig = _extract_matrixplot_figure(mp)
    axes_dict = getattr(mp, "ax_dict", {}) or (mp.get_axes() if hasattr(mp, "get_axes") else {})
    if not axes_dict:
        logger.warning(
            "[plot_professional_matrix] Warning! MatrixPlot axes were not exposed as expected. Fallback to current figure."
        )
        axes_dict = {f"ax_{index}": ax for index, ax in enumerate(fig.get_axes())}

    gene_group_ax = axes_dict.get("gene_group_ax")
    if gene_group_ax is not None:
        if not show_gene_groups:
            gene_group_ax.set_visible(False)
        elif not show_group_names:
            for text in gene_group_ax.texts:
                text.set_visible(False)

    if save_addr:
        os.makedirs(save_addr, exist_ok=True)
        abs_path = os.path.join(save_addr, filename)
        matplotlib_savefig(fig, abs_path, close_after=False)
        logger.info(f"[plot_professional_matrix] Figure was saved to: '{abs_path}'.")

    plt.close(fig)
    return filtered_gene_dict
