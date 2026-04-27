import logging
import os
import re
from typing import Mapping, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.patches import Patch

from src.core.handlers.plot_wrapper import ScanpyPlotWrapper
from src.core.plot.utils import jitter_color, matplotlib_savefig
from src.utils.hier_logger import logged

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


@logged
def process_resolution_umaps(adata, output_dir, resolutions, use_raw=True, **kwargs):
    """批量绘制不同 Leiden resolution 的 UMAP 对比图。

    Args:
        adata: 输入 AnnData 对象。
        output_dir: 输出目录。
        resolutions: 需要展示的 resolution 列表，例如 `[0.2, 0.5, 1.0]`。
        use_raw: 是否优先使用 `adata.raw`。
        **kwargs: 透传给 `scanpy.pl.umap` 的参数。

    Returns:
        None

    Example:
        process_resolution_umaps(
            adata=adata,
            output_dir=save_addr,
            resolutions=[0.2, 0.5, 1.0],
            use_raw=False,
            legend_loc="on data",
        )
        # 输出图可用于直观比较不同 Leiden resolution 下的聚类分辨率
    """
    if "X_umap" not in adata.obsm:
        raise KeyError("Key `X_umap` was not found in `adata.obsm`.")
    if resolutions is None or len(resolutions) == 0:
        raise ValueError("Argument `resolutions` must not be empty.")

    color_keys = []
    for resolution in resolutions:
        color_key = f"leiden_res{resolution}"
        if color_key in adata.obs.columns:
            color_keys.append(color_key)
        else:
            logger.warning(
                f"[process_resolution_umaps] Warning! Column `{color_key}` was not found in `adata.obs`. Skip it."
            )

    if not color_keys:
        raise ValueError("No valid Leiden resolution columns were found in `adata.obs`.")

    if use_raw and getattr(adata, "raw", None) is None:
        logger.warning("[process_resolution_umaps] Warning! `use_raw` is True but `adata.raw` is not available. Fallback to processed data.")
        use_raw = False

    umap_plot = ScanpyPlotWrapper(sc.pl.umap)
    umap_plot(
        save_addr=output_dir,
        filename="Res_Comparison",
        adata=adata,
        color=color_keys,
        legend_loc="on data",
        use_raw=use_raw,
        **kwargs,
    )


@logged
def plot_QC_umap(adata, save_addr, filename_prefix):
    """按 QC 指标批量绘制 UMAP。

    该函数会自动在 `adata.obs` 中搜索与 organelle、cell cycle phase、
    counts / tissue / disease 等相关的列，并分组绘制 UMAP。

    Args:
        adata: 输入 AnnData 对象。
        save_addr: 输出目录。
        filename_prefix: 输出文件名前缀。

    Returns:
        dict，记录每个主题实际使用了哪些 `obs` 列。

    Example:
        used_keys = plot_QC_umap(
            adata=adata,
            save_addr=save_addr,
            filename_prefix="SampleA",
        )
        # 返回值 used_keys 可帮助检查哪些 QC 相关列被实际识别并用于绘图
    """
    if "X_umap" not in adata.obsm:
        raise KeyError("Key `X_umap` was not found in `adata.obsm`.")

    umap_plot = ScanpyPlotWrapper(sc.pl.umap)
    key_dict = {
        "organelles": [col for col in adata.obs.columns if re.search(r"mt|mito|rb|ribo", col, re.IGNORECASE)],
        "phase": [col for col in adata.obs.columns if re.search(r"phase", col, re.IGNORECASE)],
        "counts": [col for col in adata.obs.columns if re.search(r"count|disease|tissue", col, re.IGNORECASE)],
    }

    cleaned = {}
    for group_name, cols in key_dict.items():
        valid_cols = []
        for col in cols:
            dtype = adata.obs[col].dtype
            if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                continue
            valid_cols.append(col)
        if valid_cols:
            cleaned[group_name] = valid_cols

    key_list = [col for cols in cleaned.values() for col in cols]
    if len(key_list) == 0:
        raise ValueError("No suitable QC-related `obs` columns were found for UMAP plotting.")

    logger.info(f"[plot_QC_umap] Selected QC-related `obs` columns: {key_list}")
    for group_name, cols in cleaned.items():
        umap_plot(
            save_addr=save_addr,
            filename=f"{filename_prefix}_UMAP_{group_name}",
            adata=adata,
            color=cols,
        )
    return cleaned


@logged
def plot_hierarchical_umap(
    adata,
    save_addr,
    filename,
    hierarchy_dict,
    color_key="Subset_Identity",
    special_celltype_colors=None,
    figsize=(20, 10),
    umap_size=50,
    legend_cols=3,
    jitter_scale=0.15,
    random_seed=42,
    save=True,
    plot=False,
):
    """绘制带分层图例的 UMAP。

    Args:
        adata: 输入 AnnData 对象。
        save_addr: 输出目录。
        filename: 输出文件名。
        hierarchy_dict: 形如 `{major_type: [subset1, subset2]}` 的层级字典。
        color_key: `adata.obs` 中用于着色的列名。
        special_celltype_colors: 指定某些大类的固定颜色。
        figsize: 画布大小。
        umap_size: UMAP 点大小。
        legend_cols: 图例列数。
        jitter_scale: 同一大类下子群颜色扰动幅度。
        random_seed: 颜色扰动随机种子。
        save: 是否保存图像。
        plot: 是否显示图像。

    Returns:
        子群颜色映射字典。

    Example:
        hierarchy = {
            "T cell": ["CD4 T", "CD8 T", "Treg"],
            "Myeloid": ["Mono", "Macro", "DC"],
        }
        subset_colors = plot_hierarchical_umap(
            adata=adata,
            save_addr=save_addr,
            filename="hierarchical_umap",
            hierarchy_dict=hierarchy,
            color_key="Subset_Identity",
            special_celltype_colors={"Proliferative Cell": (0, 0, 0)},
        )
        # 返回的 subset_colors 可复用于其他图，保证层级配色一致
    """
    if "X_umap" not in adata.obsm:
        raise KeyError("Key `X_umap` was not found in `adata.obsm`.")
    if color_key not in adata.obs.columns:
        raise KeyError(f"Column `{color_key}` was not found in `adata.obs`.")
    if not isinstance(hierarchy_dict, Mapping) or len(hierarchy_dict) == 0:
        raise ValueError("Argument `hierarchy_dict` must be a non-empty dictionary.")

    special_celltype_colors = special_celltype_colors or {"Proliferative Cell": (0, 0, 0)}
    rng = np.random.default_rng(random_seed)
    subset_colors = {}
    abs_fig_path = os.path.join(save_addr, filename)

    normal_celltypes = [celltype for celltype in hierarchy_dict if celltype not in special_celltype_colors]
    base_palette = sns.color_palette("tab10", n_colors=max(len(normal_celltypes), 1))

    base_index = 0
    for celltype, subsets in hierarchy_dict.items():
        if celltype in special_celltype_colors:
            color = special_celltype_colors[celltype]
            for subset in subsets:
                subset_colors[subset] = color
        else:
            base_color = base_palette[base_index % len(base_palette)]
            for subset in subsets:
                subset_colors[subset] = jitter_color(base_color, scale=jitter_scale, rng=rng)
            base_index += 1

    missing_subsets = [subset for subsets in hierarchy_dict.values() for subset in subsets if subset not in adata.obs[color_key].astype(str).unique()]
    if missing_subsets:
        logger.warning(
            f"[plot_hierarchical_umap] Warning! Some subsets in `hierarchy_dict` were not found in `{color_key}`: "
            f"{missing_subsets}"
        )

    fig = plt.figure(figsize=figsize)
    ax_umap = fig.add_axes([0.0, 0.0, 0.70, 1.0])
    ax_leg = fig.add_axes([0.75, 0.0, 0.24, 1.0])
    ax_leg.axis("off")

    sc.pl.umap(
        adata,
        color=color_key,
        palette=subset_colors,
        ax=ax_umap,
        size=umap_size,
        alpha=0.8,
        legend_loc="none",
        show=False,
    )

    legend_elements = []
    for celltype, subsets in hierarchy_dict.items():
        legend_elements.append(Patch(facecolor="white", edgecolor="none", label=celltype))
        for subset in subsets:
            legend_elements.append(Patch(facecolor=subset_colors[subset], label=f"  {subset}"))

    legend = ax_leg.legend(
        handles=legend_elements,
        loc="center",
        frameon=False,
        fontsize=9,
        ncol=legend_cols,
        title="Cell Type Hierarchy",
        title_fontsize=10,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    for text in legend.get_texts():
        if not text.get_text().startswith("  "):
            text.set_fontweight("bold")

    if save:
        matplotlib_savefig(fig, abs_fig_path, close_after=False)
    if plot:
        plt.show()
    else:
        plt.close(fig)
    return subset_colors
