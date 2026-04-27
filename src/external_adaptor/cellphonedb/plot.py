"""CellPhoneDB 结果可视化工具。

本模块主要封装 CellPhoneDB / ktplotspy 的常用可视化接口，并补充若干
适用于项目内下游整合图的辅助函数。整体设计偏向保守兼容：

1. 尽量保持现有公开函数名与主要参数不变，减少对旧脚本的影响。
2. 对常见输入缺失、空结果和导出失败场景增加兜底与更清晰的英文提示。
3. 对抽象或高频函数补充更详细的 `Example`，方便后续直接复用。
"""

import logging
import os
from typing import Any, Iterable, Optional, Sequence, Tuple

import ktplotspy as kpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.path import Path

from src.core.plot.utils import matplotlib_savefig
from src.external_adaptor.cellphonedb.settings import DEFAULT_CPDB_SEP
from src.external_adaptor.cellphonedb.toolkit import size_map
from src.utils.hier_logger import logged

plt.rcParams["font.family"] = "monospace"

logger = logging.getLogger(__name__)


def _validate_output_dir(output_dir: str) -> str:
    """验证并创建输出目录。"""
    if not isinstance(output_dir, str) or output_dir.strip() == "":
        raise ValueError("Argument `output_dir` must be a non-empty string.")
    output_dir = output_dir.strip()
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _build_base_path(output_dir: str, filename_prefix: Optional[str], suffix: str) -> Tuple[str, str]:
    """构造导出文件的基础路径与基础文件名。"""
    output_dir = _validate_output_dir(output_dir)
    prefix = ""
    if filename_prefix is not None:
        if not isinstance(filename_prefix, str):
            raise TypeError("Argument `filename_prefix` must be a string or `None`.")
        cleaned = filename_prefix.strip().strip("_")
        prefix = f"{cleaned}_" if cleaned else ""

    filename = f"{prefix}{suffix}"
    return os.path.join(output_dir, filename), filename


def _save_plotnine_plot(plot_obj: Any, base_path: str, save_pdf: bool, save_png: bool) -> None:
    """保存 plotnine 对象。"""
    if save_pdf:
        plot_obj.save(f"{base_path}.pdf", dpi=300, bbox_inches="tight")
    if save_png:
        plot_obj.save(f"{base_path}.png", dpi=300, bbox_inches="tight")


def _require_columns(df: pd.DataFrame, required_columns: Iterable[str], arg_name: str = "df") -> None:
    """检查 DataFrame 是否包含必需列。"""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(
            f"Columns {missing} were not found in `{arg_name}`. "
            f"Available columns are: {list(df.columns)}."
        )


def _extract_expression(adata: Any, gene_name: str) -> np.ndarray:
    """从 AnnData 或其 `raw` 中提取单基因表达向量。"""
    source = adata.raw if getattr(adata, "raw", None) is not None else adata
    if gene_name not in source.var_names:
        return np.zeros(adata.n_obs, dtype=float)

    matrix = source[:, gene_name].X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix).ravel().astype(float)


def _safe_norm_range(values: Sequence[float]) -> Tuple[float, float]:
    """为颜色映射提供稳定的上下界。"""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 1.0

    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return vmin, vmax


@logged
def cpdb_heatmap(
    CIObject,
    output_dir,
    filename_prefix,
    save_pdf=True,
    save_png=True,
    show=False,
    **kwargs,
):
    """绘制 CellPhoneDB 交互热图。

    该函数对 `ktplotspy.plot_cpdb_heatmap()` 进行轻量封装，负责从
    `CellphoneInspector` 中提取 `pvals` 与 `deg` 信息，并统一处理结果导出。

    Args:
        CIObject: `CellphoneInspector` 实例，需至少包含 `data["pvals"]`。
        output_dir: 图像输出目录。
        filename_prefix: 输出文件名前缀；若为 `None` 或空字符串，则仅使用默认文件名。
        save_pdf: 是否导出 `.pdf`。
        save_png: 是否导出 `.png`。
        show: 是否在当前会话中显示图像。
        **kwargs: 透传给 `ktplotspy.plot_cpdb_heatmap()` 的其他参数。

    Returns:
        `seaborn.matrix.ClusterGrid` 对象。

    Example:
        heatmap = cpdb_heatmap(
            CIObject=ci,
            output_dir=save_addr,
            filename_prefix="Th17_vs_Bcell",
            alpha=0.05,
            row_cluster=True,
            col_cluster=True,
        )
        # 返回值仍可用于继续访问 fig.fig、ax_heatmap 等内部对象
        heatmap.fig.suptitle("CPDB interaction heatmap")
    """
    pvals = CIObject.data.get("pvals", None)
    degs = getattr(CIObject, "deg", None)
    if pvals is None:
        raise ValueError("Key `pvals` was not found in `CIObject.data`.")

    default_param = {
        "pvals": pvals,
        "cell_types": None,
        "degs_analysis": degs,
        "log1p_transform": False,
        "alpha": 0.05,
        "linewidths": 0.05,
        "row_cluster": True,
        "col_cluster": True,
        "low_col": "#104e8b",
        "mid_col": "#ffdab9",
        "high_col": "#8b0a50",
        "cmap": None,
        "title": "",
        "return_tables": False,
        "symmetrical": True,
        "default_sep": DEFAULT_CPDB_SEP,
    }
    default_param.update(kwargs)

    fig = kpy.plot_cpdb_heatmap(**default_param)
    base_path, filename = _build_base_path(output_dir, filename_prefix, "CPDB_Heatmap")

    if save_pdf:
        fig.savefig(f"{base_path}.pdf", dpi=300, bbox_inches="tight")
    if save_png:
        fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")

    logger.info(f"[cpdb_heatmap] Figure was saved with base filename: '{filename}'.")
    if show:
        plt.show()
    else:
        plt.close(fig.fig)

    return fig


@logged
def cpdb_dotplot(
    CIObject,
    AnndataObject,
    cell_type1,
    cell_type2,
    celltype_key,
    output_dir,
    filename_prefix,
    interacting_pairs=None,
    save_pdf=True,
    save_png=True,
    show=False,
    **kwargs,
):
    """绘制 CellPhoneDB dotplot。

    该函数适合快速查看两个 cell subtype/subpopulation 之间的配体-受体显著性、
    均值表达和可选的 interaction score。若 `interacting_pairs` 未提供，通常会退回到
    ktplotspy 的默认筛选逻辑，仅展示显著互作。

    Args:
        CIObject: `CellphoneInspector` 实例，需包含 `data["pvals"]` 与 `data["means"]`。
        AnndataObject: 原始 AnnData 对象。
        cell_type1: 第一组 cell subtype/subpopulation。
        cell_type2: 第二组 cell subtype/subpopulation。
        celltype_key: `adata.obs` 中表示 cell subtype 的列名。
        output_dir: 图像输出目录。
        filename_prefix: 输出文件名前缀。
        interacting_pairs: 需要重点展示的 interaction 查询结果、基因列表或配体-受体对。
        save_pdf: 是否导出 `.pdf`。
        save_png: 是否导出 `.png`。
        show: 是否在当前会话中显示或打印图对象。
        **kwargs: 透传给 `ktplotspy.plot_cpdb()` 的其他参数。

    Returns:
        底层 `plotnine` 图对象。

    Example:
        gene_query = prepare_gene_query(ci, genes=["IL2", "IL6"])
        dotplot = cpdb_dotplot(
            CIObject=ci,
            AnndataObject=adata,
            cell_type1="Th17",
            cell_type2="B cell",
            celltype_key="Subset_Identity",
            output_dir=save_addr,
            filename_prefix="Th17_Bcell",
            interacting_pairs=gene_query,
            keep_significant_only=True,
            alpha=0.05,
        )
        # 在 notebook 或后续脚本中，可继续使用返回值做额外保存或显示
        dotplot
    """
    pvals = CIObject.data.get("pvals", None)
    means = CIObject.data.get("means", None)
    interaction_scores = CIObject.data.get("interaction_scores", None)
    cellsign = CIObject.data.get("cellsign_interactions", None)

    if pvals is None or means is None:
        raise ValueError("Required keys `pvals` and `means` must exist in `CIObject.data`.")
    if celltype_key not in AnndataObject.obs.columns:
        raise KeyError(
            f"Column `{celltype_key}` was not found in `AnndataObject.obs`. "
            f"Available columns are: {list(AnndataObject.obs.columns)}."
        )

    default_param = {
        "adata": AnndataObject,
        "cell_type1": cell_type1,
        "cell_type2": cell_type2,
        "celltype_key": celltype_key,
        "means": means,
        "pvals": pvals,
        "interaction_scores": interaction_scores,
        "cellsign": cellsign,
        "degs_analysis": getattr(CIObject, "deg", None),
        "alpha": 0.05,
        "keep_significant_only": True,
        "interacting_pairs": interacting_pairs,
        "cmap_name": "cividis",
        "highlight_col": "#cf5c60",
        "title": "CellphoneDB Dotplot",
        "min_interaction_score": 0,
    }
    default_param.update(kwargs)

    g = kpy.plot_cpdb(**default_param)
    base_path, filename = _build_base_path(output_dir, filename_prefix, "CPDB_Dotplot")
    _save_plotnine_plot(g, base_path, save_pdf, save_png)

    logger.info(f"[cpdb_dotplot] Figure was saved with base filename: '{filename}'.")
    if show:
        print(f"[cpdb_dotplot] Plot object preview: {g}")

    return g


@logged
def cpdb_chordplot(
    CIObject,
    AnndataObject,
    cell_type1,
    cell_type2,
    celltype_key,
    interaction,
    output_dir,
    filename_prefix,
    save_pdf=True,
    save_png=True,
    show=False,
    **kwargs,
):
    """绘制 CellPhoneDB chordplot。

    `interaction` 既可以是单个基因、基因列表，也可以是上游 `prepare_gene_query()`
    的输出。对于名称较抽象的场景，推荐优先传入少量基因或已经整理好的查询列表，
    这样更容易控制最终显示内容。

    Args:
        CIObject: `CellphoneInspector` 实例，需包含 `data["pvals"]` 与 `data["means"]`。
        AnndataObject: 原始 AnnData 对象。
        cell_type1: 第一组 cell subtype/subpopulation。
        cell_type2: 第二组 cell subtype/subpopulation。
        celltype_key: `adata.obs` 中表示 cell subtype 的列名。
        interaction: 单个基因、基因列表或预先构造好的 interaction 查询。
        output_dir: 图像输出目录。
        filename_prefix: 输出文件名前缀。
        save_pdf: 是否导出 `.pdf`。
        save_png: 是否导出 `.png`。
        show: 是否在当前会话中显示或打印图对象。
        **kwargs: 透传给 `ktplotspy.plot_cpdb_chord()` 的其他参数。

    Returns:
        底层绘图对象。

    Example:
        chordplot = cpdb_chordplot(
            CIObject=ci,
            AnndataObject=adata,
            cell_type1="Th17",
            cell_type2="B cell",
            celltype_key="Subset_Identity",
            interaction=["PTPRC", "CD40", "CLEC2D"],
            output_dir=save_addr,
            filename_prefix="Th17_Bcell_chord",
        )
        # 若 adata.uns 中已有现成颜色信息，ktplotspy 往往会自动复用
        chordplot
    """
    pvals = CIObject.data.get("pvals", None)
    means = CIObject.data.get("means", None)
    decon = CIObject.data.get("deconvoluted", None)

    if pvals is None or means is None:
        raise ValueError("Required keys `pvals` and `means` must exist in `CIObject.data`.")
    if celltype_key not in AnndataObject.obs.columns:
        raise KeyError(
            f"Column `{celltype_key}` was not found in `AnndataObject.obs`. "
            f"Available columns are: {list(AnndataObject.obs.columns)}."
        )

    default_param = {
        "adata": AnndataObject,
        "cell_type1": cell_type1,
        "cell_type2": cell_type2,
        "celltype_key": celltype_key,
        "means": means,
        "pvals": pvals,
        "deconvoluted": decon,
        "interaction": interaction,
        "link_kwargs": {"direction": 1, "allow_twist": True, "r1": 95, "r2": 90},
        "sector_text_kwargs": {"color": "black", "size": 12, "r": 105, "adjust_rotation": True},
        "legend_kwargs": {"loc": "center", "bbox_to_anchor": (1, 1), "fontsize": 8},
        "link_offset": 1,
    }
    default_param.update(kwargs)

    g = kpy.plot_cpdb_chord(**default_param)
    base_path, filename = _build_base_path(output_dir, filename_prefix, "CPDB_Chordplot")
    _save_plotnine_plot(g, base_path, save_pdf, save_png)

    logger.info(f"[cpdb_chordplot] Figure was saved with base filename: '{filename}'.")
    if show:
        print(f"[cpdb_chordplot] Plot object preview: {g}")

    return g


@logged
def draw_combine_dotplot(
    df_full,
    save_addr,
    filename,
    vline_pairs,
    interaction_order,
    facet_aspect=1,
    facet_height=0.9,
    fig_width=20,
    fig_height=16,
    left=0.08,
    right=0.82,
    bottom=0.2,
    top=0.95,
    hspace=0.12,
    group_list=["HC", "Colitis", "BD", "CD", "UC"],
):
    """绘制多分组组合 dotplot。

    该函数通常用于 `search_df()` 等上游整理结果的后续展示。左侧为不同组别下的
    interaction dotplot，右侧补充统一的颜色条与气泡大小图例，便于跨疾病状态比较。

    Args:
        df_full: 已整理好的长表，至少包含 `group`、`celltype_group`、`interaction_group`、
            `scaled_means`、`dot_size` 和 `significant` 列。
        save_addr: 输出目录。
        filename: 输出文件名主体，不带扩展名。
        vline_pairs: 需要在 x 轴上插入虚线分隔的位置对，例如 `[("A", "B")]`。
        interaction_order: interaction 在 y 轴上的显示顺序。
        facet_aspect: 每个 facet 的宽高比。
        facet_height: 每个 facet 的高度。
        fig_width: 总图宽度。
        fig_height: 总图高度。
        left: 图左边距。
        right: 图右边距。
        bottom: 图下边距。
        top: 图上边距。
        hspace: facet 之间的垂直间距。
        group_list: facet 的组别显示顺序。

    Returns:
        `None`。图像与数据表会被导出到 `save_addr`。

    Example:
        draw_combine_dotplot(
            df_full=df_full,
            save_addr=save_addr,
            filename="Th17_Bcell_all_groups",
            vline_pairs=[("HC>@<B cell", "Colitis>@<B cell")],
            interaction_order=interaction_order,
            fig_width=18,
            fig_height=14,
        )
        # 同目录下会额外输出一个 `(data).xlsx` 便于人工核对绘图输入
    """
    if not isinstance(df_full, pd.DataFrame):
        raise TypeError("Argument `df_full` must be a pandas DataFrame.")
    if df_full.empty:
        raise ValueError("Argument `df_full` must not be empty.")

    _require_columns(
        df_full,
        ["group", "celltype_group", "interaction_group", "scaled_means", "dot_size", "significant"],
        arg_name="df_full",
    )
    save_addr = _validate_output_dir(save_addr)
    if not isinstance(filename, str) or filename.strip() == "":
        raise ValueError("Argument `filename` must be a non-empty string.")

    plot_df = df_full.copy()
    if (plot_df["scaled_means"] < 0).any():
        logger.info(
            "[draw_combine_dotplot] Warning! Negative values were found in `scaled_means`; "
            "they will be clipped to 0 before color mapping."
        )
        plot_df["scaled_means"] = plot_df["scaled_means"].clip(lower=0)

    color_vals = np.sqrt(plot_df["scaled_means"].astype(float))
    vmin, vmax = _safe_norm_range(color_vals)
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.cm.cividis

    group_list = [group for group in group_list if group in plot_df["group"].astype(str).unique().tolist()]
    if not group_list:
        group_list = plot_df["group"].astype(str).drop_duplicates().tolist()
        logger.info(
            "[draw_combine_dotplot] Warning! Argument `group_list` did not match the data. "
            "The group order was inferred from `df_full`."
        )

    if not interaction_order:
        interaction_order = plot_df["interaction_group"].astype(str).drop_duplicates().tolist()
        logger.info(
            "[draw_combine_dotplot] Warning! Argument `interaction_order` was empty. "
            "The interaction order was inferred from `df_full`."
        )

    g = sns.FacetGrid(
        plot_df,
        row="group",
        sharex=True,
        sharey=True,
        height=facet_height,
        aspect=facet_aspect,
        row_order=group_list,
    )

    def panel(data, color=None, vline_pairs=None):
        """绘制单个 facet。"""
        del color
        ax = plt.gca()

        x_labels = data["celltype_group"].astype(str).unique().tolist()
        x_mapping = {label: index for index, label in enumerate(x_labels)}
        y_mapping = {label: index for index, label in enumerate(interaction_order)}

        x_vals = data["celltype_group"].astype(str).map(x_mapping)
        y_vals = data["interaction_group"].astype(str).map(y_mapping)
        ax.scatter(
            x_vals,
            y_vals,
            s=data["dot_size"].astype(float),
            c=cmap(norm(np.sqrt(data["scaled_means"].astype(float)))),
            linewidth=0,
        )

        sig_mask = data["significant"].fillna("").astype(str).str.lower().eq("yes")
        sig = data.loc[sig_mask]
        if not sig.empty:
            ax.scatter(
                sig["celltype_group"].astype(str).map(x_mapping),
                sig["interaction_group"].astype(str).map(y_mapping),
                s=sig["dot_size"].astype(float) + 150,
                facecolors="none",
                edgecolors="#C21E56",
                linewidth=0.4,
            )

        if vline_pairs is not None:
            for left_label, right_label in vline_pairs:
                if left_label in x_mapping and right_label in x_mapping:
                    x0 = (x_mapping[left_label] + x_mapping[right_label]) / 2
                    ax.axvline(x=x0, color="gray", linestyle="--", linewidth=1)

        ax.set_ylim(-0.5, len(interaction_order) - 0.5)
        is_bottom = str(data["group"].iloc[0]) == str(group_list[-1])
        if not is_bottom:
            ax.set_xticks([])
            ax.set_xlabel("")
        else:
            ax.set_xticks(list(range(len(x_labels))))
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=14)

        ax.set_yticks(list(range(len(interaction_order))))
        ax.set_yticklabels(interaction_order, fontsize=18)

    g.map_dataframe(panel, vline_pairs=vline_pairs)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_left = right + 0.01
    cbar_width = 0.015
    cbar_bottom = bottom + 0.1
    cbar_height = top - bottom - 0.2
    cax = g.fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = g.fig.colorbar(sm, cax=cax)
    cbar.set_label("sqrt(scaled_means)", fontsize=10)

    cbar_pos = cax.get_position()
    legend_ax = g.fig.add_axes([cbar_pos.x1 + 0.1, cbar_pos.y0 - 0.1, 0.3, cbar_pos.height * 2])
    legend_ax.axis("off")

    size_vals = [0.2, 0.5, 1.0]
    y_pos = np.linspace(0.5, 0.3, len(size_vals))
    for value, ypos in zip(size_vals, y_pos):
        legend_ax.scatter([0.1], [ypos], s=size_map(value), color="gray", alpha=0.6)
        legend_ax.text(0.2, ypos, f"{value:.2f}", va="center", fontsize=10)
    legend_ax.text(0.2, 0.55, "-log(10)p", ha="center", va="bottom", fontsize=12, fontweight="bold")
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)

    fig = g.fig
    fig.set_size_inches(w=fig_width, h=fig_height)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=hspace)

    abs_path = os.path.join(save_addr, filename.strip())
    matplotlib_savefig(fig, abs_path)
    plot_df.to_excel(os.path.join(save_addr, f"{filename.strip()}(data).xlsx"), index=False)
    logger.info(f"[draw_combine_dotplot] Figure and source table were saved with base filename: '{filename.strip()}'.")


@logged
def plot_gene_bubble_with_cell_fraction(
    adata,
    save_addr,
    filename,
    gene,
    hk_genes=None,
    celltype_col="Subset_Identity",
    celltype_exclude=None,
    celltype_order=None,
    disease_col="disease",
    disease_order=["HC", "Colitis", "BD", "CD", "UC"],
    topN=10,
    min_frac=0.15,
    out_frac=0.05,
    bubble_size_range=(20, 200),
    figsize=(12, 6),
    cmap="Reds",
):
    """绘制“表达气泡图 + 分子贡献堆叠图”的组合图。

    该函数会先按 `gene` 在不同 cell subtype/subpopulation 中的表达与表达比例做初筛，
    再结合 `adata.uns["weighted_cell_prop"]` 估算跨疾病状态的加权分子贡献。对于名称
    相对抽象的参数，如 `min_frac`、`out_frac` 和 `topN`，更建议直接参考示例理解其联动效果。

    Args:
        adata: AnnData 对象。`obs` 中需包含 `celltype_col`、`disease_col`；
            `uns` 中需包含 `weighted_cell_prop`。
        save_addr: 输出目录。
        filename: 输出文件名主体，不带扩展名。
        gene: 单个基因名或基因列表。若为列表，则使用这些基因表达的均值。
        hk_genes: 可选的 housekeeping genes 列表，用于对表达做轻微归一化矫正。
        celltype_col: `adata.obs` 中表示 cell subtype/subpopulation 的列名。
        celltype_exclude: 需要排除的 cell subtype/subpopulation 列表。
        celltype_order: 若提供，则尽量按给定顺序展示最终筛出的 cell subtype。
        disease_col: `adata.obs` 中表示疾病状态的列名。
        disease_order: 疾病状态显示顺序。
        topN: 最终展示的 cell subtype 数量上限。
        min_frac: 初筛时最小表达比例阈值。
        out_frac: 若某 cell subtype 在所有组中最大表达比例仍低于该值，则其贡献记为 0。
        bubble_size_range: 气泡面积范围。
        figsize: 画布大小。
        cmap: 气泡图颜色映射。

    Returns:
        `pd.DataFrame`。返回用于左侧气泡图的整理后长表。

    Example:
        result_df = plot_gene_bubble_with_cell_fraction(
            adata=adata,
            save_addr=save_addr,
            filename="IL7R_expression_summary",
            gene="IL7R",
            hk_genes=["ACTB", "GAPDH", "RPL13A"],
            celltype_col="Subset_Identity",
            disease_col="disease",
            topN=8,
            min_frac=0.10,
            out_frac=0.03,
        )
        # 返回的 result_df 可继续用于自定义排序、显著性标记或补充表格输出
        result_df.head()
    """
    if celltype_col not in adata.obs.columns:
        raise KeyError(
            f"Column `{celltype_col}` was not found in `adata.obs`. "
            f"Available columns are: {list(adata.obs.columns)}."
        )
    if disease_col not in adata.obs.columns:
        raise KeyError(
            f"Column `{disease_col}` was not found in `adata.obs`. "
            f"Available columns are: {list(adata.obs.columns)}."
        )
    if "weighted_cell_prop" not in adata.uns:
        raise KeyError("Key `weighted_cell_prop` was not found in `adata.uns`.")

    weighted_df = adata.uns["weighted_cell_prop"]
    if not isinstance(weighted_df, pd.DataFrame):
        raise TypeError("Object `adata.uns['weighted_cell_prop']` must be a pandas DataFrame.")
    if "weight" not in weighted_df.columns:
        raise KeyError("Column `weight` was not found in `adata.uns['weighted_cell_prop']`.")
    if "if" not in weighted_df.columns:
        logger.info(
            "[plot_gene_bubble_with_cell_fraction] Warning! Column `if` was not found in "
            "`adata.uns['weighted_cell_prop']`; a default offset of 0 will be used."
        )
        weighted_df = weighted_df.copy()
        weighted_df["if"] = 0.0

    save_addr = _validate_output_dir(save_addr)
    if not isinstance(filename, str) or filename.strip() == "":
        raise ValueError("Argument `filename` must be a non-empty string.")

    celltype_exclude = celltype_exclude or []
    adata_use = adata[~adata.obs[celltype_col].isin(celltype_exclude)].copy() if celltype_exclude else adata.copy()
    if adata_use.n_obs == 0:
        raise ValueError("No cells remained after applying `celltype_exclude`.")

    plot_df = adata_use.obs.copy()
    genes_to_fetch = [gene] if isinstance(gene, str) else list(gene)
    if not genes_to_fetch:
        raise ValueError("Argument `gene` must contain at least one gene name.")

    source = adata_use.raw if getattr(adata_use, "raw", None) is not None else adata_use
    missing_genes = [gene_name for gene_name in genes_to_fetch if gene_name not in source.var_names]
    if missing_genes:
        logger.info(
            f"[plot_gene_bubble_with_cell_fraction] Warning! Genes {missing_genes} were not found "
            "in the expression matrix; zero vectors will be used for them."
        )

    plot_df["expr_raw"] = np.mean([_extract_expression(adata_use, gene_name) for gene_name in genes_to_fetch], axis=0)

    if hk_genes:
        valid_hk = [gene_name for gene_name in hk_genes if gene_name in source.var_names]
        if len(valid_hk) >= 3:
            hk_mean = np.mean([_extract_expression(adata_use, gene_name) for gene_name in valid_hk], axis=0)
            limit = 0.8
            delta_raw = hk_mean - np.median(hk_mean)
            delta = limit * np.tanh(delta_raw / limit)
            plot_df["expr"] = (plot_df["expr_raw"] - delta).clip(lower=0)
        else:
            logger.info(
                "[plot_gene_bubble_with_cell_fraction] Warning! Fewer than 3 valid housekeeping "
                "genes were found; raw expression values will be used."
            )
            plot_df["expr"] = plot_df["expr_raw"]
    else:
        plot_df["expr"] = plot_df["expr_raw"]

    summary = (
        plot_df.groupby([celltype_col, disease_col])
        .agg(
            n_cells=("expr", "size"),
            sum_expr=("expr", "sum"),
            frac_expr=("expr_raw", lambda values: (values > 0).mean()),
        )
        .reset_index()
    )
    summary["pseudo_bulk_log"] = summary["sum_expr"] / summary["n_cells"]

    ct_stats = summary.groupby(celltype_col).agg({"pseudo_bulk_log": "max", "frac_expr": "max"})
    ct_stats["score"] = ct_stats["pseudo_bulk_log"] * ct_stats["frac_expr"]

    potential_candidates = ct_stats[ct_stats["frac_expr"] > min_frac].index.tolist()
    if len(potential_candidates) < 3:
        logger.info(
            "[plot_gene_bubble_with_cell_fraction] Warning! Too few cell subtypes passed `min_frac`; "
            "the fallback top candidates by `frac_expr` will be used."
        )
        potential_candidates = ct_stats.sort_values("frac_expr", ascending=False).head(5).index.tolist()

    candidates = ct_stats.loc[potential_candidates].copy()
    candidates["score"] = candidates["pseudo_bulk_log"] * candidates["frac_expr"]
    valid_celltypes = candidates.sort_values("score", ascending=False).head(max(topN * 2, topN)).index.tolist()

    cell_weight_records = []
    for celltype_name, row in weighted_df.iterrows():
        for disease_name in disease_order:
            if disease_name == "HC":
                weight = row["weight"]
            elif disease_name in row.index:
                weight = np.exp(row[disease_name] + row["if"]) * row["weight"]
            else:
                logger.info(
                    f"[plot_gene_bubble_with_cell_fraction] Warning! Disease label '{disease_name}' was "
                    "not found in `weighted_cell_prop`; a weight of 0 will be used for that group."
                )
                weight = 0.0
            cell_weight_records.append(
                {celltype_col: celltype_name, disease_col: disease_name, "cell_weight": weight}
            )

    cell_weight_df = pd.DataFrame(cell_weight_records)
    full_contribution = summary.merge(cell_weight_df, on=[celltype_col, disease_col], how="left")
    full_contribution["cell_weight"] = full_contribution["cell_weight"].fillna(0)

    ct_max_frac = full_contribution.groupby(celltype_col)["frac_expr"].transform("max")
    full_contribution["clean_contribution"] = np.where(
        ct_max_frac < out_frac,
        0,
        (np.exp(full_contribution["pseudo_bulk_log"]) - 1) * full_contribution["cell_weight"],
    )

    contribution_rank = (
        full_contribution[full_contribution[celltype_col].isin(valid_celltypes)]
        .groupby(celltype_col)["clean_contribution"]
        .sum()
        .sort_values(ascending=False)
        .head(topN)
    )
    valid_celltypes = contribution_rank.index.tolist()
    if celltype_order is not None:
        ordered = [celltype_name for celltype_name in celltype_order if celltype_name in valid_celltypes]
        remaining = [celltype_name for celltype_name in valid_celltypes if celltype_name not in ordered]
        valid_celltypes = ordered + remaining

    if not valid_celltypes:
        raise ValueError("No valid cell subtypes were selected for plotting.")

    full_contribution["plot_label"] = full_contribution[celltype_col].apply(
        lambda value: value if value in valid_celltypes else "Others"
    )
    stack_df = (
        full_contribution.groupby(["plot_label", disease_col])["clean_contribution"].sum().reset_index()
    )
    stack_df.columns = [celltype_col, disease_col, "clean_contribution"]

    summary_filtered = summary[summary[celltype_col].isin(valid_celltypes)].copy()
    summary_filtered[celltype_col] = pd.Categorical(
        summary_filtered[celltype_col], categories=valid_celltypes, ordered=True
    )
    summary_filtered[disease_col] = pd.Categorical(
        summary_filtered[disease_col], categories=disease_order, ordered=True
    )
    summary_filtered = summary_filtered.sort_values([celltype_col, disease_col])

    stack_categories = valid_celltypes + ["Others"]
    stack_df[celltype_col] = pd.Categorical(stack_df[celltype_col], categories=stack_categories, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(wspace=0.5, right=0.85, left=0.1)

    sns.scatterplot(
        data=summary_filtered,
        x=disease_col,
        y=celltype_col,
        size="frac_expr",
        hue="pseudo_bulk_log",
        sizes=bubble_size_range,
        palette=cmap,
        ax=axes[0],
        edgecolor="0.3",
    )
    legend = axes[0].get_legend()
    if legend is not None:
        legend.remove()

    if isinstance(gene, list):
        if len(gene) > 3:
            gene_name = f"{gene[0]}+{gene[1]}...({len(gene)} genes)"
        else:
            gene_name = "+".join(gene)
    else:
        gene_name = str(gene)
    axes[0].set_title(f"Expression Per Subset\n{gene_name}")

    norm = plt.Normalize(*_safe_norm_range(summary_filtered["pseudo_bulk_log"]))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes[0], fraction=0.03, pad=0.08)

    stack_pivot = (
        stack_df.pivot_table(index=disease_col, columns=celltype_col, values="clean_contribution", aggfunc="sum")
        .fillna(0)
        .reindex(disease_order)
    )

    if "HC" in stack_pivot.index:
        hc_total = float(stack_pivot.loc["HC"].sum())
        if hc_total > 0:
            stack_pivot = stack_pivot / hc_total
        else:
            logger.info(
                "[plot_gene_bubble_with_cell_fraction] Warning! The total contribution of 'HC' was 0; "
                "bar values will not be normalized by the HC baseline."
            )

    base_colors = sns.color_palette("tab20", n_colors=len(valid_celltypes))
    plot_colors = list(base_colors) + ["#D3D3D3"]
    stack_pivot.plot(
        kind="bar",
        stacked=True,
        ax=axes[1],
        color=plot_colors,
        width=0.7,
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1].set_title("Molecular Contribution\n(Abundance Weighted)")
    axes[1].legend(title="Cell type", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, fontsize=10)
    axes[1].set_ylabel("Relative to HC Total")

    abs_path = os.path.join(save_addr, filename.strip())
    matplotlib_savefig(fig, abs_path)
    summary_filtered.to_excel(os.path.join(save_addr, f"{filename.strip()}_data(dot).xlsx"), index=False)
    stack_pivot.to_excel(os.path.join(save_addr, f"{filename.strip()}_data(bar).xlsx"))
    logger.info(
        f"[plot_gene_bubble_with_cell_fraction] Figure and source tables were saved with base filename: "
        f"'{filename.strip()}'."
    )
    return summary_filtered


@logged
def plot_universal_bubble_legend(save_addr, bubble_size_range=(20, 200)):
    """绘制通用气泡图图例。

    该函数适合与其他散点图或 dotplot 分开排版时复用，单独导出一个“Fraction Expressed”
    参考图例，避免每张图都重复放置完整图例。

    Args:
        save_addr: 输出目录。
        bubble_size_range: 气泡最小和最大面积，需与主图设置保持一致。

    Returns:
        `None`。图例会被导出到 `save_addr`。

    Example:
        plot_universal_bubble_legend(
            save_addr=save_addr,
            bubble_size_range=(30, 220),
        )
        # 该函数会输出 `Universal_Bubble_Legend.pdf` 和 `Universal_Bubble_Legend.png`
    """
    save_addr = _validate_output_dir(save_addr)
    if len(bubble_size_range) != 2:
        raise ValueError("Argument `bubble_size_range` must contain exactly 2 numeric values.")

    fig, ax = plt.subplots(figsize=(2, 3))
    ax.axis("off")

    labels = [0.0, 0.25, 0.5, 0.75, 1.0]
    sizes = [
        bubble_size_range[0] + (bubble_size_range[1] - bubble_size_range[0]) * label_value
        for label_value in labels
    ]

    for label_value, size_value in zip(labels, sizes):
        ax.scatter([], [], s=size_value, c="gray", edgecolors="0.3", label=f"{int(label_value * 100)}%")

    ax.legend(title="Fraction Expressed", labelspacing=1.2, handletextpad=1.5, loc="center", frameon=False)
    matplotlib_savefig(fig, os.path.join(save_addr, "Universal_Bubble_Legend"))
    logger.info("[plot_universal_bubble_legend] Figure was saved with base filename: 'Universal_Bubble_Legend'.")


@logged
def plot_chord_diagram(
    mat,
    cell_colors=None,
    min_weight=0.0,
    title=None,
    figsize=(8, 8),
    chord_radius=1.0,
    control_radius=0.45,
    self_loop_radius=1.25,
    label_radius=1.35,
    label_padding=0.03,
    width_scale=6.0,
    normalize=True,
    group_cells=None,
    group_colors=None,
    group_arc_radius=None,
    group_arc_width=8,
):
    """绘制自定义 chord diagram。

    与 `cpdb_chordplot()` 不同，这个函数直接接受一个细胞间权重矩阵，更适合在
    CellPhoneDB 结果、加权交互矩阵或其他汇总表上做二次展示。

    Args:
        mat: 方阵形式的 `pd.DataFrame`，行列均表示 cell subtype/subpopulation。
        cell_colors: 每个细胞类群对应的颜色字典；若不提供则自动使用 `tab20`。
        min_weight: 低于该阈值的边将被忽略。
        title: 图标题。
        figsize: 画布大小。
        chord_radius: 弦起点所在圆的半径。
        control_radius: 弦中间控制点所在圆的半径。
        self_loop_radius: 自连边控制半径。
        label_radius: 标签半径。
        label_padding: 标签外侧额外偏移。
        width_scale: 线宽缩放系数。
        normalize: 是否按最大边权做归一化。
        group_cells: 可选分组字典，用于给一组细胞添加外侧分组弧线。
        group_colors: `group_cells` 对应的分组颜色。
        group_arc_radius: 分组弧线半径；若不提供则自动推断。
        group_arc_width: 分组弧线线宽。

    Returns:
        `(fig, ax)` 元组。

    Example:
        fig, ax = plot_chord_diagram(
            mat=interaction_matrix,
            min_weight=0.05,
            title="T cell communication overview",
            group_cells={
                "T lineage": ["CD4 T", "CD8 T", "Treg"],
                "Myeloid": ["Mono", "Macrophage"],
            },
            group_colors={"T lineage": "#1f77b4", "Myeloid": "#d62728"},
        )
        matplotlib_savefig(fig, os.path.join(save_addr, "custom_chord"))
    """
    if not isinstance(mat, pd.DataFrame):
        raise TypeError("Argument `mat` must be a pandas DataFrame.")
    if mat.empty:
        raise ValueError("Argument `mat` must not be empty.")

    if list(mat.index) != list(mat.columns):
        logger.info(
            "[plot_chord_diagram] Warning! The row and column labels of `mat` were not identical; "
            "the matrix will be reindexed to the union of both axes with missing values filled by 0."
        )
        labels = list(dict.fromkeys(list(mat.index) + list(mat.columns)))
        mat = mat.reindex(index=labels, columns=labels, fill_value=0)

    cells = list(mat.index)
    if group_cells:
        grouped = []
        used = set()
        for members in group_cells.values():
            members = [cell_name for cell_name in members if cell_name in cells]
            grouped.extend(members)
            used.update(members)
        rest = [cell_name for cell_name in cells if cell_name not in used]
        cells = grouped + rest

    if cell_colors is None:
        cmap = plt.get_cmap("tab20")
        cell_colors = {cell_name: cmap(index % 20) for index, cell_name in enumerate(cells)}

    def blend(color1, color2, ratio=0.4):
        """混合两种颜色。"""
        r1, g1, b1 = to_rgb(color1)
        r2, g2, b2 = to_rgb(color2)
        return (r1 * (1 - ratio) + r2 * ratio, g1 * (1 - ratio) + g2 * ratio, b1 * (1 - ratio) + b2 * ratio)

    edges = []
    max_weight = 0.0
    for source_cell in cells:
        for target_cell in cells:
            weight = float(mat.loc[source_cell, target_cell])
            if weight > min_weight:
                edges.append((source_cell, target_cell, weight))
                max_weight = max(max_weight, weight)

    if not edges:
        logger.info(
            "[plot_chord_diagram] Warning! No edges passed the `min_weight` threshold. "
            "An empty frame with labels will be returned."
        )

    if normalize and max_weight > 0:
        edges = [(source_cell, target_cell, weight / max_weight) for source_cell, target_cell, weight in edges]

    angles = np.linspace(0, 2 * np.pi, len(cells), endpoint=False)
    angle_map = dict(zip(cells, angles))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    for cell_name in cells:
        angle = angle_map[cell_name]
        x = label_radius * np.cos(angle)
        y = label_radius * np.sin(angle)
        if np.cos(angle) >= 0:
            ha = "left"
            x += label_padding
        else:
            ha = "right"
            x -= label_padding

        rotation = np.degrees(angle)
        if np.cos(angle) < 0:
            rotation += 180

        ax.text(
            x,
            y,
            cell_name,
            ha=ha,
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=9,
            zorder=5,
        )

    for source_cell, target_cell, weight in edges:
        angle1 = angle_map[source_cell]
        line_width = 0.8 + weight * width_scale

        if source_cell == target_cell:
            x0 = chord_radius * np.cos(angle1)
            y0 = chord_radius * np.sin(angle1)
            control1 = (self_loop_radius * np.cos(angle1 - 0.3), self_loop_radius * np.sin(angle1 - 0.3))
            control2 = (self_loop_radius * np.cos(angle1 + 0.3), self_loop_radius * np.sin(angle1 + 0.3))
            verts = [(x0, y0), control1, control2, (x0, y0)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            ax.add_patch(
                patches.PathPatch(
                    Path(verts, codes),
                    facecolor="none",
                    edgecolor=cell_colors[source_cell],
                    lw=line_width,
                    alpha=0.9,
                    zorder=3,
                )
            )
            continue

        angle2 = angle_map[target_cell]
        delta = angle2 - angle1
        if delta > np.pi:
            delta -= 2 * np.pi
        elif delta < -np.pi:
            delta += 2 * np.pi
        middle = angle1 + delta / 2

        x1, y1 = chord_radius * np.cos(angle1), chord_radius * np.sin(angle1)
        x2, y2 = chord_radius * np.cos(angle2), chord_radius * np.sin(angle2)
        cx, cy = control_radius * np.cos(middle), control_radius * np.sin(middle)
        path = Path([(x1, y1), (cx, cy), (x2, y2)], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        ax.add_patch(
            patches.PathPatch(
                path,
                facecolor="none",
                edgecolor=blend(cell_colors[source_cell], cell_colors[target_cell]),
                lw=line_width,
                alpha=0.7,
                zorder=2,
            )
        )

    if group_cells:
        if group_arc_radius is None:
            group_arc_radius = label_radius - 0.12
        for group_name, members in group_cells.items():
            members = [cell_name for cell_name in members if cell_name in angle_map]
            if not members:
                continue
            group_angles = np.array([angle_map[cell_name] for cell_name in members])
            arc = patches.Arc(
                (0, 0),
                2 * group_arc_radius,
                2 * group_arc_radius,
                theta1=np.degrees(group_angles.min()),
                theta2=np.degrees(group_angles.max()),
                lw=group_arc_width,
                color=group_colors.get(group_name, "black") if group_colors else "black",
                alpha=0.95,
                zorder=1,
            )
            ax.add_patch(arc)

    if title:
        ax.set_title(title, fontsize=14, pad=20)

    lim = max(self_loop_radius, label_radius, group_arc_radius or 0) + 0.35
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    return fig, ax


@logged
def prune_cells_by_activity(mat, min_value=0.0):
    """删除在行和列方向都不活跃的细胞类群。

    该函数常用于在绘制热图、弦图或 Sankey 图前做快速剪枝，避免大量全零或极低权重
    的 cell subtype/subpopulation 干扰主图阅读。

    Args:
        mat: 方阵形式的 `pd.DataFrame`，行列均表示 cell subtype/subpopulation。
        min_value: 活跃阈值。若某细胞在所在行和所在列的最大值都小于等于该阈值，则会被移除。

    Returns:
        剪枝后的方阵。

    Example:
        pruned_mat = prune_cells_by_activity(
            mat=interaction_matrix,
            min_value=0.02,
        )
        # 结果可直接送入 plot_chord_diagram() 或其他矩阵可视化函数
        pruned_mat.shape
    """
    if not isinstance(mat, pd.DataFrame):
        raise TypeError("Argument `mat` must be a pandas DataFrame.")
    if mat.empty:
        raise ValueError("Argument `mat` must not be empty.")

    if list(mat.index) != list(mat.columns):
        logger.info(
            "[prune_cells_by_activity] Warning! The row and column labels of `mat` were not identical; "
            "the matrix will be reindexed to the union of both axes with missing values filled by 0."
        )
        labels = list(dict.fromkeys(list(mat.index) + list(mat.columns)))
        mat = mat.reindex(index=labels, columns=labels, fill_value=0)

    row_max = mat.max(axis=1)
    col_max = mat.max(axis=0)
    keep_cells = mat.index[(row_max > min_value) | (col_max > min_value)]
    if len(keep_cells) == 0:
        logger.info(
            "[prune_cells_by_activity] Warning! No cells passed the `min_value` threshold; "
            "an empty matrix will be returned."
        )
    return mat.loc[keep_cells, keep_cells]


@logged
def CCI_sankey_plot_top5(
    nodes,
    links_df,
    save_addr,
    filename,
    title=None,
):
    """绘制按中间层细胞平衡后的 CCI Sankey 图，并输出 Top 5 贡献表。

    该函数通常配合 `CCI_sankey_table()` 使用。它会分别对每个 middle cell 的流入和流出做
    局部平衡，从而避免单个极大值完全主导布局。同时会把每个 middle cell 的前 5 个上游
    与下游贡献者导出为 Excel，便于人工检查结果。

    Args:
        nodes: 节点字典，需至少包含 `left`、`middle` 和 `right` 三个键。
        links_df: 边表，需包含 `source`、`target` 和 `value` 列，可选 `label` 列。
        save_addr: 输出目录。
        filename: 输出文件名主体，不带扩展名。
        title: 图标题。

    Returns:
        `plotly.graph_objects.Figure` 对象。

    Example:
        fig = CCI_sankey_plot_top5(
            nodes=nodes,
            links_df=links_df,
            save_addr=save_addr,
            filename="Th17_signal_flow",
            title="Th17 centered communication flow",
        )
        # 若静态导出环境不完整，函数会回退保存 HTML 版本
        fig
    """
    import plotly.graph_objects as go

    if not isinstance(nodes, dict):
        raise TypeError("Argument `nodes` must be a dictionary.")
    for key in ["left", "middle", "right"]:
        if key not in nodes:
            raise KeyError(f"Key `{key}` was not found in `nodes`.")
        if not isinstance(nodes[key], list):
            raise TypeError(f"Value of `nodes['{key}']` must be a list.")

    if not isinstance(links_df, pd.DataFrame):
        raise TypeError("Argument `links_df` must be a pandas DataFrame.")
    if links_df.empty:
        raise ValueError("Argument `links_df` must not be empty.")
    _require_columns(links_df, ["source", "target", "value"], arg_name="links_df")

    save_addr = _validate_output_dir(save_addr)
    if not isinstance(filename, str) or filename.strip() == "":
        raise ValueError("Argument `filename` must be a non-empty string.")

    nodes_copy = {key: list(value) for key, value in nodes.items()}
    middle_cells = nodes_copy["middle"]
    middle_set = set(middle_cells)
    positive_values = pd.to_numeric(links_df["value"], errors="coerce")
    positive_values = positive_values[positive_values > 0]
    epsilon = max(float(positive_values.min()) * 1e-6, 1e-6) if not positive_values.empty else 1e-6

    plot_links_all = []
    top_records = []

    for middle_cell in middle_cells:
        left_links_m = links_df[
            (links_df["target"] == middle_cell) & (~links_df["source"].isin(middle_set))
        ].copy()
        right_links_m = links_df[
            (links_df["source"] == middle_cell) & (~links_df["target"].isin(middle_set))
        ].copy()

        sum_left = float(left_links_m["value"].sum())
        sum_right = float(right_links_m["value"].sum()) if not right_links_m.empty else 0.0
        total_flow = min(sum_left, sum_right) if sum_right > 0 else sum_left

        if sum_left > 0:
            left_top5 = (
                left_links_m.groupby("source", as_index=False)["value"].sum().sort_values("value", ascending=False).head(5)
            )
            for _, row in left_top5.iterrows():
                top_records.append(
                    {
                        "middle_cell": middle_cell,
                        "direction": "Left-to-Middle",
                        "partner_cell": row["source"],
                        "value": row["value"],
                        "percentage": row["value"] / sum_left * 100,
                    }
                )

        if sum_right > 0:
            right_top5 = (
                right_links_m.groupby("target", as_index=False)["value"].sum().sort_values("value", ascending=False).head(5)
            )
            for _, row in right_top5.iterrows():
                top_records.append(
                    {
                        "middle_cell": middle_cell,
                        "direction": "Middle-to-Right",
                        "partner_cell": row["target"],
                        "value": row["value"],
                        "percentage": row["value"] / sum_right * 100,
                    }
                )
        else:
            logger.info(
                f"[CCI_sankey_plot_top5] Warning! Middle cell '{middle_cell}' had no downstream links; "
                "a dummy edge will be added to stabilize the layout."
            )
            dummy_row = pd.DataFrame(
                [{"source": middle_cell, "target": "__dummy_right__", "value": epsilon, "label": "dummy"}]
            )
            right_links_m = pd.concat([right_links_m, dummy_row], ignore_index=True)
            if "__dummy_right__" not in nodes_copy["right"]:
                nodes_copy["right"].append("__dummy_right__")

        if sum_left > 0:
            left_links_m["value_scaled"] = left_links_m["value"] / sum_left * total_flow
        else:
            left_links_m["value_scaled"] = left_links_m["value"]

        if sum_right > 0:
            right_links_m["value_scaled"] = right_links_m["value"] / sum_right * total_flow
        else:
            right_links_m["value_scaled"] = right_links_m["value"]

        plot_links_all.append(left_links_m)
        plot_links_all.append(right_links_m)

    plot_links = pd.concat(plot_links_all, axis=0, ignore_index=True)
    all_nodes = nodes_copy["left"] + nodes_copy["middle"] + nodes_copy["right"]
    node_idx = {node_name: index for index, node_name in enumerate(all_nodes)}

    sources = plot_links["source"].map(node_idx).tolist()
    targets = plot_links["target"].map(node_idx).tolist()
    values = plot_links["value_scaled"].tolist()

    colors = []
    for _, row in plot_links.iterrows():
        if row.get("label") == "dummy":
            colors.append("rgba(0,0,0,0)")
        elif row["target"] in middle_set:
            colors.append("rgba(55,126,184,0.6)")
        else:
            colors.append("rgba(228,26,28,0.6)")

    node_labels = ["" if node_name == "__dummy_right__" else node_name for node_name in all_nodes]
    fig = go.Figure(
        go.Sankey(
            node=dict(label=node_labels, pad=15, thickness=15),
            link=dict(source=sources, target=targets, value=values, color=colors),
        )
    )

    if title:
        fig.update_layout(title_text=title)

    base_path = os.path.join(save_addr, filename.strip())
    if top_records:
        top_df = pd.DataFrame(top_records)
        top_df["percentage"] = top_df["percentage"].round(5)
        top_df.to_excel(f"{base_path}(data).xlsx", index=False)

    try:
        fig.write_image(f"{base_path}.pdf", width=800, height=600, engine="kaleido")
        fig.write_image(f"{base_path}.svg", width=800, height=600, engine="kaleido")
        fig.write_image(f"{base_path}.png", width=800, height=600, scale=3)
        logger.info(f"[CCI_sankey_plot_top5] Figure was saved with base filename: '{filename.strip()}'.")
    except Exception as exc:
        logger.info(
            "[CCI_sankey_plot_top5] Warning! Static image export failed. "
            f"An HTML fallback will be saved instead. Details: '{exc}'."
        )
        fig.write_html(f"{base_path}.html")
        logger.info(f"[CCI_sankey_plot_top5] HTML fallback was saved to: '{base_path}.html'.")

    return fig
