import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.core.plot.utils import matplotlib_savefig


def plot_nmf_heatmap(
        df,
        save_path=None,
        title="Module activity per cluster",
        color_threshold=0.2,
        label_threshold=1.0,
        vmin=0.5,
        vmax=4.0,
        figsize=(10, 8),
        cmap='Reds'
):
    """绘制 NMF 模块活性热图。

    该函数使用两个阈值分别控制颜色和数字标签：低于 ``color_threshold`` 的格子
    会显示为背景色，低于 ``label_threshold`` 的格子不显示数值。保存时使用
    ``matplotlib_savefig``，因此 ``save_path`` 默认不需要后缀，会自动保存 png 和 pdf。

    Args:
        df: 输入矩阵，行通常是 cluster，列通常是 NMF module。
        save_path: 可选输出路径，默认不包含后缀。
        title: 图标题。
        color_threshold: 低于该值的单元格不着色。
        label_threshold: 低于该值的单元格不显示数字标签。
        vmin: 热图颜色映射下限。
        vmax: 热图颜色映射上限。
        figsize: 图大小。
        cmap: 热图色板。

    Example:
        >>> plot_nmf_heatmap(
        ...     module_activity_df,
        ...     save_path="figures/nmf_module_activity",
        ...     color_threshold=0.2,
        ...     label_threshold=1.0,
        ... )
        # 会保存 figures/nmf_module_activity.png 和 .pdf。
    """
    if df.empty:
        raise ValueError("Argument `df` must not be empty.")

    # 复制一份绘图数据，避免修改原矩阵。
    df_plot = df.copy()

    # 颜色过滤：低于阈值的位置设为 NaN，热图中显示为背景色。
    df_plot[df_plot < color_threshold] = np.nan

    # 数字标签过滤：低于阈值的位置显示为空字符串，减少低活性模块的视觉噪音。
    annot_labels = df.applymap(lambda x: f"{x:.1f}" if x >= label_threshold else "")

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(
        df_plot,
        annot=annot_labels,
        fmt="",
        cmap=cmap,
        cbar=True,
        linewidths=0,
        linecolor=None,
        square=False,
        vmin=vmin,
        vmax=vmax,
        mask=df_plot.isnull(),
        ax=ax
    )

    # 去除四周边框和坐标轴短线，使图面更干净。
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)

    ax.set_title(title)
    ax.set_ylabel("Cluster")
    ax.set_xlabel("Module")
    fig.tight_layout()

    if save_path:
        matplotlib_savefig(fig, save_path, dpi=300)
    else:
        plt.close(fig)


def plot_module_ranked_loading(df_H, module, gene_sets,
                               save_addr, filename,
                               module_name=None, top_n_label=10,
                               figsize=(8, 4)):
    """绘制 NMF 模块基因 loading 排序图。

    背景点显示该模块中所有基因的 loading 排名，指定 ``gene_sets`` 中的基因会被
    高亮并标注。保存时使用 ``matplotlib_savefig``，``filename`` 默认不需要后缀。

    Args:
        df_H: NMF H 矩阵，行为 module，列为 gene。
        module: 要绘制的模块名，必须存在于 ``df_H.index``。
        gene_sets: 基因集字典，例如 ``{"stem": ["LGR5", "OLFM4"]}``。
        save_addr: 输出目录。
        filename: 输出文件名，默认不包含后缀。
        module_name: 可选模块生物学名称，用于标题补充。
        top_n_label: 每个基因集最多标注前 N 个基因。
        figsize: 图大小。

    Example:
        >>> plot_module_ranked_loading(
        ...     df_H,
        ...     module="Module_1",
        ...     gene_sets={"stem": ["LGR5", "OLFM4"]},
        ...     save_addr="figures",
        ...     filename="module_1_loading",
        ... )
        # 会保存 figures/module_1_loading.png 和 .pdf。
    """
    if module not in df_H.index:
        raise ValueError(f"Argument `module` was not found in `df_H.index`: '{module}'.")
    if top_n_label < 0:
        raise ValueError("Argument `top_n_label` must be greater than or equal to 0.")

    # 取模块 loading 并排序。
    module_loading = df_H.loc[module]
    module_sorted = module_loading.sort_values(ascending=False)
    ranks = np.arange(1, len(module_sorted) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(ranks, module_sorted.values, c='lightgray', s=15, label='Other genes', zorder=1)

    x_range = ranks.max() - ranks.min()
    x_offset = x_range * 0.02

    # 高亮指定基因集，忽略不在当前模块 loading 中的基因，兼容不同基因过滤结果。
    color_map = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (label, genes) in enumerate(gene_sets.items()):
        genes_in_module = [g for g in genes if g in module_sorted.index]
        if not genes_in_module:
            continue
        idxs = [module_sorted.index.get_loc(g) + 1 for g in genes_in_module]
        vals = module_sorted.loc[genes_in_module].values
        ax.scatter(idxs, vals, c=color_map[i % len(color_map)], s=50, label=label, zorder=10)

        # 添加文字标注。
        for j, g in enumerate(genes_in_module[:top_n_label]):
            ax.text(
                idxs[j] + x_offset,
                vals[j],
                g,
                fontsize=8,
                color=color_map[i % len(color_map)],
                zorder=15
            )

    title = module_name if module_name else module
    ax.set_title(f"Rankings of genes in {module} ({title})")
    ax.set_xlabel("Ranked genes in module")
    ax.set_ylabel("Module loading")
    ax.legend()
    fig.tight_layout()

    abs_file_path = os.path.join(save_addr, filename)
    matplotlib_savefig(fig, abs_file_path, dpi=300)
