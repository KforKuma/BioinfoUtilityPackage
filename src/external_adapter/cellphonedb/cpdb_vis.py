import numpy as np
import pandas as pd


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用无GUI的后端
plt.rcParams['font.family'] = 'monospace'  # 全局设置等宽字体

import os, re

from src.external_adapter.cellphonedb.settings import DEFAULT_CPDB_SEP
from src.external_adapter.cellphonedb.cpdb_ops import size_map

import ktplotspy as kpy

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)

@logged
def cpdb_heatmap(CIObject,
                 output_dir,
                 filename_prefix,
                 save_pdf=True,
                 save_png=True,
                 show=False,
                 **kwargs):
    """
    绘制 CellPhoneDB 的配体-受体交互热图（基于 ktplotspy）。

    Parameters
    ----------
    CIObject : object
        CellphoneInspector 类实例，需包含 .data["pvalues"] 与 .deg。
    output_dir : str
        图像输出目录。
    filename_prefix : str
        输出文件名前缀（不带扩展名）。
    save_pdf : bool, default True
        是否保存 PDF 格式。
    save_png : bool, default True
        是否保存 PNG 格式。
    show : bool, default False
        是否在绘制后显示图像（适合 notebook）。
    **kwargs :
        传递给 `ktplotspy.plot_cpdb_heatmap()` 的其他参数。

    Returns
    -------
    fig : sns.matrix.ClusterGrid
        seaborn Clustermap 对象，可进一步操作或展示。
    """
    
    # --- 安全获取必要数据 ---
    pvals = CIObject.data.get("pvals", None)
    degs = getattr(CIObject, "deg", None)
    
    if pvals is None:
        raise ValueError("[cpdb_heatmap] CIObject.data['pvalues'] is missing.")
    
    default_param = {"pvals": pvals,
                     "cell_types": None,
                     "degs_analysis": degs,
                     "log1p_transform": False,
                     "alpha": 0.05,
                     "linewidths": 0.05,
                     "row_cluster": True, "col_cluster": True,
                     "low_col": "#104e8b", "mid_col": "#ffdab9", "high_col": "#8b0a50", "cmap": None,
                     "title": "",
                     "return_tables": False,
                     "symmetrical": True,
                     "default_sep": DEFAULT_CPDB_SEP,
                     }
    
    default_param.update(kwargs)
    
    # --- 绘制 ---
    fig = kpy.plot_cpdb_heatmap(**default_param)  # 本质是 sns.clustermap
    
    # --- 输出路径与保存 ---
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    filename = f"{prefix}CPDB_Heatmap"
    base_path = os.path.join(output_dir, filename)
    if save_pdf:
        fig.savefig(f"{base_path}.pdf", dpi=300, bbox_inches="tight")
    if save_png:
        fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
    
    logger.info(f"Saved: {filename} (.pdf/.png)")
    if show:
        plt.show()
    else:
        plt.close(fig.fig)
    
    return fig

@logged
def cpdb_dotplot(CIObject, AnndataObject,
                 cell_type1, cell_type2, celltype_key,
                 output_dir,
                 filename_prefix,
                 interacting_pairs=None,
                 save_pdf=True,
                 save_png=True,
                 show=False,
                 **kwargs):
    '''
     
     
     Example
     -------
     # 为了保证 genes/gene_family 符合规范，便于管理，建议采用 prep_gene_query
     gene_query = prepare_gene_query(ci, genes=["IL2", "IL6"])
     gene_query = prepare_gene_query(ci, genes="IL2|IL6")
     gene_query = prepare_gene_query(ci, gene_family=“th17”)

     # 不提供 interacting_pairs, genes, gene_family 会只打印显著项
     
    :param CIObject: 
    :param AnndataObject: 
    :param cell_type1: 
    :param cell_type2: 
    :param celltype_key:
    :param output_dir: 
    :param filename_prefix:
    :param interacting_pairs:
    :param save_pdf: 
    :param save_png: 
    :param show: 
    :param kwargs: 
    :return: 
    '''
    
    pvals = CIObject.data.get("pvals", None)
    means = CIObject.data.get("means", None)
    interaction_scores = CIObject.data.get("interaction_scores", None)  # 可以为空
    cellsign = CIObject.data.get("cellsign_interactions", None)
    
    if pvals is None or means is None:
        raise ValueError("[cpdb_heatmap] CIObject.data is incomplete.")
    
    default_param = {"adata": AnndataObject,
                     "cell_type1": cell_type1, "cell_type2": cell_type2, "celltype_key": celltype_key, "means": means,
                     "pvals": pvals,
                     "interaction_scores": interaction_scores, "cellsign": cellsign,
                     "degs_analysis": CIObject.deg,
                     "alpha": 0.05, "keep_significant_only": True,
                     "interacting_pairs": interacting_pairs,
                     "cmap_name": "cividis", "highlight_col": "#cf5c60",
                     "title": "CellphoneDB Dotplot",
                     "min_interaction_score": 0}
    
    default_param.update(kwargs)
    
    # --- 绘制 ---
    g = kpy.plot_cpdb(**default_param)  # 底层是使用 plotnine
    
    # --- 输出路径与保存 ---
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    filename = f"{prefix}CPDB_Heatmap"
    base_path = os.path.join(output_dir, filename)
    if save_pdf:
        g.save(f"{base_path}.pdf", dpi=300, bbox_inches="tight")
    if save_png:
        g.save(f"{base_path}.png", dpi=300, bbox_inches="tight")
    
    logger.info(f"Saved: {filename} (.pdf/.png)")
    if show:
        print(g)
    
    return g






@logged
def cpdb_chordplot(CIObject, AnndataObject,
                   cell_type1, cell_type2, celltype_key, interaction,
                   output_dir, filename_prefix,
                   save_pdf=True,
                   save_png=True,
                   show=False, **kwargs):
    '''

    Example
    -------
    cpdb_chordplot(interaction = ["PTPRC", "CD40", "CLEC2D"])
    "If your adata already has e.g. adata.uns['celltype_colors'], it will retrieve the sector_colours correctly”

    :param CIObject: 
    :param AnndataObject: 
    :param cell_type1: 
    :param cell_type2: 
    :param celltype_key: 
    :param interaction: str 或 list，也可以使用 prepare_gene_query 的输出，但是会对任何形式输出中的 "-" 进行拆分成 genes 列表形式
        因此，推荐的默认输入是单个基因
    :return: 
    '''
    pvals = CIObject.data.get("pvalues", None)
    means = CIObject.data.get("means", None)
    decon = CIObject.data.get("deconvoluted", None)  # 可以为空
    
    default_param = {"adata": AnndataObject,
                     "cell_type1": cell_type1, "cell_type2": cell_type2, "celltype_key": celltype_key,
                     "means": means,
                     "pvals": pvals,
                     "deconvoluted": decon,
                     "interaction": interaction,
                     "link_kwargs": {"direction": 1, "allow_twist": True, "r1": 95, "r2": 90},
                     "sector_text_kwargs": {"color": "black", "size": 12, "r": 105, "adjust_rotation": True},
                     "legend_kwargs": {"loc": "center", "bbox_to_anchor": (1, 1), "fontsize": 8},
                     "link_offset": 1}
    
    default_param.update(kwargs)
    
    # --- 绘制 ---
    g = kpy.plot_cpdb_chord(**default_param)  # 底层是使用 plotnine
    
    # --- 输出路径与保存 ---
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    filename = f"{prefix}CPDB_Heatmap"
    base_path = os.path.join(output_dir, filename)
    if save_pdf:
        g.save(f"{base_path}.pdf", dpi=300, bbox_inches="tight")
    if save_png:
        g.save(f"{base_path}.png", dpi=300, bbox_inches="tight")
    
    logger.info(f"Saved: {filename} (.pdf/.png)")
    if show:
        print(g)
    
    return g


def draw_combine_dotplot(df_full, save_addr, filename,
                         vline_pairs,
                         interaction_order, facet_aspect=1,
                         facet_height=0.9, fig_width=20, fig_height=16,
                         left=0.08, right=0.82, bottom=0.2, top=0.95, hspace=0.12,
                         group_list=['HC', 'Colitis', 'BD', 'CD', 'UC']):
    # 为全局 colormap & size scaling 做准备
    vmin = df_full["scaled_means"].min()
    vmax = df_full["scaled_means"].max()
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.cm.cividis
    
    # ----------------------------
    # FacetGrid（单列，5 行，从上到下）
    # ----------------------------
    n_interactions = len(interaction_order)
    
    g = sns.FacetGrid(
        df_full,
        row="group",
        sharex=True,
        sharey=True,
        height=facet_height,  # 把每个 facet 变小
        aspect=facet_aspect,
        row_order=group_list
    )
    
    def panel(data, color, vline_pairs=None):
        """
        vline_pairs: list of tuples, 每个 tuple 指定需要画竖线的两组 x tick 名称，例如
                     vline_pairs=[("x2","x3"), ("x10","x11")]
        """
        ax = plt.gca()
        
        # --------------------------
        # 1. x 和 y 映射为数值
        # --------------------------
        x_labels = data["celltype_group"].unique().tolist()
        x_mapping = {label: i for i, label in enumerate(x_labels)}
        y_mapping = {label: i for i, label in enumerate(interaction_order)}
        
        x_vals = data["celltype_group"].map(x_mapping)
        y_vals = data["interaction_group"].map(y_mapping)
        
        # --------------------------
        # 2. 绘制 scatter 点
        # --------------------------
        ax.scatter(
            x_vals,
            y_vals,
            s=data["dot_size"],
            c=cmap(norm(data["scaled_means"])),
            linewidth=0
        )
        
        # 绘制显著点的外圈
        sig = data[data["significant"].str.lower() == "yes"]
        if len(sig) > 0:
            ax.scatter(
                sig["celltype_group"].map(x_mapping),
                sig["interaction_group"].map(y_mapping),
                s=sig["dot_size"] + 150,
                facecolors="none",
                edgecolors="#C21E56",
                linewidth=2.4
            )
        
        # --------------------------
        # 3. 绘制竖虚线
        # --------------------------
        if vline_pairs is not None:
            for left_label, right_label in vline_pairs:
                if left_label in x_mapping and right_label in x_mapping:
                    x0 = (x_mapping[left_label] + x_mapping[right_label]) / 2
                    ax.axvline(x=x0, color='gray', linestyle='--', linewidth=1)
        
        # --------------------------
        # 4. 设置坐标轴
        # --------------------------
        ax.set_ylim(-0.5, len(interaction_order) - 0.5)
        
        # 只显示最底部 facet 的 xticks
        is_bottom = data["group"].iloc[0] == group_list[-1]
        if not is_bottom:
            ax.set_xticks([])
            ax.set_xlabel("")
        else:
            ax.set_xticks(list(range(len(x_labels))))
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=14)
        
        # y 轴统一 tick
        ax.set_yticks(list(range(len(interaction_order))))
        ax.set_yticklabels(interaction_order, fontsize=18)
    
    g.map_dataframe(panel, vline_pairs=vline_pairs)
    
    # colorbar（全局一致）
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = g.fig.colorbar(sm,
                          ax=g.axes.ravel().tolist(),
                          shrink=0.6,
                          aspect=20,
                          pad=0.02,
                          label="scaled_means")
    
    # colorbar 的位置
    cbar_pos = cbar.ax.get_position()  # Bbox(x0, y0, width, height)
    
    # 在 colorbar 正右侧放 size legend
    legend_width = 0.1
    legend_height = cbar_pos.height
    legend_left = cbar_pos.x1 + 0.01  # 紧挨 colorbar 右边
    legend_bottom = cbar_pos.y0
    
    legend_ax = g.fig.add_axes([legend_left, legend_bottom, legend_width, legend_height])
    legend_ax.axis("off")
    
    # 绘制 size legend
    for s in [0.2, 0.5, 1.0]:
        legend_ax.scatter([], [], s=size_map(s), color="gray", alpha=0.6, label=f"{s:.2f}")
    
    legend_ax.legend(title="size", loc="center left", fontsize=8)
    
    # 尺寸调整
    g.fig.set_size_inches(w=fig_width, h=fig_height)
    # 布局调整
    g.fig.subplots_adjust(left=left, right=right,
                          bottom=bottom, top=top,
                          hspace=hspace)
    
    plt.savefig(f"{save_addr}/{filename}.png")
    plt.savefig(f"{save_addr}/{filename}.pdf")
    plt.close()
