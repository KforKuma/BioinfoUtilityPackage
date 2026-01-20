# ===== Standard library =====
import os
import re
import logging

# ===== Third-party =====
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 使用无 GUI 后端
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.colors import to_rgb

import seaborn as sns
import ktplotspy as kpy

# ===== Project-specific =====
from src.external_adaptor.cellphonedb.settings import DEFAULT_CPDB_SEP
from src.external_adaptor.cellphonedb.cpdb_ops import size_map
from src.utils.hier_logger import logged

# ===== Matplotlib global settings =====
plt.rcParams["font.family"] = "monospace"

# ===== Logger =====
logger = logging.getLogger(__name__)

##################################################################
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
    color_vals = np.sqrt(df_full["scaled_means"])
    
    vmin = color_vals.min()
    vmax = color_vals.max()
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
    
    def panel(data, color=None, vline_pairs=None):
        """
        seaborn.FacetGrid.map_dataframe 会“强行”传一个 color= 关键字参数。因此必须吃掉
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
            c=cmap(norm(np.sqrt(data["scaled_means"]))),
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
    
    # ----------------------------
    # 手动放置 colorbar（关键）
    # ----------------------------
    cbar_left = right + 0.01  # ⭐ 比 right 稍微靠右，但不侵入 facet
    cbar_width = 0.015
    cbar_bottom = bottom + 0.1
    cbar_height = top - bottom - 0.2
    
    cax = g.fig.add_axes([
        cbar_left,
        cbar_bottom,
        cbar_width,
        cbar_height
    ])
    
    cbar = g.fig.colorbar(
        sm,
        cax=cax
    )
    cbar.set_label("sqrt(scaled_means)", fontsize=10)
    
    cbar_pos = cax.get_position()
    
    # 调整 legend 位置和大小
    legend_width = 0.3  # 横向拉宽
    legend_height = cbar_pos.height * 2  # 高度放大一点
    legend_left = cbar_pos.x1 + 0.1  # 靠右一点，不挤 colorbar
    legend_bottom = cbar_pos.y0 - 0.1  # 往下微调，避免顶端与 colorbar 重叠
    
    legend_ax = g.fig.add_axes([
        legend_left,
        legend_bottom,
        legend_width,
        legend_height
    ])
    legend_ax.axis("off")
    
    # 绘制 size legend
    size_vals = [0.2, 0.5, 1.0]
    
    # 分布在 0.5~0.3 之间，留出顶部空间给 title
    y_pos = np.linspace(0.5, 0.3, len(size_vals))
    
    for v, y in zip(size_vals, y_pos):
        legend_ax.scatter([0.1], [y], s=size_map(v), color="gray", alpha=0.6)
        legend_ax.text(0.2, y, f"{v:.2f}", va="center", fontsize=10)
    
    # 手动加 title
    legend_ax.text(0.2, 0.55, "-log(10)p", ha="center", va="bottom", fontsize=12, fontweight="bold")
    
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")
    
    # 尺寸调整
    g.fig.set_size_inches(w=fig_width, h=fig_height)
    # 布局调整
    g.fig.subplots_adjust(left=left, right=right,
                          bottom=bottom, top=top,
                          hspace=hspace)
    
    plt.savefig(f"{save_addr}/{filename}.png")
    plt.savefig(f"{save_addr}/{filename}.pdf")
    plt.close()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_gene_bubble_with_cell_fraction(
        adata, save_addr, filename,
        gene,
        celltype_col="celltype",
        disease_col="disease",
        disease_order=['HC', 'Colitis', 'BD', 'CD', 'UC'],
        sample_col="sample",  # ← 新增：用于 per-sample / per-patient 归一化
        pseudo_cutoff=0.1,
        frac_cutoff=0.05,
        bubble_size_range=(20, 200),
        figsize=(10, 6),
        cmap="Reds"
):
    """
    左侧：
        bubble plot
        - color: pseudo-bulk per-cell
        - size : fraction of expressing cells

    右侧：
        细胞数量加权、但先 per-sample 归一化后的
        基因来源比例（百分比）
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MultipleLocator
    
    # -----------------------------
    # Step1: 构建基础 dataframe
    # -----------------------------
    df = adata.obs.copy()
    
    # raw 表达
    if isinstance(gene, str):  # 单基因情况
        X_raw = adata.raw[:, gene].X
        df["expr"] = X_raw.ravel() if isinstance(X_raw, np.ndarray) else X_raw.toarray().ravel()
    else:  # 多基因情况，取平均
        X_raw = adata.raw[:, gene].X
        if isinstance(X_raw, np.ndarray):
            df["expr"] = X_raw.mean(axis=1)
        else:
            df["expr"] = X_raw.toarray().mean(axis=1)
    
    # -----------------------------
    # Step2: celltype × disease summary
    # -----------------------------
    summary = df.groupby([celltype_col, disease_col]).agg(
        n_cells=("expr", "size"),
        sum_expr=("expr", "sum"),
        frac_expr=("expr", lambda x: (x > 0).mean())
    ).reset_index()
    
    summary["pseudo_bulk_per_cell"] = summary["sum_expr"] / summary["n_cells"]
    
    # -----------------------------
    # Step3: 过滤 celltype
    # -----------------------------
    summary_filtered = summary.groupby(celltype_col).filter(
        lambda x: ((x["pseudo_bulk_per_cell"] >= pseudo_cutoff) &
                   (x["frac_expr"] >= frac_cutoff)).any()
    )
    
    # 排序
    total_expr = summary_filtered.groupby(celltype_col)["pseudo_bulk_per_cell"].sum()
    order_celltype = total_expr.sort_values(ascending=False).index.tolist()
    # print(order_celltype)
    summary_filtered[celltype_col] = pd.Categorical(
        summary_filtered[celltype_col],
        categories=order_celltype,
        ordered=True
    )
    summary_filtered = summary_filtered.sort_values(celltype_col)
    summary_filtered[disease_col] = pd.Categorical(
        summary_filtered[disease_col],
        categories=disease_order,
        ordered=True
    )
    
    valid_celltypes = summary_filtered[celltype_col].astype(str).unique().tolist()
    # -----------------------------
    # Step4: 作图
    # -----------------------------
    fig, axes = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [4.5, 2]}
    )
    fig.subplots_adjust(
        wspace=0.35,  # ⭐ 拉开左右子图
        right=0.88  # ⭐ 给右侧 legend 留白
    )
    
    # ===== 左侧 bubble plot =====
    scatter = sns.scatterplot(
        data=summary_filtered,
        x=disease_col,
        y=celltype_col,
        size="frac_expr",
        hue="pseudo_bulk_per_cell",
        sizes=bubble_size_range,
        palette=cmap,
        ax=axes[0]
    )
    
    axes[0].set_title(f"{'+'.join(gene) if isinstance(gene, list) else gene}")
    axes[0].set_xlabel("Disease")
    axes[0].set_ylabel("Cell type")
    
    # ---- 移除 seaborn 默认 legend
    axes[0].legend_.remove()
    axes[0].grid(False)
    
    # ---- colorbar（pseudo-bulk）
    norm = plt.Normalize(
        summary_filtered["pseudo_bulk_per_cell"].min(),
        summary_filtered["pseudo_bulk_per_cell"].max()
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0], fraction=0.03, pad=0.015)
    # cbar.set_label("Pseudo-bulk per cell")
    
    # ---- size legend（frac_expr）
    size_vals = [0.1, 0.3, 0.6]  # 你可以按需要改
    size_handles = [
        plt.scatter(
            [], [],
            s=np.interp(v, [0, 1], bubble_size_range),
            color="gray",
            alpha=0.6
        )
        for v in size_vals
    ]
    size_legend = axes[0].legend(
        size_handles,
        [f"{int(v * 100)}%" for v in size_vals],
        # title="Fraction\nexpressing",
        loc="upper left",
        bbox_to_anchor=(1.1, 0.6),  # ⭐ 在 colorbar 下方
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.6
    )
    
    axes[0].add_artist(size_legend)  # ⭐ 这是关键
    
    # =============================
    # ===== 右侧：基因来源比例 =====
    # =============================
    
    # ---- per-sample × celltype 的表达贡献
    df_expr = df.copy()
    df_expr["expr_pos"] = df_expr["expr"] > 0
    
    sample_cell_contrib = (
        df_expr
        .groupby([sample_col, celltype_col])["expr"]
        .sum()
        .reset_index()
    )
    
    # ---- 每个 sample 内归一化为比例
    sample_cell_contrib["fraction"] = (
        sample_cell_contrib
        .groupby(sample_col)["expr"]
        .transform(lambda x: x / x.sum())
    )
    
    # ---- 对 sample 取平均
    celltype_fraction = (
            sample_cell_contrib
            .groupby(celltype_col)["fraction"]
            .mean()
            .reindex(order_celltype)
            * 100
    )
    
    print(celltype_fraction.sort_values(ascending=False))
    celltype_fraction = celltype_fraction[celltype_fraction.index.isin(valid_celltypes)]
    
    sns.barplot(
        x=celltype_fraction.values,
        y=celltype_fraction.index,
        orient="h",
        color="gray",
        ax=axes[1]
    )
    
    axes[1].set_xlabel("Gene contribution (%)")
    axes[1].set_ylabel("")
    axes[1].set_title("Gene contribution")
    
    axes[1].set_yticklabels([])
    axes[1].tick_params(axis="y", length=0)
    
    axes[1].set_xlim(0, celltype_fraction.max() * 1.05)
    axes[1].margins(x=0)
    
    axes[1].xaxis.set_major_locator(MultipleLocator(5))
    axes[1].tick_params(axis="x", labelsize=9)
    
    # -----------------------------
    # 保存
    # -----------------------------
    if filename.endswith(".png") or filename.endswith(".pdf"):
        plt.savefig(f"{save_addr}/{filename}")
    else:
        plt.savefig(f"{save_addr}/{filename}.png")
        plt.savefig(f"{save_addr}/{filename}.pdf")
    
    plt.close("all")
    print(f"Picture saved at {save_addr}/{filename}")
    
    return summary_filtered



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
        group_arc_width=8
):
    def blend(c1, c2, t=0.4):
        r1, g1, b1 = to_rgb(c1)
        r2, g2, b2 = to_rgb(c2)
        return (r1 * (1 - t) + r2 * t, g1 * (1 - t) + g2 * t, b1 * (1 - t) + b2 * t)
    
    # -----------------------------
    # Step 0: cell order (group 聚集)
    # -----------------------------
    cells = list(mat.index)
    
    if group_cells:
        grouped = []
        used = set()
        for members in group_cells.values():
            members = [c for c in members if c in cells]
            grouped.extend(members)
            used.update(members)
        rest = [c for c in cells if c not in used]
        cells = grouped + rest
    
    n = len(cells)
    
    if cell_colors is None:
        cmap = plt.get_cmap("tab20")
        cell_colors = {c: cmap(i % 20) for i, c in enumerate(cells)}
    
    # -----------------------------
    # collect edges
    # -----------------------------
    edges = []
    max_w = 0
    for i in cells:
        for j in cells:
            w = mat.loc[i, j]
            if w > min_weight:
                edges.append((i, j, w))
                max_w = max(max_w, w)
    
    if normalize and max_w > 0:
        edges = [(i, j, w / max_w) for i, j, w in edges]
    
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angle_map = dict(zip(cells, angles))
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")
    
    # -----------------------------
    # labels
    # -----------------------------
    for c in cells:
        a = angle_map[c]
        x = label_radius * np.cos(a)
        y = label_radius * np.sin(a)
        
        if np.cos(a) >= 0:
            ha = "left"
            x += label_padding
        else:
            ha = "right"
            x -= label_padding
        
        rot = np.degrees(a)
        if np.cos(a) < 0:
            rot += 180
        
        ax.text(
            x, y, c,
            ha=ha, va="center",
            rotation=rot,
            rotation_mode="anchor",
            fontsize=9,
            zorder=5
        )
    
    # -----------------------------
    # chords
    # -----------------------------
    for src, tgt, w in edges:
        a1 = angle_map[src]
        color = cell_colors[src]
        lw = 0.8 + w * width_scale
        
        # ---- self-loop ----
        if src == tgt:
            theta = a1
            x0 = chord_radius * np.cos(theta)
            y0 = chord_radius * np.sin(theta)
            
            c1 = (self_loop_radius * np.cos(theta - 0.3),
                  self_loop_radius * np.sin(theta - 0.3))
            c2 = (self_loop_radius * np.cos(theta + 0.3),
                  self_loop_radius * np.sin(theta + 0.3))
            
            verts = [(x0, y0), c1, c2, (x0, y0)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            
            patch = patches.PathPatch(
                Path(verts, codes),
                facecolor="none",
                edgecolor=color,
                lw=lw,
                alpha=0.9,
                zorder=3
            )
            ax.add_patch(patch)
            continue
        
        # ---- inter-cell chord ----
        a2 = angle_map[tgt]
        delta = a2 - a1
        if delta > np.pi:
            delta -= 2 * np.pi
        elif delta < -np.pi:
            delta += 2 * np.pi
        mid = a1 + delta / 2
        
        x1, y1 = chord_radius * np.cos(a1), chord_radius * np.sin(a1)
        x2, y2 = chord_radius * np.cos(a2), chord_radius * np.sin(a2)
        cx, cy = control_radius * np.cos(mid), control_radius * np.sin(mid)
        
        path = Path(
            [(x1, y1), (cx, cy), (x2, y2)],
            [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        )
        
        edge_color = blend(cell_colors[src], cell_colors[tgt])
        patch = patches.PathPatch(
            path,
            facecolor="none",
            edgecolor=edge_color,
            lw=lw,
            alpha=0.7,
            zorder=2
        )
        ax.add_patch(patch)
    
    # -----------------------------
    # group outer arcs
    # -----------------------------
    if group_cells:
        if group_arc_radius is None:
            group_arc_radius = label_radius - 0.12
        
        for gname, members in group_cells.items():
            members = [c for c in members if c in angle_map]
            if not members:
                continue
            
            angs = np.array([angle_map[c] for c in members])
            a1, a2 = angs.min(), angs.max()
            
            arc = patches.Arc(
                (0, 0),
                2 * group_arc_radius,
                2 * group_arc_radius,
                theta1=np.degrees(a1),
                theta2=np.degrees(a2),
                lw=group_arc_width,
                color=group_colors.get(gname, "black") if group_colors else "black",
                alpha=0.95,
                zorder=1
            )
            ax.add_patch(arc)
    
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    
    lim = max(self_loop_radius, group_arc_radius or 0) + 0.35
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    
    return fig, ax


def prune_cells_by_activity(mat, min_value=0.0):
    """
    剔除在行和列上都 <= min_value 的细胞

    Parameters
    ----------
    mat : pd.DataFrame
        方阵（行列都是细胞）
    min_value : float
        阈值，行最大值和列最大值都 <= 该值的细胞会被移除

    Returns
    -------
    pd.DataFrame
        剪枝后的矩阵
    """
    import numpy as np
    
    row_max = mat.max(axis=1)
    col_max = mat.max(axis=0)
    
    keep_cells = mat.index[
        (row_max > min_value) | (col_max > min_value)
        ]
    
    return mat.loc[keep_cells, keep_cells]




def CCI_sankey_plot_top2(nodes, links_df, title=None):
    """
    Sankey plot with:
    1) per-middle-cell balanced flow
    2) print top2 contributors (percentage) on each side (per middle cell)
    3) layout stabilization for middle cells without downstream interactions
    """
    
    import plotly.graph_objects as go
    
    # -------------------------
    # identify middle set
    # -------------------------
    middle_cells = nodes["middle"]
    middle_set = set(middle_cells)
    
    plot_links_all = []
    
    # epsilon for dummy edges
    epsilon = max(links_df["value"].min() * 1e-6, 1e-6)
    
    # =========================
    # per-middle-cell balancing
    # =========================
    for m in middle_cells:
        
        left_links_m = links_df[
            (links_df["target"] == m) &
            (~links_df["source"].isin(middle_set))
            ].copy()
        
        right_links_m = links_df[
            (links_df["source"] == m) &
            (~links_df["target"].isin(middle_set))
            ].copy()
        
        sum_left = left_links_m["value"].sum()
        sum_right = right_links_m["value"].sum() if not right_links_m.empty else 0.0
        total_flow = min(sum_left, sum_right) if sum_right > 0 else sum_left
        
        # -------------------------
        # print top2 (per middle)
        # -------------------------
        if sum_left > 0:
            print(f"\n[{m}] Left → Middle Top2:")
            left_top2 = (
                left_links_m
                .groupby("source", as_index=False)["value"]
                .sum()
                .sort_values("value", ascending=False)
                .head(2)
            )
            for _, r in left_top2.iterrows():
                print(f"  {r['source']}: {r['value'] / sum_left * 100:.1f}%")
        
        if sum_right > 0:
            print(f"\n[{m}] Middle → Right Top2:")
            right_top2 = (
                right_links_m
                .groupby("target", as_index=False)["value"]
                .sum()
                .sort_values("value", ascending=False)
                .head(2)
            )
            for _, r in right_top2.iterrows():
                print(f"  {r['target']}: {r['value'] / sum_right * 100:.1f}%")
        else:
            # -------------------------
            # add dummy edge for layout
            # -------------------------
            dummy_row = pd.DataFrame([{
                "source": m,
                "target": "__dummy_right__",
                "value": epsilon,
                "label": "dummy"
            }])
            right_links_m = pd.concat([right_links_m, dummy_row], ignore_index=True)
            # add dummy node if not exists
            if "__dummy_right__" not in nodes["right"]:
                nodes["right"].append("__dummy_right__")
        
        # -------------------------
        # rescale (per middle)
        # -------------------------
        if sum_left > 0:
            left_links_m["value_scaled"] = (
                    left_links_m["value"] / sum_left * total_flow
            )
        else:
            left_links_m["value_scaled"] = left_links_m["value"]
        
        if sum_right > 0:
            right_links_m["value_scaled"] = (
                    right_links_m["value"] / sum_right * total_flow
            )
        else:
            right_links_m["value_scaled"] = right_links_m["value"]
        
        plot_links_all.append(left_links_m)
        plot_links_all.append(right_links_m)
    
    plot_links = pd.concat(plot_links_all, axis=0)
    
    # -------------------------
    # node list & index mapping
    # -------------------------
    all_nodes = nodes["left"] + nodes["middle"] + nodes["right"]
    node_idx = {n: i for i, n in enumerate(all_nodes)}
    
    sources = plot_links["source"].map(node_idx).tolist()
    targets = plot_links["target"].map(node_idx).tolist()
    values = plot_links["value_scaled"].tolist()
    
    # -------------------------
    # color by direction (dummy invisible)
    # -------------------------
    colors = []
    for _, r in plot_links.iterrows():
        if r.get("label") == "dummy":
            colors.append("rgba(0,0,0,0)")
        elif r["target"] in middle_set:
            colors.append("rgba(55,126,184,0.6)")
        else:
            colors.append("rgba(228,26,28,0.6)")
    
    # -------------------------
    # node labels (hide dummy)
    # -------------------------
    node_labels = [
        "" if n == "__dummy_right__" else n
        for n in all_nodes
    ]
    
    # -------------------------
    # plot
    # -------------------------
    fig = go.Figure(
        go.Sankey(
            node=dict(
                label=node_labels,
                pad=15,
                thickness=15
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=colors
            )
        )
    )
    
    if title:
        fig.update_layout(title_text=title)
    
    return fig
