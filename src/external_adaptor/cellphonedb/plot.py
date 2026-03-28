# ===== Standard library =====
import os
import re
import logging
from site import abs_paths

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
from src.external_adaptor.cellphonedb.toolkit import size_map
from src.utils.hier_logger import logged
from src.core.plot.utils import matplotlib_savefig
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
                linewidth=0.4
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
    
    fig = g.fig
    # 尺寸调整
    fig.set_size_inches(w=fig_width, h=fig_height)
    # 布局调整
    fig.subplots_adjust(left=left, right=right,
                          bottom=bottom, top=top,
                          hspace=hspace)
    abs_path = os.path.join(save_addr, filename)
    matplotlib_savefig(fig, abs_path)
    
    df_full.to_excel(f"{save_addr}/{filename}(data).xlsx", index=False)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.path import Path
import matplotlib.patches as patches
from src.core.plot.utils import matplotlib_savefig


def plot_gene_bubble_with_cell_fraction(
        adata, save_addr, filename,
        gene,
        hk_genes=None,
        celltype_col="Subset_Identity",
        celltype_exclude=None,
        celltype_order=None,
        disease_col="disease",
        disease_order=['HC', 'Colitis', 'BD', 'CD', 'UC'],
        topN=10,min_frac=0.15,out_frac=0.05,
        bubble_size_range=(20, 200),
        figsize=(12, 6),
        cmap="Reds"
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # -----------------------------
    # Step1: 数据提取与 HKG 校准
    # -----------------------------
    adata_use = adata[~adata.obs[celltype_col].isin(celltype_exclude)] if celltype_exclude else adata
    df = adata_use.obs.copy()
    
    def get_expr_safe(ad, g):
        source = ad.raw if ad.raw is not None else ad
        if g not in source.var_names: return np.zeros(ad.n_obs)
        X = source[:, g].X
        return X.ravel() if isinstance(X, np.ndarray) else X.toarray().ravel()
    
    genes_to_fetch = [gene] if isinstance(gene, str) else gene
    df["expr_raw"] = np.mean([get_expr_safe(adata_use, g) for g in genes_to_fetch], axis=0)
    
    if hk_genes:
        ref_obj = adata_use.raw if adata_use.raw is not None else adata_use
        valid_hk = [h for h in hk_genes if h in ref_obj.var_names]
        if len(valid_hk) >= 3:
            hk_mean = np.mean([get_expr_safe(adata_use, h) for h in valid_hk], axis=0)
            limit = 0.8
            delta_raw = hk_mean - np.median(hk_mean)
            # 使用 tanh 进行软截断
            delta = limit * np.tanh(delta_raw / limit)
            df["expr"] = (df["expr_raw"] - delta).clip(lower=0)
        else:
            df["expr"] = df["expr_raw"]
    else:
        df["expr"] = df["expr_raw"]
    
    # -----------------------------
    # Step2: 初步汇总与 TopN 筛选 (按 Score 初筛)
    # -----------------------------
    summary = df.groupby([celltype_col, disease_col]).agg(
        n_cells=("expr", "size"),
        sum_expr=("expr", "sum"),
        frac_expr=("expr_raw", lambda x: (x > 0).mean())
    ).reset_index()
    summary["pseudo_bulk_log"] = summary["sum_expr"] / summary["n_cells"]
    
    ct_stats = summary.groupby(celltype_col).agg({"pseudo_bulk_log": "max", "frac_expr": "max"})
    ct_stats["score"] = ct_stats["pseudo_bulk_log"] * ct_stats["frac_expr"]
    
    # --- 新增筛选条件 ---
    potential_candidates = ct_stats[ct_stats["frac_expr"] > min_frac].index.tolist()
    
    # 如果符合比例条件的太少，则降低门槛（保底机制）
    if len(potential_candidates) < 3:
        potential_candidates = ct_stats.sort_values("frac_expr", ascending=False).head(5).index.tolist()
    
    # 在符合比例条件的里面，再按之前的逻辑选出初筛名单
    candidates = ct_stats.loc[potential_candidates].copy()
    candidates["score"] = candidates["pseudo_bulk_log"] * candidates["frac_expr"]
    if celltype_order is not None:
        valid_celltypes = candidates.sort_values("score", ascending=False).head(topN * 2).index.tolist()
    else:
        # 初步筛选出有表达潜力的细胞类型
        candidates = ct_stats.sort_values("score", ascending=False).head(topN * 2)  # 多取一点用于后续贡献度排序
        valid_celltypes = candidates.index.tolist()
    
    # -----------------------------
    # Step3: 计算贡献度并“重新排序”
    # -----------------------------
    weighted_df = adata.uns["weighted_cell_prop"].copy()
    cell_weight_records = []
    for ct, row in weighted_df.iterrows():
        # 此时先记录原始名称，稍后统一处理 Others
        for dis in disease_order:
            if dis == "HC":
                w = row["weight"]
            elif dis in row.index:
                w = np.exp(row[dis] + row["if"]) * row["weight"]
            else:
                w = 0
            cell_weight_records.append({celltype_col: ct, disease_col: dis, "cell_weight": w})
    
    cell_weight_df = pd.DataFrame(cell_weight_records)
    # 这一步计算每个细胞类型在所有疾病状态下的“理论总分子产出”
    full_contribution = summary.merge(cell_weight_df, on=[celltype_col, disease_col])
    # 如果该细胞类型在任何组中的最大表达比例都 < out_frac，则该细胞类型全局置 0
    ct_max_frac = full_contribution.groupby(celltype_col)["frac_expr"].transform("max")
    full_contribution["clean_contribution"] = np.where(
        ct_max_frac < out_frac,
        0,
        (np.exp(full_contribution["pseudo_bulk_log"]) - 1) * full_contribution["cell_weight"]
    )
    
    
    # 【核心逻辑 1】：按右侧分子表达量（clean_contribution）的总和重新排序
    # 我们只对初筛通过的 celltypes 进行排序
    contribution_rank = (
        full_contribution[full_contribution[celltype_col].isin(valid_celltypes)]
        .groupby(celltype_col)["clean_contribution"].sum()
        .sort_values(ascending=False)  # 贡献最大的排最前面
        .head(topN)
    )
    valid_celltypes = contribution_rank.index.tolist()
    
    # -----------------------------
    # Step4: 构建绘图用的 stack_df (处理 Others)
    # -----------------------------
    # 重新打标签
    full_contribution["plot_label"] = full_contribution[celltype_col].apply(
        lambda x: x if x in valid_celltypes else "Others")
    stack_df = full_contribution.groupby(["plot_label", disease_col])["clean_contribution"].sum().reset_index()
    stack_df.columns = [celltype_col, disease_col, "clean_contribution"]
    
    # -----------------------------
    # Step5: 设置 Categorical 顺序
    # -----------------------------
    # 左图 Y 轴：贡献最大的在最上方，Others 不显示在左图气泡中
    summary_filtered = summary[summary[celltype_col].isin(valid_celltypes)].copy()
    summary_filtered[celltype_col] = pd.Categorical(summary_filtered[celltype_col], categories=valid_celltypes,
                                                    ordered=True)
    summary_filtered[disease_col] = pd.Categorical(summary_filtered[disease_col], categories=disease_order,
                                                   ordered=True)
    
    # 【核心逻辑 2】：右图堆叠顺序
    # Pandas 绘图从底向上堆叠。为了让 Others 在顶端，它必须是最后一个 Category
    # 顺序：[贡献最小, ..., 贡献最大, Others] -> 对应 [底部, ..., 中间, 顶端]
    stack_categories = valid_celltypes + ["Others"]
    stack_df[celltype_col] = pd.Categorical(stack_df[celltype_col], categories=stack_categories, ordered=True)
    
    # -----------------------------
    # Step6: 绘图
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(wspace=0.5, right=0.85, left=0.1)
    
    # --- Bubble Plot ---
    sns.scatterplot(data=summary_filtered, x=disease_col, y=celltype_col, size="frac_expr", hue="pseudo_bulk_log",
                    sizes=bubble_size_range, palette=cmap, ax=axes[0], edgecolor='0.3')
    axes[0].get_legend().remove()
    if isinstance(gene, list):
        if len(gene) > 3:
            gene_name = f"{gene[0]}+{gene[1]}...({len(gene)} genes)"
        else:
            gene_name = "+".join(gene)
    else:
        gene_name = gene
    
    axes[0].set_title(f"Expression Per Subset\n {gene_name}")
    
    # 颜色条
    norm = plt.Normalize(summary_filtered["pseudo_bulk_log"].min(), summary_filtered["pseudo_bulk_log"].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes[0], fraction=0.03, pad=0.08)
    
    # --- Stacked Bar ---
    stack_pivot = stack_df.pivot_table(index=disease_col, columns=celltype_col, values="clean_contribution",
                                       aggfunc="sum").fillna(0)
    stack_pivot = stack_pivot.reindex(disease_order)
    
    # 归一化
    if "HC" in stack_pivot.index:
        hc_total = float(stack_pivot.loc["HC"].sum())
        if hc_total > 0: stack_pivot = stack_pivot / hc_total
    
    # 配色：前 N 个彩色，最后一个 Others 灰色
    base_colors = sns.color_palette("tab20", n_colors=len(valid_celltypes))
    # 注意这里颜色顺序要对应 stack_categories: [贡献小->大(Set2), Others(Gray)]
    # 我们希望贡献大的颜色显眼，Others 永远在最后
    plot_colors = list(base_colors) + ["#D3D3D3"]
    
    stack_pivot.plot(kind='bar', stacked=True, ax=axes[1], color=plot_colors, width=0.7, edgecolor='white',
                     linewidth=0.5)
    
    axes[1].set_title("Molecular Contribution\n(Abundance Weighted)")
    axes[1].legend(title="Cell type", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False,
                   fontsize=10, title_fontsize=10)
    axes[1].set_ylabel("Relative to HC Total")
    
    # -----------------------------
    # 保存与导出
    # -----------------------------
    abs_path = os.path.join(save_addr, filename)
    matplotlib_savefig(fig, abs_path)
    summary_filtered.to_excel(f"{save_addr}/{filename}_data.xlsx", index=False)
    plt.close()
    
    return summary_filtered


def plot_universal_bubble_legend(save_addr, bubble_size_range=(20, 200)):
    
    # 创建一个小画布
    fig, ax = plt.subplots(figsize=(2, 3))
    ax.axis('off')
    
    # 定义参照点：0%, 25%, 50%, 75%, 100%
    labels = [0.0, 0.25, 0.5, 0.75, 1.0]
    # 计算对应的像素大小 (线性插值)
    sizes = [bubble_size_range[0] + (bubble_size_range[1] - bubble_size_range[0]) * l for l in labels]
    
    # 画出参考气泡
    for i, (l, s) in enumerate(zip(labels, sizes)):
        ax.scatter([], [], s=s, c='gray', edgecolors='0.3', label=f"{int(l * 100)}%")
    
    ax.legend(
        title="Fraction Expressed",
        labelspacing=1.2,
        handletextpad=1.5,
        loc='center',
        frameon=False
    )
    
    plt.savefig(f"{save_addr}/Universal_Bubble_Legend.pdf", bbox_inches='tight')
    plt.savefig(f"{save_addr}/Universal_Bubble_Legend.png", bbox_inches='tight', dpi=300)
    plt.close()




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
    
    row_max = mat.max(axis=1)
    col_max = mat.max(axis=0)
    
    keep_cells = mat.index[
        (row_max > min_value) | (col_max > min_value)
        ]
    
    return mat.loc[keep_cells, keep_cells]


def CCI_sankey_plot_top5(
        nodes,
        links_df,
        save_addr,
        filename,
        title=None
):
    """
    Sankey plot with:
    1) per-middle-cell balanced flow
    2) save top5 contributors (percentage) per middle cell to xlsx
    3) layout stabilization for middle cells without downstream interactions
    """
    
    import pandas as pd
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
    # collect top5 info
    # =========================
    top_records = []
    
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
        # top5: Left → Middle
        # -------------------------
        if sum_left > 0:
            left_top5 = (
                left_links_m
                .groupby("source", as_index=False)["value"]
                .sum()
                .sort_values("value", ascending=False)
                .head(5)
            )
            for _, r in left_top5.iterrows():
                top_records.append({
                    "middle_cell": m,
                    "direction": "Left→Middle",
                    "partner_cell": r["source"],
                    "value": r["value"],
                    "percentage": r["value"] / sum_left * 100
                })
        
        # -------------------------
        # top5: Middle → Right
        # -------------------------
        if sum_right > 0:
            right_top5 = (
                right_links_m
                .groupby("target", as_index=False)["value"]
                .sum()
                .sort_values("value", ascending=False)
                .head(5)
            )
            for _, r in right_top5.iterrows():
                top_records.append({
                    "middle_cell": m,
                    "direction": "Middle→Right",
                    "partner_cell": r["target"],
                    "value": r["value"],
                    "percentage": r["value"] / sum_right * 100
                })
        else:
            # dummy edge for layout
            dummy_row = pd.DataFrame([{
                "source": m,
                "target": "__dummy_right__",
                "value": epsilon,
                "label": "dummy"
            }])
            right_links_m = pd.concat([right_links_m, dummy_row], ignore_index=True)
            if "__dummy_right__" not in nodes["right"]:
                nodes["right"].append("__dummy_right__")
        
        # -------------------------
        # rescale (per middle)
        # -------------------------
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
    
    plot_links = pd.concat(plot_links_all, axis=0)
    
    # -------------------------
    # save top5 to xlsx
    # -------------------------
    if len(top_records) > 0:
        top_df = pd.DataFrame(top_records)
        top_df["percentage"] = top_df["percentage"].round(5)
        
        xlsx_path = f"{save_addr}/{filename}(data).xlsx"
        top_df.to_excel(xlsx_path, index=False)
    
    # -------------------------
    # node list & index mapping
    # -------------------------
    all_nodes = nodes["left"] + nodes["middle"] + nodes["right"]
    node_idx = {n: i for i, n in enumerate(all_nodes)}
    
    sources = plot_links["source"].map(node_idx).tolist()
    targets = plot_links["target"].map(node_idx).tolist()
    values = plot_links["value_scaled"].tolist()
    
    # -------------------------
    # color by direction
    # -------------------------
    colors = []
    for _, r in plot_links.iterrows():
        if r.get("label") == "dummy":
            colors.append("rgba(0,0,0,0)")
        elif r["target"] in middle_set:
            colors.append("rgba(55,126,184,0.6)")
        else:
            colors.append("rgba(228,26,28,0.6)")
    
    node_labels = ["" if n == "__dummy_right__" else n for n in all_nodes]
    
    fig = go.Figure(
        go.Sankey(
            node=dict(label=node_labels, pad=15, thickness=15),
            link=dict(source=sources, target=targets, value=values, color=colors)
        )
    )
    
    if title:
        fig.update_layout(title_text=title)
    
    fig.write_image(f"{save_addr}/{filename}.pdf", width=800, height=600, engine="kaleido")
    fig.write_image(f"{save_addr}/{filename}.svg", width=800, height=600, engine="kaleido")
    
    # PNG 保持高分辨率用于快速预览
    fig.write_image(f"{save_addr}/{filename}.png", width=800, height=600, scale=3)
    
    # 保存 Excel 数据
    if len(top_records) > 0:
        pd.DataFrame(top_records).to_excel(f"{save_addr}/{filename}(data).xlsx", index=False)