# ===== Standard library =====
import os
from typing import Tuple, Callable

# ===== Third-party =====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

from src.core.plot.utils import matplotlib_savefig


def draw_lorenz(freq_series, label, ax=None):
    '''
    现在支持显式传入 ax 对象。如果不传，则回退到当前活动坐标轴。
    '''
    if ax is None:
        ax = plt.gca()
    
    x = freq_series.sort_values().cumsum()
    # 归一化到 0-1 之间（Lorenz 曲线的标准定义）
    x = x / x.max() if x.max() != 0 else x
    y = np.linspace(0, 1, len(x))
    
    ax.plot(y, x, label=label)


def plot_lorenz(
        tcr_usage_df,
        save_addr,
        filename,
        groups,
        chain: str = "TRAV",
        freq_col: str = "freq",
        figsize=(4, 4),
):
    # 1. 显式创建 fig 和 ax
    fig, ax = plt.subplots(figsize=figsize)
    
    for g in groups:
        # 2. 将 ax 传给子函数
        draw_lorenz(
            tcr_usage_df.loc[g, (chain, freq_col)].dropna(),
            g,
            ax=ax
        )
    
    # 3. 使用 ax 对象进行装饰
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("Cumulative V genes")
    ax.set_ylabel("Cumulative frequency")
    ax.set_title(f"Lorenz curve of {chain} usage")
    ax.legend()
    
    fig.tight_layout()
    
    # 4. 显式通过 fig 对象保存
    abs_path = os.path.join(save_addr, filename)
    matplotlib_savefig(fig, abs_path)
    

def plot_simpson_index(simpson_mat, save_addr, filename,figsize=(6, 7)):
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        simpson_mat,
        cmap="Reds",
        linewidths=0.5,
        linecolor="gray"
    )
    ax.grid(False)
    
    fig = ax.get_figure()
    ax.set_title("TCR usage skewness (Simpson index)")
    fig.tight_layout()
    abs_path = os.path.join(save_addr, filename)
    matplotlib_savefig(fig, abs_path)
    

def color_to_rgb_tuple(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def rgb_tuple_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


def blend_color(c1, c2):
    """RGB 平均出中间色"""
    r1, g1, b1 = color_to_rgb_tuple(c1)
    r2, g2, b2 = color_to_rgb_tuple(c2)
    return rgb_tuple_to_hex((
        int((r1 + r2) / 2),
        int((g1 + g2) / 2),
        int((b1 + b2) / 2)
    ))


def generate_tcr_sankey_colors(
    top5_trav,
    top5_trbv,
    other_color: str = "#BBBBBB"
):
    red_palette = ["#ff9999", "#ff6666", "#ff4d4d", "#ff1a1a", "#e60000"]
    blue_palette = ["#9999ff", "#6666ff", "#4d4dff", "#1a1aff", "#0000e6"]

    trav_color_map = {g: red_palette[i] for i, g in enumerate(top5_trav)}
    trbv_color_map = {g: blue_palette[i] for i, g in enumerate(top5_trbv)}

    def get_trav_color(g):
        return trav_color_map.get(g, other_color)

    def get_trbv_color(g):
        return trbv_color_map.get(g, other_color)

    def link_color(trav, trbv):
        return blend_color(get_trav_color(trav), get_trbv_color(trbv))

    return (
        trav_color_map,
        trbv_color_map,
        get_trav_color,
        get_trbv_color,
        link_color,
    )


# ---------------------------------------------------------------------------------
# 主函数：TCR Sankey
# ---------------------------------------------------------------------------------

def plot_tcr_sankey(
    adata,save_addr=None,filename=None,
    tcr_alpha_col: str = "TRAV_call",
    tcr_beta_col: str = "TRBV_call",
    unassigned_omit: bool = True,
    auto_top_n: int = 5,
    save=True, do_return=False
):
    """
    使用 adata.obs 中的 TRAV / TRBV 调用结果绘制 Sankey 图

    Parameters
    ----------
    adata : AnnData
    tcr_alpha_col : str
    tcr_beta_col : str
    unassigned_omit : bool
        是否剔除 Unassigned
    auto_top_n : int
        自动选前 n 个 TRAV / TRBV
    """
    if save:
        if save_addr is None or filename is None:
            raise ValueError("Save address must be provided if `save=True`.")
    
    df = adata.obs[[tcr_alpha_col, tcr_beta_col]].copy()
    df = df.dropna(subset=[tcr_alpha_col, tcr_beta_col])

    df_count = (
        df.groupby([tcr_alpha_col, tcr_beta_col])
        .size()
        .reset_index(name="count")
    )

    if unassigned_omit:
        df_count = df_count[
            (df_count[tcr_alpha_col] != "Unassigned")
            & (df_count[tcr_beta_col] != "Unassigned")
        ]

    # ------------------------------------------------------------------
    # 自动选择 top TRAV / TRBV
    # ------------------------------------------------------------------
    edge_df = df_count.copy()

    top_trav = (
        edge_df.groupby(tcr_alpha_col)["count"]
        .sum()
        .sort_values(ascending=False)
        .head(auto_top_n + 1)
        .index.tolist()
    )
    top_trbv = (
        edge_df.groupby(tcr_beta_col)["count"]
        .sum()
        .sort_values(ascending=False)
        .head(auto_top_n + 1)
        .index.tolist()
    )

    if "Unassigned" in top_trav:
        top_trav.remove("Unassigned")
    else:
        top_trav = top_trav[:auto_top_n]

    if "Unassigned" in top_trbv:
        top_trbv.remove("Unassigned")
    else:
        top_trbv = top_trbv[:auto_top_n]

    trav_map, trbv_map, get_trav_color, get_trbv_color, link_color = \
        generate_tcr_sankey_colors(top_trav, top_trbv)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    trav_nodes = sorted(edge_df[tcr_alpha_col].unique())
    trbv_nodes = sorted(edge_df[tcr_beta_col].unique())
    nodes = trav_nodes + trbv_nodes

    node_index = {n: i for i, n in enumerate(nodes)}
    node_colors = [
        get_trav_color(n) if n in trav_nodes else get_trbv_color(n)
        for n in nodes
    ]

    # ------------------------------------------------------------------
    # Links
    # ------------------------------------------------------------------
    sources, targets, values, link_colors = [], [], [], []

    for _, row in edge_df.iterrows():
        trav, trbv, count = row[tcr_alpha_col], row[tcr_beta_col], row["count"]
        sources.append(node_index[trav])
        targets.append(node_index[trbv])
        values.append(count)
        link_colors.append(link_color(trav, trbv))

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            label=nodes,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        ),
    ))

    fig.update_layout(
        title="TCR V-region usage Sankey",
        font=dict(size=12),
    )
    
    
    if save:
        fig.write_image(f"{save_addr}/{filename}.png", scale=4)
        fig.write_image(f"{save_addr}/{filename}.pdf", scale=4)
    
    if do_return:
        return fig
    else:
        return

