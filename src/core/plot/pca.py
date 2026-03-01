import anndata
import pandas as pd
import numpy as np
import scanpy as sc
import os

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用无GUI的后端

from sklearn.cluster import KMeans

from src.core.plot.utils import matplotlib_savefig

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)

def _set_plot_style():
    """
    设置统一绘图风格，使PCA和Cluster图视觉一致。
    """
    sns.set_theme(
        context="talk",          # 字体较大，适合展示
        style="whitegrid",       # 背景白色带浅网格
        palette="tab10",         # 默认色板
        font="Arial",            # 统一字体
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300
        }
    )


@logged
def _elbow_detector(ts, cluster_counts, method="kneed", default_cluster=2):
    """
    :param ts: x轴，簇数列表
    :param cluster_counts: y轴，对应的聚类指标（如 inertia）
    :param method: "MSD" or "kneed"
    :param min_cluster: 最小簇数下限
    :param default_cluster: 检测失败时默认返回值
    :return: optimal cluster number
    """
    optimal_t = None

    if method == "MSD":
        # 简单拐点检测：最大二阶差分
        first_diff = np.diff(cluster_counts)
        second_diff = np.diff(first_diff)

        # 拐点 = 最大弯曲点
        elbow_idx = np.argmax(np.abs(second_diff)) + 2  # +2 to align with ts index after 2 diffs
        optimal_t = ts[elbow_idx]

    elif method == "kneed":
        ensure_package(kneed)
        from kneed import KneeLocator
        kneedle = KneeLocator(ts, cluster_counts, curve='convex', direction='decreasing')
        optimal_t = kneedle.knee

    # 检查有效性
    if optimal_t is None:
        logger.info(f"Failed to fetch the elbow, using default elbow number: {default_cluster}")
        optimal_t = default_cluster
    else:
        logger.info(f"Optimal cluster {optimal_t} found with elbow method.")

    return optimal_t

@logged
def _format_labels_in_lines(labels, max_line_length=60, max_label=None):
    '''
    为图注自动换行：限制每行最大字符数
    :param labels: list of str
    :param max_line_length: 每行最多字符数
    :param max_label: 最多展示多少个 label
    :return: formatted string with \n
    '''
    if max_label:
        labels = labels[:max_label]
        if len(labels) < len(labels):
            labels.append("...")

    lines = []
    current_line = ""

    for label in labels:
        label_str = label if current_line == "" else ", " + label
        # 如果当前加上这个 label 会超限，先收行
        if len(current_line + label_str) > max_line_length:
            lines.append(current_line)
            current_line = label
        else:
            current_line += label_str

    if current_line:
        lines.append(current_line)

    return "  " + "\n  ".join(lines) + "\n  "

@logged
def _format_tidy_label(cluster_to_labels):
    '''
    返回一个重整细胞名的字典，格式为 "[disease] subtype"
    '''
    new_dict = {}
    for cluster_id, labels in cluster_to_labels.items():
        new_labels = []
        labels.sort()
        for label in labels:
            try:
                dis = "_".join(label.split("_")[:-2])
                celltype = label.split("_")[-2]
                cellsubtype = label.split("_")[-1]
                new_label = f"[{dis}] {cellsubtype}"
            except ValueError:
                # 如果格式不符，保持原样
                new_label = label
            new_labels.append(new_label)
        new_dict[cluster_id] = new_labels
    return new_dict

@logged
def _plot_pca_with_cluster_legend(
    result_df,
    cluster_to_labels,
    only_show=5,
    figsize=(10, 6),
    save_addr=None,
    filename=None,
    save=True,
    plot=False,
):

    # ---------- 路径与文件 ----------
    """
    Plot PCA result with KMeans clustering and add a legend on the right side.
    绘制带右侧注释的 PCA 聚类散点图。

    Parameters
    ----------
    result_df : pandas.DataFrame
        Result of PCA, with columns 'PC1', 'PC2', and 'cluster'.
    cluster_to_labels : dict
        A dictionary mapping cluster index to a list of cell types.
    only_show : int, default 5
        Only show the first `only_show` number of cell types in the legend.
    figsize : tuple, default (10, 6)
        Figure size.
    save_addr : str, default None
        Path to save the figure.
    filename : str, default None
        Filename of the figure.
    save : bool, default True
        Whether to save the figure.
    plot : bool, default False
        Whether to plot the figure.

    Returns
    -------
    str
        Path to the saved figure.

    """
    _set_plot_style()

    if save_addr is None:
        save_addr = os.getcwd()
    os.makedirs(save_addr, exist_ok=True)

    if filename is None:
        filename = "PCA"

    abs_fig_path = os.path.join(save_addr, filename)

    # ---------- 图像创建 ----------
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Step 1: 绘制 PCA 散点图
    sns.scatterplot(
        data=result_df,
        x="PC1", y="PC2",
        hue="cluster",
        palette="tab10",
        s=100, edgecolor="black", linewidth=0.5, ax=ax
    )

    # Step 2: 构造右侧注释文字
    legend_text = ""
    cluster_to_labels = _format_tidy_label(cluster_to_labels)
    for cluster_id, labels in cluster_to_labels.items():
        label_str = _format_labels_in_lines(labels, max_label=only_show)
        legend_text += f"Cluster {cluster_id}:\n{label_str}\n\n"

    # Step 3: 添加文字说明
    fig.text(
        0.8, 0.5, legend_text,
        fontsize=10, linespacing=1.6,
        va='center', ha='left'
    )

    # Step 4: 调整图像与标题
    ax.set_title("PCA with KMeans Clustering")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')

    fig.tight_layout(rect=[0, 0, 0.75, 1])

    # Step 5: 保存与显示逻辑
    if save:
        matplotlib_savefig(fig, abs_fig_path)

    if plot:
        plt.show()
    else:
        plt.close(fig)

    return abs_fig_path + ".png"

@logged
def _pca_cluster_process(result_df, save_addr, filename, figsize=(10, 6)):

    # 使用 PC1 和 PC2
    X = result_df[["PC1", "PC2"]]
    max_k = min(10, X.shape[0])
    cluster_seq = [i for i in range(2, max_k + 1)]
    inertia_seq = [KMeans(n_clusters=k, random_state=0).fit(X).inertia_ for k in cluster_seq]

    optimal_cluster = _elbow_detector(cluster_seq, inertia_seq)

    kmeans = KMeans(n_clusters=optimal_cluster, random_state=0)  # 可改成你认为合适的簇数
    result_df['cluster'] = kmeans.fit_predict(X)

    # 整理出一个 cluster: celltype list 的字典
    # Step 1: 去重（保留第一个出现的 label）
    dedup_df = result_df.drop_duplicates(subset='label', keep='first')
    # Step 2: 设置 label 为索引，只保留 cluster 列
    label_cluster_map = dedup_df.set_index('cluster')['label']
    cluster_to_labels = label_cluster_map.groupby(label_cluster_map.index).apply(list).to_dict()

    _plot_pca_with_cluster_legend(result_df, cluster_to_labels,
                                 save_addr=save_addr, filename=filename, only_show=100, figsize=figsize)

    return cluster_to_labels

@logged
def _plot_pca(result_df, pca,  color_by,
              figsize=(12, 10),
              save_addr=None, filename_prefix=None):

    """
    Plot PCA result and explained variance.
    绘制带有 PCA 解释力（Variance）的图。

    Parameters
    ----------
    result_df : pandas.DataFrame
        Result of PCA, with columns 'PC1', 'PC2', and 'group'.
    pca : sklearn.decomposition.PCA
        PCA object.
    color_by : str
        Column name to color by.
    figsize : tuple, default (12, 10)
        Figure size.
    save_addr : str, default None
        Path to save the figure.
    filename_prefix : str, default None
        Prefix of the filename.

    Returns
    -------
    str
        Path to the saved figure.

    """
    _set_plot_style()

    if save_addr is None:
        save_addr = os.getcwd()
    os.makedirs(save_addr, exist_ok=True)

    prefix = f"{filename_prefix}_" if filename_prefix else ""
    filename1 = f"{prefix}PCA_Explanation"
    filename2 = f"{prefix}PCA"

    # 图 1
    fig, ax = plt.subplots(figsize=(6,6), dpi=300) # 这是固定尺寸的小图
    explained_var = pca.explained_variance_ratio_

    bars = ax.bar(
        range(1, len(explained_var) + 1),
        explained_var * 100,
        color=sns.color_palette("tab10")[0]
    )
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)
    ax.set_title("PCA Explained Variance", fontsize=12)
    fig.tight_layout()
    matplotlib_savefig(fig, os.path.join(save_addr, filename1),close_after=True)

    # 图 2
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    PC1 = f"{pca.explained_variance_ratio_[0]:.2%}"
    PC2 = f"{pca.explained_variance_ratio_[1]:.2%}"

    sns.scatterplot(
        data=result_df,
        x="PC1", y="PC2",
        hue=color_by,
        style="group",
        s=100, edgecolor="black", linewidth=0.5, ax=ax
    )

    ax.set_title(f"PCA of Cell-Disease DEG Patterns")
    ax.set_xlabel(f'PC1({PC1})')
    ax.set_ylabel(f'PC2({PC2})')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)

    ncol = (len(result_df[color_by].unique()) + len(result_df["group"].unique())) // 25 + 1
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=ncol)
    fig.tight_layout()
    matplotlib_savefig(fig, os.path.join(save_addr, filename2),close_after=True)


@logged
def pca_cluster_process(result_df, save_addr, filename, **kwargs):
    """
    Compute optimal cluster number using Elbow method and perform K-means clustering.

    Parameters
    ----------
    result_df : pandas.DataFrame
        Result of PCA, with columns 'PC1', 'PC2', and 'label'.
    save_addr : str
        Path to save the figure.
    filename : str
        Filename of the figure.
    **kwargs : for plot_pca_with_cluster_legend

    Returns
    -------
    cluster_to_labels : dict
        A dictionary mapping cluster index to a list of cell types.

    Notes
    -----
    The function first computes the optimal cluster number using the Elbow method,
    and then performs K-means clustering with the optimal number of clusters.
    The result is a dictionary mapping cluster index to a list of cell types.
    """
    

    # 使用 PC1 和 PC2
    X = result_df[["PC1", "PC2"]]
    max_k = min(10, X.shape[0])
    cluster_seq = [i for i in range(2, max_k + 1)]
    inertia_seq = [KMeans(n_clusters=k, random_state=0).fit(X).inertia_ for k in cluster_seq]

    optimal_cluster = _elbow_detector(cluster_seq, inertia_seq)

    kmeans = KMeans(n_clusters=optimal_cluster, random_state=0)
    result_df['cluster'] = kmeans.fit_predict(X)

    # 整理出一个 cluster: celltype list 的字典
    # Step 1: 去重（保留第一个出现的 label）
    dedup_df = result_df.drop_duplicates(subset='label', keep='first')
    # Step 2: 设置 label 为索引，只保留 cluster 列
    label_cluster_map = dedup_df.set_index('cluster')['label']
    cluster_to_labels = label_cluster_map.groupby(label_cluster_map.index).apply(list).to_dict()

    _plot_pca_with_cluster_legend(result_df, cluster_to_labels, save_addr=save_addr,
                                 filename=filename, **kwargs)

    return cluster_to_labels
