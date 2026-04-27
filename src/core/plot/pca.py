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

import matplotlib.pyplot as plt
import seaborn as sns


def plot_pca_results(save_addr, filename,
                     pca_df, exp_var, ax=None, figsize=(10, 7), title="PCA Analysis", show_labels=True,
                     color_by='cluster'):
    """
    使用显式 ax 调用可视化 PCA 结果

    参数:
    - pca_df: run_pca_analysis 返回的坐标表
    - exp_var: 各主成分的解释变异度
    - ax: (可选) 传入已有的 matplotlib axes 对象
    - figsize: 仅在 ax 为 None 时生效
    - title: 图表标题
    - show_labels: 是否标注细胞亚群名称
    - color_by: 按照哪一列进行上色（默认为 'cluster'）
    """
    # 1. 如果没有传入 ax，则创建一个新的 fig 和 ax
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # 2. 计算变异度百分比
    pc1_var = exp_var[0] * 100
    pc2_var = exp_var[1] * 100
    
    # 3. 绘图 (显式指定 ax=ax，并增加 hue 参数)
    # 检查 color_by 是否在列中，避免报错
    hue_param = color_by if (color_by is not None and color_by in pca_df.columns) else None
    
    sns.scatterplot(
        data=pca_df,
        x='PC1',
        y='PC2',
        hue=hue_param,
        palette='viridis' if hue_param else None,  # 如果有颜色列，使用 viridis 色板
        s=120,
        edgecolor='w',
        alpha=0.8,
        ax=ax
    )
    
    # 4. 添加标签 (显式使用 ax.annotate)
    if show_labels:
        for i in range(len(pca_df)):
            ax.annotate(
                pca_df.index[i],
                (pca_df.PC1.iloc[i], pca_df.PC2.iloc[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                alpha=0.8
            )
    
    # 5. 设置图表属性 (使用 ax.set_xxx)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f'PC1 ({pc1_var:.2f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pc2_var:.2f}%)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 如果有图例，调整位置避免遮挡
    if hue_param:
        ax.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    abs_path = os.path.join(save_addr, filename)
    matplotlib_savefig(fig, abs_path)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def plot_cluster_expression_heatmap(merged_df, result_df, save_addr, filename,
                                    top_n_genes=40, figsize=(12, 10), title="Cluster Specific Expression Pattern"):
    """
    展现每个 Cluster 的基因表达模式热图

    参数:
    - merged_df: 包含原始 scores 和 names 的数据框
    - result_df: run_pca_and_clustering 返回的包含 cluster 列的数据框
    - top_n_genes: 选取多少个基因进行展示 (默认选变异度最大的)
    """
    # 1. 将聚类信息映射回 merged_df
    # 建立映射字典 {cell_type: cluster}
    result_df = result_df.reset_index()
    
    cluster_map = result_df.set_index('cell_type')['cluster'].to_dict()
    
    # 确保 merged_df 里的 cell_type 是干净的 (去掉下划线)
    plot_df = merged_df.copy()
    plot_df['cluster'] = plot_df['cell_type'].map(cluster_map)
    
    # 2. 计算每个 Cluster 中每个基因的平均表达量
    # 矩阵形状：Index=基因名, Columns=Cluster, Values=平均Score
    cluster_matrix = plot_df.pivot_table(
        index='names',
        columns='cluster',
        values='scores',
        aggfunc='mean'
    ).fillna(0)
    
    # 3. 筛选基因 (如果基因太多，热图会挤在一起)
    # 这里选择在不同 Cluster 之间差异最大的前 top_n_genes 个基因
    gene_variance = cluster_matrix.var(axis=1).sort_values(ascending=False)
    top_genes = gene_variance.head(top_n_genes).index
    cluster_matrix_filtered = cluster_matrix.loc[top_genes]
    
    # 4. 绘图
    plt.figure(figsize=figsize)
    # 使用 clustermap 可以自动对基因进行聚类，观察哪些基因在特定 Cluster 中高表达
    g = sns.clustermap(
        cluster_matrix_filtered,
        cmap='RdYlBu_r',  # 红黄蓝配色，红色代表高表达
        standard_scale=0,  # 对行(基因)进行 Z-score 标准化，突出跨 Cluster 的相对差异
        figsize=figsize,
        annot=False,  # 基因多时建议不显示数值
        cbar_kws={'label': 'Relative Expression (Row Scaled)'}
    )
    
    g.fig.suptitle(title, y=1.02, fontsize=16)
    
    # 保存图片
    abs_path = os.path.join(save_addr, filename)
    matplotlib_savefig(g.fig, abs_path)
    
    return cluster_matrix_filtered