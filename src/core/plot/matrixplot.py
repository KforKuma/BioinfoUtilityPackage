import os
import warnings
import anndata
import pandas as pd
import numpy as np
import scanpy as sc

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用无GUI的后端

from src.core.plot.utils import *
from src.core.adata.deg import easy_DEG

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)

@logged
def plot_deg_matrixplot(
        adata,
        save_addr,
        filename,
        obs_key="Celltype",
        col_order=None,
        exclude_groups=["Proliferative Cell"],
        rename_map=None,
        top_n_genes=20,
        label_step=5,
        save=True,
        show=False,
        **kwargs
):
    """
    执行 DEG 分析并生成美化的 MatrixPlot，遵循标准保存逻辑。
    """
    # 1. 数据预过滤
    if exclude_groups:
        adata = adata[~adata.obs[obs_key].isin(exclude_groups)].copy()
    
    # 2. 调用 easy_DEG
    adata = easy_DEG(
        adata,
        save_addr=save_addr,
        filename_prefix=filename,
        obs_key=obs_key,
        downsample=6000
    )
    
    # 3. 提取 DEG 结果
    uns_key = f"deg_{obs_key}"
    groups = adata.uns[uns_key]['names'].dtype.names
    
    df_all = pd.concat([
        sc.get.rank_genes_groups_df(adata, group=grp, key=uns_key).assign(cluster=grp)
        for grp in groups
    ])
    
    # 4. 筛选 Marker 基因
    marker_dict = {}
    # 如果没传 order，就按默认顺序
    target_order = col_order if col_order else groups
    
    # 改进的 marker 筛选逻辑：去重并保持独特性
    all_candidates = []
    for cl in target_order:
        # 先多拿一点候选，比如前 200 个
        genes = df_all.query("cluster == @cl & logfoldchanges > 1 & pvals_adj < 0.05").head(200)
        all_candidates.append(genes)
    
    df_candidates = pd.concat(all_candidates)
    
    # 逻辑：如果一个基因在多个 cluster 中都是 Top，只保留它 logfoldchanges 最大的那个
    df_unique = df_candidates.sort_values("logfoldchanges", ascending=False).drop_duplicates(subset=["names"],
                                                                                             keep="first")
    
    marker_dict = {}
    for cl in target_order:
        # 在去重后的集合里取每个 cluster 的前 N 个
        cl_unique_genes = df_unique[df_unique["cluster"] == cl].head(top_n_genes)["names"].tolist()
        if cl_unique_genes:
            marker_dict[cl] = cl_unique_genes
        
    # 5. 批量重命名 (用于绘图展示)
    if rename_map:
        marker_dict = {rename_map.get(k, k): v for k, v in marker_dict.items()}
    
    # 6. 绘制 Matrixplot
    base_params = {
        'adata':adata,
        "var_names" : marker_dict,
        "groupby" : obs_key,
        "use_raw" : True,
        "standard_scale" : "var",
        "cmap" : "magma",
        "figsize" : (12, 6),
        "return_fig" : True
    }
    base_params.update(**kwargs)
    
    mp = sc.pl.matrixplot(**base_params)
    
    # 7. 关键美化：处理 X 轴标签密度
    axes_dict = mp.get_axes()
    ax = axes_dict["mainplot_ax"]
    labels = ax.get_xticklabels()
    
    new_labels = []
    for i, lab in enumerate(labels):
        # 每隔 label_step 个基因显示一个标签，其余为空
        if i % label_step == 0:
            new_labels.append(lab.get_text())
        else:
            new_labels.append("")
    
    ax.set_xticklabels(new_labels, rotation=90)
    fig = plt.gcf()
    
    # 8. 严格执行标准保存逻辑
    if save:
        # 自动生成绝对路径
        abs_fig_path = os.path.join(save_addr, filename)
        
        # 假设 matplotlib_savefig 是你环境中的全局函数
        matplotlib_savefig(fig, abs_fig_path)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return marker_dict


@logged
def plot_matrixplot(
        adata,marker_dict,
        save_addr,
        filename,
        obs_key="Celltype",
        label_step=5,
        save=True,
        show=False,
        **kwargs
):
    """
    执行 DEG 分析并生成美化的 MatrixPlot，遵循标准保存逻辑。
    """
    # 1. 绘制 Matrixplot
    base_params = {
        'adata': adata,
        "var_names": marker_dict,
        "groupby": obs_key,
        "use_raw": True,
        "standard_scale": "var",
        "cmap": "magma",
        "figsize": (12, 6),
        "return_fig": True
    }
    base_params.update(**kwargs)
    
    mp = sc.pl.matrixplot(**base_params)
    
    # 2. 关键美化：处理 X 轴标签密度
    axes_dict = mp.get_axes()
    ax = axes_dict["mainplot_ax"]
    labels = ax.get_xticklabels()
    
    new_labels = []
    for i, lab in enumerate(labels):
        # 每隔 label_step 个基因显示一个标签，其余为空
        if i % label_step == 0:
            new_labels.append(lab.get_text())
        else:
            new_labels.append("")
    
    ax.set_xticklabels(new_labels, rotation=90)
    fig = plt.gcf()
    
    # 3. 严格执行标准保存逻辑
    if save:
        # 自动生成绝对路径
        abs_fig_path = os.path.join(save_addr, filename)
        
        # 假设 matplotlib_savefig 是你环境中的全局函数
        matplotlib_savefig(fig, abs_fig_path)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return marker_dict




def plot_professional_matrix(adata, gene_dict, groupby,
                             save_addr=None,
                             filename="matrixplot",
                             dendrogram=False,
                             cmap='RdYlBu_r',
                             show_gene_groups=True,  # 新增：是否显示顶部的基因组 Bracket 名称
                             show_group_names=False,
                             **kwargs):
    """
    修正后的 Matrixplot 函数，具备功能模块标注和高级感配色
    """
    warnings.filterwarnings('ignore', category=FutureWarning)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('fontTools').setLevel(logging.ERROR)
    
    # 1. 预处理：确保基因存在
    filtered_gene_dict = {}
    for k, v in gene_dict.items():
        existing_genes = [g for g in v if g in adata.var_names]
        if existing_genes:
            filtered_gene_dict[k] = existing_genes
    
    # 如果要隐藏基因组名称，直接传入展平后的基因列表，Scanpy 就不会绘制 Brackets
    var_names_input = filtered_gene_dict if show_gene_groups else [g for v in filtered_gene_dict.values() for g in v]
    
    mp = sc.pl.matrixplot(
        adata,
        var_names=var_names_input,
        groupby=groupby,
        standard_scale='var',
        use_raw=False,
        cmap=cmap,
        return_fig=True,
        dendrogram=dendrogram,
        **kwargs
    )
    
    # 3. 核心步骤：手动触发绘图渲染，确保 ax_dict 被填充
    mp.swap_axes = kwargs.get('swap_axes', False)  # 保持参数一致性
    mp.style(cmap=cmap, edge_color='white')
    
    # 手动获取图表对象，这是解决 'No axes' 的关键
    # 在某些版本中，必须调用这个方法才会生成 fig 和 axes
    mp_fig = mp.make_figure()
    
    # 4. 安全地获取 Figure 和 Axes
    # 现在 mp.ax_dict 应该已经填充完毕
    axes_dict = mp.ax_dict
    if not axes_dict:
        # 最后的兜底：如果还不行，从 plt 现存的 figure 中抓取
        fig = plt.gcf()
        all_axes = fig.get_axes()
        if not all_axes:
            print("Critical Error: Matplotlib failed to generate any axes.")
            return
    else:
        # 正常逻辑：获取 Figure
        any_ax = axes_dict.get('main_plot_ax') or list(axes_dict.values())[0]
        fig = any_ax.get_figure()
    
    # 5. 处理顶部的 Bracket 和 文字
    gene_group_ax = axes_dict.get('gene_group_ax')
    if gene_group_ax:
        if not show_gene_groups:
            gene_group_ax.set_visible(False)
        elif not show_group_names:
            # 保留括号线条，隐藏文字
            for txt in gene_group_ax.texts:
                txt.set_visible(False)
    
    # 5. 保存逻辑
    if save_addr:
        if not os.path.exists(save_addr):
            os.makedirs(save_addr)
        abs_path = os.path.join(save_addr, filename)
        matplotlib_savefig(fig, abs_path, close_after=True)
        print(f"Figures saved to: {abs_path}")
    
    plt.close(fig)
