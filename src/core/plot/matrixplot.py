import os
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