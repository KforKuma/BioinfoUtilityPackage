import anndata
import pandas as pd
import numpy as np
import scanpy as sc

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用无GUI的后端

from src.core.plot.utils import *


import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)






@logged
def plot_stacked_bar(cluster_counts,
                     cluster_palette=None,
                     xlabel_rotation=0,
                     plot=True,
                     save_addr=None,
                     filename_prefix=None,
                     save=True):
    """
    绘制堆叠条形图，可选择保存为PNG和PDF格式。
    一般配合 get_cluster_counts / get_cluster_props 使用。

    Examples
    --------
    counts = get_cluster_counts(adata,obs_key="Subset_Identity", group_by="disease")
    props = get_cluster_proportions(adata,obs_key="Subset_Identity", group_by="disease")

    plot_stacked_bar(cluster_counts,
                     cluster_palette=adata.uns["leiden_res1_colors"],
                     filename_prefix="AllSample_Counts",save=True)


    Parameters
    ----------
    cluster_counts : pd.DataFrame
        行为组别（如样本、疾病类型），列为子群或类别（如细胞类型）。
    cluster_palette : list or dict, optional
        自定义颜色方案。
    xlabel_rotation : int, optional
        X轴标签旋转角度。
    plot : bool, default True
        是否直接显示图像（Jupyter中）。
    filename_prefix : str, optional
        保存文件的路径（不带后缀时会自动生成 .png/.pdf）。
    save : bool, default True
        是否保存图像。

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        当 plot=True 时返回 Figure 对象，否则返回 None。
    """
    if not plot and not save:
        raise ValueError("At least one of `plot` or `save` must be True.")

    if save_addr is None:
        save_addr = os.getcwd()

    prefix = f"{filename_prefix}_" if filename_prefix else ""
    filename = f"{prefix}Stacked_Barplot"
    abs_fig_path = os.path.join(save_addr, filename)

    fig, ax = plt.subplots(figsize=(6, 6),dpi=300)
    fig.patch.set_facecolor("white")

    # 绘图部分
    cluster_counts.plot(kind="bar", stacked=True, ax=ax, color=cluster_palette)
    ax.legend(bbox_to_anchor=(1.01, 1), frameon=False, title="Cluster")
    sns.despine(fig, ax)
    ax.tick_params(axis="x", rotation=xlabel_rotation)
    ax.set_xlabel(cluster_counts.index.name.capitalize() if cluster_counts.index.name else "")
    ax.set_ylabel("Counts")
    fig.tight_layout()

    # 保存图像部分
    if save:
        matplotlib_savefig(fig, abs_fig_path)
    # 返回或关闭图像
    if plot:
        fig.show()
    else:
        plt.close(fig)

@logged
def plot_stacked_violin(adata,
                      output_dir,filename_prefix,save_addr,
                      gene_dict,
                      cell_type,obs_key="Subset_Identity",
                      group_by="disease",split=False,**kwargs):
    '''

    :param adata:
    :param output_dir:
    :param file_suffix:
    :param save_addr:
    :param gene_dict:
    :param cell_type:
    :param obs_key:
    :param group_by:
    :param kwargs:
    :return:
    '''

    if len(gene_dict) == 0 or next(iter(gene_dict.values())) is None:
        raise ValueError("[easy_stack_violin] gene_dict must contain at least one gene.")

    stacked_violin = ScanpyPlotWrapper(func=sc.pl.stacked_violin)

    for k, v in gene_dict.items():
        gene_name = k
        gene_list = v

        prefix = f"{filename_prefix}_" if filename_prefix else ""
        filename = f"{prefix}{gene_name}_Stacked_Violin{'(split)' if split else ''}.png"

        if isinstance(cell_type, list):
            adata_subset = adata[adata.obs[obs_key].isin(cell_type)]
        elif isinstance(cell_type, str):
            adata_subset = adata[adata.obs[obs_key] == cell_type]
        else:
            raise ValueError("Cell type must be a list or string.")

        default_params = {"swap_axes":False,
                          "cmap":"viridis_r",
                          "use_raw":False,
                          "layer":"log1p_norm",
                          "show":False
        }
        default_params.update(kwargs)
        if kwargs:
            logger.info(f"Overriding defaults with: {kwargs}")

        stacked_violin(
            filename=filename,save_addr=output_dir,
            adata=adata_subset,var_names=gene_list,groupby=group_by,
            **default_params
            )



@logged
def plot_piechart(outer_count, inner_count, colormaplist,
                  plot_title=None, plot=False, save=True, save_path=None,filename=None):
    """
    绘制内外双层饼图（OO风格）
    """
    if plot_title is None:
        plot_title = "Piechart"

    if save_path is None:
        save_path=os.getcwd()

    if filename is None:
        filename="Piechart"

    abs_fig_path=os.path.join(save_path,filename)

    # 创建 Figure 与 Axes 对象
    fig, ax = plt.subplots(figsize=(6, 6),dpi=300)

    # 绘制外环
    ax.pie(
        x=outer_count,
        colors=colormaplist,
        radius=0.8,
        pctdistance=0.765,
        autopct='%3.1f%%',
        labels=outer_count.index.tolist(),
        textprops=dict(color="w"),
        wedgeprops=dict(width=0.3, edgecolor='w')
    )

    # 绘制内环
    ax.pie(
        x=inner_count,
        autopct="%3.1f%%",
        radius=1.0,
        pctdistance=0.85,
        colors=colormaplist,
        textprops=dict(color="w"),
        labels=inner_count.index.tolist(),
        wedgeprops=dict(width=0.3, edgecolor='w')
    )

    # 标题和图例
    ax.set_title(plot_title, fontsize=10)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(1, 0.5)
    )

    # 保存或显示
    if save:
        matplotlib_savefig(fig,abs_fig_path)

    if plot:
        plt.show()
    else:
        plt.close(fig)




