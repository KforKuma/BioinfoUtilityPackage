import anndata
import pandas as pd
import numpy as np
import scanpy as sc
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用无GUI的后端

from src.core.utils.plot_wrapper import ScanpyPlotWrapper
from src.core.base_anndata_ops import sanitize_filename
# from src.utils.geneset_editor import Geneset

def geneset_dotplot(adata,
                    markers, marker_sheet,
                    output_dir, filename_prefix, groupby_key, use_raw=True, **kwargs):
    """

    :param adata:
    :param markers: Markers 类对象
    :param marker_sheet:  Markers 的 sheet 名
    :param output_dir:
    :param filename_prefix:
    :param groupby_key:
    :param use_raw:
    :param kwargs:
    :return:
    """

    def _log(msg):
        print(f"[geneset_dotplot] {msg}")

    dotplot = ScanpyPlotWrapper(func=sc.pl.dotplot)

    if isinstance(marker_sheet, pd.Series):
        raise ValueError("marker_sheet is pd.Series, please recheck input.")

    gene_dicts = markers.get_gene_dict(marker_sheet=marker_sheet, facet_split=True)

    for facet_name, gene_list_dict in gene_dicts.items():
        # 构造文件名
        filename = sanitize_filename(f"{filename_prefix}_{groupby_key}_{marker_sheet}_{facet_name}")

        # 获取有效基因名
        if use_raw and adata.raw is not None:
            valid_genes = adata.raw.var_names
        else:
            valid_genes = adata.var_names

        # 检查并过滤子基因集
        cleaned_gene_list_dict = {}
        for subcat, genes in gene_list_dict.items():
            missing_genes = [gene for gene in genes if gene not in valid_genes]
            if missing_genes:
                print(f"[Warning] Genes missing in '{subcat}' ({facet_name}): {missing_genes}")

            # 保留有效基因
            valid_sublist = [gene for gene in genes if gene in valid_genes]
            if valid_sublist:
                cleaned_gene_list_dict[subcat] = valid_sublist

        if not cleaned_gene_list_dict:
            print(f"[Info] All gene groups for facet '{facet_name}' are empty after filtering. Skipping this plot.")
            continue

        # 构造 kwargs（传入 dotplot）
        dotplot_kwargs = dict(
            save_addr=output_dir,
            filename=filename,
            adata=adata,
            groupby=groupby_key,
            standard_scale="var",
            var_names=cleaned_gene_list_dict,  # 注意这里传的是 dict
            use_raw=use_raw,
        )

        if use_raw:
            print("Now using raw data of anndata object.")
        if not use_raw:
            if "scvi_normalized" in adata.layers.keys():
                print("Using layer 'scvi_normalized'.")
                dotplot_kwargs["layer"] = "scvi_normalized"

        # 删除外部可能传入的 layer
        if "layer" in kwargs and use_raw:
            print("Warning: Ignoring 'layer' argument because use_raw=True.")
            kwargs.pop("layer")

        dotplot_kwargs.update(kwargs)

        dotplot(**dotplot_kwargs)
        print(f"--> Dotplot saved: {filename}")



def plot_stacked_bar(cluster_counts,
                     cluster_palette=None,
                     xlabel_rotation=0,
                     plot=True,
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
    filename : str, optional
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

    fig, ax = plt.subplots(dpi=300)
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
        filename = "StackedBarplot" if filename_prefix is None else f"{filename_prefix}_StackedBarplot"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        base, ext = os.path.splitext(filename)
        if ext.lower() not in [".png", ".pdf"]:
            fig.savefig(base + ".png", bbox_inches="tight")
            fig.savefig(base + ".pdf", bbox_inches="tight")
        else:
            fig.savefig(filename, bbox_inches="tight")

    # 返回或关闭图像
    if plot:
        return fig
    else:
        plt.close(fig)


def plot_stacked_violin(adata,
                      output_dir,filename_prefix,save_addr,
                      gene_dict,
                      cell_type,obs_key="Subset_Identity",
                      group_by="disease",**kwargs):
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

    from src.core.utils.plot_wrapper import ScanpyPlotWrapper
    stacked_violin = ScanpyPlotWrapper(func=sc.pl.stacked_violin)

    for k, v in gene_dict.items():
        gene_name = k
        gene_list = v

        filename = f"{filename_prefix}_{gene_name}_StViolin{'(split)' if split else ''}.png"

        if isinstance(cell_type, list):
            adata_subset = adata[adata.obs[obs_key].isin(cell_type)]
        elif isinstance(cell_type, str):
            adata_subset = adata[adata.obs[obs_key] == cell_type]
        else:
            raise ValueError("[easy_stack_violin] cell type must be a list or string.")

        default_params = {"swap_axes":False,
                          "cmap":"viridis_r",
                          "use_raw":False,
                          "layer":"log1p_norm",
                          "show":False
        }
        default_params.update(kwargs)
        if kwargs:
            print(f"[easy_stack_violin] Overriding defaults with: {kwargs}")

        stacked_violin(
            filename=filename,save_addr=output_dir,
            adata=adata_subset,var_names=gene_list,groupby=group_by,
            **default_params
            )


