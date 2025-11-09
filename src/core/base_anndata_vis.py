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

def _matplotlib_savefig(fig, abs_file_path):
    os.makedirs(os.path.dirname(abs_file_path), exist_ok=True)
    base, ext = os.path.splitext(abs_file_path)
    if ext.lower() not in [".png", ".pdf"]:
        fig.savefig(base + ".png", bbox_inches="tight", dpi=300)
        fig.savefig(base + ".pdf", bbox_inches="tight", dpi=300)
    else:
        fig.savefig(abs_file_path, bbox_inches="tight", dpi=300)


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

    # 处理图像名
    if save_addr is None:
        save_addr=os.getcwd()
    filename = "Stacked_Barplot" if filename_prefix is None else f"{filename_prefix}_Stacked_Barplot"
    abs_fig_path = os.path.join(save_addr, filename)

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
        _matplotlib_savefig(fig, abs_fig_path)

    # 返回或关闭图像
    if plot:
        fig.show()
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


def plot_cosg_rankplot(adata, groupby, save_addr=None,csv_name=None, filename=None, top_n=5, do_return=False):
    """
    Plot the rank plot for COSG marker genes.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    groupby : str
        Key for grouping the data.
    csv_name : str
        The file name for saving the COSG marker genes.
    save_addr : str, optional
        The directory for saving the rank plot.
    filename : str, optional
        The file name for saving the rank plot.
    top_n : int, optional
        The number of top genes to select for each group.
    do_return : bool, optional
        Whether to return the data and markers.

    Returns
    -------
    adata : AnnData
        The data with COSG marker genes.
    markers : dict
        The top genes for each group.

    """
    import cosg

    dotplot = ScanpyPlotWrapper(func=sc.pl.dotplot)

    if filename is None:
        filename="Cosg_HVG_Dotplot"

    if save_addr is None:
        save_addr=os.getcwd()

    if csv_name is None:
        csv_name="Cosg_HVG.csv"

    abs_csv_path = os.path.join(save_addr, csv_name)

    # 1. 计算COSG marker
    cosg.cosg(adata, key_added='cosg', mu=1, n_genes_user=100, groupby=groupby)

    # 2. 保存结果
    result = adata.uns['cosg']
    df = pd.concat(result, axis=1)
    df.to_csv(abs_csv_path)

    # 3. 提取top_n
    markers = {c: df.loc[:top_n-1, ('names', c)].tolist()
               for c in adata.obs[groupby].cat.categories}

    # 4. 绘图
    dotplot(save_addr, filename,
            cmap='Spectral_r', use_raw=False, standard_scale='var',show=False)

    if do_return:
        return adata, markers


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

    # 1️⃣ 创建 Figure 与 Axes 对象
    fig, ax = plt.subplots(figsize=(6, 6))

    # 2️⃣ 绘制外环
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

    # 3️⃣ 绘制内环
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

    # 4️⃣ 标题和图例
    ax.set_title(plot_title, fontsize=10)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(1, 0.5)
    )

    # 5️⃣ 保存或显示
    if save:
        _matplotlib_savefig(fig,abs_fig_path)

    if plot:
        plt.show()
    else:
        plt.close(fig)


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


def _formate_tidy_label(cluster_to_labels):
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


def plot_pca_with_cluster_legend(result_df, cluster_to_labels, save_dir, figname, only_show=5, figsize=(10, 6)):
    plt.figure(figsize=figsize)

    # Step 1: 画 PCA 聚类散点图
    sns.scatterplot(
        data=result_df,
        x="PC1", y="PC2",
        hue="cluster",
        palette="tab10",
        s=100,
        legend='full'
    )

    # Step 2: 构造右侧图注文字
    legend_text = ""
    cluster_to_labels = _formate_tidy_label(cluster_to_labels)
    for cluster_id, labels in cluster_to_labels.items():
        label_str = _format_labels_in_lines(labels, max_label=only_show)
        legend_text += f"Cluster {cluster_id}: \n{label_str}\n"

    # Step 3: 添加注释文字（图右侧）
    plt.gcf().text(0.8, 0.5, legend_text,
                   fontsize=10, linespacing=1.8,
                   va='center', ha='left')

    # Step 4: 调整图布局
    plt.title("PCA with KMeans Clustering")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # 让右侧有空间
    plt.savefig(f"{save_dir}/{figname}_PCA_cluster.png", dpi=300, bbox_inches="tight")
