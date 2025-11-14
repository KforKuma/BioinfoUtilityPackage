import anndata
import pandas as pd
import numpy as np
import scanpy as sc

import re,os

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用无GUI的后端

from src.core.utils.plot_wrapper import ScanpyPlotWrapper
from src.core.base_anndata_ops import _elbow_detector
from src.utils.env_utils import ensure_package, sanitize_filename

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)

def _matplotlib_savefig(fig, abs_file_path, close_after=False):
    os.makedirs(os.path.dirname(abs_file_path), exist_ok=True)
    
    # 定义 Matplotlib 常见支持的格式
    valid_exts = {".png", ".pdf", ".svg", ".eps", ".jpg", ".jpeg", ".tif", ".tiff"}
    
    # 手动拆分文件名并识别扩展
    filename = os.path.basename(abs_file_path)
    dirname = os.path.dirname(abs_file_path)
    name_parts = filename.rsplit('.', 1)  # 只从右边拆一次
    
    if len(name_parts) == 2 and f".{name_parts[1].lower()}" in valid_exts:
        base = os.path.join(dirname, name_parts[0])
        ext = f".{name_parts[1].lower()}"
    else:
        base = os.path.join(dirname, filename)
        ext = ""
    
    # 根据是否识别到扩展名决定保存方式
    if ext == "":
        # 未识别到文件类型 → 默认导出 png 和 pdf
        fig.savefig(base + ".png", bbox_inches="tight", dpi=300)
        fig.savefig(base + ".pdf", bbox_inches="tight", dpi=300)
    else:
        # 识别到合法扩展名 → 按原路径保存
        fig.savefig(base + ext, bbox_inches="tight", dpi=300)
    
    if close_after:
        plt.close(fig)

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
def geneset_dotplot(adata,
                    markers, marker_sheet,
                    save_addr, filename_prefix, groupby_key, use_raw=True, **kwargs):
    """

    :param adata:
    :param markers: Markers 类对象
    :param marker_sheet:  Markers 的 sheet 名
    :param save_addr:
    :param filename_prefix:
    :param groupby_key:
    :param use_raw:
    :param kwargs:
    :return:
    """

    dotplot = ScanpyPlotWrapper(func=sc.pl.dotplot)

    if isinstance(marker_sheet, pd.Series):
        raise ValueError("marker_sheet is pd.Series, please recheck input.")

    gene_dicts = markers.get_gene_dict(marker_sheet=marker_sheet, facet_split=True)

    for facet_name, gene_list_dict in gene_dicts.items():
        # 构造文件名
        prefix = f"{filename_prefix}_" if filename_prefix else ""
        filename = sanitize_filename(f"{prefix}{groupby_key}[{marker_sheet}-{facet_name}]")

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
                logger.info(f"Genes missing in '{subcat}' ({facet_name}): {missing_genes}")

            # 保留有效基因
            valid_sublist = [gene for gene in genes if gene in valid_genes]
            if valid_sublist:
                cleaned_gene_list_dict[subcat] = valid_sublist

        if not cleaned_gene_list_dict:
            logger.info(f"All gene groups for facet '{facet_name}' are empty after filtering. Skipping this plot.")
            continue

        # 构造 kwargs（传入 dotplot）
        dotplot_kwargs = dict(
            save_addr=save_addr,
            filename=filename,
            adata=adata,
            groupby=groupby_key,
            standard_scale="var",
            var_names=cleaned_gene_list_dict,  # 注意这里传的是 dict
            use_raw=use_raw,
        )

        if use_raw:
            logger.info("Now using raw data of anndata object.")
        if not use_raw:
            if "scvi_normalized" in adata.layers.keys():
                logger.info("Using layer 'scvi_normalized'.")
                dotplot_kwargs["layer"] = "scvi_normalized"

        # 删除外部可能传入的 layer
        if "layer" in kwargs and use_raw:
            logger.info("Warning: Ignoring 'layer' argument because use_raw=True.")
            kwargs.pop("layer")

        dotplot_kwargs.update(kwargs)
        dotplot(**dotplot_kwargs)


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
        _matplotlib_savefig(fig, abs_fig_path)
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
    ensure_package(cosg)
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
        _matplotlib_savefig(fig,abs_fig_path)

    if plot:
        plt.show()
    else:
        plt.close(fig)

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
        _matplotlib_savefig(fig, abs_fig_path)

    if plot:
        plt.show()
    else:
        plt.close(fig)

    return abs_fig_path + ".png"

@logged
def _pca_cluster_process(result_df, save_addr, filename, figsize=(10, 6)):
    from sklearn.cluster import KMeans

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
    _matplotlib_savefig(fig, os.path.join(save_addr, filename1),close_after=True)

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
    _matplotlib_savefig(fig, os.path.join(save_addr, filename2),close_after=True)

@logged
def process_resolution_umaps(adata, output_dir, resolutions,use_raw=True,**kwargs):
    """
    生成 UMAP 图像，用于不同 Leiden 分辨率对比。
    """
    umap_plot = ScanpyPlotWrapper(sc.pl.umap)
    color_keys = [f"leiden_res{res}" for res in resolutions]
    umap_plot(
        save_addr=output_dir,
        filename="Res_Comparison",
        adata=adata,
        color=color_keys,
        legend_loc="on data",
        use_raw=use_raw,
        **kwargs
    )

@logged
def plot_QC_umap(adata, save_addr, filename_prefix):
    umap_plot = ScanpyPlotWrapper(sc.pl.umap)
    key_dict = {
        "organelles": [i for i in adata.obs.columns if re.search(r'mt|mito|rb|ribo', i)],
        "phase": [i for i in adata.obs.columns if re.search(r'phase', i)],
        "counts": [i for i in adata.obs.columns if re.search(r'disease|tissue', i)]
    }
    key_list = [item for sublist in key_dict.values() for item in sublist if sublist]
    if len(key_list) == 0:
        raise ValueError("[plot_QC_umap] No QC obs_key founded, unable to draw QC umap plot.")

    # 对每类调用一次
    for name, cols in key_dict.items():
        if not cols:
            continue
        umap_plot(
            save_addr=save_addr,
            filename=f"{filename_prefix}_UMAP_{name}",
            adata=adata,
            color=cols
        )