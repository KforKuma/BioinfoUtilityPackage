import anndata
import pandas as pd
import numpy as np
import scanpy as sc

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用无GUI的后端

from src.core.plot.utils import *
from src.core.handlers.plot_wrapper import *
from src.utils.env_utils import sanitize_filename,ensure_package

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)


@logged
def geneset_dotplot(adata,
                    markers, marker_sheet,
                    save_addr, filename_prefix, groupby_key, use_raw=True, **kwargs):
    """

    :param adata:
    :param markers: Geneset 类对象
    :param marker_sheet:  Geneset 的 sheet_name 参数
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

    gene_dicts = markers.get(sheet_name=marker_sheet, facet_split=True)

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
