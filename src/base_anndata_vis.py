import anndata
import pandas as pd
import numpy as np
import scanpy as sc
import sys
sys.path.append("/data/HeLab/bio/IBD_analysis/")

import matplotlib
matplotlib.use('Agg')  # 使用无GUI的后端

from src.utils.plot_wrapper import ScanpyPlotWrapper
from src.base_anndata_ops import sanitize_filename
from src.utils.geneset_editor import Geneset

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