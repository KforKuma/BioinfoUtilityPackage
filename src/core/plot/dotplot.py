import logging
import os
from typing import Optional

import pandas as pd
import scanpy as sc

from src.core.handlers.plot_wrapper import ScanpyPlotWrapper
from src.utils.env_utils import ensure_package, sanitize_filename
from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


@logged
def geneset_dotplot(
    adata,
    markers,
    marker_sheet,
    save_addr,
    filename_prefix,
    groupby_key,
    use_raw: bool = True,
    **kwargs,
):
    """按基因集批量绘制 dotplot。

    该函数通常配合 `Geneset` 对象使用，按 sheet 和 facet 将 marker genes
    分组后，为不同 cell subtype/subpopulation 或 sample 分组绘制 dotplot。

    Args:
        adata: 输入 AnnData 对象。
        markers: 具有 `get()` 方法的基因集对象，通常为 `Geneset`。
        marker_sheet: 需要提取的 sheet 名称。
        save_addr: 输出目录。
        filename_prefix: 输出文件名前缀。
        groupby_key: `scanpy.pl.dotplot` 的分组列名。
        use_raw: 是否优先使用 `adata.raw`。
        **kwargs: 透传给 `scanpy.pl.dotplot` 的参数。

    Returns:
        None

    Example:
        # 按 geneset sheet 中不同 facet 的定义，分别绘制 dotplot
        geneset_dotplot(
            adata=adata,
            markers=my_markers,
            marker_sheet="Immune",
            save_addr=save_addr,
            filename_prefix="Tcell",
            groupby_key="Subset_Identity",
            use_raw=True,
            dendrogram=True,
        )
    """
    if isinstance(marker_sheet, pd.Series):
        raise TypeError("Argument `marker_sheet` must be a sheet name string, not a pandas Series.")
    if groupby_key not in adata.obs.columns:
        raise KeyError(f"Column `{groupby_key}` was not found in `adata.obs`.")
    if not hasattr(markers, "get"):
        raise TypeError("Argument `markers` must provide a callable `get()` method.")

    dotplot = ScanpyPlotWrapper(func=sc.pl.dotplot)
    gene_dicts = markers.get(sheet_name=marker_sheet, facet_split=True)
    if not gene_dicts:
        logger.warning(f"[geneset_dotplot] Warning! No genesets were returned for `marker_sheet`: '{marker_sheet}'.")
        return

    valid_genes = adata.raw.var_names if use_raw and getattr(adata, "raw", None) is not None else adata.var_names
    if use_raw and getattr(adata, "raw", None) is None:
        logger.warning("[geneset_dotplot] Warning! `use_raw` is True but `adata.raw` is not available. Fallback to `adata.var_names`.")
        use_raw = False

    for facet_name, gene_list_dict in gene_dicts.items():
        prefix = f"{filename_prefix}_" if filename_prefix else ""
        filename = sanitize_filename(f"{prefix}{groupby_key}[{marker_sheet}-{facet_name}]")

        cleaned_gene_list_dict = {}
        for subcat, genes in gene_list_dict.items():
            missing_genes = [gene for gene in genes if gene not in valid_genes]
            if missing_genes:
                logger.info(
                    f"[geneset_dotplot] Missing genes for subcategory '{subcat}' in facet '{facet_name}': "
                    f"{missing_genes}"
                )

            valid_sublist = [gene for gene in genes if gene in valid_genes]
            if valid_sublist:
                cleaned_gene_list_dict[subcat] = valid_sublist

        if not cleaned_gene_list_dict:
            logger.warning(
                f"[geneset_dotplot] Warning! All genes were filtered out for facet '{facet_name}'. Skip plotting."
            )
            continue

        dotplot_kwargs = dict(
            save_addr=save_addr,
            filename=filename,
            adata=adata,
            groupby=groupby_key,
            standard_scale="var",
            var_names=cleaned_gene_list_dict,
            use_raw=use_raw,
        )

        if not use_raw and "layer" not in kwargs and "scvi_normalized" in adata.layers:
            logger.info("[geneset_dotplot] Using layer `scvi_normalized` because `use_raw` is False.")
            dotplot_kwargs["layer"] = "scvi_normalized"

        if "layer" in kwargs and use_raw:
            logger.warning("[geneset_dotplot] Warning! Ignore `layer` because `use_raw` is True.")
            kwargs = dict(kwargs)
            kwargs.pop("layer", None)

        dotplot_kwargs.update(kwargs)
        dotplot(**dotplot_kwargs)


@logged
def plot_cosg_rankplot(
    adata,
    groupby,
    save_addr: Optional[str] = None,
    csv_name: Optional[str] = None,
    filename: Optional[str] = None,
    top_n: int = 5,
    do_return: bool = False,
):
    """运行 COSG 并绘制 top marker dotplot。

    Args:
        adata: 输入 AnnData 对象。
        groupby: 用于分组的 `obs` 列名。
        save_addr: 输出目录。
        csv_name: 保存 COSG 结果表的文件名。
        filename: 绘图输出文件名。
        top_n: 每个分组保留的 top genes 数量。
        do_return: 是否返回更新后的 `adata` 和 marker 字典。

    Returns:
        当 `do_return=True` 时，返回 `(adata, markers)`；否则返回 `None`。

    Example:
        adata, markers = plot_cosg_rankplot(
            adata=adata,
            groupby="Subset_Identity",
            save_addr=save_addr,
            csv_name="Cosg_HVG.csv",
            filename="Cosg_HVG_Dotplot",
            top_n=8,
            do_return=True,
        )
        # markers 的格式通常为 {cluster_name: [gene1, gene2, ...]}
    """
    if groupby not in adata.obs.columns:
        raise KeyError(f"Column `{groupby}` was not found in `adata.obs`.")
    if top_n <= 0:
        raise ValueError("Argument `top_n` must be greater than 0.")

    ensure_package("cosg")
    import cosg

    dotplot = ScanpyPlotWrapper(func=sc.pl.dotplot)
    filename = filename or "Cosg_HVG_Dotplot"
    csv_name = csv_name or "Cosg_HVG.csv"
    save_addr = save_addr or os.getcwd()
    abs_csv_path = os.path.join(save_addr, csv_name)

    cosg.cosg(adata, key_added="cosg", mu=1, n_genes_user=max(100, top_n), groupby=groupby)
    result = adata.uns.get("cosg")
    if result is None:
        raise KeyError("Key `cosg` was not found in `adata.uns` after COSG analysis.")

    df = pd.concat(result, axis=1)
    os.makedirs(save_addr, exist_ok=True)
    df.to_csv(abs_csv_path)
    logger.info(f"[plot_cosg_rankplot] COSG result table was saved to: '{abs_csv_path}'.")

    group_values = adata.obs[groupby]
    if pd.api.types.is_categorical_dtype(group_values):
        categories = group_values.cat.categories.tolist()
    else:
        categories = sorted(group_values.dropna().astype(str).unique().tolist())
        logger.warning(
            f"[plot_cosg_rankplot] Warning! Column `{groupby}` is not categorical. Fallback to sorted unique values."
        )

    markers = {}
    for category in categories:
        if ("names", category) not in df.columns:
            logger.warning(
                f"[plot_cosg_rankplot] Warning! Category '{category}' was not found in COSG result table. Skip it."
            )
            continue
        markers[category] = df.loc[: top_n - 1, ("names", category)].dropna().tolist()

    if not markers:
        raise ValueError("No valid marker genes were extracted from COSG results.")

    dotplot(
        save_addr=save_addr,
        filename=filename,
        adata=adata,
        var_names=markers,
        groupby=groupby,
        cmap="Spectral_r",
        use_raw=False,
        standard_scale="var",
        show=False,
    )

    if do_return:
        return adata, markers
    return None
