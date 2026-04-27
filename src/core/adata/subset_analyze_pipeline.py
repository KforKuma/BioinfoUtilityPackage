import os
import sys
from typing import Any, Optional, Sequence

from anndata import AnnData
import scanpy as sc

from src.core.adata.deg import easy_DEG
from src.core.adata.ops import subcluster
from src.core.handlers.plot_wrapper import ScanpyPlotWrapper
from src.core.plot.dotplot import geneset_dotplot
from src.core.plot.umap import plot_QC_umap, process_resolution_umaps

import logging
from src.utils.hier_logger import logged

sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


def _validate_pipeline_inputs(
        adata_subset: AnnData,
        save_addr: str,
        DEG_enrich_key: str,
        resolutions_list: Optional[Sequence[float]],
        use_rep: str,
) -> None:
    """检查主流程输入参数是否满足分析要求。"""
    if adata_subset.n_obs == 0:
        raise ValueError("`adata_subset` contains no cells.")
    if not save_addr:
        raise ValueError("`save_addr` cannot be empty.")
    if DEG_enrich_key == "leiden_res" and not resolutions_list:
        raise ValueError("`resolutions_list` is required when `DEG_enrich_key='leiden_res'`.")
    if use_rep != "X" and use_rep not in adata_subset.obsm:
        raise KeyError(f"`use_rep`: '{use_rep}' does not exist in `adata_subset.obsm`.")


@logged
def adata_subset_analyze_pipeline(
        adata_subset: AnnData,
        filename_prefix: str,
        my_markers: Any,
        marker_sheet: Any,
        save_addr: str,
        do_subcluster: bool = True,
        do_DEG_enrich: bool = True,
        downsample: bool | int | float = False,
        DEG_enrich_key: str = "leiden_res",
        resolutions_list: Optional[Sequence[float]] = None,
        use_rep: str = "X_scVI",
        use_raw: bool = True,
        **kwargs,
) -> AnnData:
    """执行单个 AnnData 子集的自动化分析流程。

    流程包括：
    1. 可选的子聚类；
    2. UMAP 与 QC 可视化；
    3. marker dotplot；
    4. 可选的 DEG 分析。

    Args:
        adata_subset: 待分析的 AnnData 子集。
        filename_prefix: 输出文件名前缀。
        my_markers: marker 基因字典或 marker 定义对象。
        marker_sheet: marker 注释表或名称。
        save_addr: 输出目录。
        do_subcluster: 是否执行子聚类。
        do_DEG_enrich: 是否执行 DEG 分析。
        downsample: DEG 分析时的下采样参数。
        DEG_enrich_key: 进行后续分析时使用的分组键。若为 `"leiden_res"`，
            则会遍历 `resolutions_list`。
        resolutions_list: Leiden 分辨率列表。
        use_rep: 子聚类时使用的表示名称。
        use_raw: 绘图和 DEG 时是否优先使用 `adata.raw`。
        **kwargs: 透传给绘图函数的额外参数。

    Returns:
        分析完成后的 AnnData 对象。
    """
    _validate_pipeline_inputs(adata_subset, save_addr, DEG_enrich_key, resolutions_list, use_rep)
    os.makedirs(save_addr, exist_ok=True)

    n_cells = adata_subset.n_obs
    logger.info(
        f"[adata_subset_analyze_pipeline] Starting analysis for subset: '{filename_prefix}'. "
        f"Total cells: {n_cells}."
    )

    umap_plot = ScanpyPlotWrapper(func=sc.pl.umap)

    if do_subcluster:
        logger.info(
            f"[adata_subset_analyze_pipeline] Starting subclustering with `use_rep`: '{use_rep}' "
            f"and `resolutions_list`: {list(resolutions_list) if resolutions_list is not None else None}."
        )
        n_pcs_current = min(adata_subset.n_vars, 50) if use_rep == "X" else min(adata_subset.obsm[use_rep].shape[1], 50)
        adata_subset = subcluster(
            adata_subset,
            n_neighbors=20,
            n_pcs=n_pcs_current,
            resolutions=resolutions_list,
            use_rep=use_rep,
        )
    else:
        logger.info("[adata_subset_analyze_pipeline] Warning! Subclustering is disabled.")

    if DEG_enrich_key == "leiden_res":
        logger.info("[adata_subset_analyze_pipeline] Running multi-resolution workflow for Leiden clusters.")
        process_resolution_umaps(
            adata_subset,
            save_addr,
            resolutions_list,
            use_raw=use_raw,
            **kwargs,
        )
        plot_QC_umap(adata_subset, save_addr, filename_prefix=filename_prefix)

        for resolution in resolutions_list:
            groupby_key = f"leiden_res{resolution}"
            _execute_cluster_analysis_workflow(
                adata=adata_subset,
                key=groupby_key,
                prefix=filename_prefix,
                markers=my_markers,
                sheet=marker_sheet,
                addr=save_addr,
                plot_func=umap_plot,
                do_deg=do_DEG_enrich,
                ds=downsample,
                raw=use_raw,
                **kwargs,
            )
    elif DEG_enrich_key in adata_subset.obs.columns:
        logger.info(
            f"[adata_subset_analyze_pipeline] Running single-key workflow with `DEG_enrich_key`: "
            f"'{DEG_enrich_key}'."
        )
        _execute_cluster_analysis_workflow(
            adata=adata_subset,
            key=DEG_enrich_key,
            prefix=filename_prefix,
            markers=my_markers,
            sheet=marker_sheet,
            addr=save_addr,
            plot_func=umap_plot,
            do_deg=do_DEG_enrich,
            ds=downsample,
            raw=use_raw,
            **kwargs,
        )
        plot_QC_umap(adata_subset, save_addr, filename_prefix=filename_prefix)
    else:
        raise KeyError(f"`DEG_enrich_key`: '{DEG_enrich_key}' does not exist in `adata_subset.obs`.")

    logger.info(f"[adata_subset_analyze_pipeline] Analysis completed for subset: '{filename_prefix}'.")
    return adata_subset


@logged
def _execute_cluster_analysis_workflow(
        adata: AnnData,
        key: str,
        prefix: str,
        markers: Any,
        sheet: Any,
        addr: str,
        plot_func,
        do_deg: bool,
        ds: bool | int | float,
        raw: bool,
        **kwargs,
) -> None:
    """执行单个分组键下的可视化与 DEG 分析流程。

    Args:
        adata: 输入的 AnnData 对象。
        key: 当前分析使用的 `adata.obs` 分组列名。
        prefix: 输出文件名前缀。
        markers: marker 基因定义。
        sheet: marker 注释信息。
        addr: 输出目录。
        plot_func: UMAP 绘图包装函数。
        do_deg: 是否执行 DEG 分析。
        ds: DEG 分析使用的下采样参数。
        raw: 是否优先使用 `adata.raw`。
        **kwargs: 透传给绘图函数的额外参数。
    """
    if key not in adata.obs.columns:
        raise KeyError(f"`key`: '{key}' does not exist in `adata.obs`.")

    logger.info(f"[_execute_cluster_analysis_workflow] Processing group key: '{key}'.")

    try:
        plot_func(
            adata=adata,
            save_addr=addr,
            filename=f"{prefix}_{key}_UMAP",
            color=key,
            legend_loc="right margin",
            use_raw=raw,
            **kwargs,
        )
    except Exception as exc:
        logger.info(f"[_execute_cluster_analysis_workflow] Warning! UMAP plotting failed for `key`: '{key}'. Details: {exc}")

    try:
        # 为输出文件名补充分组键，便于后续追踪不同 cell subtype/subpopulation 结果。
        geneset_dotplot(
            adata=adata,
            markers=markers,
            marker_sheet=sheet,
            save_addr=addr,
            filename_prefix=f"{prefix}_{key}_MarkerDotplot",
            groupby_key=key,
            use_raw=raw,
            **kwargs,
        )
    except Exception as exc:
        logger.info(
            f"[_execute_cluster_analysis_workflow] Warning! Marker dotplot generation failed for "
            f"`key`: '{key}'. Details: {exc}"
        )

    if do_deg:
        logger.info(f"[_execute_cluster_analysis_workflow] Starting DEG analysis for `key`: '{key}'.")
        easy_DEG(
            adata,
            save_addr=addr,
            filename_prefix=f"{prefix}_{key}",
            obs_key=key,
            save_plot=True,
            plot_gene_num=5,
            downsample=ds,
            use_raw=raw,
        )
    else:
        logger.info(f"[_execute_cluster_analysis_workflow] Warning! DEG analysis is disabled for `key`: '{key}'.")
