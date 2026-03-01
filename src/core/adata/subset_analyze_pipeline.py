import pandas as pd
import anndata
import scanpy as sc
import os,gc

from src.core.adata.deg import easy_DEG
from src.core.adata.ops import subcluster

from src.core.plot.umap import process_resolution_umaps,plot_QC_umap
from src.core.plot.dotplot import geneset_dotplot

from src.core.handlers.plot_wrapper import ScanpyPlotWrapper

import logging
from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)



@logged
def adata_subset_analyze_pipeline(
        adata_subset,
        filename_prefix,
        my_markers,
        marker_sheet,
        save_addr,
        do_subcluster=True,
        do_DEG_enrich=True,
        downsample=False,
        DEG_enrich_key="leiden_res",
        resolutions_list=None,
        use_rep="X_scVI",
        use_raw=True,
        **kwargs
):
    """
    针对 adata 子集的自动化处理流水线：
    包含：子聚类(Subclustering) -> 多维度可视化 -> 差异表达(DEG) -> 标记物表征。
    """
    os.makedirs(save_addr, exist_ok=True)
    n_cells = adata_subset.n_obs
    logger.info(f"--- [START] Processing Subset: {filename_prefix} ({n_cells} cells) ---")
    
    umap_plot = ScanpyPlotWrapper(func=sc.pl.umap)
    
    # ==== 1. 子聚类模块 (可选) ====
    if do_subcluster:
        logger.info(f"[STAGE 1] Subclustering with {use_rep} | Resolutions: {resolutions_list}")
        adata_subset = subcluster(
            adata_subset,
            n_neighbors=20,
            n_pcs=min(adata_subset.obsm[use_rep].shape[1], 50),
            resolutions=resolutions_list,
            use_rep=use_rep
        )
    
    # ==== 2. 核心分析模块 (分发逻辑) ====
    # 根据用户指定的 key 进行迭代或单次处理
    if DEG_enrich_key == "leiden_res":
        if not resolutions_list:
            raise ValueError("resolutions_list is required when DEG_enrich_key='leiden_res'.")
        
        logger.info("[STAGE 2] Running multi-resolution analysis for Leiden clusters")
        # 预处理：分辨率对比图和基础 QC
        process_resolution_umaps(adata_subset, save_addr, resolutions_list, use_raw=use_raw, **kwargs)
        plot_QC_umap(adata_subset, save_addr, filename_prefix=filename_prefix)
        
        for res in resolutions_list:
            groupby_key = f"leiden_res{res}"
            _execute_cluster_analysis_workflow(
                adata_subset, groupby_key, filename_prefix, my_markers, marker_sheet,
                save_addr, umap_plot, do_DEG_enrich, downsample, use_raw, **kwargs
            )
    
    elif DEG_enrich_key in adata_subset.obs.columns:
        logger.info(f"[STAGE 2] Running single-key analysis for '{DEG_enrich_key}'")
        _execute_cluster_analysis_workflow(
            adata_subset, DEG_enrich_key, filename_prefix, my_markers, marker_sheet,
            save_addr, umap_plot, do_DEG_enrich, downsample, use_raw, **kwargs
        )
        plot_QC_umap(adata_subset, save_addr, filename_prefix=filename_prefix)
    else:
        raise KeyError(f"Key '{DEG_enrich_key}' not found in adata.obs.")
    
    logger.info(f"--- [FINISH] All tasks for {filename_prefix} completed. ---")
    return adata_subset
    
    # ------------------------------------------------------------------
    # 辅助逻辑：统一绘图与分析出口
    # ------------------------------------------------------------------

@logged
def _execute_cluster_analysis_workflow(
        adata, key, prefix, markers, sheet, addr, plot_func, do_deg, ds, raw, **kwargs
):
    """私有函数：执行特定分组下的绘图、Dotplot 和 DEG 分析"""
    logger.info(f"  > Processing group: {key}")
    
    # 1. UMAP 可视化
    plot_func(
        save_addr=addr,
        filename=f"{prefix}_{key}_UMAP",
        adata=adata,
        color=key,
        legend_loc="right margin",
        use_raw=raw,
        **kwargs
    )
    
    # 2. Marker Dotplot
    # 为文件名增加分组标识，方便识别
    geneset_dotplot(
        adata=adata,
        markers=markers,
        marker_sheet=sheet,
        save_addr=addr,
        filename_prefix=f"{prefix}_{key}_MarkerDotplot",
        groupby_key=key,
        use_raw=raw,
        **kwargs
    )
    
    # 3. 差异表达分析 (DEG)
    if do_deg:
        logger.info(f"    - Calculating DEGs for {key}...")
        easy_DEG(
            adata,
            save_addr=addr,
            filename_prefix=f"{prefix}_{key}",
            obs_key=key,
            save_plot=True,
            plot_gene_num=5,
            downsample=ds,
            use_raw=raw
        )
