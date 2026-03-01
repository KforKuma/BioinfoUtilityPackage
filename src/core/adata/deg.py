import anndata

import pandas as pd
import scanpy as sc
import numpy as np
import scipy as sp

import os, gc, sys

sys.stdout.reconfigure(encoding='utf-8')

from src.core.adata.ops import obs_keywise_downsample

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)

@logged
def easy_DEG(
        adata, save_addr, filename_prefix,
        obs_key="Subset_Identity",
        save_plot=True, plot_gene_num=5, downsample=False,
        method='wilcoxon', use_raw=False,
        min_cells=3,  # <<< 新增参数，可调
):
    """
    快速进行差异基因富集（DEG）
    自动处理以下情况：
        - 某些 group 的细胞数为 0（不会出现，但稳妥）
        - 某些 group 细胞数过小（< min_cells）
        - group 数不足 2 → 打印 warning，并跳过 DEG
    """
    
    if use_raw and adata.raw is None:
        logger.info("Warning: use_raw=True, but .raw not found. Using .X instead.")
    
    # --- 检查 obs_key 是否是 category ---
    if not pd.api.types.is_categorical_dtype(adata.obs[obs_key]):
        adata.obs[obs_key] = adata.obs[obs_key].astype("category")
    
    # --- 统计每个 group 细胞数 ---
    vc = adata.obs[obs_key].value_counts()
    logger.info(f"[easy_DEG] Initial group sizes for '{obs_key}':\n{vc.to_dict()}")
    
    # --- 去掉 extremely small group ---
    small_groups = vc[vc < min_cells].index.tolist()
    if len(small_groups) > 0:
        logger.info(f"[easy_DEG] Removing small groups (<{min_cells} cells): {small_groups}")
        adata = adata[~adata.obs[obs_key].isin(small_groups)].copy()
    
    # --- 重新统计剩余 group ---
    vc2 = adata.obs[obs_key].value_counts()
    logger.info(f"[easy_DEG] Remaining group sizes: {vc2.to_dict()}")
    
    # --- 检查 group 是否 >=2 ---
    if vc2.size < 2:
        logger.info(f"[easy_DEG] Only one group left after filtering. "
                    f"DEG skipped. Remaining categories: {vc2.to_dict()}")
        return adata
    
    # --- 正常执行下游 DE 流程 ---
    deg_key = "deg_" + obs_key
    save_addr = save_addr if save_addr.endswith("/") else save_addr + "/"
    os.makedirs(save_addr, exist_ok=True)
    
    if isinstance(downsample, (float, int)) and downsample > 0:
        logger.info(f"Downsampling enabled: {downsample}")
        adata = obs_keywise_downsample(adata, obs_key, downsample)
    else:
        logger.info("No downsampling performed.")
    
    logger.info(f"Starting DEG ranking for '{obs_key}'...")
    sc.tl.rank_genes_groups(
        adata, groupby=obs_key,
        use_raw=use_raw, method=method, key_added=deg_key
    )
    
    filename_prefix = f"{filename_prefix}_" if filename_prefix is not None else filename_prefix
    
    # --- 绘图部分 ---
    if save_plot:
        try:
            from src.core.handlers.plot_wrapper import ScanpyPlotWrapper
            rank_genes_groups_dotplot = ScanpyPlotWrapper(sc.pl.rank_genes_groups_dotplot)
            rank_genes_groups_dotplot(
                save_addr=save_addr,
                filename=f"{filename_prefix}{obs_key}_HVG_Dotplot",
                adata=adata,
                groupby=obs_key,
                key=deg_key,
                standard_scale="var",
                n_genes=plot_gene_num,
                dendrogram=False,
                use_raw=use_raw,
                show=False
            )
            logger.info("Dotplot saved successfully.")
        except Exception as e:
            logger.info(f"Dotplot generation failed: {e}")
    
    # --- 整理 DataFrame ---
    groups = adata.uns[deg_key]['names'].dtype.names
    df_all = pd.concat([
        sc.get.rank_genes_groups_df(adata, group=grp, key=deg_key).assign(cluster=grp)
        for grp in groups
    ])
    
    # 排序1
    df_sorted_logfc = df_all.sort_values(
        by=['names', 'logfoldchanges'], ascending=[True, False]
    )
    
    # 排序2
    df_sorted_pval = (
        df_all[df_all["scores"] > 0]  # 过滤
        .sort_values(
            by=["cluster", "pvals_adj"],
            ascending=[True, True]
        )
    )
    
    # 保存 Excel
    excel_path = os.path.join(save_addr, f"{filename_prefix}{obs_key}_HVG.xlsx")
    try:
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            df_sorted_logfc.to_excel(writer, sheet_name='Sorted_by_logFC', index=False)
            df_sorted_pval.to_excel(writer, sheet_name='Sorted_by_pval', index=False)
            logger.info("Excel file saved successfully.")
    except Exception as e:
        logger.info(f"Error saving Excel file: {e}")
    
    logger.info("DEG completed successfully.")
    return adata


@logged
def run_deg_on_subsets(
        cell_idents_list: list = None,
        resolutions: list = None,
        base_input_path: str = ".",
        base_output_path: str = ".",
        deg_method: str = "wilcoxon",
        save_plot: bool = True,
        plot_gene_num: int = 5,
        use_raw: bool = True,
        downsample: bool = False,
        save_prefix: str = None,
        output_suffix: str = None,
        easy_deg_func=None,
):
    """
    对 AnnData 中不同子集执行 DEG 分析（基于 leiden 聚类）。
    和 generate_subclusters_by_identity 是兄弟函数。

    Parameters:
        resolutions: list
            需要运行 DEG 的 leiden 分辨率列表。
        base_input_path: str
            子集 h5ad 文件的基础目录。
        base_output_path: str
            输出 DEG 文件的基础目录。
        deg_method: str
            使用的 DEG 方法，例如 "wilcoxon"。
        save_plot: bool
            是否保存 DEG 的 marker gene 图。
        plot_gene_num: int
            每个 cluster 显示的 top marker gene 数量。
        use_raw: bool
            是否使用原始表达值进行 DEG。
        downsample: bool
            计算 DEG 时，是否进行下采样处理；不影响 adata 数据的保存
        save_prefix: str
            输入输出文件名的前缀部分。
        output_suffix: str
            输出 DEG 后缀，例如 "_DEG.h5ad"。
        easy_deg_func: callable
            你自己的 easy_DEG 函数，必须作为参数传入。
    """
    
    assert easy_deg_func is not None, "请传入 easy_DEG 函数作为参数 easy_deg_func"
    
    if save_prefix is None:
        save_prefix = "SubsetSplit_"
    
    if output_suffix is None:
        output_suffix = "_DEG"
    
    if not output_suffix.endswith(".h5ad"):
        output_suffix = f"{output_suffix}.h5ad"
    
    if resolutions is None:
        resolutions = [0.5, 1.0]
    
    for cell_ident in cell_idents_list:
        logger.info(f"Now processing subset: {cell_ident}")
        
        input_file = os.path.join(base_input_path, f"{save_prefix}_{cell_ident}.h5ad")
        logger.info(f"Loading file: {input_file}")
        adata_subset = anndata.read_h5ad(input_file)
        
        for res in resolutions:
            group_key = f"leiden_res{res}"
            logger.info(f"Running easy_DEG for resolution {res} with group key '{group_key}'...")
            os.makedirs(base_output_path, exist_ok=True)
            adata_subset = easy_deg_func(
                adata_subset,
                save_addr=base_output_path,
                filename_prefix=f"Secondary_Cluster_{cell_ident}(For clean up)",
                obs_key=group_key,
                save_plot=save_plot,
                plot_gene_num=plot_gene_num,
                downsample=downsample,
                method=deg_method,
                use_raw=use_raw
            )
            logger.info(f"Finished DEG at resolution {res}.")
        
        output_file = os.path.join(base_output_path, f"{save_prefix}_{cell_ident}{output_suffix}")
        adata_subset.write_h5ad(output_file)
        logger.info(f"Saved DEG results to: {output_file}")
        
@logged
def split_and_DEG(adata, subset_list, obs_key, groupby_key, output_dir, count_thr=30, downsample=5000):
    '''
    【探索】 对每个亚群进行分组拆分，观察其 DEG

    Example
    -------
    celllist = adata.obs["Subset_Identity"].unique().tolist()
    split_and_DEG(subset_list=celllist,subset_key="Subset_Identity", split_by_key="disease", output_dir=output_dir)

    :param adata:
    :param subset_list:
    :param obs_key:
    :param groupby_key:
    :param output_dir:
    :param count_thr:
    :param downsample:
    :return:
    '''
    for subset in subset_list:
        logger.info(f"Processing subset: {subset}")

        save_dir = f"{output_dir}/_{subset}"
        logger.info(f"Creating output directory: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)  # 避免目录已存在时报错

        logger.info(f"Subsetting data for: {subset}")
        adata_subset = adata[adata.obs[obs_key] == subset]

        # 筛选掉计数小于 30 的疾病亚群；目的是其存在影响在后续 PCA 聚类中对其意义进行挖掘，而且可能存在较大的偏倚
        value_count_df = adata_subset.obs[groupby_key].value_counts()
        disease_accountable = value_count_df.index[value_count_df >= count_thr]
        logger.info(f"Disease group cell counts in {subset}:\n{value_count_df}")

        adata_subset = adata_subset[adata_subset.obs[groupby_key].isin(disease_accountable)]

        logger.info(f"Running easy_DEG for: {subset}")
        if adata_subset.n_obs < (2*count_thr):
            logger.info(f"Skipped DEG for {subset}: too few cells after filtering.")
            continue
        else:
            easy_DEG(
                adata_subset,
                save_addr=save_dir,
                filename_prefix=f"{subset}",
                obs_key=groupby_key,
                save_plot=True,
                plot_gene_num=10,
                downsample=downsample,
                use_raw=True
            )

        logger.info(f"Completed DEG analysis for: {subset}\n")
        write_path = f"{save_dir}/Subset_by_disease.h5ad"
        adata_subset.write(write_path)
        del adata_subset
        gc.collect()


