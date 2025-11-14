import pandas as pd
import anndata
import scanpy as sc
import os,gc

# from src.utils.env_utils import count_element_list_occurrence
from src.utils.env_utils import sanitize_filename
from src.core.base_anndata_ops import easy_DEG, remap_obs_clusters, _run_pca, subcluster
from src.core.base_anndata_vis import _pca_cluster_process, process_resolution_umaps, geneset_dotplot, plot_QC_umap,_plot_pca
from src.core.utils.plot_wrapper import ScanpyPlotWrapper

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)

@logged
def generate_subclusters_by_identity(
        adata: anndata.AnnData,
        identity_key: str = "Subset_Identity",
        cell_idents_list: list = None,
        resolutions: list = None,
        output_dir: str = ".",
        use_rep: str = "X_scVI",
        subcluster_func=None,
        n_neighbors: int = 20,
        filename_prefix: str = None
):
    """
    对指定的细胞群体进行子聚类分析并保存为独立文件。
    和 run_deg_on_subsets 是兄弟函数。

    Parameters:
        adata: AnnData
            原始 AnnData 数据对象。
        identity_key: str
            用于选择子集的 obs 列名，默认 "Subset_Identity"。
        cell_idents_list: list
            需要处理的细胞身份列表，默认使用该列中的所有唯一值。
        resolutions: list
            聚类分辨率列表，例如 [0.5, 1.0]。
        output_dir: str
            子集 h5ad 文件的保存目录。
        use_rep: str
            用于聚类的表示空间（例如 "X_scVI"）。
        subcluster_func: callable
            聚类函数，例如 subcluster(adata_subset, ...)，必须传入。
        n_neighbors: int
            聚类时使用的邻居数。
        filename_prefix: str
            输出文件名前缀。
    """
    assert subcluster_func is not None, "请传入 subcluster 函数作为参数 subcluster_func"
    os.makedirs(output_dir, exist_ok=True)
    
    if filename_prefix is None:
        filename_prefix = "SubsetSplit_"
    
    if cell_idents_list is None:
        cell_idents_list = adata.obs[identity_key].unique()
    if resolutions is None:
        resolutions = [0.5, 1.0]

    for ident in cell_idents_list:
        logger.info(f"Now processing subset: {ident}")
        adata_subset = adata[adata.obs[identity_key] == ident].copy()

        # 删除 leiden_res 相关列（obs）
        leiden_cols = [col for col in adata_subset.obs.columns if 'leiden_res' in col]
        if leiden_cols:
            adata_subset.obs.drop(columns=leiden_cols, inplace=True)

        # 删除 leiden_res 相关项（uns）
        leiden_keys = [key for key in adata_subset.uns.keys() if 'leiden_res' in key]
        for key in leiden_keys:
            del adata_subset.uns[key]

        # 子聚类
        adata_subset = subcluster_func(
            adata_subset,
            n_neighbors=n_neighbors,
            n_pcs=min(adata.obsm[use_rep].shape[1], 50),
            resolutions=resolutions,
            use_rep=use_rep
        )

        # 保存
        filename = os.path.join(output_dir, f"{filename_prefix}_{ident}.h5ad")
        adata_subset.write(filename)
        logger.info(f"Saved to {filename}")

        # 清理内存
        del adata_subset
        gc.collect()

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
    
    if resolutions is None:
        resolutions = [0.5,1.0]
    
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

@logged
def _pca_process(merged_df, save_addr, filename_prefix, figsize=(12, 10)):

    if merged_df.columns.duplicated().any():
        logger.info("Warning: There are duplicated column names!")
        # 可加前缀防止冲突，例如按df编号
        df_list_renamed = [
            df.add_prefix(f"df{i}_") for i, df in enumerate(merged_df)
        ]
        merged_df = pd.concat(df_list_renamed, axis=1)

    result_df, pca = _run_pca(merged_df, n_components=3)
    explained_var = pca.explained_variance_ratio_
    logger.info(f"PC1 explains {explained_var[0]:.2%} of variance")
    logger.info(f"PC2 explains {explained_var[1]:.2%} of variance")
    logger.info(f"PC3 explains {explained_var[2]:.2%} of variance")

    _plot_pca(result_df, pca,
              save_addr=save_addr, filename_prefix=filename_prefix, figsize=figsize,
              color_by='cell_type')
    return result_df, pca

@logged
def run_pca_and_deg_for_celltype(celltype, merged_df_filtered, adata, save_addr,
                                 figsize=(12, 10),
                                 file_prefix="20251110"):
    '''
    对每个/每组细胞亚群按照分组信进行拆分后，进行 PCA 聚类，观察其模式

    :param celltype: list or tuple or str
    :param merged_df_filtered:
    :param adata:
    :param save_addr:
    :param figsize:
    :param file_prefix: 探索性任务推荐用时间批次进行文件管理
    :return:
    '''
    if isinstance(celltype, (list, tuple)):
        logger.info(f"Processing multiple celltypes.")
        column_mask = [col for col in merged_df_filtered.columns if col.split("_")[-2] in celltype]
        celltype_use_as_name = "-".join(celltype)
    else:
        logger.info(f"Processing {celltype}")
        column_mask = [col for col in merged_df_filtered.columns if col.split("_")[-2] == celltype]
        celltype_use_as_name = celltype

    celltype_use_as_name = celltype_use_as_name.replace(" ", "-")
    celltype_use_as_name = sanitize_filename(celltype_use_as_name)

    if not column_mask:
        logger.info(f"No columns found for {celltype}")
        return None

    df_split = merged_df_filtered.loc[:, column_mask]
    result_df, pca = _pca_process(df_split,
                                  save_addr=save_addr,
                                  filename_prefix=f"{file_prefix}({celltype_use_as_name})",
                                  figsize=figsize)

    cluster_to_labels = _pca_cluster_process(result_df,
                                             save_addr=save_addr,
                                             filename=f"{file_prefix}({celltype_use_as_name})",
                                             figsize=figsize)

    if not cluster_to_labels:
        logger.info(f"{celltype} cannot be clustered, skipped.")
        return None

    # 进行多对一的映射
    adata_combined = remap_obs_clusters(adata, mapping=cluster_to_labels,
                                        obs_key="tmp", new_key="cluster")

    easy_DEG(
        adata_combined,
        save_addr=save_addr,
        filename_prefix=f"{file_prefix}_{celltype_use_as_name})",
        obs_key="cluster",
        save_plot=True,
        plot_gene_num=10,
        downsample=5000,
        use_raw=True
    )

@logged
def process_adata(
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
    主流程：处理子集 adata，对其进行子聚类、DEG富集、绘图。
    """
    os.makedirs(save_addr, exist_ok=True)

    umap_plot = ScanpyPlotWrapper(func=sc.pl.umap)

    # ==== 1. 可选：降维聚类 ====
    if do_subcluster:
        logger.info("Starting subclustering...")
        adata_subset = subcluster(
            adata_subset,
            n_neighbors=20,
            n_pcs=min(adata_subset.obsm[use_rep].shape[1], 50),
            resolutions=resolutions_list,
            use_rep=use_rep
        )
        logger.info("Subclustering completed.")

    # ==== 2.1 使用 leiden_res 作为分组方式；如果省略第一步则依赖原有adata.obs中的列，需要确保`resolutions_list`能对应实际存在的列 ====
    if DEG_enrich_key == "leiden_res":
        if not resolutions_list:
            raise ValueError("[process_adata] resolutions_list cannot be empty when using 'leiden_res' as DEG enrichment key.")
        if not all(isinstance(res, (int, float)) for res in resolutions_list):
            raise TypeError("[process_adata] All elements in resolutions_list must be integers or floats.")


        # 2.1.1 分辨率比较图，和基础 QC 图
        process_resolution_umaps(adata_subset, save_addr, resolutions_list, use_raw=use_raw, **kwargs)

        # 自动识别 QC 关键字
        plot_QC_umap(adata_subset,save_addr,filename_prefix=filename_prefix )

        # 2.1.2 每个分辨率进行绘图 + DEG
        for res in resolutions_list:
            groupby_key = f"leiden_res{res}"

            logger.info(f"Creating UMAP plot for key '{groupby_key}'...")
            umap_plot(
                save_addr=save_addr, filename=f"{filename_prefix}_{groupby_key}",
                adata=adata_subset,
                color=groupby_key,
                legend_loc="right margin",
                use_raw=use_raw,
                **kwargs
            )

            logger.info(f"Drawing gene marker dotplot for key '{groupby_key}'...")
            geneset_dotplot(
                adata=adata_subset,
                markers=my_markers,
                marker_sheet=marker_sheet,
                output_dir=save_addr,
                filename_prefix=f"{filename_prefix}_Geneset({marker_sheet})",
                groupby_key=groupby_key,
                use_raw=use_raw,
                **kwargs
            )

            if do_DEG_enrich:
                logger.info(f"Running DEG enrichment for '{groupby_key}'...")
                easy_DEG(adata_subset,
                         save_addr=save_addr, filename_prefix=filename_prefix,
                         obs_key=groupby_key,
                         save_plot=True, plot_gene_num=5, downsample=downsample, use_raw=use_raw)


    # ==== 2.2 其他 obs 中的分组变量 ====
    elif DEG_enrich_key in adata_subset.obs.columns:
        logger.info(f"Creating UMAP plot for key '{DEG_enrich_key}'...")
        umap_plot(
            save_addr=save_addr,filename=f"{filename_prefix}_{DEG_enrich_key}",
            adata=adata_subset,
            color=DEG_enrich_key,
            legend_loc="right margin",
            use_raw=use_raw,
            **kwargs
        )

        logger.info(f"Drawing gene marker dotplot for key '{DEG_enrich_key}'...")
        geneset_dotplot(
            adata=adata_subset,
            markers=my_markers,
            marker_sheet=marker_sheet,
            save_addr=save_addr,
            filename_prefix=filename_prefix,
            groupby_key=DEG_enrich_key,
            use_raw=use_raw,
            **kwargs
        )

        if do_DEG_enrich:
            logger.info(f"Running DEG enrichment for '{DEG_enrich_key}'...")
            easy_DEG(adata_subset,
                     save_addr=save_addr, filename_prefix=filename_prefix,
                     obs_key=DEG_enrich_key,
                     save_plot=True, plot_gene_num=5, downsample=downsample, use_raw=use_raw)
            plot_QC_umap(adata_subset, save_addr, filename_prefix=filename_prefix)

    else:
        raise ValueError("Please recheck the `DEG_enrich_key`.")
