import pandas as pd
import anndata
import scanpy as sc
import re
import os,gc

# from src.utils.env_utils import count_element_list_occurrence
from src.core.base_anndata_ops import easy_DEG, remap_obs_clusters, sanitize_filename, _run_pca, subcluster
from src.core.base_anndata_vis import _pca_cluster_process, process_resolution_umaps, geneset_dotplot, plot_QC_umap
from src.core.utils.plot_wrapper import ScanpyPlotWrapper
def generate_subclusters_by_identity(
        adata: anndata.AnnData,
        identity_key: str = "Subset_Identity",
        cell_idents_list: list = None,
        resolutions: list = None,
        output_dir: str = ".",
        use_rep: str = "X_scVI",
        subcluster_func=None,
        n_neighbors: int = 20,
        filename_prefix: str = "Step06_Subset"
):
    """
    对指定的细胞群体进行子聚类分析并保存为独立文件。

    Parameters:
        adata: AnnData
            原始 AnnData 数据对象。
        identity_key: str
            用于选择子集的 obs 列名，默认 "Subset_Identity"。
        identities: list
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
    if cell_idents_list is None:
        cell_idents_list = adata.obs[identity_key].unique()
    if resolutions is None:
        resolutions = [0.5, 1.0]

    for ident in cell_idents_list:
        print(f"\n🔍 Now processing subset: {ident}")
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
        print(f"💾 Saved to {filename}")

        # 清理内存
        del adata_subset
        gc.collect()


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
        print(f"[split_and_DEG] Processing subset: {subset}")

        save_dir = f"{output_dir}/_{subset}"
        print(f"[split_and_DEG] Creating output directory: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)  # 避免目录已存在时报错

        print(f"[split_and_DEG] Subsetting data for: {subset}")
        adata_subset = adata[adata.obs[obs_key] == subset]

        # 筛选掉计数小于 30 的疾病亚群；目的是其存在影响在后续 PCA 聚类中对其意义进行挖掘，而且可能存在较大的偏倚
        value_count_df = adata_subset.obs[groupby_key].value_counts()
        disease_accountable = value_count_df.index[value_count_df >= count_thr]
        print(f"Disease group cell counts in {subset}:\n{value_count_df}")

        adata_subset = adata_subset[adata_subset.obs[groupby_key].isin(disease_accountable)]

        print(f"[split_and_DEG] Running easy_DEG for: {subset}")
        if adata_subset.n_obs < (2*count_thr):
            print(f"[split_and_DEG] Skipped DEG for {subset}: too few cells after filtering.")
            continue
        else:
            easy_DEG(
                adata_subset,
                save_addr=save_dir,
                filename=f"{subset}",
                obs_key=groupby_key,
                save_plot=True,
                plot_gene_num=10,
                downsample=downsample,
                use_raw=True
            )

        print(f"[split_and_DEG] Completed DEG analysis for: {subset}\n")
        write_path = f"{save_dir}/Subset_by_disease.h5ad"
        adata_subset.write(write_path)
        del adata_subset
        gc.collect()


def _pca_process(merged_df, save_addr, filename_prefix, figsize=(12, 10)):

    if merged_df.columns.duplicated().any():
        print("[pca_process] Warning: There are duplicated column names!")
        # 可加前缀防止冲突，例如按df编号
        df_list_renamed = [
            df.add_prefix(f"df{i}_") for i, df in enumerate(merged_df)
        ]
        merged_df = pd.concat(df_list_renamed, axis=1)

    result_df, pca = _run_pca(merged_df, n_components=3)
    explained_var = pca.explained_variance_ratio_
    print(f"[pca_process] PC1 explains {explained_var[0]:.2%} of variance")
    print(f"[pca_process] PC2 explains {explained_var[1]:.2%} of variance")
    print(f"[pca_process] PC3 explains {explained_var[2]:.2%} of variance")

    _plot_pca(result_df, pca,
              save_addr=save_addr, filename_prefix=filename_prefix, figsize=figsize,
              color_by='cell_type')
    return result_df, pca

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
        print(f"[run_pca_and_deg_for_celltype] Processing multiple celltypes.")
        column_mask = [col for col in merged_df_filtered.columns if col.split("_")[-2] in celltype]
        celltype_use_as_name = "-".join(celltype)
    else:
        print(f"[run_pca_and_deg_for_celltype] Processing {celltype}")
        column_mask = [col for col in merged_df_filtered.columns if col.split("_")[-2] == celltype]
        celltype_use_as_name = celltype

    celltype_use_as_name = celltype_use_as_name.replace(" ", "-")
    celltype_use_as_name = sanitize_filename(celltype_use_as_name)

    if not column_mask:
        print(f"[run_pca_and_deg_for_celltype] No columns found for {celltype}")
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
        print(f"[run_pca_and_deg_for_celltype] {celltype} cannot be clustered, skipped.")
        return None

    # 进行多对一的映射
    adata_combined = remap_obs_clusters(adata, mapping=cluster_to_labels,
                                        obs_key="tmp", new_key="cluster")

    easy_DEG(
        adata_combined,
        save_addr=save_addr,
        filename=f"{file_prefix}_{celltype_use_as_name})",
        obs_key="cluster",
        save_plot=True,
        plot_gene_num=10,
        downsample=5000,
        use_raw=True
    )


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
        print("[process_adata] Starting subclustering...")
        adata_subset = subcluster(
            adata_subset,
            n_neighbors=20,
            n_pcs=min(adata_subset.obsm[use_rep].shape[1], 50),
            resolutions=resolutions_list,
            use_rep=use_rep
        )
        print("[process_adata] Subclustering completed.")

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

            print(f"[process_adata] Creating UMAP plot for key '{groupby_key}'...")
            umap_plot(
                save_addr=save_addr, filename=f"{filename_prefix}_{groupby_key}",
                adata=adata_subset,
                color=groupby_key,
                legend_loc="right margin",
                use_raw=use_raw,
                **kwargs
            )

            print(f"[process_adata] Drawing gene marker dotplot for key '{groupby_key}'...")
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
                print(f"[process_adata] Running DEG enrichment for '{groupby_key}'...")
                easy_DEG(adata_subset,
                         save_addr=save_addr, filename_prefix=filename_prefix,
                         obs_key=groupby_key,
                         save_plot=True, plot_gene_num=5, downsample=downsample, use_raw=use_raw)


    # ==== 2.2 其他 obs 中的分组变量 ====
    elif DEG_enrich_key in adata_subset.obs.columns:
        print(f"[process_adata] Creating UMAP plot for key '{DEG_enrich_key}'...")
        umap_plot(
            save_addr=save_addr,filename=f"{filename_prefix}_{DEG_enrich_key}",
            adata=adata_subset,
            color=DEG_enrich_key,
            legend_loc="right margin",
            use_raw=use_raw,
            **kwargs
        )

        print(f"[process_adata] Drawing gene marker dotplot for key '{DEG_enrich_key}'...")
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
            print(f"[process_adata] Running DEG enrichment for '{DEG_enrich_key}'...")
            easy_DEG(adata_subset,
                     save_addr=save_addr, filename_prefix=filename_prefix,
                     obs_key=DEG_enrich_key,
                     save_plot=True, plot_gene_num=5, downsample=downsample, use_raw=use_raw)
            plot_QC_umap(adata_subset, save_addr, filename_prefix=filename_prefix)

    else:
        raise ValueError("[process_adata] Please recheck the `DEG_enrich_key`.")
