import pandas as pd
import anndata
import os,gc

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


def analysis_DEG(adata_subset, file_name, groupby_key, output_dir,downsample,use_raw,skip_QC=False):
    from src.core.base_anndata_ops import easy_DEG
    # from
    print(f"--> Starting differential expression analysis for group '{groupby_key}'...")
    easy_DEG(adata_subset, save_addr=output_dir, filename=file_name, obs_key=groupby_key,
             save_plot=True, plot_gene_num=5, downsample=downsample,use_raw=use_raw)
    # 基础QC图
    Basic_QC_Plot(
        adata_subset,
        prefixx=f"{file_name}_{groupby_key}",
        out_dir=output_dir
    )

from src.utils.env_utils import count_element_list_occurrence
from src.core.base_anndata_ops import easy_DEG, remap_obs_clusters


def run_pca_and_deg_for_celltype(celltype, merged_df_filtered, adata, save_dir,
                                 figsize=(12, 10),
                                 pca_fig_prefix="among_disease", DEG_file_suffix="by_PCA_cluster"):
    if isinstance(celltype, (list, tuple)):
        print(f"Processing multiple celltypes.")
        column_mask = [col for col in merged_df_filtered.columns if col.split("_")[-2] in celltype]
        celltype_use_as_name = "-".join(celltype)
    else:
        print(f"Processing {celltype}")
        column_mask = [col for col in merged_df_filtered.columns if col.split("_")[-2] == celltype]
        celltype_use_as_name = celltype

    celltype_use_as_name = celltype_use_as_name.replace(" ", "-")

    if not column_mask:
        print(f"No columns found for {celltype}")
        return None

    df_split = merged_df_filtered.loc[:, column_mask]
    result_df, pca = pca_process(df_split, save_dir, figname=f"{pca_fig_prefix}({celltype_use_as_name})",
                                 figsize=figsize)
    cluster_to_labels = pca_cluster_process(result_df, save_dir,
                                            figname=f"{pca_fig_prefix}({celltype_use_as_name})", figsize=figsize)

    if not cluster_to_labels:
        print(f"!{celltype} cannot be clustered, skipped.")
        return None

    print(cluster_to_labels)
    adata_combined = remap_obs_clusters(adata, cluster_to_labels)

    easy_DEG(
        adata_combined,
        save_addr=save_dir,
        filename=f"{pca_fig_prefix}_{celltype_use_as_name}({DEG_file_suffix})",
        obs_key="cluster",
        save_plot=True,
        plot_gene_num=10,
        downsample=5000,
        use_raw=True
    )
