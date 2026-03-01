import anndata
import pandas as pd
import scanpy as sc
import numpy as np
import scipy as sp

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from src.core.plot.pca import _plot_pca, _pca_cluster_process
from src.utils.env_utils import sanitize_filename

import sys
sys.stdout.reconfigure(encoding='utf-8')



import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)

@logged
def _run_pca(logfc_matrix, n_components=2):
    

    # 转置 → 每行是一个“celltype-disease”样本，每列是基因
    df_T = logfc_matrix.T  # shape: [samples x genes]

    # 标准化（按列，即基因）
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df_T)

    # PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)

    # 构造结果 dataframe
    result_df = pd.DataFrame(
        pca_result,
        columns=[f"PC{i + 1}" for i in range(n_components)],
        index=df_T.index  # 每个index是 like "UC_T Cell_NK.CD16+"
    )
    result_df = result_df.copy()
    result_df['label'] = result_df.index
    result_df = result_df.reset_index(drop=True)

    # 解析 label 中的疾病 & 细胞类型（可根据你的格式微调）
    result_df['group'] = result_df['label'].apply(lambda x: '_'.join(x.split('_')[:-2]))
    result_df['cell_type'] = result_df['label'].apply(lambda x: '_'.join(x.split('_')[-2:]))

    return result_df, pca

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
    from src.core.adata.ops import remap_obs_clusters
    from src.core.adata.deg import easy_DEG
    
    
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