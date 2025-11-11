import anndata
import pandas as pd
import numpy as np
import scanpy as sc
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用无GUI的后端

from src.core.utils.plot_wrapper import ScanpyPlotWrapper
from src.core.base_anndata_ops import _elbow_detector
from src.core.base_anndata_vis import _plot_pca_with_cluster_legend

def pca_cluster_process(result_df, save_addr, filename, **kwargs):
    """
    Compute optimal cluster number using Elbow method and perform K-means clustering.

    Parameters
    ----------
    result_df : pandas.DataFrame
        Result of PCA, with columns 'PC1', 'PC2', and 'label'.
    save_addr : str
        Path to save the figure.
    filename : str
        Filename of the figure.
    **kwargs : for plot_pca_with_cluster_legend

    Returns
    -------
    cluster_to_labels : dict
        A dictionary mapping cluster index to a list of cell types.

    Notes
    -----
    The function first computes the optimal cluster number using the Elbow method,
    and then performs K-means clustering with the optimal number of clusters.
    The result is a dictionary mapping cluster index to a list of cell types.
    """
    from sklearn.cluster import KMeans

    # 使用 PC1 和 PC2
    X = result_df[["PC1", "PC2"]]
    max_k = min(10, X.shape[0])
    cluster_seq = [i for i in range(2, max_k + 1)]
    inertia_seq = [KMeans(n_clusters=k, random_state=0).fit(X).inertia_ for k in cluster_seq]

    optimal_cluster = _elbow_detector(cluster_seq, inertia_seq)

    kmeans = KMeans(n_clusters=optimal_cluster, random_state=0)
    result_df['cluster'] = kmeans.fit_predict(X)

    # 整理出一个 cluster: celltype list 的字典
    # Step 1: 去重（保留第一个出现的 label）
    dedup_df = result_df.drop_duplicates(subset='label', keep='first')
    # Step 2: 设置 label 为索引，只保留 cluster 列
    label_cluster_map = dedup_df.set_index('cluster')['label']
    cluster_to_labels = label_cluster_map.groupby(label_cluster_map.index).apply(list).to_dict()

    _plot_pca_with_cluster_legend(result_df, cluster_to_labels, save_addr=save_addr,
                                 filename=filename, **kwargs)

    return cluster_to_labels
