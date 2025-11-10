import anndata
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import scanpy as sc
import numpy as np
import scipy as sp

import gc, os
import re
import time
import sys
sys.stdout.reconfigure(encoding='utf-8')
matplotlib.use('Agg')  # 使用无GUI的后端

from src.utils.env_utils import ensure_package
from src.core.base_anndata_vis import _plot_pca

def subcluster(adata, n_neighbors=20, n_pcs=50, skip_DR=False, resolutions=None, use_rep="X_scVI"):
    """
    执行Scanpy的降维与Leiden聚类流程，可选多分辨率。

    Parameters
    ----------
    adata : AnnData
        输入的 anndata 对象
    n_neighbors : int, optional
        KNN近邻数量
    n_pcs : int, optional
        PCA维度数量
    skip_DR : bool, optional
        是否跳过降维与UMAP
    resolutions : float | list[float], optional
        Leiden分辨率，可为单值或列表
    use_rep : str, optional
        用于计算邻域的表示（默认 'X_scVI'）

    Returns
    -------
    AnnData
        聚类降维后的 anndata 对象
    """

    def _log(msg):
        print(f"[subcluster] {msg}")

    # 处理 use_rep 参数
    if use_rep != "X" and use_rep not in adata.obsm:
        _log(f"[Warning] '{use_rep}' not found in adata.obsm.")
        available_keys = list(adata.obsm.keys())
        if available_keys:
            _log(f"Available options in adata.obsm: {available_keys}")
        else:
            _log("adata.obsm is empty.")
        raise KeyError(f"use_rep='{use_rep}' not found in adata.obsm.")

    # 处理 resolutions
    if resolutions is None:
        resolutions = [1.0, 1.5]
        _log(f"No resolutions provided. Using default: {resolutions}")
    elif isinstance(resolutions, (int, float)):
        resolutions = [resolutions]
    elif not isinstance(resolutions, (list, tuple, np.ndarray)):
        raise ValueError("resolutions must be a float, int, or list of floats.")

    # 降维
    if not skip_DR:
        _log(f"Computing neighbors (n_neighbors={n_neighbors}, n_pcs={n_pcs}, use_rep='{use_rep}')...")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep)
        sc.tl.umap(adata)
        _log("UMAP completed.")
    else:
        _log("Skipping dimensional reduction.")

    # 聚类
    for res in resolutions:
        key = f"leiden_res{res}"
        _log(f"Running Leiden clustering (resolution={res}) -> '{key}'")
        sc.tl.leiden(adata, key_added=key, resolution=res)

    _log("Subclustering process finished.")
    return adata


def obs_keywise_downsample(adata, obs_key, downsample=1000):
    """
    下采样 adata，保证obs_key所限定的列中每一个值（身份）都至少有N个样本（N>1）；
    或对每个值所对应的样本量以N为缩放系数进行缩放（0<N<1)。
    函数曾用名： obs_key_wise_subsampling。


    :param obs_key: 分组键
    :param adata: 需要下采样的anndata文件
    :param downsample: 下采样系数；当N>1时为最大样本数，当0<N<1时为缩放系数
    :return: 返回下采样后的 adata
    """
    counts = adata.obs[obs_key].value_counts()
    indices = []  # 用列表存储选取的样本索引

    def _log(msg):
        print(f"[obs_keywise_downsample] {msg}")

    if downsample > 1 and downsample % 1 == 0:  # 截断模式
        _log(f"mode: cutoff; threshold: {downsample}")
        for group, count in counts.items():
            use_size = min(count, downsample)  # 限制样本数不超过
            selected_indices = np.random.choice(
                adata.obs_names[adata.obs[obs_key] == group],
                size=use_size,
                replace=False
            )
            indices.extend(selected_indices)  # 使用extend避免嵌套列表
            _log(f"{group}: {use_size}/{count} selected (total={len(indices)})")
    elif 0 < downsample < 1:  # 缩放模式
        _log(f"mode: zoom; ratio: {downsample:.2%}.")
        for group, count in counts.items():
            use_size = max(1, round(count * downsample))  # 确保至少选择1个样本
            selected_indices = np.random.choice(
                adata.obs_names[adata.obs[obs_key] == group],
                size=use_size,
                replace=False
            )
            indices.extend(selected_indices)
            _log(f"{group}: {use_size}/{count} selected (total={len(indices)})")
    else:
        raise ValueError("Please recheck parameter `downsample`. It must be >1 (integer) or 0<N<1 (float).")

    _log(f"Final sample size: {len(indices)} (from {len(adata)} cells).")
    return adata[indices].copy()


def easy_DEG(adata, save_addr, filename_prefix,
             obs_key="Subset_Identity",
             save_plot=True, plot_gene_num=5, downsample=False,
             method='wilcoxon', use_raw=False):
    """
    快速进行差异基因富集（DEG）
    """

    def _log(msg):
        print(f"[easy_DEG] {msg}")

    if use_raw and adata.raw is None:
        _log("Warning: use_raw=True, but .raw not found in AnnData. Will fallback to .X.")

    deg_key = "deg_" + obs_key
    save_addr = save_addr if save_addr.endswith("/") else save_addr + "/"
    os.makedirs(save_addr, exist_ok=True)

    if isinstance(downsample, (float, int)) and downsample > 0:
        _log(f"Downsampling enabled: {downsample}")
        adata = obs_keywise_downsample(adata, obs_key, downsample)
    else:
        _log("No downsampling performed.")

    _log(f"Starting DEG ranking for '{obs_key}'...")
    sc.tl.rank_genes_groups(adata, groupby=obs_key, use_raw=use_raw,
                            method=method, key_added=deg_key)

    if save_plot:
        try:
            from src.core.utils.plot_wrapper import ScanpyPlotWrapper
            rank_genes_groups_dotplot = ScanpyPlotWrapper(sc.pl.rank_genes_groups_dotplot)
            rank_genes_groups_dotplot(save_addr=save_addr,filename=f"{filename_prefix}_HVG_dotplot",
                                      adata=adata, groupby=obs_key, key=deg_key, standard_scale="var",
                                      n_genes=plot_gene_num, dendrogram=False, use_raw=use_raw, show=False)
            _log("Dotplot saved successfully.")
        except Exception as e:
            _log(f"Plot generation failed: {e}")

    groups = adata.uns[deg_key]['names'].dtype.names
    # 合并所有 group 的结果
    df_all = pd.concat([
        sc.get.rank_genes_groups_df(adata, group=grp, key=deg_key).assign(cluster=grp)
        for grp in groups
    ])

    # 第一种排序方式：按 logfoldchanges 降序，再按 names 升序
    df_sorted_logfc = df_all.sort_values(by=['names', 'logfoldchanges'], ascending=[True, False])

    # 第二种排序方式：按 pvals_adj 升序，再按 cluster 升序
    df_sorted_pval = df_all.sort_values(by=['cluster', 'pvals_adj'], ascending=[True, True])

    # 保存到 Excel 两个 sheet 中
    abs_csv_path = os.path.join(save_addr,f"{filename_prefix}_HVG.xlsx")
    try:
        with pd.ExcelWriter(abs_csv_path, engine='xlsxwriter') as writer:
            df_sorted_logfc.to_excel(writer, sheet_name='Sorted_by_logFC', index=False)
            df_sorted_pval.to_excel(writer, sheet_name='Sorted_by_pval', index=False)
            _log("Excel file saved successfully.")
    except Exception as e:
        _log(f"Error saving Excel file: {e}")

    _log("Successfully saved.")
    return adata


def score_gene_analysis(marker_dict, adata_subset,
                        downsample=False, plot=False, obs_key=None, save_addr=None):
    """
    对多个基因集进行打分，并返回仅包含打分结果的 AnnData。
    可选绘制 dotplot 图。

    Parameters
    ----------
    marker_dict : dict
        {'GeneSetName1': ['Gene1', 'Gene2', ...], ...}
    adata_subset : AnnData
        待分析数据
    downsample : bool | float | int
        是否下采样；传数值时为下采样参数
    plot : bool
        是否绘制 dotplot
    obs_key : str
        用于分组作图的列名（plot=True,或 downsample 运算时均必需）
    save_addr : str
        作图保存目录

    Returns
    -------
    AnnData
        新 AnnData，仅包含打分结果和必要元数据
    """
    def _log(msg):
        print(f"[score_gene_analysis] {msg}")

    # 参数检查
    if (plot or downsample) and (obs_key is None):
        raise ValueError("obs_key must be provided when using downsample or plot=True")

    # 下采样
    if isinstance(downsample, (float, int)) and downsample > 0:
        _log(f"Downsampling with parameter={downsample}")
        adata_subset = obs_keywise_downsample(adata_subset, obs_key, downsample)

    # Step 1: 计算每个基因集分数
    _log("Scoring gene sets...")
    score_cols = []
    for key, genes in marker_dict.items():
        score_name = f"score_{key}"
        sc.tl.score_genes(adata=adata_subset, gene_list=genes, score_name=score_name, use_raw=False)
        score_cols.append(score_name)

    # Step 2: 仅保留有用的 obs 列
    def auto_col_selector(obs_columns):
        return (
                [x for x in obs_columns if x.startswith(("orig", "disease", "Subset"))] +
                [x for x in obs_columns if x.endswith("Identity")] +
                [x for x in obs_columns if x.startswith('leiden')]
        )

    col_list = auto_col_selector(adata_subset.obs.columns)
    if obs_key is not None and obs_key not in col_list:
        col_list.append(obs_key)

    _log(f"Keeping {len(col_list)} columns in obs.")

    # Step 3: 构建新的 AnnData
    var_df = pd.DataFrame(
        index=score_cols,
        data={'features': [marker_dict.get(k.replace("score_", ""), []) for k in score_cols]}
    )
    adata_score = anndata.AnnData(
        X=adata_subset.obs[score_cols].values,
        obs=adata_subset.obs[col_list],
        var=var_df
    )

    # Step 4: 绘图
    if plot:
        if save_addr is None:
            save_addr = "./fig/"
        os.makedirs(save_addr, exist_ok=True)

        _log("Plotting dotplot...")
        try:
            with plt.rc_context():
                sc.pl.dotplot(adata_subset, groupby=obs_key, var_names=score_cols,
                              standard_scale="var", show=False)
                fname_base = os.path.join(save_addr, f"GeneScore_by_{obs_key}")
                plt.savefig(f"{fname_base}.pdf", bbox_inches="tight", dpi=300)
                plt.savefig(f"{fname_base}.png", bbox_inches="tight", dpi=300)
                _log(f"Saved dotplot to {fname_base}.pdf/png")
        except Exception as e:
            _log(f"Plot failed: {e}")

    _log("Finished scoring.")
    return adata_score


def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def get_cluster_counts(adata, obs_key="Subset_Identity", group_by="orig.project",
                       drop_values=None, drop_axis="index"):
    """
    统计每个 group_by 下各个 obs_key 的细胞数量。

    Examples
    --------
    counts = get_cluster_counts(adata,obs_key="Subset_Identity", group_by="disease")

    """
    counts = (
        adata.obs.groupby([group_by, obs_key])
        .size()
        .unstack(fill_value=0)
    )

    if drop_values is not None:
        counts.drop(drop_values, axis=0 if drop_axis=="index" else 1, inplace=True)

    counts.attrs["obs_key"] = obs_key
    counts.attrs["group_by"] = group_by
    return counts


def get_cluster_proportions(adata, obs_key="Subset_Identity", group_by="orig.project",
                            drop_values=None, drop_axis="index"):
    """
    统计每个 group_by 下各个 obs_key 的百分比（行和为100%）。

    Examples
    --------
    props = get_cluster_proportions(adata,obs_key="Subset_Identity", group_by="disease")


    """
    props = (
        adata.obs.groupby([group_by, obs_key])
        .size()
        .groupby(level=0)
        .apply(lambda x: 100 * x / x.sum())
        .unstack(fill_value=0)
    )

    if drop_values is not None:
        props.drop(drop_values, axis=0 if drop_axis=="index" else 1, inplace=True)

    props.attrs["obs_key"] = obs_key
    props.attrs["group_by"] = group_by
    return props


def check_counts_layer(adata, layer="counts"):
    """
    检查 adata.layers 中的 'counts' 层是否满足以下条件：
    - 存在于 adata.layers 中
    - 数据类型为整数
    - 不包含负值
    - 不包含 NaN 和 Inf
    """

    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' 不存在于 adata.layers 中。")

    counts = adata.layers[layer]

    # 数据类型检查
    if isinstance(counts, np.ndarray):
        dtype = counts.dtype
    elif sp.isspmatrix(counts):
        dtype = counts.dtype
    else:
        raise TypeError(f"{layer} 既不是 ndarray 也不是稀疏矩阵")
    print(f"{layer}.dtype = {dtype}")
    if not np.issubdtype(dtype, np.integer):
        raise TypeError(f"{layer} 不是整数类型！")

    # 非负性检查
    if isinstance(counts, np.ndarray):
        min_val = counts.min()
    else:
        min_val = counts.data.min()
    print(f"{layer} 最小值 = {min_val}")
    if min_val < 0:
        raise ValueError(f"{layer} 存在负值 {min_val}！")

    # NaN 与 Inf 检查
    if isinstance(counts, np.ndarray):
        has_nan = np.isnan(counts).any()
        has_inf = np.isinf(counts).any()
    else:
        has_nan = np.isnan(counts.data).any()
        has_inf = np.isinf(counts.data).any()
    print(f"{layer} 含 NaN？{has_nan}, 含 Inf？{has_inf}")
    if has_nan:
        raise ValueError(f"{layer} 中存在 NaN！")
    if has_inf:
        raise ValueError(f"{layer} 中存在 Inf！")

    print(f"{layer} 检查通过：未归一化、整数、无 NaN/Inf。")


def remap_obs_clusters(adata, mapping, obs_key="tmp", new_key="cluster"):
    """
    将 adata.obs[obs_key] 中的多个标签映射到新的聚类标签（多对一映射）。

    Example
    -------
    mapping = ["Immune":["T Cell", "B Cell"], "Non-Immune":["Fibroblast","Epithelium"]]
    adata = remap_obs_cluster(adata, mapping, obs_key="Celltype", new_key="isImmune")

    Parameters
    ----------
    adata : anndata.AnnData
        输入 AnnData 对象。
    mapping : dict[str, list[str]]
        映射关系，例如 {"clusterA": ["a1", "a2"], "clusterB": ["b1"]}。
    obs_key : str, default "tmp"
        原始 obs 列名。
    new_key : str, default "cluster"
        新的 obs 列名。

    Returns
    -------
    anndata.AnnData
        返回带有新 obs 列的 AnnData 对象。
    """
    # 反转 mapping：label -> cluster
    label_to_cluster = {
        label: cluster for cluster, labels in mapping.items() for label in labels
    }

    adata = adata.copy()
    adata.obs[new_key] = adata.obs[obs_key].map(label_to_cluster)
    adata.obs[new_key] = adata.obs[new_key].astype("category")

    return adata


def _elbow_detector(ts, cluster_counts, method="kneed", default_cluster=2):
    """
    :param ts: x轴，簇数列表
    :param cluster_counts: y轴，对应的聚类指标（如 inertia）
    :param method: "MSD" or "kneed"
    :param min_cluster: 最小簇数下限
    :param default_cluster: 检测失败时默认返回值
    :return: optimal cluster number
    """
    optimal_t = None

    if method == "MSD":
        # 简单拐点检测：最大二阶差分
        first_diff = np.diff(cluster_counts)
        second_diff = np.diff(first_diff)

        # 拐点 = 最大弯曲点
        elbow_idx = np.argmax(np.abs(second_diff)) + 2  # +2 to align with ts index after 2 diffs
        optimal_t = ts[elbow_idx]

    elif method == "kneed":
        ensure_package(kneed)
        from kneed import KneeLocator
        kneedle = KneeLocator(ts, cluster_counts, curve='convex', direction='decreasing')
        optimal_t = kneedle.knee

    # 检查有效性
    if optimal_t is None:
        print(f"[_elbow_detector] Failed to fetch the elbow, using default elbow number: {default_cluster}")
        optimal_t = default_cluster

    return optimal_t


def _run_pca(logfc_matrix, n_components=2):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler

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





