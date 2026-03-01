from typing import Dict, List

import anndata
from anndata import AnnData
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import scanpy as sc
import numpy as np
import scipy as sp

import os, gc, sys

sys.stdout.reconfigure(encoding='utf-8')
matplotlib.use('Agg')  # 使用无GUI的后端


import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)

@logged
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
    # 处理 use_rep 参数
    if use_rep != "X" and use_rep not in adata.obsm:
        logger.info(f"Warning:'{use_rep}' not found in adata.obsm.")
        available_keys = list(adata.obsm.keys())
        if available_keys:
            logger.info(f"Available options in adata.obsm: {available_keys}")
        else:
            logger.info("Notice adata.obsm is empty.")
        raise KeyError(f"use_rep='{use_rep}' not found in adata.obsm.")

    # 处理 resolutions
    if resolutions is None:
        resolutions = [1.0, 1.5]
        logger.info(f"No resolutions provided. Using default: {resolutions}")
    elif isinstance(resolutions, (int, float)):
        resolutions = [resolutions]
    elif not isinstance(resolutions, (list, tuple, np.ndarray)):
        raise ValueError("Resolutions must be a float, int, or list of floats.")

    # 降维
    if not skip_DR:
        logger.info(f"Computing neighbors (n_neighbors={n_neighbors}, n_pcs={n_pcs}, use_rep='{use_rep}')...")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep)
        sc.tl.umap(adata)
        logger.info("UMAP completed.")
    else:
        logger.info("Skipping dimensional reduction.")

    # 聚类
    for res in resolutions:
        key = f"leiden_res{res}"
        logger.info(f"Running Leiden clustering (resolution={res}) -> '{key}'")
        sc.tl.leiden(adata, key_added=key, resolution=res)

    logger.info("Subclustering down.")
    return adata


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

    if downsample > 1 and downsample % 1 == 0:  # 截断模式
        logger.info(f"mode: cutoff; threshold: {downsample}")
        for group, count in counts.items():
            use_size = min(count, downsample)  # 限制样本数不超过
            selected_indices = np.random.choice(
                adata.obs_names[adata.obs[obs_key] == group],
                size=use_size,
                replace=False
            )
            indices.extend(selected_indices)  # 使用extend避免嵌套列表
            logger.info(f"{group}: {use_size}/{count} selected (total={len(indices)})")
    elif 0 < downsample < 1:  # 缩放模式
        logger.info(f"mode: zoom; ratio: {downsample:.2%}.")
        for group, count in counts.items():
            use_size = max(1, round(count * downsample))  # 确保至少选择1个样本
            selected_indices = np.random.choice(
                adata.obs_names[adata.obs[obs_key] == group],
                size=use_size,
                replace=False
            )
            indices.extend(selected_indices)
            logger.info(f"{group}: {use_size}/{count} selected (total={len(indices)})")
    else:
        raise ValueError("Please recheck parameter `downsample`. It must be >1 (integer) or 0<N<1 (float).")

    logger.info(f"Final sample size: {len(indices)} (from {len(adata)} cells).")
    return adata[indices].copy()



@logged
def score_gene_analysis(marker_dict, adata_subset,
                        downsample=False, plot=False, obs_key=None, save_addr=None):
    """
    对多个基因集进行打分，并返回仅包含打分结果的 AnnData。
    可选绘制 dotplot 图。
    todo: 拆分掉绘图部分

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
    # 参数检查
    if (plot or downsample) and (obs_key is None):
        raise ValueError("obs_key must be provided when using downsample or plot=True")

    # 下采样
    if isinstance(downsample, (float, int)) and downsample > 0:
        logger.info(f"Downsampling with parameter={downsample}")
        adata_subset = obs_keywise_downsample(adata_subset, obs_key, downsample)

    # Step 1: 计算每个基因集分数
    logger.info("Scoring gene sets...")
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

    logger.info(f"Keeping {len(col_list)} columns in obs.")

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

        logger.info("Plotting dotplot...")
        try:
            with plt.rc_context():
                sc.pl.dotplot(adata_subset, groupby=obs_key, var_names=score_cols,
                              standard_scale="var", show=False)
                fname_base = os.path.join(save_addr, f"GeneScore_by_{obs_key}")
                plt.savefig(f"{fname_base}.pdf", bbox_inches="tight", dpi=300)
                plt.savefig(f"{fname_base}.png", bbox_inches="tight", dpi=300)
                logger.info(f"Saved dotplot to {fname_base}.pdf/png")
        except Exception as e:
            logger.info(f"Plot failed: {e}")

    logger.info("Finished scoring.")
    return adata_score



@logged
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
        raise TypeError(f"{layer} is neither ndarray nor sparse matrix.")
    logger.info(f"{layer}.dtype = {dtype}")
    if not np.issubdtype(dtype, np.integer):
        raise TypeError(f"{layer} is not Int.")

    # 非负性检查
    if isinstance(counts, np.ndarray):
        min_val = counts.min()
    else:
        min_val = counts.data.min()
    logger.info(f"{layer} minimum = {min_val}")
    if min_val < 0:
        raise ValueError(f"{layer} has negative value {min_val}！")

    # NaN 与 Inf 检查
    if isinstance(counts, np.ndarray):
        has_nan = np.isnan(counts).any()
        has_inf = np.isinf(counts).any()
    else:
        has_nan = np.isnan(counts.data).any()
        has_inf = np.isinf(counts.data).any()
    logger.info(f"{layer} has NaN? {has_nan}, \nhas Inf?{has_inf}")
    if has_nan:
        raise ValueError(f"{layer} has NaN.")
    if has_inf:
        raise ValueError(f"{layer} has Inf.")

    logger.info(f"{layer} checedk: unscaled, int-type, has no NaN/Inf.")


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


def make_raw_adata(adata):
    X_raw = adata.raw.X
    var_raw = adata.raw.var.copy()
    obs = adata.obs.copy()
    obsm = adata.obsm.copy()
    # 新建 AnnData
    adata_full = anndata.AnnData(
        X=X_raw.copy(),
        var=var_raw.copy(),
        obs=obs,
        obsm=obsm,
    )
    
    sc.pp.normalize_total(adata_full, target_sum=1e4)
    sc.pp.log1p(adata_full)
    return adata_full


def annotate_by_mapping(
        adata: AnnData,
        mapping: Dict[str, List[str]],
        col2: str,
        col1: str,
        *,
        inplace: bool = True,
        check_unmapped: bool = True,
):
    """
    Annotate adata.obs[col1] by reverse-mapping from adata.obs[col2],
    and set ordered categorical levels for both columns.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    mapping : dict[str, list[str]]
        Standard dictionary mapping col1 -> list of col2.
    col2 : str
        Source column in adata.obs (fine-grained labels).
    col1 : str
        Target column in adata.obs (coarse labels).
    inplace : bool, default True
        Modify adata in place.
    check_unmapped : bool, default True
        Warn if some col2 values are not covered by mapping.

    Returns
    -------
    adata or pd.Series
        If inplace=True, returns adata; otherwise returns the annotated Series.
    """
    
    # -------- 1. flatten mapping: col2 -> col1 --------
    flat = {
        v: k
        for k, values in mapping.items()
        for v in values
    }
    
    # -------- 2. reverse annotation --------
    annotated = adata.obs[col2].map(flat)
    
    n_total = annotated.shape[0]
    n_mapped = annotated.notna().sum()
    n_unmapped = n_total - n_mapped
    
    if check_unmapped and n_unmapped > 0:
        unmapped = adata.obs.loc[annotated.isna(), col2].unique()
        print(
            f"[annotate_by_mapping] Warning: {n_unmapped} / {n_total} "
            f"`{col2}` values are unmapped:\n"
            f"{list(unmapped)}"
        )
    
    if inplace:
        adata.obs[col1] = annotated
    else:
        return annotated
    
    # -------- 3. set category order for col1 --------
    col1_order = list(mapping.keys())
    adata.obs[col1] = (
        adata.obs[col1]
        .astype("category")
        .cat.set_categories(col1_order, ordered=True)
    )
    
    # -------- 4. set category order for col2 --------
    col2_order = [v for key in mapping for v in mapping[key]]
    adata.obs[col2] = (
        adata.obs[col2]
        .astype("category")
        .cat.set_categories(col2_order, ordered=True)
    )
    
    # -------- 5. completion message --------
    print(
        f"[annotate_by_mapping] Done: `{col2}` → `{col1}` | "
        f"mapped {n_mapped}/{n_total} "
        f"({n_mapped / n_total:.1%})"
    )
    
    return adata
