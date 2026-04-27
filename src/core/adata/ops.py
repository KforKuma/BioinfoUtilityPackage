from typing import Callable, Dict, List, Optional, Sequence, Union

import gc
import os
import sys

import anndata
from anndata import AnnData
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp

import logging
from src.utils.hier_logger import logged

sys.stdout.reconfigure(encoding="utf-8")
matplotlib.use("Agg")  # 使用无 GUI 的后端，便于在服务器环境中保存图片。

logger = logging.getLogger(__name__)


def _normalize_filename_prefix(filename_prefix: Optional[str], default: str) -> str:
    """规范化输出文件名前缀。"""
    if filename_prefix is None:
        return default
    return filename_prefix.rstrip("_")


def _should_downsample(downsample: Union[bool, int, float]) -> bool:
    """判断是否需要执行下采样。"""
    # bool 是 int 的子类，这里显式排除，避免 True 被当成 1。
    return not isinstance(downsample, bool) and isinstance(downsample, (int, float)) and downsample > 0


@logged
def subcluster(
        adata: AnnData,
        n_neighbors: int = 20,
        n_pcs: int = 50,
        skip_DR: bool = False,
        resolutions: Optional[Sequence[Union[int, float]]] = None,
        use_rep: str = "X_scVI",
) -> AnnData:
    """执行降维邻接图构建与 Leiden 聚类。

    Args:
        adata: 输入的 AnnData 对象。
        n_neighbors: 构图时使用的近邻数。
        n_pcs: 参与构图的维度数；当 `use_rep="X"` 时对应 PCA 维度数。
        skip_DR: 是否跳过 `neighbors` 与 `umap` 计算。
        resolutions: Leiden 聚类分辨率；可传入单个数值或数值列表。
        use_rep: 构图使用的表示。若不为 `"X"`，则必须存在于 `adata.obsm` 中。

    Returns:
        聚类完成后的 AnnData 对象。

    Raises:
        KeyError: 当 `use_rep` 不存在于 `adata.obsm` 中时抛出。
        ValueError: 当分辨率参数类型不合法，或邻居数/维度数不合法时抛出。
    """
    if n_neighbors < 1:
        raise ValueError("`n_neighbors` must be greater than or equal to 1.")
    if n_pcs < 1:
        raise ValueError("`n_pcs` must be greater than or equal to 1.")

    if use_rep != "X" and use_rep not in adata.obsm:
        logger.info(f"[subcluster] Warning! Cannot find `use_rep`: '{use_rep}' in `adata.obsm`.")
        available_keys = list(adata.obsm.keys())
        if available_keys:
            logger.info(f"[subcluster] Available keys in `adata.obsm`: {available_keys}")
        else:
            logger.info("[subcluster] Warning! `adata.obsm` is empty.")
        raise KeyError(f"`use_rep`: '{use_rep}' does not exist in `adata.obsm`.")

    if resolutions is None:
        resolution_list = [1.0, 1.5]
        logger.info(f"[subcluster] `resolutions` is not provided. Using default values: {resolution_list}")
    elif isinstance(resolutions, (int, float)):
        resolution_list = [resolutions]
    elif isinstance(resolutions, (list, tuple, np.ndarray)):
        resolution_list = list(resolutions)
    else:
        raise ValueError("`resolutions` must be a numeric value, or a list/tuple/ndarray of numeric values.")

    if not resolution_list:
        raise ValueError("`resolutions` cannot be empty.")

    if not skip_DR:
        logger.info(
            f"[subcluster] Computing neighbors and UMAP with `n_neighbors`: {n_neighbors}, "
            f"`n_pcs`: {n_pcs}, `use_rep`: '{use_rep}'."
        )
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep)
        sc.tl.umap(adata)
        logger.info("[subcluster] UMAP computation completed.")
    else:
        logger.info("[subcluster] Warning! Skipping neighbors and UMAP computation.")

    for resolution in resolution_list:
        key = f"leiden_res{resolution}"
        logger.info(
            f"[subcluster] Running Leiden clustering with `resolution`: {resolution}. "
            f"Results will be written to `adata.obs['{key}']`."
        )
        sc.tl.leiden(adata, key_added=key, resolution=resolution)

    logger.info("[subcluster] Subclustering completed.")
    return adata


@logged
def generate_subclusters_by_identity(
        adata: anndata.AnnData,
        identity_key: str = "Subset_Identity",
        cell_idents_list: Optional[Sequence[str]] = None,
        resolutions: Optional[Sequence[Union[int, float]]] = None,
        output_dir: str = ".",
        use_rep: str = "X_scVI",
        subcluster_func: Optional[Callable] = None,
        n_neighbors: int = 20,
        filename_prefix: Optional[str] = None,
) -> None:
    """按细胞身份拆分子集，并对每个子集执行子聚类。

    该函数通常与 `run_deg_on_subsets` 搭配使用：本函数负责生成各子集的 h5ad，
    后续再在这些文件上执行多分辨率 DEG 分析。

    Args:
        adata: 输入的 AnnData 对象。
        identity_key: `adata.obs` 中用于拆分子集的列名。
        cell_idents_list: 需要处理的子集名称列表；为空时默认使用 `identity_key`
            列中的全部唯一值。
        resolutions: 子聚类使用的 Leiden 分辨率列表。
        output_dir: 输出 h5ad 文件的目录。
        use_rep: 子聚类时使用的表示名称。
        subcluster_func: 实际执行聚类的函数；为空时默认使用本模块中的 `subcluster`。
        n_neighbors: 子聚类时使用的近邻数。
        filename_prefix: 输出文件名前缀。

    Raises:
        KeyError: 当 `identity_key` 不存在于 `adata.obs` 中时抛出。
    """
    if identity_key not in adata.obs.columns:
        raise KeyError(f"`identity_key`: '{identity_key}' does not exist in `adata.obs`.")

    subcluster_func = subcluster if subcluster_func is None else subcluster_func
    filename_prefix = _normalize_filename_prefix(filename_prefix, default="SubsetSplit")
    cell_idents = adata.obs[identity_key].unique() if cell_idents_list is None else list(cell_idents_list)
    resolution_list = [0.5, 1.0] if resolutions is None else list(resolutions)

    os.makedirs(output_dir, exist_ok=True)

    for ident in cell_idents:
        logger.info(
            f"[generate_subclusters_by_identity] Starting processing for cell subtype/subpopulation: '{ident}'."
        )
        adata_subset = adata[adata.obs[identity_key] == ident].copy()

        leiden_cols = [col for col in adata_subset.obs.columns if "leiden_res" in col]
        if leiden_cols:
            adata_subset.obs.drop(columns=leiden_cols, inplace=True)

        leiden_keys = [key for key in adata_subset.uns.keys() if "leiden_res" in key]
        for key in leiden_keys:
            del adata_subset.uns[key]

        if use_rep == "X":
            n_pcs_current = min(n_pcs_from_adata(adata_subset), 50)
        else:
            n_pcs_current = min(adata_subset.obsm[use_rep].shape[1], 50)

        adata_subset = subcluster_func(
            adata_subset,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs_current,
            resolutions=resolution_list,
            use_rep=use_rep,
        )

        filename = os.path.join(output_dir, f"{filename_prefix}_{ident}.h5ad")
        adata_subset.write(filename)
        logger.info(f"[generate_subclusters_by_identity] Results were saved to: {filename}")

        del adata_subset
        gc.collect()


def n_pcs_from_adata(adata: AnnData) -> int:
    """为使用 `X` 表示时提供保守的默认维度数。"""
    return min(adata.n_vars, 50)


@logged
def obs_keywise_downsample(
        adata: AnnData,
        obs_key: str,
        downsample: Union[int, float] = 1000,
) -> AnnData:
    """按分组对 AnnData 执行下采样。

    支持两种模式：
    1. `downsample > 1` 且为整数时，表示每个分组最多保留指定数量的细胞。
    2. `0 < downsample < 1` 时，表示每个分组按比例进行下采样。

    Args:
        adata: 待下采样的 AnnData 对象。
        obs_key: `adata.obs` 中用于分组的列名。
        downsample: 下采样参数，含义见上文。

    Returns:
        下采样后的新 AnnData 对象。

    Raises:
        KeyError: 当 `obs_key` 不存在于 `adata.obs` 中时抛出。
        ValueError: 当 `downsample` 取值非法时抛出。
    """
    if obs_key not in adata.obs.columns:
        raise KeyError(f"`obs_key`: '{obs_key}' does not exist in `adata.obs`.")
    if isinstance(downsample, bool):
        raise ValueError("`downsample` cannot be a boolean value. Please provide an integer or a float.")

    counts = adata.obs[obs_key].value_counts()
    indices = []

    if downsample > 1 and float(downsample).is_integer():
        logger.info(
            f"[obs_keywise_downsample] Downsampling mode: cutoff. "
            f"Maximum cells kept for each cell subtype/subpopulation: {int(downsample)}."
        )
        cutoff = int(downsample)
        for group, count in counts.items():
            use_size = min(count, cutoff)
            selected_indices = np.random.choice(
                adata.obs_names[adata.obs[obs_key] == group],
                size=use_size,
                replace=False,
            )
            indices.extend(selected_indices)
            logger.info(
                f"[obs_keywise_downsample] Cell subtype/subpopulation '{group}': "
                f"kept {use_size}/{count} cells."
            )
    elif 0 < downsample < 1:
        logger.info(
            f"[obs_keywise_downsample] Downsampling mode: ratio. "
            f"`downsample`: {downsample:.2%}."
        )
        for group, count in counts.items():
            # 每组至少保留 1 个细胞，避免极小分组在缩放后被完全丢弃。
            use_size = max(1, round(count * downsample))
            selected_indices = np.random.choice(
                adata.obs_names[adata.obs[obs_key] == group],
                size=use_size,
                replace=False,
            )
            indices.extend(selected_indices)
            logger.info(
                f"[obs_keywise_downsample] Cell subtype/subpopulation '{group}': "
                f"kept {use_size}/{count} cells."
            )
    else:
        raise ValueError("`downsample` must be an integer greater than 1, or a float between 0 and 1.")

    logger.info(
        f"[obs_keywise_downsample] Downsampling completed. "
        f"Final cells kept: {len(indices)}/{len(adata)}."
    )
    return adata[indices].copy()


@logged
def score_gene_analysis(
        marker_dict: Dict[str, List[str]],
        adata_subset: AnnData,
        downsample: Union[bool, int, float] = False,
        plot: bool = False,
        obs_key: Optional[str] = None,
        save_addr: Optional[str] = None,
) -> AnnData:
    """对多个基因集进行打分，并按需绘制 dotplot。

    Args:
        marker_dict: 基因集字典，格式如 `{"GeneSetA": ["Gene1", "Gene2"]}`。
        adata_subset: 待分析的 AnnData 子集。
        downsample: 下采样参数；若为 `False` 则不执行下采样。
        plot: 是否绘制 dotplot。
        obs_key: 绘图或下采样时使用的分组列名。
        save_addr: 绘图输出目录。

    Returns:
        新的 AnnData 对象，其 `X` 为各基因集的得分矩阵。

    Raises:
        ValueError: 当绘图或下采样需要 `obs_key` 但未提供时抛出。
    """
    if (plot or _should_downsample(downsample)) and obs_key is None:
        raise ValueError("`obs_key` must be provided when `plot=True` or downsampling is enabled.")

    if _should_downsample(downsample):
        logger.info(f"[score_gene_analysis] Downsampling enabled with `downsample`: {downsample}.")
        adata_subset = obs_keywise_downsample(adata_subset, obs_key, downsample)

    logger.info("[score_gene_analysis] Starting gene set scoring.")
    score_cols = []
    for key, genes in marker_dict.items():
        score_name = f"score_{key}"
        sc.tl.score_genes(
            adata=adata_subset,
            gene_list=genes,
            score_name=score_name,
            use_raw=False,
        )
        score_cols.append(score_name)

    def auto_col_selector(obs_columns: Sequence[str]) -> List[str]:
        """自动挑选适合保留在输出中的 obs 列。"""
        return (
            [col for col in obs_columns if col.startswith(("orig", "disease", "Subset"))]
            + [col for col in obs_columns if col.endswith("Identity")]
            + [col for col in obs_columns if col.startswith("leiden")]
        )

    col_list = auto_col_selector(list(adata_subset.obs.columns))
    if obs_key is not None and obs_key not in col_list:
        col_list.append(obs_key)

    logger.info(f"[score_gene_analysis] Keeping {len(col_list)} columns in `adata.obs`.")

    var_df = pd.DataFrame(
        index=score_cols,
        data={"features": [marker_dict.get(col.replace("score_", ""), []) for col in score_cols]},
    )
    adata_score = anndata.AnnData(
        X=adata_subset.obs[score_cols].values,
        obs=adata_subset.obs[col_list].copy(),
        var=var_df,
    )

    if plot:
        if save_addr is None:
            save_addr = "./fig/"
        os.makedirs(save_addr, exist_ok=True)

        logger.info("[score_gene_analysis] Starting dotplot generation for gene set scores.")
        try:
            with plt.rc_context():
                sc.pl.dotplot(
                    adata_subset,
                    groupby=obs_key,
                    var_names=score_cols,
                    standard_scale="var",
                    show=False,
                )
                fname_base = os.path.join(save_addr, "GeneScore_by_{0}".format(obs_key))
                plt.savefig(f"{fname_base}.pdf", bbox_inches="tight", dpi=300)
                plt.savefig(f"{fname_base}.png", bbox_inches="tight", dpi=300)
                logger.info(
                    f"[score_gene_analysis] Dotplot was saved to: "
                    f"{fname_base}.pdf and {fname_base}.png"
                )
        except Exception as exc:
            logger.info(f"[score_gene_analysis] Warning! Dotplot generation failed: {exc}")

    logger.info("[score_gene_analysis] Gene set scoring completed.")
    return adata_score


@logged
def check_counts_layer(adata: AnnData, layer: str = "counts") -> None:
    """检查指定 layer 是否符合原始计数矩阵的基本要求。

    检查内容包括：
    1. layer 是否存在；
    2. 数据类型是否为整数；
    3. 是否包含负值；
    4. 是否包含 NaN 或 Inf。

    Args:
        adata: 输入的 AnnData 对象。
        layer: 待检查的 layer 名称。

    Raises:
        KeyError: 当 layer 不存在时抛出。
        TypeError: 当数据类型不符合要求时抛出。
        ValueError: 当存在负值、NaN 或 Inf 时抛出。
    """
    if layer not in adata.layers:
        raise KeyError(f"`layer`: '{layer}' does not exist in `adata.layers`.")

    counts = adata.layers[layer]

    if isinstance(counts, np.ndarray):
        dtype = counts.dtype
    elif sp.sparse.isspmatrix(counts):
        dtype = counts.dtype
    else:
        raise TypeError(f"`layer`: '{layer}' is neither an `ndarray` nor a sparse matrix.")

    logger.info(f"[check_counts_layer] `dtype` of layer '{layer}': {dtype}")
    if not np.issubdtype(dtype, np.integer):
        raise TypeError(f"`layer`: '{layer}' is not an integer type.")

    if isinstance(counts, np.ndarray):
        min_val = counts.min()
    else:
        # 稀疏矩阵未显式存储的元素默认为 0；若 data 为空，则最小值视为 0。
        min_val = counts.data.min() if counts.data.size > 0 else 0
    logger.info(f"[check_counts_layer] Minimum value in layer '{layer}': {min_val}")
    if min_val < 0:
        raise ValueError(f"`layer`: '{layer}' contains negative values: {min_val}.")

    if isinstance(counts, np.ndarray):
        has_nan = np.isnan(counts).any()
        has_inf = np.isinf(counts).any()
    else:
        has_nan = np.isnan(counts.data).any()
        has_inf = np.isinf(counts.data).any()

    logger.info(f"[check_counts_layer] Layer '{layer}' has NaN: {has_nan}; has Inf: {has_inf}")
    if has_nan:
        raise ValueError(f"`layer`: '{layer}' contains NaN.")
    if has_inf:
        raise ValueError(f"`layer`: '{layer}' contains Inf.")

    logger.info(
        f"[check_counts_layer] Layer '{layer}' passed validation: integer type, non-negative, "
        f"and without NaN/Inf."
    )


def remap_obs_clusters(
        adata: AnnData,
        mapping: Dict[str, List[str]],
        obs_key: str = "tmp",
        new_key: str = "cluster",
) -> AnnData:
    """将原始标签按映射关系重编码为新的聚类标签。

    Args:
        adata: 输入的 AnnData 对象。
        mapping: 映射字典，格式如 `{"Immune": ["T Cell", "B Cell"]}`。
        obs_key: 原始标签所在的 `adata.obs` 列名。
        new_key: 新标签写入的 `adata.obs` 列名。

    Returns:
        带有新标签列的 AnnData 副本。
    """
    label_to_cluster = {
        label: cluster
        for cluster, labels in mapping.items()
        for label in labels
    }

    adata = adata.copy()
    adata.obs[new_key] = adata.obs[obs_key].map(label_to_cluster)
    adata.obs[new_key] = adata.obs[new_key].astype("category")
    return adata


def make_raw_adata(adata: AnnData) -> AnnData:
    """基于 `adata.raw` 构造新的表达矩阵对象，并完成标准化与对数化。"""
    if adata.raw is None:
        raise ValueError("`adata.raw` does not exist, so raw AnnData cannot be constructed.")

    adata_full = anndata.AnnData(
        X=adata.raw.X.copy(),
        var=adata.raw.var.copy(),
        obs=adata.obs.copy(),
        obsm=adata.obsm.copy(),
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
    """根据细粒度标签反向映射出粗粒度标签，并设置分类顺序。

    Args:
        adata: 输入的 AnnData 对象。
        mapping: 标准映射字典，格式为 `col1 -> list[col2]`。
        col2: 源列名，通常为更细粒度的标签列。
        col1: 目标列名，通常为更粗粒度的标签列。
        inplace: 是否直接写回 `adata.obs[col1]`。
        check_unmapped: 是否提示未被映射覆盖的 `col2` 标签。

    Returns:
        若 `inplace=True`，返回更新后的 AnnData；否则返回映射后的 `pd.Series`。
    """
    flat = {
        value: key
        for key, values in mapping.items()
        for value in values
    }

    annotated = adata.obs[col2].map(flat)
    n_total = annotated.shape[0]
    n_mapped = annotated.notna().sum()
    n_unmapped = n_total - n_mapped

    if check_unmapped and n_unmapped > 0:
        unmapped = adata.obs.loc[annotated.isna(), col2].unique()
        print(
            f"[annotate_by_mapping] Warning! {n_unmapped} / {n_total} values in `{col2}` are unmapped:\n"
            f"{list(unmapped)}"
        )

    if not inplace:
        return annotated

    adata.obs[col1] = annotated

    col1_order = list(mapping.keys())
    adata.obs[col1] = (
        adata.obs[col1]
        .astype("category")
        .cat.set_categories(col1_order, ordered=True)
    )

    col2_order = [value for key in mapping for value in mapping[key]]
    adata.obs[col2] = (
        adata.obs[col2]
        .astype("category")
        .cat.set_categories(col2_order, ordered=True)
    )

    print(
        f"[annotate_by_mapping] Completed mapping from `{col2}` to `{col1}`. "
        f"Mapped: {n_mapped}/{n_total} ({n_mapped / n_total:.1%})"
    )
    return adata
