"""CellRank 相关辅助工具函数。"""

import logging
import re
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


def _require_obs_column(adata, column_name: str) -> None:
    """检查 `adata.obs` 是否包含指定列。"""
    if column_name not in adata.obs.columns:
        raise KeyError(
            f"Column `{column_name}` was not found in `adata.obs`. "
            f"Available columns are: {list(adata.obs.columns)}."
        )


@logged
def gimme_random_cb(adata, obs_key, identity, random_seed):
    """随机选取一个根细胞，并返回其整数索引。

    Args:
        adata: 输入 AnnData 对象。
        obs_key: `adata.obs` 中用于筛选 cell subtype/subpopulation 的列名。
        identity: 单个或多个待筛选的 cell subtype/subpopulation。
        random_seed: 随机种子。

    Returns:
        随机抽中的细胞在当前 `adata` 中的整数索引。

    Example:
        root_idx = gimme_random_cb(
            adata=adata,
            obs_key="Subset_Identity",
            identity=["Stem-like", "Transit-amplifying"],
            random_seed=42,
        )
    """
    _require_obs_column(adata, obs_key)

    if isinstance(identity, str):
        mask = adata.obs[obs_key] == identity
    elif isinstance(identity, (list, tuple, set, np.ndarray)):
        if len(identity) == 0:
            raise ValueError("Argument `identity` must not be an empty sequence.")
        mask = adata.obs[obs_key].isin(identity)
    else:
        raise TypeError("Argument `identity` must be a string or a sequence of strings.")

    cells = adata.obs_names[mask]
    if len(cells) < 1:
        raise ValueError("No cells matched the requested `identity`.")

    rng = np.random.default_rng(seed=random_seed)
    root_cell = rng.choice(cells)
    root_index = int(np.where(adata.obs_names == root_cell)[0][0])
    logger.info(f"[gimme_random_cb] Selected root cell: '{root_cell}' with index: {root_index}.")
    return root_index


@logged
def select_cell_specific_genes(
    df: pd.DataFrame,
    target_cell: str,
    corr_min: float = 0.3,
    qval_max: float = 0.05,
    delta_corr: float = 0.2,
    other_corr_max: Optional[float] = None,
    min_frac_non_sig: float = 0.8,
):
    """筛选与目标 cell subtype 强相关且相对特异的基因。

    Args:
        df: 行为基因、列包含 `*_corr` 与 `*_qval` 的 DataFrame。
        target_cell: 目标 cell subtype 名称。
        corr_min: 目标细胞的最小相关系数。
        qval_max: 目标细胞允许的最大 q-value。
        delta_corr: 目标细胞相关性相对其他细胞的最小领先幅度。
        other_corr_max: 若提供，则限制其他细胞的最大绝对相关系数。
        min_frac_non_sig: 其他细胞中“不显著”比例的最小阈值。

    Returns:
        按特异性评分降序排列的筛选结果 DataFrame。

    Example:
        selected = select_cell_specific_genes(
            df=merged_corr_df,
            target_cell="Enterocyte",
            corr_min=0.4,
            qval_max=0.01,
            delta_corr=0.25,
        )
        selected.head()
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Argument `df` must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Argument `df` must not be empty.")

    corr_cols = [column for column in df.columns if column.endswith("_corr")]
    qval_cols = [column for column in df.columns if column.endswith("_qval")]
    target_corr_col = f"{target_cell}_corr"
    target_qval_col = f"{target_cell}_qval"

    if target_corr_col not in df.columns or target_qval_col not in df.columns:
        raise KeyError(
            f"Columns `{target_corr_col}` and/or `{target_qval_col}` were not found in `df`."
        )

    other_corr_cols = [column for column in corr_cols if column != target_corr_col]
    other_qval_cols = [column for column in qval_cols if column != target_qval_col]
    if not other_corr_cols or not other_qval_cols:
        logger.info(
            "[select_cell_specific_genes] Warning! No comparison cell subtypes were available. "
            "Only the target-cell constraints will be applied."
        )

    cond_target = (df[target_corr_col] >= corr_min) & (df[target_qval_col] <= qval_max)
    max_other_corr = df[other_corr_cols].abs().max(axis=1) if other_corr_cols else pd.Series(0, index=df.index)
    max_corr_gap = df[target_corr_col] - (df[other_corr_cols].max(axis=1) if other_corr_cols else 0)
    frac_other_non_sig = (
        (df[other_qval_cols] > qval_max).sum(axis=1) / max(len(other_qval_cols), 1)
        if other_qval_cols
        else pd.Series(1.0, index=df.index)
    )

    cond_specific = (max_corr_gap >= delta_corr) & (frac_other_non_sig >= min_frac_non_sig)
    if other_corr_max is not None:
        cond_specific &= max_other_corr <= other_corr_max

    selected = df[cond_target & cond_specific].copy()
    selected["specificity_score"] = (
        selected[target_corr_col]
        - 0.5 * max_other_corr.loc[selected.index]
        - 0.2 * (max(len(other_qval_cols), 1) - (selected[other_qval_cols] > qval_max).sum(axis=1) if other_qval_cols else 0)
    )

    logger.info(
        f"[select_cell_specific_genes] Selected {selected.shape[0]} genes for target cell subtype: '{target_cell}'."
    )
    return selected.sort_values("specificity_score", ascending=False)


@logged
def standardize_obs_names(adata, ident_col="orig.ident", prefix_mode=True):
    """标准化 `adata.obs_names` 为统一命名格式。

    默认策略是把 `ident_col` 中的样本标识与条形码片段拼接为新名字，便于不同对象之间对齐。
    对于没有匹配到条形码模式的细胞，函数不会报错中断，而是保留原始名称并打印 warning。

    Args:
        adata: 输入 AnnData 对象。
        ident_col: `adata.obs` 中存储样本来源的列名。
        prefix_mode: 是否使用 `<sample>_<barcode>` 格式命名。

    Returns:
        修改后的 AnnData 对象本身。

    Example:
        adata = standardize_obs_names(
            adata=adata,
            ident_col="orig.ident",
            prefix_mode=True,
        )
        adata.obs_names[:5]
    """
    _require_obs_column(adata, ident_col)
    barcode_pattern = re.compile(r"[ACGT]{12,16}")
    new_names: List[str] = []

    for index in range(adata.n_obs):
        original_id = str(adata.obs_names[index])
        sample_id = str(adata.obs.iloc[index][ident_col])
        match = barcode_pattern.search(original_id)
        if match:
            barcode = match.group(0)
            new_id = f"{sample_id}_{barcode}" if prefix_mode else barcode
            new_names.append(new_id)
        else:
            logger.info(
                f"[standardize_obs_names] Warning! No barcode-like pattern was found in obs name: '{original_id}'."
            )
            new_names.append(original_id)

    adata.obs_names = new_names
    adata.obs_names_make_unique()
    logger.info(f"[standardize_obs_names] Standardized {adata.n_obs} obs names.")
    return adata


@logged
def align_anndata(adata1, adata2, ident_col="orig.ident"):
    """标准化两个 AnnData 对象的 `obs_names` 并汇报交集。

    Args:
        adata1: 第一个 AnnData 对象。
        adata2: 第二个 AnnData 对象。
        ident_col: 用于生成标准化样本前缀的 `obs` 列名。

    Returns:
        二元组 `(adata1, adata2)`，其中二者已完成 `obs_names` 标准化。

    Example:
        adata1, adata2 = align_anndata(
            adata1=adata_rna,
            adata2=adata_velocity,
            ident_col="orig.ident",
        )
    """
    adata1 = standardize_obs_names(adata1, ident_col=ident_col)
    adata2 = standardize_obs_names(adata2, ident_col=ident_col)
    common_cells = adata1.obs_names.intersection(adata2.obs_names)
    print(f"[align_anndata] `adata1` total cells: {adata1.n_obs}")
    print(f"[align_anndata] `adata2` total cells: {adata2.n_obs}")
    print(f"[align_anndata] Shared cells after alignment: {len(common_cells)}")
    if len(common_cells) == 0:
        print("[align_anndata] Warning! No shared cells were found after name standardization.")
    return adata1, adata2


@logged
def compare_index(df1: pd.DataFrame, df2: pd.DataFrame, name1="df1", name2="df2"):
    """比较两个 DataFrame 的索引重叠情况。

    Args:
        df1: 第一个 DataFrame。
        df2: 第二个 DataFrame。
        name1: 第一个对象的显示名称。
        name2: 第二个对象的显示名称。

    Returns:
        包含公共索引、独有索引和 Jaccard 相似度的字典。

    Example:
        result = compare_index(
            df1=gene_corr_df,
            df2=gene_expr_df,
            name1="corr",
            name2="expr",
        )
        result["summary"]
    """
    if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
        raise TypeError("Arguments `df1` and `df2` must both be pandas DataFrames.")

    idx1 = df1.index
    idx2 = df2.index
    common = idx1.intersection(idx2)
    only1 = idx1.difference(idx2)
    only2 = idx2.difference(idx1)
    similarity = len(common) / max(len(idx1.union(idx2)), 1)

    summary = {
        name1: len(idx1),
        name2: len(idx2),
        "common": len(common),
        f"only_{name1}": len(only1),
        f"only_{name2}": len(only2),
        "jaccard_similarity": similarity,
    }
    return {
        "common_index": common,
        "only_in_df1": only1,
        "only_in_df2": only2,
        "similarity_jaccard": similarity,
        "summary": summary,
    }


@logged
def build_top_corr_gene_dict(df, top_n=10):
    """按 `*_corr` 列提取每个 lineage 的 Top 相关基因。

    Args:
        df: 行为基因、列包含 `*_corr` 的 DataFrame。
        top_n: 每列返回的 Top 基因数。

    Returns:
        字典，键为 lineage 名称，值为对应 Top 基因列表。

    Example:
        top_dict = build_top_corr_gene_dict(
            df=driver_corr_df,
            top_n=20,
        )
        top_dict["Enterocyte"][:5]
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Argument `df` must be a pandas DataFrame.")
    corr_cols = [column for column in df.columns if column.endswith("_corr")]
    if not corr_cols:
        raise ValueError("No columns ending with `_corr` were found in `df`.")

    gene_dict = {}
    for column in corr_cols:
        top_genes = df[column].dropna().sort_values(ascending=False).head(top_n).index.tolist()
        gene_dict[column.replace("_corr", "")] = top_genes
    logger.info(f"[build_top_corr_gene_dict] Generated Top gene lists for {len(gene_dict)} lineages.")
    return gene_dict


@logged
def quantify_backflow(adata, basis="umap", t_key="palantir_pseudotime", cluster_key="Subset_Identity"):
    """量化 velocity 与拟时序梯度相反的“逆流”比例。

    Args:
        adata: 输入 AnnData 对象。
        basis: 嵌入空间名称，例如 `'umap'`。
        t_key: `adata.obs` 中的拟时序列名。
        cluster_key: `adata.obs` 中表示 cell subtype 的列名。

    Returns:
        以 cell subtype 为索引、按逆流比例降序排列的 `pd.Series`。

    Example:
        backflow = quantify_backflow(
            adata=adata,
            basis="umap",
            t_key="palantir_pseudotime",
            cluster_key="Subset_Identity",
        )
        backflow.head()
    """
    _require_obs_column(adata, t_key)
    _require_obs_column(adata, cluster_key)

    velocity_key = f"T_fwd_{basis}"
    embedding_key = f"X_{basis}"
    if velocity_key not in adata.obsm:
        raise KeyError(f"Key `{velocity_key}` was not found in `adata.obsm`.")
    if embedding_key not in adata.obsm:
        raise KeyError(f"Key `{embedding_key}` was not found in `adata.obsm`.")

    v_emb = np.asarray(adata.obsm[velocity_key])
    x_emb = np.asarray(adata.obsm[embedding_key])
    t_val = adata.obs[t_key].to_numpy()

    nn = NearestNeighbors(n_neighbors=min(10, adata.n_obs)).fit(x_emb)
    _, neighs = nn.kneighbors(x_emb)
    grad_t = np.zeros_like(v_emb, dtype=float)

    for i in range(len(adata)):
        dt = t_val[neighs[i]] - t_val[i]
        dx = x_emb[neighs[i]] - x_emb[i]
        grad_t[i] = np.mean(dx * dt[:, np.newaxis], axis=0)

    norm_v = np.linalg.norm(v_emb, axis=1)
    norm_g = np.linalg.norm(grad_t, axis=1)
    valid = (norm_v > 0) & (norm_g > 0)
    cos_sim = np.zeros(len(adata), dtype=float)
    cos_sim[valid] = np.einsum("ij,ij->i", v_emb[valid], grad_t[valid]) / (norm_v[valid] * norm_g[valid])
    adata.obs["velocity_pseudotime_cosine"] = cos_sim

    backflow_stats = {}
    for cluster in adata.obs[cluster_key].astype(str).unique():
        cluster_mask = adata.obs[cluster_key].astype(str) == cluster
        total = int(np.sum(cluster_mask))
        backflow = int(np.sum((adata.obs["velocity_pseudotime_cosine"] < -0.3) & cluster_mask))
        backflow_stats[cluster] = 0.0 if total == 0 else (backflow / total) * 100

    logger.info(f"[quantify_backflow] Computed backflow ratios for {len(backflow_stats)} cell subtypes.")
    return pd.Series(backflow_stats).sort_values(ascending=False)


@logged
def screen_phase_drivers(
    adata,
    cluster_key="Subset_Identity",
    source="Stressed epithelium",
    target="Ion-transport colonocyte CFTR+",
):
    """筛选 source 与 target 阶段之间 velocity 变化显著的候选驱动基因。

    Args:
        adata: 输入 AnnData 对象。
        cluster_key: `adata.obs` 中表示 cell subtype 的列名。
        source: 源阶段 cell subtype。
        target: 目标阶段 cell subtype。

    Returns:
        按 `vel_diff` 降序排列的候选基因 DataFrame。

    Example:
        drivers = screen_phase_drivers(
            adata=adata,
            cluster_key="Subset_Identity",
            source="Stem-like",
            target="Enterocyte",
        )
        drivers.head()
    """
    _require_obs_column(adata, cluster_key)
    if "velocity" not in adata.layers:
        raise KeyError("Layer `velocity` was not found in `adata.layers`.")

    clusters = adata.obs[cluster_key].astype(str)
    if source not in clusters.unique():
        raise ValueError(f"Source cell subtype '{source}' was not found in `adata.obs['{cluster_key}']`.")
    if target not in clusters.unique():
        raise ValueError(f"Target cell subtype '{target}' was not found in `adata.obs['{cluster_key}']`.")

    vel_mtx = adata.to_df(layer="velocity")
    source_vel = vel_mtx.loc[clusters == source].mean()
    target_vel = vel_mtx.loc[clusters == target].mean()
    source_exp = adata.to_df().loc[clusters == source].mean()

    df = pd.DataFrame(
        {
            "source_velocity": source_vel,
            "target_velocity": target_vel,
            "source_expression": source_exp,
            "vel_diff": source_vel - target_vel,
        }
    )
    interesting_genes = df[(df["source_expression"] > 0.1)].sort_values(by="vel_diff", ascending=False)
    logger.info(
        f"[screen_phase_drivers] Screened {interesting_genes.shape[0]} candidate driver genes from "
        f"'{source}' to '{target}'."
    )
    return interesting_genes
