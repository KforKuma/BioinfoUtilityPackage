import pandas as pd
import numpy as np
import re

def call_major_tcr_v(adata, threshold=0.55, trav_genes=None, colname_1="TRAV_call",
                     trbv_genes=None, colname_2="TRBV_call",use_raw=False):
    """为每个细胞识别主要 TRAV/TRBV V 基因。

    函数会在 ``adata.var_names`` 或 ``adata.raw.var_names`` 中查找 ``TRAV*`` 和
    ``TRBV*`` 基因，对每个细胞分别计算 V 基因表达占比。若最高占比达到
    ``threshold``，则写入对应基因名；否则写入 ``"Unassigned"``。结果会同步写入
    ``adata.obs[colname_1]`` 和 ``adata.obs[colname_2]``。

    Args:
        adata: AnnData 对象。
        threshold: 主要 V 基因占比阈值，默认 ``0.55``。
        trav_genes: 可选 TRAV 基因列表；为 ``None`` 时自动从 var_names 中匹配。
        colname_1: 写入 ``adata.obs`` 的 TRAV call 列名。
        trbv_genes: 可选 TRBV 基因列表；为 ``None`` 时自动从 var_names 中匹配。
        colname_2: 写入 ``adata.obs`` 的 TRBV call 列名。
        use_raw: 是否从 ``adata.raw`` 读取表达矩阵。

    Returns:
        ``(trav_call, trbv_call)``，均为以 ``adata.obs_names`` 为索引的 Series。

    Example:
        >>> trav_call, trbv_call = call_major_tcr_v(
        ...     adata,
        ...     threshold=0.6,
        ...     colname_1="TRAV_call",
        ...     colname_2="TRBV_call",
        ...     use_raw=True,
        ... )
        >>> adata.obs[["TRAV_call", "TRBV_call"]].head()
        # 每个细胞的主要 alpha/beta V 基因；低置信度细胞为 Unassigned。
    """
    if not 0 <= threshold <= 1:
        raise ValueError("`threshold` must be between 0 and 1.")
    if use_raw and getattr(adata, "raw", None) is None:
        raise ValueError("`adata.raw` is required when `use_raw=True`.")

    if use_raw:
        var_names = adata.raw.var_names
    else:
        var_names = adata.var_names
    
    if trav_genes is None:
        pattern_trav = re.compile(r"^TRAV")
        trav_genes = [g for g in var_names if pattern_trav.match(g)]
    if trbv_genes is None:
        pattern_trbv = re.compile(r"^TRBV")
        trbv_genes = [g for g in var_names if pattern_trbv.match(g)]
    
    # 没有对应链基因时保守地全部设为 Unassigned，避免空矩阵 argmax 崩溃。
    if len(trav_genes) == 0:
        print("[call_major_tcr_v] Warning! No TRAV genes found in var_names.")
    if len(trbv_genes) == 0:
        print("[call_major_tcr_v] Warning! No TRBV genes found in var_names.")
    
    def _call_chain(genes, colname):
        if len(genes) == 0:
            return pd.Series("Unassigned", index=adata.obs_names, name=colname)

        # 稀疏矩阵转 dense 是为了后续逐行比例计算；该函数只切 TCR V 基因，规模通常较小。
        if use_raw:
            matrix = adata.raw[:, genes].X
        else:
            matrix = adata[:, genes].X
        matrix = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        matrix = np.asarray(matrix)

        chain_sum = matrix.sum(axis=1)
        chain_fraction = matrix / (chain_sum[:, None] + 1e-9)
        max_idx = chain_fraction.argmax(axis=1)
        max_fraction = chain_fraction[np.arange(len(chain_fraction)), max_idx]

        return pd.Series(
            [
                genes[i] if frac >= threshold else "Unassigned"
                for i, frac in zip(max_idx, max_fraction)
            ],
            index=adata.obs_names,
            name=colname
        )

    trav_call = _call_chain(trav_genes, colname_1)
    trbv_call = _call_chain(trbv_genes, colname_2)
    
    # --- 写入 adata.obs ---
    adata.obs[colname_1] = trav_call
    adata.obs[colname_2] = trbv_call
    
    return trav_call, trbv_call


def classify_subset(row, cutoff=0.1):
    """按 alpha-beta / gamma-delta 使用比例粗分 TCR subset。

    Args:
        row: 包含 ``ab_frac`` 和 ``gd_frac`` 的 Series 或 dict-like 对象。
        cutoff: 判定为对应 TCR user 的最低比例。

    Returns:
        ``"ab_user"``、``"gd_user"`` 或 ``"others"``。

    Example:
        >>> classify_subset({"ab_frac": 0.2, "gd_frac": 0.0}, cutoff=0.1)
        'ab_user'
    """
    if cutoff < 0:
        raise ValueError("`cutoff` must be non-negative.")
    if row["ab_frac"] >= cutoff:
        return "ab_user"
    
    if row["gd_frac"] >= cutoff:
        return "gd_user"
    
    return "others"


def cluster_top_usage(
        df,
        cluster_col="cluster",
        v_col="v_gene",
        unassigned_label="Unassigned"
):
    """统计每个 cluster 中最常用的 V 基因。

    Args:
        df: 每行一个细胞的表，至少包含 ``cluster_col`` 和 ``v_col``。
        cluster_col: cluster 标签列。
        v_col: TCR V 基因 call 列。
        unassigned_label: 未检测到 V 区时使用的标签。

    Returns:
        每个 cluster 的 top1/top2 V 基因、比例、top2 合计比例和 unassigned 比例。

    Example:
        >>> usage = cluster_top_usage(
        ...     adata.obs,
        ...     cluster_col="Subset_Identity",
        ...     v_col="TRBV_call",
        ... )
        >>> usage[["cluster", "top1_v", "top1_prop"]].head()
        # 快速查看每个 cell subtype/subpopulation 的优势 TRBV 使用。
    """
    required_cols = {cluster_col, v_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")
    out = []
    
    for cluster, subdf in df.groupby(cluster_col):
        # 计算 Unassigned 占比
        unassigned_prop = (subdf[v_col] == unassigned_label).mean()
        
        # 对非 Unassigned 的 V 区基因做排名
        vc = (
            subdf.loc[subdf[v_col] != unassigned_label, v_col]
            .value_counts(normalize=True)
        )
        
        # 如果一个都没有，避免报错
        if len(vc) == 0:
            out.append({
                "cluster": cluster,
                "top1_v": None,
                "top1_prop": 0.0,
                "top2_v": None,
                "top2_prop": 0.0,
                "top2_sum_prop": 0.0,
                "unassigned_prop": unassigned_prop,
            })
            continue
        
        top2 = vc.iloc[:2]
        
        out.append({
            "cluster": cluster,
            "top1_v": top2.index[0] if len(top2) > 0 else None,
            "top1_prop": top2.iloc[0] if len(top2) > 0 else 0,
            "top2_v": top2.index[1] if len(top2) > 1 else None,
            "top2_prop": top2.iloc[1] if len(top2) > 1 else 0,
            "top2_sum_prop": top2.sum(),
            "unassigned_prop": unassigned_prop,
        })
    
    # 根据 top2_sum_prop 排序
    return pd.DataFrame(out).sort_values("top2_sum_prop", ascending=False)


def topN_usage(usage_merged, TCR="TRAV", N=3):
    """计算每行前 N 个 TCR V 基因频率之和。

    Args:
        usage_merged: 宽表，列通常为 MultiIndex，例如 ``("TRAV", "freq")``。
        TCR: 链名称，默认 ``"TRAV"``。
        N: 取前 N 个频率。

    Returns:
        以 ``f"{TCR}_Top{N}_usage"`` 命名的 Series。

    Example:
        >>> top3 = topN_usage(usage_merged, TCR="TRBV", N=3)
        >>> top3.head()
        # 数值越高表示该 subset 的 TRBV usage 越集中。
    """
    if N <= 0:
        raise ValueError("`N` must be greater than 0.")
    freq = usage_merged[TCR, "freq"]
    return (
        freq
        .apply(lambda x: x.sort_values(ascending=False).head(N).sum(), axis=1)
        .rename(f"{TCR}_Top{N}_usage")
    )



def tcr_usage_and_simpson(
        adata,
        subset_col="Subset_Identity",
        tcr_col="TRBV_call",
        exclude_unassigned=True
):
    """统计每个 subset 的 TCR usage 频率和 Simpson index。

    Args:
        adata: AnnData 对象，使用 ``adata.obs`` 中的 subset 和 TCR call。
        subset_col: cell subtype/subpopulation 列名。
        tcr_col: TCR V gene call 列名。
        exclude_unassigned: 是否排除 ``"Unassigned"``。

    Returns:
        ``(usage_df, simpson_df)``。``usage_df`` 是长表 usage 频率；
        ``simpson_df`` 保存每个 subset 的 Simpson index。

    Example:
        >>> usage_df, simpson_df = tcr_usage_and_simpson(
        ...     adata,
        ...     subset_col="Subset_Identity",
        ...     tcr_col="TRBV_call",
        ... )
        >>> simpson_df.sort_values("Simpson_index", ascending=False).head()
        # Simpson index 越高表示 TCR usage 越偏向少数 V 基因。
    """
    required_cols = {subset_col, tcr_col}
    missing_cols = required_cols - set(adata.obs.columns)
    if missing_cols:
        raise ValueError(f"Missing required obs columns: {sorted(missing_cols)}")
    
    obs = adata.obs[[subset_col, tcr_col]].copy()
    
    if exclude_unassigned:
        obs = obs[obs[tcr_col] != "Unassigned"]
    
    # -------- usage 统计 --------
    usage_df = (
        obs
        .groupby([subset_col, tcr_col])
        .size()
        .reset_index(name="count")
    )
    
    # 计算频率
    usage_df["freq"] = (
        usage_df
        .groupby(subset_col)["count"]
        .transform(lambda x: x / x.sum())
    )
    
    # -------- Simpson index --------
    simpson_df = (
        usage_df
        .assign(freq_sq=lambda x: x["freq"] ** 2)
        .groupby(subset_col)["freq_sq"]
        .sum()
        .reset_index(name="Simpson_index")
    )
    
    return usage_df, simpson_df
