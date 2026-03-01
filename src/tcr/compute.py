import pandas as pd

def call_major_tcr_v(adata, threshold=0.55, trav_genes=None, colname_1="TRAV_call",
                     trbv_genes=None, colname_2="TRBV_call"):
    """
    为每个细胞 call 出主要 TRAV / TRBV 基因（占比 > threshold）

    参数：
        adata: AnnData 对象
        threshold: 占比阈值（默认 0.55 = 55%）

    返回：
        trav_call: pd.Series
        trbv_call: pd.Series
    """
    var_names = adata.raw.var_names
    
    if trav_genes is None:
        pattern_trav = re.compile(r"^TRAV")
        trav_genes = [g for g in var_names if pattern_trav.match(g)]
    if trbv_genes is None:
        pattern_trbv = re.compile(r"^TRBV")
        trbv_genes = [g for g in var_names if pattern_trbv.match(g)]
    
    # 如果没有找到，提醒用户
    if len(trav_genes) == 0:
        print("Warning: No TRAV genes found in adata.var_names.")
    if len(trbv_genes) == 0:
        print("Warning: No TRBV genes found in adata.var_names.")
    
    # --- 高效从矩阵切片 ---
    X_trav = adata.raw[:, trav_genes].X.toarray() \
        if hasattr(adata.raw[:, trav_genes].X, "toarray") else adata.raw[:, trav_genes].X
    X_trbv = adata.raw[:, trbv_genes].X.toarray() \
        if hasattr(adata.raw[:, trbv_genes].X, "toarray") else adata.raw[:, trbv_genes].X
    
    # --- 计算每细胞各 V 基因占比 ---
    trav_sum = X_trav.sum(axis=1)
    trbv_sum = X_trbv.sum(axis=1)
    
    # 避免除零
    trav_fraction = X_trav / (trav_sum[:, None] + 1e-9)
    trbv_fraction = X_trbv / (trbv_sum[:, None] + 1e-9)
    
    # --- 每细胞找最大占比的基因 ---
    trav_max_idx = trav_fraction.argmax(axis=1)
    trbv_max_idx = trbv_fraction.argmax(axis=1)
    
    trav_max_fraction = trav_fraction[np.arange(len(trav_fraction)), trav_max_idx]
    trbv_max_fraction = trbv_fraction[np.arange(len(trbv_fraction)), trbv_max_idx]
    
    # --- 生成 call ---
    trav_call = pd.Series(
        [
            trav_genes[i] if frac >= threshold else "Unassigned"
            for i, frac in zip(trav_max_idx, trav_max_fraction)
        ],
        index=adata.obs_names,
        name=colname_1
    )
    
    trbv_call = pd.Series(
        [
            trbv_genes[i] if frac >= threshold else "Unassigned"
            for i, frac in zip(trbv_max_idx, trbv_max_fraction)
        ],
        index=adata.obs_names,
        name=colname_2
    )
    
    # --- 写入 adata.obs ---
    adata.obs[colname_1] = trav_call
    adata.obs[colname_2] = trbv_call
    
    return trav_call, trbv_call


def classify_subset(row, cutoff=0.1):
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
    """
    df: 包含 cluster 和 v_gene 的 DataFrame（每行 = 一个细胞）
    unassigned_label: 在 v_gene 中代表未检测到 V 区的标签
    """
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
    """
    对每个 subset 统计 TCR usage 频率，并计算 Simpson 指数

    返回：
    1) usage_df: long-format usage 表
    2) simpson_df: 每个 subset 的 Simpson index
    """
    
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
