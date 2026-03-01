import numpy as np
import pandas as pd
import re

def gimme_random_cb(adata, obs_key, identity, random_seed):
    if isinstance(identity, list):
        mask = adata.obs[obs_key].isin(identity)
    elif isinstance(identity, str):
        mask = adata.obs[obs_key] == identity
    else:
        raise ValueError("Please recheck the identity.")
    
    cells = adata.obs_names[mask]
    if len(cells) < 1:
        raise ValueError("Warning! No cell found.")
    
    rng = np.random.default_rng(seed=random_seed)
    root_cell = rng.choice(cells)
    
    # 3. 转成 adata_sub 中的整数 index
    root_ixs = np.where(adata.obs_names == root_cell)[0][0]
    
    return root_ixs


def select_cell_specific_genes(
        df: pd.DataFrame,
        target_cell: str,
        corr_min: float = 0.3,
        qval_max: float = 0.05,
        delta_corr: float = 0.2,
        other_corr_max: float | None = None,
        min_frac_non_sig: float = 0.8,
):
    """
    筛选与 target_cell 强相关、且对其他细胞相对特异的基因
    """
    
    # --- 列名解析 ---
    corr_cols = [c for c in df.columns if c.endswith("_corr")]
    qval_cols = [c for c in df.columns if c.endswith("_qval")]
    
    target_corr_col = f"{target_cell}_corr"
    target_qval_col = f"{target_cell}_qval"
    
    other_corr_cols = [c for c in corr_cols if c != target_corr_col]
    other_qval_cols = [c for c in qval_cols if c != target_qval_col]
    
    # --- 目标细胞条件 ---
    cond_target = (
            (df[target_corr_col] >= corr_min) &
            (df[target_qval_col] <= qval_max)
    )
    
    # --- 其他细胞统计 ---
    max_other_corr = df[other_corr_cols].abs().max(axis=1)
    max_corr_gap = df[target_corr_col] - df[other_corr_cols].max(axis=1)
    
    frac_other_non_sig = (
            (df[other_qval_cols] > qval_max).sum(axis=1)
            / len(other_qval_cols)
    )
    
    cond_specific = (
            (max_corr_gap >= delta_corr) &
            (frac_other_non_sig >= min_frac_non_sig)
    )
    
    if other_corr_max is not None:
        cond_specific &= (max_other_corr <= other_corr_max)
    
    # --- 组合 ---
    selected = df[cond_target & cond_specific].copy()
    
    # --- 可选：打分排序 ---
    selected["specificity_score"] = (
            selected[target_corr_col]
            - 0.5 * max_other_corr.loc[selected.index]
            - 0.2 * (len(other_qval_cols) - (selected[other_qval_cols] > qval_max).sum(axis=1))
    )
    
    return selected.sort_values("specificity_score", ascending=False)

def align_anndata(adata1,adata2,ident_col="orig.ident"):
    adata1 = standardize_obs_names(adata1, ident_col)
    adata2 = standardize_obs_names(adata2, ident_col)
    common_cells = adata1.obs_names.intersection(adata2.obs_names)
    print(f"adata1 细胞数: {adata1.n_obs}")
    print(f"adata2 细胞数: {adata2.n_obs}")
    print(f"成功对齐的共有细胞数: {len(common_cells)}")
    return adata1, adata2


def standardize_obs_names(adata, ident_col='orig.ident', prefix_mode=True):
    """
    将 adata 的 obs_names 统一为 'ident_16位碱基' 格式。

    参数:
    adata: AnnData 对象
    ident_col: 存储样本来源的 obs 列名
    prefix_mode: 如果为 True，输出 'Sample_ATGC...'; 否则根据需要自定义
    """
    new_names = []
    # 正则表达式：匹配连续的 16 位 ACGT 碱基
    barcode_pattern = re.compile(r'[ACGT]{12}')
    
    for i in range(adata.n_obs):
        original_id = adata.obs_names[i]
        sample_id = str(adata.obs[ident_col][i])
        
        # 提取 16 位碱基
        match = barcode_pattern.search(original_id)
        if match:
            barcode_16 = match.group(0)
            # 拼接新 ID：这里使用 '_' 连接，确保清晰且唯一
            new_id = f"{sample_id}_{barcode_16}"
            new_names.append(new_id)
        else:
            # 如果没找到 16 位碱基（报错或记录）
            print(f"Warning: No 16-mer barcode found in {original_id}")
            new_names.append(original_id)
    
    adata.obs_names = new_names
    # 确保唯一性（处理极少数同样本同序列情况）
    adata.obs_names_make_unique()
    return adata


def align_anndata(adata1, adata2, ident_col="orig.ident"):
    adata1 = standardize_obs_names(adata1, ident_col)
    adata2 = standardize_obs_names(adata2, ident_col)
    common_cells = adata1.obs_names.intersection(adata2.obs_names)
    print(f"adata1 细胞数: {adata1.n_obs}")
    print(f"adata2 细胞数: {adata2.n_obs}")
    print(f"成功对齐的共有细胞数: {len(common_cells)}")
    return adata1, adata2


def compare_index(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        name1="df1",
        name2="df2"
):
    """
    比较两个 DataFrame 的 index 相似度。

    返回:
        dict:
            - common_index: Index
            - only_in_df1: Index
            - only_in_df2: Index
            - similarity_jaccard: float
            - summary: dict
    """
    idx1 = df1.index
    idx2 = df2.index
    
    common = idx1.intersection(idx2)
    only1 = idx1.difference(idx2)
    only2 = idx2.difference(idx1)
    
    # Jaccard 相似度
    similarity = len(common) / len(idx1.union(idx2))
    
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


def build_top_corr_gene_dict(df, top_n=10):
    gene_dict = {}
    
    # 1. 找到所有以 _corr 结尾的列
    corr_cols = [c for c in df.columns if c.endswith("_corr")]
    
    for col in corr_cols:
        # 2. 按相关系数从大到小排序，自动忽略 NaN
        top_genes = (
            df[col]
            .dropna()
            .sort_values(ascending=False)
            .head(top_n)
            .index
            .tolist()
        )
        
        # 3. 去掉 _corr 作为 key
        key = col.replace("_corr", "")
        
        gene_dict[key] = top_genes
    
    return gene_dict


def quantify_backflow(adata, basis='umap', t_key='palantir_pseudotime'):
    """
    统计逆拟时序流向的细胞比例
    """
    # 1. 提取投影后的速度向量 (V_emb) 和坐标 (X_emb)
    # 确保之前已经运行过 scv.tl.velocity_embedding
    v_emb = adata.obsm[f'T_fwd_{basis}']
    x_emb = adata.obsm[f'X_{basis}']
    t_val = adata.obs[t_key].values
    
    # 2. 计算拟时序梯度 (基于邻居平滑)
    # 这里我们简化处理：寻找每个细胞最近的邻居，计算拟时序的变化方向
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=10).fit(x_emb)
    dists, neighs = nn.kneighbors(x_emb)
    
    grad_t = np.zeros_like(v_emb)
    for i in range(len(adata)):
        # 邻居的拟时序减去当前细胞的拟时序
        dt = t_val[neighs[i]] - t_val[i]
        # 邻居的坐标位移
        dx = x_emb[neighs[i]] - x_emb[i]
        # 梯度 = 位移 * 拟时序差的加权平均
        grad_t[i] = np.mean(dx * dt[:, np.newaxis], axis=0)
    
    # 3. 计算速度向量与梯度的余弦相似度
    norm_v = np.linalg.norm(v_emb, axis=1)
    norm_g = np.linalg.norm(grad_t, axis=1)
    
    # 避免除以零
    valid = (norm_v > 0) & (norm_g > 0)
    cos_sim = np.zeros(len(adata))
    cos_sim[valid] = np.einsum('ij,ij->i', v_emb[valid], grad_t[valid]) / (norm_v[valid] * norm_g[valid])
    
    adata.obs['velocity_pseudotime_cosine'] = cos_sim
    
    # 4. 按亚群统计逆流比例 (cos_sim < -0.3 视为明显逆流)
    backflow_stats = {}
    for cluster in adata.obs['Subset_Identity'].unique():
        cluster_mask = adata.obs['Subset_Identity'] == cluster
        total = np.sum(cluster_mask)
        backflow = np.sum((adata.obs['velocity_pseudotime_cosine'] < -0.3) & cluster_mask)
        backflow_stats[cluster] = (backflow / total) * 100
    
    return pd.Series(backflow_stats).sort_values(ascending=False)


def screen_phase_drivers(adata,
                         cluster_key='Subset_Identity',
                         source='Stressed epithelium',
                         target='Ion-transport colonocyte CFTR+'):
    # 1. 提取速率矩阵和分组信息
    vel_mtx = adata.to_df(layer='velocity')
    clusters = adata.obs[cluster_key]
    
    # 2. 计算 source 和 target 的平均速率
    source_vel = vel_mtx.loc[clusters == source].mean()
    target_vel = vel_mtx.loc[clusters == target].mean()
    
    # 3. 计算 source 的平均表达量 (用于排除低表达噪音)
    source_exp = adata.to_df().loc[clusters == source].mean()
    
    # 4. 构建筛选结果表
    df = pd.DataFrame({
        'source_velocity': source_vel,
        'target_velocity': target_vel,
        'source_expression': source_exp,
        'vel_diff': source_vel - target_vel  # 寻找速率降幅最大的基因
    })
    
    # 5. 核心筛选条件：
    # - Source 正在诱导 (vel > 0)
    # - Target 正在压制或趋于平稳 (vel < 0)
    # - 表达量不在最底层 (exp > 0.1)
    interesting_genes = df[
        # (df['source_velocity'] > 0.01) &
        # (df['target_velocity'] < 0.01) &
        (df['source_expression'] > 0.1)
    ].sort_values(by='vel_diff', ascending=False)
    
    return interesting_genes