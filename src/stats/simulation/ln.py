from __future__ import annotations
import warnings
import logging
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.utils.hier_logger import logged
from src.stats.simulation.truth_refine import refine_ground_truth_by_observation
logger = logging.getLogger(__name__)



# -----------------------
# 生成模拟数据：Logistic-Normal Multinomial 模拟
# 有利于 LMM/CLR
# -----------------------

@logged
def simulate_LogisticNormal_hierarchical(
        n_donors=8,
        n_samples_per_donor=4,
        n_celltypes=50,
        baseline_mu_scale=1.0,
        disease_effect_size=0.5,
        tissue_effect_size=0.8,
        interaction_effect_size=0.5,
        inflamed_cell_frac=0.15,
        donor_noise_sd=0.3,
        sample_noise_sd=0.5,
        total_count_mean=2e4,total_count_sd=5e2,min_count=1000,
        disease_levels=("HC", "CD", "UC"),
        tissue_levels=("nif", "if"),
        random_state=1234
):
    """生成 Logistic-Normal Multinomial 层次模拟数据。

    该模拟器在 logit/CLR-like 空间中叠加 baseline、donor 随机效应、disease
    主效应、tissue 主效应、interaction 以及 sample 噪声，随后通过 softmax 转为
    组成比例并进行 multinomial 采样。它更适合评估 LMM/CLR 类方法。

    Args:
        n_donors: donor 数量。
        n_samples_per_donor: 每个 donor 的 sample 数量。
        n_celltypes: cell subtype/subpopulation 数量。
        baseline_mu_scale: baseline logits 的标准差。
        disease_effect_size: disease 主效应大小。
        tissue_effect_size: tissue 主效应大小。
        interaction_effect_size: disease x tissue 交互效应大小。
        inflamed_cell_frac: 受 tissue/interaction 影响的亚群比例。
        donor_noise_sd: donor 层面随机效应标准差。
        sample_noise_sd: sample 层面噪声标准差。
        total_count_mean: 每个 sample 的测序深度均值。
        total_count_sd: 每个 sample 的测序深度标准差。
        min_count: 每个 sample 的最小测序深度。
        disease_levels: disease 水平，首个元素作为参考组。
        tissue_levels: tissue 水平，首个元素作为参考组。
        random_state: 随机种子。

    Returns:
        ``(df_long, df_true_refined)``，分别为模拟丰度长表和可观察 ground truth。

    Example:
        >>> df_sim, df_truth = simulate_LogisticNormal_hierarchical(
        ...     n_donors=10,
        ...     n_samples_per_donor=4,
        ...     n_celltypes=30,
        ...     disease_effect_size=0.5,
        ...     tissue_effect_size=0.8,
        ...     random_state=7,
        ... )
        >>> df_sim.groupby("sample_id")["count"].sum().head()
        # 检查每个 sample 的模拟测序深度。
    """
    if n_donors <= 0 or n_samples_per_donor <= 0 or n_celltypes <= 1:
        raise ValueError("`n_donors`, `n_samples_per_donor`, and `n_celltypes` must be positive; `n_celltypes` must be greater than 1.")
    if len(disease_levels) < 2:
        raise ValueError("`disease_levels` must contain at least two levels.")
    if len(tissue_levels) < 2:
        raise ValueError("`tissue_levels` must contain at least two levels.")
    if not 0 <= inflamed_cell_frac <= 1:
        raise ValueError("`inflamed_cell_frac` must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    ref_disease = disease_levels[0]
    ref_tissue = tissue_levels[0]
    other_tissue = tissue_levels[1]
    cell_types = [f"CT{i + 1}" for i in range(n_celltypes)]
    
    # ---------------------------
    # Step 1: 向量化构建 Metadata
    # ---------------------------
    donor_ids = [f"D{i + 1}" for i in range(n_donors)]
    # 每个 donor 固定一个疾病状态
    donor_disease_map = {d: rng.choice(disease_levels) for d in donor_ids}
    
    records = []
    for d in donor_ids:
        for s_idx in range(n_samples_per_donor):
            records.append({
                "donor_id": d,
                "disease": donor_disease_map[d],
                "tissue": rng.choice(tissue_levels),
                "sample_id": f"{d}_S{s_idx}"
            })
    df_meta = pd.DataFrame(records)
    n_samples = len(df_meta)
    
    # ---------------------------
    # Step 2-5: 定义效应向量 (保持原逻辑，向量化准备)
    # ---------------------------
    baseline_mu = rng.normal(0, baseline_mu_scale, n_celltypes)
    donor_random_effects = {d: rng.normal(0, donor_noise_sd, n_celltypes) for d in donor_ids}
    
    # 预计算 disease/tissue/interaction 效应
    disease_effects = {}
    n_main = max(1, int(n_celltypes * 0.1))
    main_indices = rng.choice(n_celltypes, size=n_main, replace=False)
    for d_level in disease_levels[1:]:
        vec = np.zeros(n_celltypes)
        vec[main_indices] = disease_effect_size * rng.uniform(0.8, 1.2) * rng.choice([-1, 1], n_main)
        disease_effects[d_level] = vec
    
    tissue_effect_vec = np.zeros(n_celltypes)
    n_inf = max(1, int(n_celltypes * inflamed_cell_frac))
    inf_indices = rng.choice(n_celltypes, size=n_inf, replace=False)
    tissue_signs = rng.choice([-1, 1], n_inf)
    tissue_effect_vec[inf_indices] = tissue_effect_size * rng.uniform(0.8, 1.2) * tissue_signs
    
    inter_effects = {}
    for d_level in disease_levels[1:]:
        vec = np.zeros(n_celltypes)
        vec[inf_indices] = interaction_effect_size * rng.uniform(0.5, 1.5) * tissue_signs
        inter_effects[d_level] = vec
    
    # ---------------------------
    # Step 6: 向量化构建 Logits (性能核心优化)
    # ---------------------------
    # 1. 基础 mu 和 Donor 效应
    donor_indices = df_meta['donor_id'].map({d: i for i, d in enumerate(donor_ids)}).values
    donor_effects_matrix = np.array([donor_random_effects[d] for d in donor_ids])
    
    logits = baseline_mu + donor_effects_matrix[donor_indices]
    
    # 2. 疾病效应
    for d_level, effect in disease_effects.items():
        mask = (df_meta['disease'] == d_level).values
        logits[mask] += effect
    
    # 3. 组织效应
    tissue_mask = (df_meta['tissue'] == other_tissue).values
    logits[tissue_mask] += tissue_effect_vec
    
    # 4. 交互效应
    for d_level, effect in inter_effects.items():
        mask = ((df_meta['disease'] == d_level) & (df_meta['tissue'] == other_tissue)).values
        logits[mask] += effect
    
    # 5. 合并样本噪声 (合并 multivariate_normal 的方差)
    # 原始 sd 为 sample_noise_sd，协方差 0.5 相当于 sd 约为 0.707
    total_noise_sd = np.sqrt(sample_noise_sd ** 2 + 0.5)
    logits += rng.normal(0, total_noise_sd, size=logits.shape)
    
    # ---------------------------
    # Step 7-9: 采样与长表构建
    # ---------------------------
    proportions = softmax(logits, axis=1)
    
    total_counts = np.maximum(
        np.rint(rng.normal(total_count_mean, total_count_sd, n_samples)).astype(int), min_count
    )
    
    # 采样
    counts = np.array([
        rng.multinomial(n=total_counts[i], pvals=proportions[i])
        for i in range(n_samples)
    ])
    
    # 直接构建长表，避开宽表和 melt
    df_long = df_meta.iloc[np.repeat(np.arange(n_samples), n_celltypes)].copy()
    df_long['cell_type'] = np.tile(cell_types, n_samples)
    df_long['count'] = counts.flatten()
    df_long['total_count'] = df_long.groupby('sample_id')['count'].transform('sum')
    df_long['prop'] = df_long['count'] / (df_long['total_count'] + 1e-12)
    
    # 真实效应表 (此处可调用你原有的 build_true_effect_table)
    # 为简洁起见，假设逻辑同原代码
    df_true_effect = build_true_effect_table(
        cell_types, ref_disease, ref_tissue,
        disease_effects, tissue_effect_vec, inter_effects, other_tissue
    )
    df_long = df_long.reset_index(drop=True)
    df_true_refined = refine_ground_truth_by_observation(df_long, df_true_effect)
    return df_long, df_true_refined

@logged
def build_true_effect_table(cell_types, ref_disease, ref_tissue, disease_effects, tissue_effect,
                            interaction_effects, other_tissue):
    """构建 Logistic-Normal 模拟的真实效应表。

    Args:
        cell_types: cell subtype/subpopulation 名称列表。
        ref_disease: disease 参考组。
        ref_tissue: tissue 参考组。
        disease_effects: disease 主效应字典。
        tissue_effect: tissue 主效应向量。
        interaction_effects: interaction 效应字典。
        other_tissue: 非参考 tissue 水平。

    Returns:
        每个 cell subtype/subpopulation 和对比组合的 ground truth DataFrame。

    Example:
        >>> truth = build_true_effect_table(
        ...     ["CT1", "CT2"],
        ...     "HC",
        ...     "nif",
        ...     {"CD": np.array([0.5, 0.0])},
        ...     np.array([0.0, -0.4]),
        ...     {"CD": np.array([0.2, 0.0])},
        ...     "if",
        ... )
        >>> truth[truth["True_Significant"]]
        # 用于评估估计结果是否命中真实注入效应。
    """
    if len(cell_types) == 0:
        raise ValueError("`cell_types` must not be empty.")
    true_effects = []
    for other_disease, E_vec in disease_effects.items():
        for i, ct_name in enumerate(cell_types):
            E_disease = E_vec[i]
            true_effects.append({
                'cell_type': ct_name, 'contrast_factor': 'disease', 'contrast_group': other_disease,
                'contrast_ref': ref_disease,
                'True_Effect': E_disease, 'True_Direction': 'other_greater' if E_disease > 0 else (
                    'ref_greater' if E_disease < 0 else 'None'),
                'True_Significant': True if E_disease != 0 else False
            })
    for i, ct_name in enumerate(cell_types):
        E_tissue = tissue_effect[i]
        true_effects.append({
            'cell_type': ct_name, 'contrast_factor': 'tissue', 'contrast_group': other_tissue,
            'contrast_ref': ref_tissue,
            'True_Effect': E_tissue,
            'True_Direction': 'other_greater' if E_tissue > 0 else ('ref_greater' if E_tissue < 0 else 'None'),
            'True_Significant': True if E_tissue != 0 else False
        })
    for other_disease, E_inter_vec in interaction_effects.items():
        E_disease_vec = disease_effects[other_disease]
        for i, ct_name in enumerate(cell_types):
            E_disease = E_disease_vec[i]
            E_tissue = tissue_effect[i]
            E_interaction = E_inter_vec[i]
            
            # 计算总效应
            total_effect = E_disease + E_tissue + E_interaction
            # 只要三者有一个注入了，就是显著的（符合 Addition 语义）
            is_truly_sig = (E_disease != 0) or (E_tissue != 0) or (E_interaction != 0)
            
            true_effects.append({
                'cell_type': ct_name,
                'contrast_factor': 'interaction',  # 以后可以考虑统一改为 'addition'
                'contrast_group': f'{other_disease} x {other_tissue}',
                'contrast_ref': f'{ref_disease} x {ref_tissue}',
                'True_Effect': total_effect,
                'True_Direction': 'other_greater' if total_effect > 0 else (
                    'ref_greater' if total_effect < 0 else 'None'),
                'True_Significant': is_truly_sig
            })
    return pd.DataFrame(true_effects)
