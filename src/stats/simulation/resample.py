from __future__ import annotations
import warnings
import logging
from typing import Sequence, Tuple

import numpy as np
import pandas as pd



from src.utils.hier_logger import logged
from src.stats.simulation.truth_refine import refine_ground_truth_by_observation
logger = logging.getLogger(__name__)

# -----------------------
# 生成模拟数据：“真实数据 resampling” 模拟
# 相对最公正
# -----------------------
@logged
def simulate_CLR_resample_data(
        count_df,
        n_donors=20,
        n_samples_per_donor=4,
        n_celltypes=30,  # 新增参数，默认设为你想要的 100
        disease_effect_size=0.5,
        tissue_effect_size=0.8,
        interaction_effect_size=0.5,
        inflamed_cell_frac=0.15,
        donor_noise_sd=0.2,
        sample_noise_sd=0.1,
        disease_levels=("HC", "CD", "UC"),
        tissue_levels=("nif", "if"),
        pseudocount=1.0,
        random_state=1234
):
    rng = np.random.default_rng(random_state)
    n_sim_samples = n_donors * n_samples_per_donor
    
    # ---------------------------
    # Step 1: 数据提取与基线构建
    # ---------------------------
    metadata_map = count_df[['sample_id', 'donor_id', 'disease', 'tissue']].drop_duplicates().set_index('sample_id')
    sample_totals = count_df.groupby('sample_id')['count'].sum()
    
    df_counts_wide = (
        count_df.groupby(['sample_id', 'cell_type'])['count']
        .sum()
        .unstack(fill_value=0)
    )
    
    # ---------------------------
    # Step 2: 关键修改 - 细胞维度重塑
    # ---------------------------
    cell_types_original = df_counts_wide.columns.tolist()
    n_orig = len(cell_types_original)
    
    # 根据目标 n_celltypes 采样或补全原始索引
    if n_celltypes <= n_orig:
        selected_orig_indices = rng.choice(n_orig, size=n_celltypes, replace=False)
    else:
        # 如果需要的比原始多，则允许重复采样原始细胞背景
        selected_orig_indices = np.concatenate([
            np.arange(n_orig),
            rng.choice(n_orig, size=n_celltypes - n_orig, replace=True)
        ])
    
    # 构建新的虚拟细胞名称列表 [CT1, CT2, ..., CTn]
    sim_cell_names = [f"CT{i + 1}" for i in range(n_celltypes)]
    
    # 获取基线样本池
    ref_disease = disease_levels[0]
    ref_tissue = tissue_levels[0]
    baseline_sample_ids = metadata_map[
        (metadata_map['disease'] == ref_disease) & (metadata_map['tissue'] == ref_tissue)
        ].index
    
    if len(baseline_sample_ids) == 0:
        raise ValueError(f"基线样本池为空。请确保数据中包含 {ref_disease} & {ref_tissue}。")
    
    # 提取 count 矩阵并根据 selected_orig_indices 进行切片/重组
    # 注意：这里我们提取了指定维度的 Logits 背景
    counts_baseline = df_counts_wide.loc[baseline_sample_ids].values[:, selected_orig_indices] + pseudocount
    log_counts = np.log(counts_baseline)
    clr_logits_baseline = log_counts - np.mean(log_counts, axis=1, keepdims=True)
    
    # ---------------------------
    # Step 3: 设计效应向量 (使用新的 sim_cell_names)
    # ---------------------------
    disease_main_effects_dict, tissue_effect, interaction_effects_dict, df_true_effect = build_CLR_effects_and_table(
        cell_types=sim_cell_names,
        disease_levels=disease_levels,
        tissue_levels=tissue_levels,
        disease_effect_size=disease_effect_size,
        interaction_effect_size=interaction_effect_size,
        tissue_effect_size=tissue_effect_size,
        inflamed_cell_frac=inflamed_cell_frac,
        rng=rng
    )
    
    # ---------------------------
    # Step 4: 层次化模拟 (与之前逻辑一致，但维度已变为 n_celltypes)
    # ---------------------------
    sim_records = []
    for d_idx in range(n_donors):
        donor_id = f"D{d_idx + 1:02d}"
        disease = rng.choice(disease_levels)
        # 生成对应维度的噪声
        donor_shift = rng.normal(0, donor_noise_sd, n_celltypes)
        
        for s_idx in range(n_samples_per_donor):
            sample_id = f"{donor_id}_S{s_idx + 1}"
            tissue = rng.choice(tissue_levels)
            
            idx_resample = rng.integers(0, len(clr_logits_baseline))
            clr_logit_sim = clr_logits_baseline[idx_resample].copy()
            
            clr_logit_sim += donor_shift
            if disease != ref_disease:
                clr_logit_sim += disease_main_effects_dict[disease]
            if tissue != ref_tissue:
                clr_logit_sim += tissue_effect
            # 修正：之前代码这里漏了 tissue 判断，现在加上
            if disease != ref_disease and tissue != ref_tissue:
                clr_logit_sim += interaction_effects_dict[disease]
            
            clr_logit_sim += rng.normal(0, sample_noise_sd, n_celltypes)
            
            sim_records.append({
                "donor_id": donor_id, "sample_id": sample_id,
                "disease": disease, "tissue": tissue,
                "clr_logit_sim": clr_logit_sim
            })
    
    df_sim_meta = pd.DataFrame(sim_records)
    
    # ---------------------------
    # Step 5: 生成 Count
    # ---------------------------
    logits_matrix = np.vstack(df_sim_meta['clr_logit_sim'].values)
    logits_matrix = np.clip(logits_matrix, -700, 700)
    exp_logits = np.exp(logits_matrix)
    proportions = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    sim_depths = rng.choice(sample_totals.values, size=n_sim_samples, replace=True)
    counts_matrix = np.array([
        rng.multinomial(n=sim_depths[i], pvals=proportions[i])
        for i in range(n_sim_samples)
    ])
    
    # ---------------------------
    # Step 6: 内存高效构建长表
    # ---------------------------
    df_sim_long = df_sim_meta.iloc[np.repeat(np.arange(n_sim_samples), n_celltypes)].copy()
    df_sim_long.drop(columns='clr_logit_sim', inplace=True)
    
    # 填充 cell_type 编号
    df_sim_long['cell_type'] = np.tile(sim_cell_names, n_sim_samples)
    df_sim_long['count'] = counts_matrix.flatten()
    
    df_sim_long['total_count'] = df_sim_long.groupby('sample_id')['count'].transform('sum')
    df_sim_long['prop'] = df_sim_long['count'] / (df_sim_long['total_count'] + 1e-9)
    df_long = df_sim_long.reset_index(drop=True)
    df_true_refined = refine_ground_truth_by_observation(df_long, df_true_effect)
    return df_long, df_true_refined

@logged
def build_CLR_effects_and_table(
        cell_types, disease_levels, tissue_levels,
        disease_effect_size, tissue_effect_size, interaction_effect_size,
        inflamed_cell_frac, rng
):
    """
    修正后的 CLR 效应生成函数：
    1. 支持不同疾病影响不同的细胞集
    2. 将交互作用与组织主效应细胞集解耦
    3. 修复显著性判定逻辑
    """
    n_celltypes = len(cell_types)
    ref_disease = disease_levels[0]  # HC
    ref_tissue = tissue_levels[0]  # nif
    other_tissue = tissue_levels[1]  # if
    
    n_disease_main_cts = max(1, int(n_celltypes * 0.1))
    n_inflamed_cts = max(1, int(n_celltypes * inflamed_cell_frac))
    
    # --- 1. 疾病主效应 (每个疾病独立采样受影响细胞) ---
    disease_main_effects_dict = {}
    for other_disease in disease_levels[1:]:
        effect_vec = np.zeros(n_celltypes)
        # 只有在 size > 0 时才采样和赋值
        if disease_effect_size > 0:
            indices = rng.choice(n_celltypes, size=n_disease_main_cts, replace=False)
            signs = rng.choice([-1, 1], size=n_disease_main_cts)
            random_multiplier = rng.uniform(0.8, 1.2)
            effect_vec[indices] = disease_effect_size * random_multiplier * signs
        disease_main_effects_dict[other_disease] = effect_vec
    
    # --- 2. 组织主效应 ---
    tissue_effect = np.zeros(n_celltypes)
    if tissue_effect_size > 0:
        inflamed_cts_indices = rng.choice(n_celltypes, size=n_inflamed_cts, replace=False)
        inflamed_signs = rng.choice([-1, 1], size=n_inflamed_cts)
        random_multiplier = rng.uniform(0.8, 1.2)
        tissue_effect[inflamed_cts_indices] = tissue_effect_size * random_multiplier * inflamed_signs
    
    # --- 3. 交互作用效应 (独立采样，不强制与组织效应细胞重合) ---
    interaction_effects_dict = {}
    for other_disease in disease_levels[1:]:
        effect_vec = np.zeros(n_celltypes)
        if interaction_effect_size > 0:
            # 独立采样交互项影响的细胞，数量仍由 inflamed_cell_frac 决定
            inter_indices = rng.choice(n_celltypes, size=n_inflamed_cts, replace=False)
            inter_signs = rng.choice([-1, 1], size=n_inflamed_cts)
            random_multiplier = rng.uniform(0.5, 1.5)
            effect_vec[inter_indices] = interaction_effect_size * random_multiplier * inter_signs
        interaction_effects_dict[other_disease] = effect_vec
    
    # --------------------
    # 构建 True Effect Table
    # --------------------
    true_effects = []
    
    # 1. Disease Main Effect
    for other_disease, E_vec in disease_main_effects_dict.items():
        for i, ct_name in enumerate(cell_types):
            val = E_vec[i]
            true_effects.append({
                'cell_type': ct_name,
                'contrast_factor': 'disease',
                'contrast_group': other_disease,
                'contrast_ref': ref_disease,
                'True_Effect': val,
                'True_Direction': 'other_greater' if val > 0 else ('ref_greater' if val < 0 else 'None'),
                'True_Significant': True if val != 0 else False
            })
    
    # 2. Tissue Main Effect
    for i, ct_name in enumerate(cell_types):
        val = tissue_effect[i]
        true_effects.append({
            'cell_type': ct_name,
            'contrast_factor': 'tissue',
            'contrast_group': other_tissue,
            'contrast_ref': ref_tissue,
            'True_Effect': val,
            'True_Direction': 'other_greater' if val > 0 else ('ref_greater' if val < 0 else 'None'),
            'True_Significant': True if val != 0 else False
        })
    
    # 3. Disease x Tissue Interaction
    for other_disease, E_inter_vec in interaction_effects_dict.items():
        # 获取疾病主效应
        E_disease_vec = disease_main_effects_dict[other_disease]
        for i, ct_name in enumerate(cell_types):
            val_disease = E_disease_vec[i]
            val_tissue = tissue_effect[i]
            val_inter = E_inter_vec[i]
            
            # 叠加
            total_val = val_disease + val_tissue + val_inter
            is_truly_sig = (val_disease != 0) or (val_tissue != 0) or (val_inter != 0)
            
            true_effects.append({
                'cell_type': ct_name,
                'contrast_factor': 'interaction',
                'contrast_group': f'{other_disease} x {other_tissue}',
                'contrast_ref': f'{ref_disease} x {ref_tissue}',
                'True_Effect': total_val,
                'True_Direction': 'other_greater' if total_val > 0 else ('ref_greater' if total_val < 0 else 'None'),
                'True_Significant': is_truly_sig
            })
    
    return disease_main_effects_dict, tissue_effect, interaction_effects_dict, pd.DataFrame(true_effects)
