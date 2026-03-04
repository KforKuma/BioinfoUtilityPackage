from __future__ import annotations
import warnings
import logging
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from scipy.stats import (
    dirichlet,
    multinomial
)


from src.utils.hier_logger import logged
from src.stats.simulation.truth_refine import refine_ground_truth_by_observation
logger = logging.getLogger(__name__)


# -----------------------
# 生成模拟数据：Dirichlet-Multinomial 模拟
# 有利于 Dirichlet 回归
# -----------------------
@logged
def simulate_DM_data(
        *,
        n_donors: int = 8,
        n_samples_per_donor: int = 4,
        n_celltypes: int = 50,
        baseline_alpha_scale: float = 30.0,
        disease_effect_size: float = 0.5,
        tissue_effect_size: float = 0.6,
        interaction_effect_size: float = 1.0,
        inflamed_cell_frac: float = 0.15,
        sampling_bias_strength: float = 0.0,
        disease_levels: Tuple[str, str, str] = ("HC", "CD", "UC"),
        tissue_levels: Tuple[str, str] = ("nif", "if"),
        total_count_mean=2e4,total_count_sd=5e2,min_count=1000,
        donor_noise_sd: float = 0.3,
        random_state: int = 1234
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    cell_type_names = [f"CT{i + 1}" for i in range(n_celltypes)]
    
    # 1. Baseline alpha
    baseline = rng.uniform(0.5, 2.0, n_celltypes)
    baseline = baseline / baseline.sum() * baseline_alpha_scale
    
    # 1.5 效应向量 (假设该函数已定义)
    disease_main_effects_dict, tissue_effect_vec, interaction_effects_dict, df_true_effect = build_DM_effects_with_main_effect(
        cell_type_names, disease_levels, tissue_levels,
        disease_effect_size, tissue_effect_size, interaction_effect_size,
        inflamed_cell_frac, rng
    )
    
    # 2. 预准备元数据和采样 (为了提速，先生成所有参数)
    ref_disease = disease_levels[0]
    ref_tissue = tissue_levels[0]
    
    # 提前生成采样偏差向量
    latent_axis = np.zeros(n_celltypes)
    if sampling_bias_strength > 0:
        latent_axis = rng.normal(0, 1, n_celltypes)
        latent_axis /= np.linalg.norm(latent_axis)
    
    meta_records = []
    counts_list = []
    
    for d_idx in range(n_donors):
        donor = f"D{d_idx + 1}"
        disease = rng.choice(disease_levels)
        # Donor 级别的噪声
        alpha_d = baseline * np.exp(rng.normal(0, donor_noise_sd, n_celltypes))
        
        for s_idx in range(n_samples_per_donor):
            tissue = rng.choice(tissue_levels)
            alpha = alpha_d.copy()
            
            # 应用效应
            if disease != ref_disease:
                alpha *= np.exp(disease_main_effects_dict[disease])
            if tissue != ref_tissue:
                alpha *= np.exp(tissue_effect_vec)
                if disease != ref_disease:
                    alpha *= np.exp(interaction_effects_dict[disease])
            
            # 采样偏差
            if sampling_bias_strength > 0:
                alpha *= np.exp(rng.normal(0, sampling_bias_strength) * latent_axis)
            
            alpha = np.maximum(alpha, 1e-6)
            
            # 采样
            N =  np.maximum(rng.normal(total_count_mean, total_count_sd), min_count)
            p = rng.dirichlet(alpha)
            counts = rng.multinomial(n=N, pvals=p)
            
            counts_list.append(counts)
            meta_records.append({
                "donor_id": donor,
                "disease": disease,
                "tissue": tissue,
                "sample_id": f"{donor}_S{s_idx + 1}"
            })
    
    # 3. 内存友好型构建长表 (核心优化点)
    df_meta = pd.DataFrame(meta_records)
    counts_matrix = np.vstack(counts_list)
    
    # 直接利用 NumPy 向量化展开，避开 melt
    df_long = df_meta.iloc[np.repeat(np.arange(len(df_meta)), n_celltypes)].copy()
    df_long['cell_type'] = np.tile(cell_type_names, len(df_meta))
    df_long['count'] = counts_matrix.flatten()
    
    # 4. 计算比例
    df_long['total_count'] = df_long.groupby('sample_id')['count'].transform('sum')
    df_long['prop'] = df_long['count'] / (df_long['total_count'] + 1e-9)
    
    df_long = df_long.reset_index(drop=True)
    df_true_refined = refine_ground_truth_by_observation(df_long,df_true_effect)
    return df_long, df_true_refined


def build_DM_effects_with_main_effect(
        cell_type_names, disease_levels, tissue_levels,
        disease_effect_size, tissue_effect_size, interaction_effect_size,
        inflamed_cell_frac, rng
):
    """
    DM 模型的效应生成函数，现在包含独立的 Disease Main Effect 和双向效应（增加或减少）。
    同时，使用全局基线 HC x nif 修正了 True Effect Table 中的交互作用参照组。
    """
    n_celltypes = len(cell_type_names)
    ref_disease = disease_levels[0]  # HC
    ref_tissue = tissue_levels[0]  # nif
    other_tissue = tissue_levels[1]  # if
    
    # ------------------------------------
    # Step 1: 确定受影响的细胞集和方向
    # ------------------------------------
    
    # 疾病主效应细胞集 (Disease Main Effect Cells)
    n_disease_main_cts = max(1, int(n_celltypes * 0.1))
    disease_main_cts_indices = rng.choice(n_celltypes, size=n_disease_main_cts, replace=False)
    # 随机分配方向 (+1 或 -1)
    disease_signs = rng.choice([-1, 1], size=n_disease_main_cts)
    
    # 组织/交互作用效应细胞集 (Tissue/Interaction Effect Cells)
    n_inflamed_cts = max(1, int(n_celltypes * inflamed_cell_frac))
    inflamed_cts_indices = rng.choice(n_celltypes, size=n_inflamed_cts, replace=False)
    # NEW: 随机分配方向 (+1 或 -1)
    inflamed_signs = rng.choice([-1, 1], size=n_inflamed_cts)
    
    # --- 2. Disease Main Effects (字典存储) ---
    disease_main_effects_dict = {}
    for other_disease in disease_levels[1:]:
        effect_vec = np.zeros(n_celltypes)
        random_multiplier = rng.uniform(0.8, 1.2)
        
        # 应用双向效应
        effect_values = disease_effect_size * random_multiplier * disease_signs
        effect_vec[disease_main_cts_indices] = effect_values
        
        disease_main_effects_dict[other_disease] = effect_vec
    
    # --- 3. Tissue Main Effect ---
    tissue_effect_vec = np.zeros(n_celltypes)
    random_multiplier = rng.uniform(0.8, 1.2)  # 同样增加随机性
    
    #  应用双向效应
    tissue_effect_values = tissue_effect_size * random_multiplier * inflamed_signs
    tissue_effect_vec[inflamed_cts_indices] = tissue_effect_values
    
    # --- 4. Disease x Tissue Interaction Effects (字典存储) ---
    interaction_effects_dict = {}
    for other_disease in disease_levels[1:]:
        effect_vec = np.zeros(n_celltypes)
        random_multiplier = rng.uniform(0.5, 1.5)
        
        # 应用双向效应 (使用与 Tissue Main 效应相同的受影响细胞集和方向，但大小独立)
        interaction_effect_values = interaction_effect_size * random_multiplier * inflamed_signs
        effect_vec[inflamed_cts_indices] = interaction_effect_values
        
        interaction_effects_dict[other_disease] = effect_vec
    
    # --------------------
    # Step 5: 构建 True Effect Table (保持先前修正的参照组和方向判断逻辑)
    # --------------------
    true_effects = []
    
    # 1. Disease Main Effect (Disease vs HC)
    for other_disease, E_vec in disease_main_effects_dict.items():
        for i, ct_name in enumerate(cell_type_names):
            E_disease = E_vec[i]
            true_effects.append({
                'cell_type': ct_name,
                'contrast_factor': 'disease',
                'contrast_group': other_disease,
                'contrast_ref': ref_disease,
                'True_Effect': E_disease,
                #  E_disease < 0 时为 ref_greater
                'True_Direction': 'other_greater' if E_disease > 0 else ('ref_greater' if E_disease < 0 else 'None'),
                'True_Significant': True if E_disease != 0 else False
            })
    
    # 2. Tissue Main Effect (if vs nif)
    for i, ct_name in enumerate(cell_type_names):
        E_tissue = tissue_effect_vec[i]
        true_effects.append({
            'cell_type': ct_name,
            'contrast_factor': 'tissue',
            'contrast_group': other_tissue,
            'contrast_ref': ref_tissue,
            'True_Effect': E_tissue,
            # E_tissue < 0 时为 ref_greater
            'True_Direction': 'other_greater' if E_tissue > 0 else ('ref_greater' if E_tissue < 0 else 'None'),
            'True_Significant': True if E_tissue != 0 else False
        })
    
    # 3. Disease x Tissue Interaction
    for other_disease, E_inter_vec in interaction_effects_dict.items():
        E_disease_vec = disease_main_effects_dict[other_disease]
        for i, ct_name in enumerate(cell_type_names):
            E_disease = E_disease_vec[i]
            E_tissue = tissue_effect_vec[i]
            E_interaction = E_inter_vec[i]
            
            # 计算总效应 (Addition 语义)
            total_effect = E_disease + E_tissue + E_interaction
            is_truly_sig = (E_disease != 0) or (E_tissue != 0) or (E_interaction != 0)
            
            true_effects.append({
                'cell_type': ct_name,
                'contrast_factor': 'interaction',
                'contrast_group': f'{other_disease} x {other_tissue}',
                'contrast_ref': f'{ref_disease} x {ref_tissue}',
                'True_Effect': E_interaction,
                # NEW: E_interaction < 0 时为 ref_greater
                'True_Direction': 'other_greater' if total_effect > 0 else (
                    'ref_greater' if total_effect < 0 else 'None'),
                'True_Significant': is_truly_sig
            })
    
    return disease_main_effects_dict, tissue_effect_vec, interaction_effects_dict, pd.DataFrame(true_effects)

