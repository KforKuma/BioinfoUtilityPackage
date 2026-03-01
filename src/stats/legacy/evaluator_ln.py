from __future__ import annotations
import warnings
import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd



from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)

##################
@logged
def estimate_ln_params(
        df_real: pd.DataFrame,
        dm_results: Dict[str, Any],
        ref_disease: str = "HC",
        ref_tissue: str = "nif",
        alpha: float = 0.05,
        min_effect_size: float = 0.1
) -> Dict[str, Any]:
    """
    从真实数据中提取层次化模拟所需的参数。
    输出字典的 Key 与 simulate_LogisticNormal_hierarchical 的输入参数完全对应。
    """
    
    # 1. 估计测序深度 (total_count_mean, total_count_sd)
    sample_depths = df_real.groupby('sample_id')['count'].sum()
    
    # 2. 构建潜在空间 (CLR) 矩阵进行噪声分解
    # 增加伪计数 0.5 以稳定 log 转换
    df_prop_wide = df_real.pivot_table(
        index=['donor_id', 'disease', 'tissue', 'sample_id'],
        columns='cell_type',
        values='count',
        fill_value=0
    )
    counts_matrix = df_prop_wide.values + 0.5
    log_counts = np.log(counts_matrix)
    clr_matrix = log_counts - np.mean(log_counts, axis=1, keepdims=True)
    df_clr = pd.DataFrame(clr_matrix, index=df_prop_wide.index, columns=df_prop_wide.columns)
    
    # 3. 估计 baseline_mu_scale
    # 逻辑：参考组内各细胞类型平均丰度的标准差
    ref_mask = (df_clr.index.get_level_values('disease') == ref_disease) & \
               (df_clr.index.get_level_values('tissue') == ref_tissue)
    ref_clr = df_clr[ref_mask]
    if ref_clr.empty: ref_clr = df_clr
    baseline_mu_scale = float(ref_clr.mean(axis=0).std())
    
    # 4. 估计供体级和样本级噪声 (donor_noise_sd, sample_noise_sd)
    # 计算每个 Donor 在特定条件下的均值
    donor_means = df_clr.groupby(['disease', 'tissue', 'donor_id']).mean()
    # Donor Noise: 同一条件下不同 Donor 之间的标准差
    donor_sd = donor_means.groupby(['disease', 'tissue']).std().median().median()
    
    # Sample Noise: 减去 Donor 均值后的残差标准差
    # 模拟器公式包含 np.sqrt(... + 0.5)，此处我们估计残差的总 SD
    residuals = df_clr - donor_means.reindex(df_clr.index, method=None).values
    residual_sd = residuals.std(axis=0).median()
    # 逆向对齐模拟器的额外噪声定义 (若残差过小则取极小值)
    sample_noise_sd = np.sqrt(max(0.01, residual_sd ** 2 - 0.5))
    
    # 5. 估计效应量 (从 dm_results['all_coefs'] 提取)
    df_coefs = dm_results.get('all_coefs', pd.DataFrame())
    
    def get_effect(factor_name):
        if df_coefs.empty: return 0.0
        # 筛选显著且属于该因子的项
        sig = df_coefs[(df_coefs['factor'] == factor_name) & (df_coefs['PValue'] < alpha)]
        if sig.empty: return 0.0
        return float(sig['LogFC_Coef'].abs().median())
    
    disease_eff = get_effect('disease')
    tissue_eff = get_effect('tissue')
    inter_eff = get_effect('interaction')
    
    # 6. 估计受影响细胞比例 (inflamed_cell_frac)
    total_cts = df_real['cell_type'].nunique()
    if not df_coefs.empty:
        sig_tissue_cts = df_coefs[(df_coefs['factor'] == 'tissue') & (df_coefs['PValue'] < alpha)][
            'cell_type'].nunique()
        inflamed_cell_frac = max(0.05, sig_tissue_cts / total_cts)
    else:
        inflamed_cell_frac = 0.1
    
    # ==========================================================================
    # 返回对齐模拟器参数名的字典
    # ==========================================================================
    simulation_params = {
        # "n_donors": int(df_real['donor_id'].nunique()),
        # "n_celltypes": int(total_cts),
        "baseline_mu_scale": baseline_mu_scale,
        "disease_effect_size": max(disease_eff, min_effect_size) if disease_eff > 0 else 0.0,
        "tissue_effect_size": max(tissue_eff, min_effect_size) if tissue_eff > 0 else 0.0,
        "interaction_effect_size": inter_eff,
        "inflamed_cell_frac": inflamed_cell_frac,
        "donor_noise_sd": donor_sd,
        "sample_noise_sd": sample_noise_sd,
        # "total_count_mean": int(sample_depths.mean()),
        # "total_count_sd": int(sample_depths.std()),
        # "disease_levels": list(df_real['disease'].unique()),
        # "tissue_levels": list(df_real['tissue'].unique())
    }
    
    print(
        f"✅ Extracted: Baseline_Scale={baseline_mu_scale:.2f}, Donor_SD={donor_sd:.2f}, Sample_SD={sample_noise_sd:.2f}")
    return simulation_params