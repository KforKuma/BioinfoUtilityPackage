from __future__ import annotations
import warnings
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.utils.hier_logger import logged
from src.stats.evaluation.evaluator_dm import *

logger = logging.getLogger(__name__)


#################
@logged
def get_refined_noise_estimates(df_real):
    """使用混合模型提取最精确的方差分量"""
    # 1. 准备 CLR 空间数据
    df_wide = df_real.pivot_table(index=['donor_id', 'sample_id', 'disease', 'tissue'],
                                  columns='cell_type', values='count', fill_value=0)
    # 增加伪计数并转换 CLR
    vals = df_wide.values + 1.0
    clr_vals = np.log(vals) - np.log(vals).mean(axis=1, keepdims=True)
    df_clr = pd.DataFrame(clr_vals, index=df_wide.index, columns=df_wide.columns).reset_index()
    
    donor_vols, resid_vols = [], []
    reliable_cts = df_wide.columns
    n_samples_per_donor = df_clr.groupby('donor_id').size().mean()
    
    
    for ct in reliable_cts:
        try:
            # 尝试线性混合模型 (LMM)
            model = smf.mixedlm(f"Q('{ct}') ~ disease + tissue", df_clr, groups=df_clr["donor_id"]).fit(reml=True)
            d_var = model.cov_re.iloc[0, 0]
            r_var = model.scale
            donor_vols.append(np.sqrt(max(d_var, 0)))
            resid_vols.append(np.sqrt(max(r_var, 0)))
        except:
            # OLS Fallback 并进行方差偏见修正
            model_ols = sm.OLS(df_clr[ct], sm.add_constant(
                pd.get_dummies(df_clr[['disease', 'tissue']], drop_first=True, dtype=float))).fit()
            s_noise = df_clr.assign(res=model_ols.resid).groupby('donor_id')['res'].std().median()
            d_var_obs = df_clr.assign(res=model_ols.resid).groupby('donor_id')['res'].mean().var()
            corrected_d_sd = np.sqrt(max(d_var_obs - (s_noise ** 2 / n_samples_per_donor), 0))
            donor_vols.append(corrected_d_sd)
            resid_vols.append(s_noise)
    
    return {
        "donor_noise_sd": float(np.nanmedian(donor_vols)),
        "sample_noise_sd": float(np.nanmedian(resid_vols))
    }

@logged
def get_all_simulation_params(df_real, collected_results, ref_disease="HC", ref_tissue="nif"):
    # A. 提取核心精确噪声
    refined_noise = get_refined_noise_estimates(df_real)
    
    # B. 计算 baseline_mu_scale (用于 LN 模拟)
    # 取参考组各细胞类型平均比例在 CLR 空间的标准差
    ref_mask = (df_real['disease'] == ref_disease) & (df_real['tissue'] == ref_tissue)
    ct_props = df_real[ref_mask].groupby('cell_type')['count'].sum() + 1
    clr_baseline = np.log(ct_props) - np.log(ct_props).mean()
    baseline_mu_scale = float(clr_baseline.std())
    
    # C. 从 DM 结果提取效应量和 alpha_sum
    dm_base = estimate_DM_parameters(collected_results)  # 你之前的重写版
    
    # 计算测序深度 (Total Count) 统计量
    sample_sums = df_real.groupby('sample_id')['count'].sum()
    total_count_mean = float(sample_sums.mean())
    total_count_sd = float(sample_sums.std())
    
    # D. 深度对齐各个字典
    n_samples_per_donor = n_samples_per_donor = int(
        np.ceil(df_real["sample_id"].nunique() / df_real["donor_id"].nunique()))
    
    # 1. 对齐 LogisticNormal 参数
    ln_params = {
        "n_donors": len(df_real['donor_id'].unique()),
        "n_samples_per_donor": n_samples_per_donor,
        "n_celltypes": len(df_real['cell_type'].unique()),
        "disease_effect_size": dm_base["disease_effect_size"],
        "tissue_effect_size": dm_base["tissue_effect_size"],
        "interaction_effect_size": dm_base.get("interaction_effect_size", 0.0),
        "baseline_mu_scale": baseline_mu_scale,
        "donor_noise_sd": refined_noise["donor_noise_sd"],
        "sample_noise_sd": refined_noise["sample_noise_sd"],
        "inflamed_cell_frac": dm_base["inflamed_cell_frac"],
        "total_count_mean": total_count_mean,
        "total_count_sd": total_count_sd,
        'disease_levels':collected_results['disease_levels']
    }
    
    # 2. 对齐 Dirichlet-Multinomial 参数
    dm_params = dm_base.copy()
    dm_params.update({
        "n_donors": len(df_real['donor_id'].unique()),
        "n_samples_per_donor": n_samples_per_donor,
        "n_celltypes": len(df_real['cell_type'].unique()),
        "donor_noise_sd": refined_noise["donor_noise_sd"],
        "sampling_bias_strength": 0.0,  # 逻辑对齐：DM 波动由 alpha_sum 承担
        "sample_size_range": (max(int(total_count_mean - total_count_sd),1000),
                              max(int(total_count_mean + total_count_sd),3000)),
        'disease_levels': collected_results['disease_levels']
    })
    
    # 3. 对齐 Resample 参数
    resample_params = {
        **refined_noise,
        "n_donors": len(df_real['donor_id'].unique()),
        "n_samples_per_donor": n_samples_per_donor,
        "n_celltypes": len(df_real['cell_type'].unique()),
        "disease_effect_size": dm_base["disease_effect_size"],
        "tissue_effect_size": dm_base["tissue_effect_size"],
        "interaction_effect_size": dm_base.get("interaction_effect_size", 0.0),
        "inflamed_cell_frac": dm_base["inflamed_cell_frac"],
        'disease_levels': collected_results['disease_levels']
    }
    
    return {"ln_params": ln_params, "dm_params": dm_params, "resample_params": resample_params}
