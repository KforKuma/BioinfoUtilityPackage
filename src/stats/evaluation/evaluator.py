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
def get_refined_noise_estimates(df_real: pd.DataFrame) -> Dict[str, float]:
    """从真实丰度数据中估计 donor 和 sample 层面的 CLR 噪声。

    函数先将长表 count 转换为 sample x cell_type 宽表，再进入 CLR 空间。
    对每个 cell subtype/subpopulation 优先拟合 ``CLR ~ disease + tissue + (1|donor)``
    的混合模型；如果 LMM 失败，则回退到 OLS 残差，并对 donor 方差做一个保守修正。

    Args:
        df_real: 真实长表丰度数据，至少包含 ``donor_id``、``sample_id``、
            ``disease``、``tissue``、``cell_type`` 和 ``count``。

    Returns:
        包含 ``donor_noise_sd`` 和 ``sample_noise_sd`` 的字典，可直接传给
        simulation 模块的模拟函数。

    Example:
        >>> noise = get_refined_noise_estimates(real_count_df)
        >>> noise["donor_noise_sd"], noise["sample_noise_sd"]
        # 用于 simulate_LogisticNormal_hierarchical 或 simulate_CLR_resample_data。
    """
    required_cols = {"donor_id", "sample_id", "disease", "tissue", "cell_type", "count"}
    missing_cols = required_cols - set(df_real.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    # 先进入 CLR 空间，是为了让噪声估计与 CLR/LMM 类方法的模拟尺度一致。
    df_wide = df_real.pivot_table(index=['donor_id', 'sample_id', 'disease', 'tissue'],
                                  columns='cell_type', values='count', fill_value=0)
    if df_wide.empty:
        raise ValueError("Input data contains no count rows after pivoting.")

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
        except Exception as e:
            # OLS Fallback 并进行方差偏见修正
            # LMM 在小样本或奇异方差时容易失败，OLS 残差提供可继续模拟的保守估计。
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
    """汇总三类模拟函数所需的参数字典。

    Args:
        df_real: 真实长表丰度数据。
        collected_results: ``collect_DM_results`` 的输出，至少包含 ``all_coefs`` 和
            ``disease_levels``。
        ref_disease: 参考 disease 水平，默认 ``"HC"``。
        ref_tissue: 参考 tissue 水平，默认 ``"nif"``。

    Returns:
        包含 ``ln_params``、``dm_params`` 和 ``resample_params`` 的字典，分别对应
        Logistic-Normal、Dirichlet-Multinomial 和 CLR resampling 模拟函数。

    Example:
        >>> collected = collect_DM_results(count_df, cell_types, run_Dirichlet_Multinomial_Wald)
        >>> params = get_all_simulation_params(count_df, collected)
        >>> simulate_DM_data(**params["dm_params"])
        # 生成与真实数据噪声和效应量大致对齐的模拟丰度长表。
    """
    required_cols = {"donor_id", "sample_id", "disease", "tissue", "cell_type", "count"}
    missing_cols = required_cols - set(df_real.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")
    if "disease_levels" not in collected_results:
        raise ValueError("Missing required key in `collected_results`: 'disease_levels'.")

    refined_noise = get_refined_noise_estimates(df_real)
    
    # B. 计算 baseline_mu_scale (用于 LN 模拟)
    # 取参考组各细胞类型平均比例在 CLR 空间的标准差
    ref_mask = (df_real['disease'] == ref_disease) & (df_real['tissue'] == ref_tissue)
    ct_props = df_real[ref_mask].groupby('cell_type')['count'].sum() + 1
    if ct_props.empty:
        raise ValueError(f"No reference samples found for disease: '{ref_disease}' and tissue: '{ref_tissue}'.")
    clr_baseline = np.log(ct_props) - np.log(ct_props).mean()
    baseline_mu_scale = float(clr_baseline.std())
    
    # C. 从 DM 结果提取效应量和 alpha_sum
    dm_base = estimate_DM_parameters(collected_results)  # 你之前的重写版
    
    # 计算测序深度 (Total Count) 统计量
    sample_sums = df_real.groupby('sample_id')['count'].sum()
    total_count_mean = float(sample_sums.mean())
    total_count_sd = min(float(sample_sums.std()), 500)
    
    # D. 深度对齐各个字典
    n_donors = df_real["donor_id"].nunique()
    if n_donors == 0:
        raise ValueError("No donor_id values found in input data.")
    n_samples_per_donor = int(np.ceil(df_real["sample_id"].nunique() / n_donors))
    
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
        "total_count_mean": total_count_mean,
        "total_count_sd": total_count_sd,
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
