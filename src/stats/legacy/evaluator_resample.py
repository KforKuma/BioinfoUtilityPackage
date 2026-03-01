from __future__ import annotations
import warnings
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


#################
@logged
def estimate_resample_params(
        df_real: pd.DataFrame,
        disease_ref: str = "HC",
        tissue_ref: str = "nif",
        min_abundance: float = 0.01,
        pseudocount: float = 1.0,
        quantile_threshold=0.8
) -> dict:
    params = {}
    
    # ==========================================================================
    # 1. 数据预处理与 CLR 空间转换 (保持不变)
    # ==========================================================================
    total_counts = df_real.groupby('cell_type')['count'].sum()
    rel_abundance = total_counts / total_counts.sum()
    reliable_cts = rel_abundance[rel_abundance > min_abundance].index.tolist()
    
    if len(reliable_cts) < 3:
        reliable_cts = rel_abundance.nlargest(5).index.tolist()
    
    df_wide = df_real.pivot_table(
        index=['donor_id', 'sample_id', 'disease', 'tissue'],
        columns='cell_type',
        values='count',
        fill_value=0
    )[reliable_cts].reset_index()
    
    data_values = df_wide[reliable_cts].values
    log_data = np.log(data_values + pseudocount)
    clr_data = log_data - log_data.mean(axis=1, keepdims=True)
    
    df_clr = df_wide[['donor_id', 'sample_id', 'disease', 'tissue']].copy()
    df_clr[reliable_cts] = clr_data
    
    # ==========================================================================
    # 2. 构建回归模型 (对齐 LMM 与 OLS)
    # ==========================================================================
    meta = df_clr[['disease', 'tissue']].copy()
    meta['disease'] = pd.Categorical(meta['disease'], categories=[disease_ref] + [c for c in meta['disease'].unique() if
                                                                                  c != disease_ref])
    meta['tissue'] = pd.Categorical(meta['tissue'],
                                    categories=[tissue_ref] + [c for c in meta['tissue'].unique() if c != tissue_ref])
    
    X = pd.get_dummies(meta, drop_first=True, dtype=float)
    X = sm.add_constant(X)
    
    results_list = []
    donor_vols = []
    resid_vols = []
    
    # 计算每个 Donor 的平均样本数，用于 OLS 下的偏差修正
    # n_k = harmonic mean 效果最好，这里简化使用平均值
    n_samples_per_donor = df_clr.groupby('donor_id').size().mean()
    
    # 我们改为逐个 cell_type 尝试 LMM，如果失败则该 cell_type 退回 OLS
    # 这样可以保留大部分 LMM 的精确估计，而不是因为一个 cell_type 报错就全盘放弃
    for ct in reliable_cts:
        lmm_success = False
        try:
            formula = (
                f"Q('{ct}') ~ "
                f"C(disease, Treatment('{disease_ref}')) + "
                f"C(tissue, Treatment('{tissue_ref}')) + "
                f"C(disease, Treatment('{disease_ref}')):"
                f"C(tissue, Treatment('{tissue_ref}'))"
            )
            
            model_lmm = smf.mixedlm(formula, df_clr, groups=df_clr["donor_id"]).fit(
                reml=True, method="lbfgs", maxiter=200, disp=False
            )
            
            # 提取噪声
            d_var = model_lmm.cov_re.iloc[0, 0]
            r_var = model_lmm.scale
            donor_vols.append(np.sqrt(max(d_var, 0)))
            resid_vols.append(np.sqrt(max(r_var, 0)))
            
            # 提取系数
            coefs, pvals = model_lmm.params, model_lmm.pvalues
            lmm_success = True
        
        except Exception as e:
            # --- OLS Fallback Branch ---
            model_ols = sm.OLS(df_clr[ct], X).fit()
            resids = model_ols.resid
            temp_df = pd.DataFrame({'resid': resids, 'donor_id': df_clr['donor_id']})
            
            # 1. 样本噪声 (Within-donor SD)
            # 在 LMM 语义下，这是 residual scale
            s_noise_ct = temp_df.groupby('donor_id')['resid'].std(ddof=1).median()
            resid_vols.append(s_noise_ct)
            
            # 2. 供体噪声 (Between-donor SD 对齐公式)
            # Var(means) = Var(donor) + Var(sample)/n
            # 因此 Var(donor) = Var(means) - Var(sample)/n
            donor_means_var = temp_df.groupby('donor_id')['resid'].mean().var()
            corrected_d_var = donor_means_var - (s_noise_ct ** 2 / n_samples_per_donor)
            donor_vols.append(np.sqrt(max(corrected_d_var, 0)))
            
            coefs, pvals = model_ols.params, model_ols.pvalues
        
        # 统一收集统计量
        coef_map = {
            'cell_type': ct,
            'disease_coef': next((v for k, v in coefs.items() if 'disease' in k and ':' not in k), 0.0),
            'tissue_coef': next((v for k, v in coefs.items() if 'tissue' in k and ':' not in k), 0.0),
            'inter_coef': next((v for k, v in coefs.items() if ':' in k), 0.0),
            'disease_p': next((v for k, v in pvals.items() if 'disease' in k and ':' not in k), 1.0),
            'tissue_p': next((v for k, v in pvals.items() if 'tissue' in k and ':' not in k), 1.0),
            'inter_p': next((v for k, v in pvals.items() if ':' in k), 1.0)
        }
        results_list.append(coef_map)
    
    df_stats = pd.DataFrame(results_list)
    params['donor_noise_sd'] = float(np.nanmedian(donor_vols))
    params['sample_noise_sd'] = float(np.nanmedian(resid_vols))
    
    # ==========================================================================
    # 3. 效应量与流行度估计 (维持你的逻辑)
    # ==========================================================================
    def summarize_factor(coef_col, p_col, q_thresh):
        abs_coefs = df_stats[coef_col].abs()
        effect_size = abs_coefs.quantile(q_thresh)
        sig_mask = df_stats[p_col] < 0.15
        frac = sig_mask.mean()
        return (float(effect_size) if frac >= 0.05 else 0.0), float(frac)
    
    params['disease_effect_size'], _ = summarize_factor('disease_coef', 'disease_p', quantile_threshold)
    params['tissue_effect_size'], t_frac = summarize_factor('tissue_coef', 'tissue_p', quantile_threshold)
    params['interaction_effect_size'], _ = summarize_factor('inter_coef', 'inter_p', quantile_threshold)
    params['inflamed_cell_frac'] = max(t_frac, 0.1)
    
    # ==========================================================================
    # 4. 打印报告
    # ==========================================================================
    print("\n" + "Integrated Residual Analysis (LMM/OLS Hybrid)".center(50, "="))
    for k, v in params.items():
        print(f"{k:25s}: {v:.4f}")
    print("=" * 50)
    
    return params