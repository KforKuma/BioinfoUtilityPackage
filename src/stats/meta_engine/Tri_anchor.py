from typing import Dict, Any
import pandas as pd
import numpy as np
from scipy.stats import norm

from src.stats.engine import *
from src.utils.env_utils import call_with_compatible_args
from src.utils.warnings import deprecated


def run_Meta_Ensemble(df_all: pd.DataFrame,
                      cell_type: str,
                      formula: str,
                      main_variable: str = "disease",
                      alpha: float = 0.05,
                      coef_threshold: float = 0.2,  # 降低硬门槛，依赖共识压制 FPR
                      **kwargs) -> Dict[str, Any]:
    """
    Robust-Consensus Meta Ensemble (V2):
    1. 采用 2/3 多数投票机制 (Majority Vote)，解决单方法作为“单点故障”的问题。
    2. 效应量取中位数 (Median Coef)，提高对单方法估计偏差的稳健性。
    3. 方向强一致性约束：只有显著的方法方向一致时才判定为显著。
    """
    sub_methods = {
        'dmw': run_Dirichlet_Multinomial_Wald,
        'clr': run_CLR_LMM,
        'deseq2': run_PyDESeq2
    }
    
    base_kwargs = {'df_all': df_all, 'cell_type': cell_type, 'formula': formula,
                   'main_variable': main_variable, 'alpha': alpha, **kwargs}
    
    # 1. 运行所有子方法
    results = {}
    for name, func in sub_methods.items():
        try:
            res = call_with_compatible_args(func, **base_kwargs)
            if res and isinstance(res.get('contrast_table'), pd.DataFrame) and not res['contrast_table'].empty:
                results[name] = res
            else:
                results[name] = None
        except Exception:
            results[name] = None
    
    # 2. 提取并对齐数据
    all_indices = [r['contrast_table'].index for r in results.values() if r is not None]
    if not all_indices:
        return {'contrast_table': pd.DataFrame(), 'summary': 'All Methods Failed'}
    
    # 取并集索引，确保即便某个方法没算出来，我们也能对比其他方法
    common_idx = all_indices[0].union(all_indices[1]) if len(all_indices) > 1 else all_indices[0]
    for i in range(2, len(all_indices)):
        common_idx = common_idx.union(all_indices[i])
    
    def get_standardized_data(res_obj, target_index):
        if res_obj is None:
            return (pd.Series(False, index=target_index), pd.Series(0, index=target_index),
                    pd.Series(1.0, index=target_index), pd.Series(0.0, index=target_index))
        
        df = res_obj['contrast_table'].reindex(target_index)
        sig = df['significant'].fillna(False).astype(bool)
        dir_map = {'other_greater': 1, 'ref_greater': -1}
        direction = df['direction'].map(dir_map).fillna(0).astype(int)
        pvals = df['P>|z|'].fillna(1.0)
        c_col = 'Coef.' if 'Coef.' in df.columns else 'Coef'
        coefs = df[c_col].fillna(0.0)
        return sig, direction, pvals, coefs
    
    s1, d1, p1, c1 = get_standardized_data(results.get('dmw'), common_idx)
    s2, d2, p2, c2 = get_standardized_data(results.get('clr'), common_idx)
    s3, d3, p3, c3 = get_standardized_data(results.get('deseq2'), common_idx)
    
    # 3. 核心集成逻辑
    # A. 显著性计数
    sig_count = s1.astype(int) + s2.astype(int) + s3.astype(int)
    
    # B. 方向一致性 (非常关键：只检查那些判定为显著的方法是否方向一致)
    # 计算显著方法的方向和：如果 2 个方法显著且方向一致，绝对值应为 2
    actual_dir_sum = (s1 * d1) + (s2 * d2) + (s3 * d3)
    is_direction_coherent = (actual_dir_sum.abs() == sig_count) & (sig_count > 0)
    
    # 增加共识
    is_not_dmw_veto = p1 < 0.2
    
    # 如果是叠加效应，Meta 估计的 Coef. 符号必须与单方法中最显著的那个一致
    anchor_dir = pd.Series(np.where(p2 < p3, d2, d3), index=common_idx)
    
    # 4. P值聚合：采用 Stouffer's 思想的简化版 (中位数 P 在集成中通常表现最稳)
    # 或者用极小值 P (如果你追求 Power)
    combined_p = pd.concat([p1, p2, p3], axis=1).median(axis=1)
    
    # 5. 构造结果表
    meta_dir_val = actual_dir_sum.apply(np.sign).astype(int)
    rev_map = {1: 'other_greater', -1: 'ref_greater', 0: 'None'}
    
    # C. 效应量中位数 (比单用 DMW 更稳健)
    median_coef = pd.concat([c1, c2, c3], axis=1).median(axis=1)
    
    # D. 最终判定：多数原则 (>=2) 且 方向一致 且 满足最小效应门槛
    # 不再强制要求 DMW (s1) 必须为 True
    meta_significant = (
            (sig_count >= 2) &  # 多数投票
            is_direction_coherent &  # 方向一致
            (median_coef.abs() >= coef_threshold) &
            is_not_dmw_veto &
            (combined_p < alpha) &
            (meta_dir_val == anchor_dir)  # 集成方向必须与最可靠的单方法方向一致
    )
    
    
    # 寻找一个非空的参考列
    ref_col = "Unknown"
    for r in results.values():
        if r is not None:
            ref_col = r['contrast_table']['ref'].iloc[0]
            break
    
    meta_table = pd.DataFrame({
        'ref': ref_col,
        'Coef.': median_coef,
        'P>|z|': combined_p,
        'direction': meta_dir_val.map(rev_map),
        'significant': meta_significant,
        'method_agreement': sig_count
    }, index=common_idx)
    
    return {
        'contrast_table': meta_table,
        'summary': f"Consensus Meta. Hits: {meta_significant.sum()}, Agreement: {sig_count.mean():.2f}",
        'raw_results': results
    }


def run_Meta_Ensemble_adaptive(df_all: pd.DataFrame,
                               cell_type: str,
                               formula: str,
                               main_variable: str = "disease",
                               alpha: float = 0.05,
                               **kwargs) -> Dict[str, Any]:
    """
    Robust-Consensus Meta Ensemble (V2):
    1. 采用 2/3 多数投票机制 (Majority Vote)，解决单方法作为“单点故障”的问题。
    2. 效应量取中位数 (Median Coef)，提高对单方法估计偏差的稳健性。
    3. 方向强一致性约束：只有显著的方法方向一致时才判定为显著。
    """
    sub_methods = {
        'dmw': run_Dirichlet_Multinomial_Wald,
        'clr': run_CLR_LMM,
        'deseq2': run_PyDESeq2
    }
    
    base_kwargs = {'df_all': df_all, 'cell_type': cell_type, 'formula': formula,
                   'main_variable': main_variable, 'alpha': alpha, **kwargs}
    
    # 1. 运行所有子方法
    results = {}
    for name, func in sub_methods.items():
        try:
            res = call_with_compatible_args(func, **base_kwargs)
            if res and isinstance(res.get('contrast_table'), pd.DataFrame) and not res['contrast_table'].empty:
                results[name] = res
            else:
                results[name] = None
        except Exception:
            results[name] = None
    
    # 2. 提取并对齐数据
    all_indices = [r['contrast_table'].index for r in results.values() if r is not None]
    if not all_indices:
        return {'contrast_table': pd.DataFrame(), 'summary': 'All Methods Failed'}
    
    # 取并集索引，确保即便某个方法没算出来，我们也能对比其他方法
    common_idx = all_indices[0].union(all_indices[1]) if len(all_indices) > 1 else all_indices[0]
    for i in range(2, len(all_indices)):
        common_idx = common_idx.union(all_indices[i])
    
    def get_standardized_data(res_obj, target_index):
        if res_obj is None:
            return (pd.Series(False, index=target_index), pd.Series(0, index=target_index),
                    pd.Series(1.0, index=target_index), pd.Series(0.0, index=target_index))
        
        df = res_obj['contrast_table'].reindex(target_index)
        sig = df['significant'].fillna(False).astype(bool)
        dir_map = {'other_greater': 1, 'ref_greater': -1}
        direction = df['direction'].map(dir_map).fillna(0).astype(int)
        pvals = df['P>|z|'].fillna(1.0)
        c_col = 'Coef.' if 'Coef.' in df.columns else 'Coef'
        coefs = df[c_col].fillna(0.0)
        return sig, direction, pvals, coefs
    
    s1, d1, p1, c1 = get_standardized_data(results.get('dmw'), common_idx)
    s2, d2, p2, c2 = get_standardized_data(results.get('clr'), common_idx)
    s3, d3, p3, c3 = get_standardized_data(results.get('deseq2'), common_idx)
    
    # 3. 核心集成逻辑
    # A. 显著性计数
    sig_count = s1.astype(int) + s2.astype(int) + s3.astype(int)
    
    # B. 方向一致性 (非常关键：只检查那些判定为显著的方法是否方向一致)
    # 计算显著方法的方向和：如果 2 个方法显著且方向一致，绝对值应为 2
    actual_dir_sum = (s1 * d1) + (s2 * d2) + (s3 * d3)
    is_direction_coherent = (actual_dir_sum.abs() == sig_count) & (sig_count > 0)
    
    # C. 效应量中位数 (比单用 DMW 更稳健)
    median_coef = pd.concat([c1, c2, c3], axis=1).median(axis=1)
    
    # 计算全局中位绝对偏差 (Median Absolute Deviation, MAD)
    # MAD 是比标准差更稳健的离散度衡量
    all_coefs = pd.concat([c1, c2, c3])
    data_scale = all_coefs.abs().median()
    
    # 动态设置门槛：基础门槛 + 比例增益
    # 基础值 0.1 保证低波动下的敏感度，0.5 * data_scale 保证高波动下的拦截力
    dynamic_threshold = 0.1 + 0.3 * data_scale
    
    # 限制门槛范围，防止极端情况下门槛过高或过低
    dynamic_threshold = np.clip(dynamic_threshold, 0.15, 0.8)
    
    # 增加共识
    is_not_dmw_veto = p1 < 0.2
    
    # 如果是叠加效应，Meta 估计的 Coef. 符号必须与单方法中最显著的那个一致
    anchor_dir = pd.Series(np.where(p2 < p3, d2, d3), index=common_idx)
    
    # 4. P值聚合：采用 Stouffer's 思想的简化版 (中位数 P 在集成中通常表现最稳)
    # 或者用极小值 P (如果你追求 Power)
    combined_p = pd.concat([p1, p2, p3], axis=1).median(axis=1)
    
    
    # 5. 构造结果表
    meta_dir_val = actual_dir_sum.apply(np.sign).astype(int)
    rev_map = {1: 'other_greater', -1: 'ref_greater', 0: 'None'}
    
    
    # 6. 最终判定：多数原则 (>=2) 且 方向一致 且 满足最小效应门槛
    # 不再强制要求 DMW (s1) 必须为 True
    meta_significant = (
            (sig_count >= 2) &  # 多数投票
            is_direction_coherent &  # 方向一致
            (median_coef.abs() >= dynamic_threshold) &
            is_not_dmw_veto &
            (combined_p < alpha) &
            (meta_dir_val == anchor_dir)  # 集成方向必须与最可靠的单方法方向一致
    )
    
    # 寻找一个非空的参考列
    ref_col = "Unknown"
    for r in results.values():
        if r is not None:
            ref_col = r['contrast_table']['ref'].iloc[0]
            break
    
    meta_table = pd.DataFrame({
        'ref': ref_col,
        'Coef.': median_coef,
        'P>|z|': combined_p,
        'direction': meta_dir_val.map(rev_map),
        'significant': meta_significant,
        'method_agreement': sig_count
    }, index=common_idx)
    
    return {
        'contrast_table': meta_table,
        'summary': f"Consensus Meta. Hits: {meta_significant.sum()}, Agreement: {sig_count.mean():.2f}",
        'raw_results': results
    }


@deprecated(alternative="run_Meta_Ensemble_adaptive")
def run_Meta_Ensemble_dynamic(
        df_all: pd.DataFrame,
        cell_type: str,
        formula: str,
        main_variable: str = "disease",
        alpha: float = 0.05,
        coef_threshold: float = 0.2,
        # 新增动态超参数
        k_penalty: float = 2.0,  # 功能1：P值惩罚强度（越小越保守）
        inflation_factor: float = 1.0,  # 功能2：手动或预估的基因组膨胀因子 lambda
        diversity_weight: float = 0.5,  # 功能3：分歧度惩罚权重
        **kwargs
) -> Dict[str, Any]:
    """
    Advanced Dynamic Meta Ensemble (V3):
    1. Soft Coef-Penalty: 基于效应量的P值软惩罚，保护显著且高效应的信号。
    2. Dispersion Correction: 引入类似 Genomic Control 的思想修正过离散导致的P值通胀。
    3. Consensus Diversity Shrinkage: 基于方法间分歧度（CV）修正共识性假阳性。
    """
    sub_methods = {
        'dmw': run_Dirichlet_Multinomial_Wald,
        'clr': run_CLR_LMM,
        'deseq2': run_PyDESeq2
    }
    
    base_kwargs = {'df_all': df_all, 'cell_type': cell_type, 'formula': formula,
                   'main_variable': main_variable, 'alpha': alpha, **kwargs}
    
    # 1. 运行所有子方法
    results = {}
    for name, func in sub_methods.items():
        try:
            res = call_with_compatible_args(func, **base_kwargs)
            if res and isinstance(res.get('contrast_table'), pd.DataFrame) and not res['contrast_table'].empty:
                results[name] = res
            else:
                results[name] = None
        except Exception:
            results[name] = None
    
    # 2. 提取并对齐数据
    all_indices = [r['contrast_table'].index for r in results.values() if r is not None]
    if not all_indices:
        return {'contrast_table': pd.DataFrame(), 'summary': 'All Methods Failed'}
    
    # 取并集索引，确保即便某个方法没算出来，我们也能对比其他方法
    common_idx = all_indices[0].union(all_indices[1]) if len(all_indices) > 1 else all_indices[0]
    for i in range(2, len(all_indices)):
        common_idx = common_idx.union(all_indices[i])
    
    def get_standardized_data(res_obj, target_index):
        if res_obj is None:
            return (pd.Series(False, index=target_index), pd.Series(0, index=target_index),
                    pd.Series(1.0, index=target_index), pd.Series(0.0, index=target_index))
        
        df = res_obj['contrast_table'].reindex(target_index)
        sig = df['significant'].fillna(False).astype(bool)
        dir_map = {'other_greater': 1, 'ref_greater': -1}
        direction = df['direction'].map(dir_map).fillna(0).astype(int)
        pvals = df['P>|z|'].fillna(1.0)
        c_col = 'Coef.' if 'Coef.' in df.columns else 'Coef'
        coefs = df[c_col].fillna(0.0)
        return sig, direction, pvals, coefs
    
    s1, d1, p1, c1 = get_standardized_data(results.get('dmw'), common_idx)
    s2, d2, p2, c2 = get_standardized_data(results.get('clr'), common_idx)
    s3, d3, p3, c3 = get_standardized_data(results.get('deseq2'), common_idx)
    
    # --- 功能 1: 基于 coef_threshold 的 P 值软惩罚 (Conservative Design) ---
    # 目的：在大样本下，如果效应量达不到门槛，即使P值很小也要拉高它。
    # 采用指数缓冲函数：如果 |beta| >= threshold，惩罚为1；如果越小，惩罚越大。
    median_coef = pd.concat([c1, c2, c3], axis=1).median(axis=1)
    abs_beta = median_coef.abs()
    
    # 只有当 abs_beta < coef_threshold 时才触发惩罚
    # penalty = exp( k * (threshold - |beta|) )，且最小为1
    soft_penalty = np.exp(np.maximum(0, k_penalty * (coef_threshold - abs_beta)))
    
    # --- 功能 2: 基于过离散/膨胀的校正 (Empirical Null / Lambda Correction) ---
    # 目的：模拟高 scale_factor 下零假设分布变宽的情况。
    # 如果外部传入了 inflation_factor (lambda > 1)，则修正 Z-score
    def adjust_p_by_lambda(p_series, lam):
        if lam <= 1.0: return p_series
        # 将 P 换算回 Z，缩小 Z 后再换回 P
        z = norm.ppf(1 - p_series / 2)
        z_adj = z / np.sqrt(lam)
        return 2 * norm.sf(np.abs(z_adj))
    
    p1_adj = adjust_p_by_lambda(p1, inflation_factor)
    p2_adj = adjust_p_by_lambda(p2, inflation_factor)
    p3_adj = adjust_p_by_lambda(p3, inflation_factor)
    
    # --- 功能 3: P 值的“共识多样性”收缩 (Consensus Diversity Shrinkage) ---
    # 目的：如果三个方法 $P$ 值高度一致且都很小，警惕系统性偏误。
    # 计算 Z-score 的变异系数 (CV)
    z_matrix = np.array([norm.ppf(1 - p_adj.clip(upper=0.999) / 2) for p_adj in [p1_adj, p2_adj, p3_adj]])
    z_mean = np.mean(z_matrix, axis=0)
    z_std = np.std(z_matrix, axis=0)
    # CV 越小（一致性越高），惩罚因子越大
    # 当一致性极高时，我们将 P 值向中值方向收缩
    cv = z_std / (np.abs(z_mean) + 1e-6)
    diversity_penalty = 1 + diversity_weight * np.exp(-cv * 3)  # CV越小，penalty越高
    
    # 3. 核心集成逻辑
    # A. 显著性计数
    sig_count = (pd.concat([p1_adj, p2_adj, p3_adj], axis=1) < alpha).sum(axis=1)
    
    # B. 方向一致性 (非常关键：只检查那些判定为显著的方法是否方向一致)
    actual_dir_sum = (s1 * d1) + (s2 * d2) + (s3 * d3)
    is_direction_coherent = (actual_dir_sum.abs() == sig_count) & (sig_count > 0)
    
    # 如果是叠加效应，Meta 估计的 Coef. 符号必须与单方法中最显著的那个一致
    anchor_dir = pd.Series(np.where(p2_adj < p3_adj, d2, d3), index=p1.index)
    
    # 4. P值聚合：采用 Stouffer's 思想的简化版 (中位数 P 在集成中通常表现最稳)
    # 或者用极小值 P (如果你追求 Power)
    combined_p_raw = pd.concat([p1_adj, p2_adj, p3_adj], axis=1).median(axis=1)
    
    # 应用功能 1 和 功能 3 的联合惩罚
    final_p = combined_p_raw * soft_penalty * diversity_penalty
    final_p = final_p.clip(upper=1.0)
    
    # 5. 构造结果表
    meta_dir_val = actual_dir_sum.apply(np.sign).astype(int)
    rev_map = {1: 'other_greater', -1: 'ref_greater', 0: 'None'}
    
    
    
    meta_significant = (
            (sig_count >= 2) &
            is_direction_coherent &
            (final_p < alpha) &  # 使用校正后的 final_p
            (abs_beta >= 0.1) &  # 保留一个极小的底线门槛
            (meta_dir_val == anchor_dir)  # 方向锁
    )
    
    # 寻找一个非空的参考列
    ref_col = "Unknown"
    for r in results.values():
        if r is not None:
            ref_col = r['contrast_table']['ref'].iloc[0]
            break

    meta_table = pd.DataFrame({
        'ref': ref_col,
        'Coef.': median_coef,
        'P>|z|': final_p,
        'direction': meta_dir_val.map(rev_map),
        'significant': meta_significant,
        'method_agreement': sig_count
    }, index=common_idx)
    
    return {
        'contrast_table': meta_table,
        'summary': f"Consensus Meta. Hits: {meta_significant.sum()}, Agreement: {sig_count.mean():.2f}",
        'raw_results': results
    }
