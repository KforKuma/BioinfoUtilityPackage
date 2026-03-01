# from typing import Dict, List, Tuple
# import re
# import warnings
from tqdm import tqdm

import inspect
import logging

import numpy as np
import pandas as pd

from src.stats.support import *
from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)



def _calculate_performance_metrics(df_all_sims: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    计算基于多次模拟结果的性能指标 (Power, FPR)。
    """
    # --- 新增：强制类型转换，确保是布尔型 ---
    for col in ['True_Significant', 'Est_Significant', 'Est_PValue']:
        if col in df_all_sims.columns:
            if col == 'Est_PValue':
                df_all_sims[col] = pd.to_numeric(df_all_sims[col], errors='coerce')
            else:
                # 处理可能存在的字符串 "True"/"False"
                df_all_sims[col] = df_all_sims[col].map({'True': True, 'False': False, True: True, False: False})
                # 填充 NaN 并强制转为 bool
                df_all_sims[col] = df_all_sims[col].fillna(False).astype(bool)
    
    # 如果 Est_Significant 已经存在（由包装函数 _collect_simulation_results 计算好了），则直接使用它
    if 'Est_Significant' in df_all_sims.columns:
        df_all_sims['Est_Significant_Alpha'] = df_all_sims['Est_Significant']
    else:
        # 否则才根据 p-value 重新计算
        df_all_sims['Est_Significant_Alpha'] = (df_all_sims['Est_PValue'] <= alpha)
    
    # 分类为 TP, FP, TN, FN
    # TP = (df_all_sims['True_Significant']) & (df_all_sims['Est_Significant_Alpha']).sum()
    # FP = (~df_all_sims['True_Significant']) & (df_all_sims['Est_Significant_Alpha']).sum()
    # TN = (~df_all_sims['True_Significant']) & (~df_all_sims['Est_Significant_Alpha']).sum()
    # FN = (df_all_sims['True_Significant']) & (~df_all_sims['Est_Significant_Alpha']).sum()
    
    # 按对比因素计算指标
    metrics = df_all_sims.groupby('contrast_factor').apply(lambda g: pd.Series({
        'TP': ((g['True_Significant']) & (g['Est_Significant_Alpha'])).sum(),
        'FP': ((~g['True_Significant']) & (g['Est_Significant_Alpha'])).sum(),
        'FN': ((g['True_Significant']) & (~g['Est_Significant_Alpha'])).sum(),
    })).reset_index()
    
    metrics['Power'] = metrics['TP'] / (metrics['TP'] + metrics['FN'])
    metrics['FPR'] = metrics['FP'] / (metrics['TP'] + metrics['FP'])
    
    # 处理除以零的情况
    metrics['Power'] = metrics['Power'].fillna(0)
    metrics['FPR'] = metrics['FPR'].fillna(0)
    
    return metrics


def _collect_simulation_results(
        df_sim: pd.DataFrame,
        df_true_effect: pd.DataFrame,
        run_stats_func,  # 传入的统计运行函数，例如 run_Dirichlet_Wald
        formula: str,
        **kwargs
) -> pd.DataFrame:
    """
    收集单个模拟数据集的统计结果，并与真实效应表合并。

    Args:
        df_sim: 模拟的计数数据 (长格式)。
        df_true_effect: 模拟的真实效应查找表。
        run_stats_func: 实际运行统计分析的函数 (如 run_Dirichlet_Wald)。
        formula: 传递给统计函数的模型公式 (e.g., "disease + C(tissue, ...)")。

    Returns:
        DataFrame, 包含所有细胞类型、所有对比的真实效应和统计估计值。
    """
    
    # 获取唯一的细胞类型列表
    cell_types = df_sim['cell_type'].unique().tolist()
    
    # 真实效应表预处理: 确保 True_Significant 基于 alpha
    df_true_effect['True_Significant'] = (df_true_effect['True_Effect'] != 0)
    
    # 存储所有对比结果
    all_results = []
    
    for ct_name in cell_types:
        try:
            # 1. 运行统计模型
            # 假设 run_stats_func(df_all, cell_type, formula) 返回结构化的结果
            stats_res = run_stats_func(df_all=df_sim, cell_type=ct_name, formula=formula, **kwargs)
            contrast_table = stats_res["contrast_table"]
        
        except Exception as e:
            # 如果统计分析失败，记录错误并跳过该细胞类型
            print(f"Warning: Stats failed for {ct_name}. Error: {e}")
            continue
        
        # 2. 提取该细胞类型的真实效应行
        df_true_ct = df_true_effect[df_true_effect['cell_type'] == ct_name].copy()
        
        # 3. 匹配真实效应和统计估计值
        for _, true_row in df_true_ct.iterrows():
            contrast_factor = true_row['contrast_factor']
            
            # 根据 Fallback 规则确定要查找的 'other' 组名称
            if contrast_factor == 'tissue':
                # Rule: contrast_factor=tissue 对应 other='if'
                target_other = true_row['contrast_group']  # 'if'
            elif contrast_factor in ('disease', 'interaction'):
                # Rule: disease/interaction 对应 other=疾病名称
                # 从 'UC x if' 或 'CD x if' 中提取疾病名称 'UC' 或 'CD'
                target_other_full = true_row['contrast_group']
                target_other = target_other_full.split(' ')[0]  # 'CD' 或 'UC'
            
            # 提取统计结果
            est_results = _extract_contrast_results(contrast_table, target_other)
            
            # 组合结果
            result_record = {
                'cell_type': ct_name,
                'contrast_factor': contrast_factor,
                'contrast_group': true_row['contrast_group'],
                'contrast_ref': true_row['contrast_ref'],
                'True_Effect': true_row['True_Effect'],
                'True_Direction': true_row['True_Direction'],
                'True_Significant': true_row['True_Significant'],
                **est_results
            }
            all_results.append(result_record)
    
    return pd.DataFrame(all_results)


def _extract_contrast_results(contrast_table: pd.DataFrame, target_other: str, alpha: float = 0.05) -> dict:
    """
    从统计模型的 contrast_table 中提取特定对比的结果 (Coef, PValue, direction, significant)。
    已修正索引查找逻辑，并增加了 P 值列名的 fallback 机制 (P>|z| -> p_adj -> pval)。
    """
    
    # 1. 重置索引以便按列名访问 'other'
    df_reset = contrast_table.reset_index()
    
    # 2. 使用布尔索引查找目标行
    result_rows = df_reset[df_reset['other'] == target_other]
    
    if result_rows.empty:
        # 如果找不到匹配的对比，返回默认值
        return {
            'Est_Coef': np.nan,
            'Est_PValue': np.nan,
            'Est_Direction': 'None',
            'Est_Significant': False
        }
    
    # 3. 提取结果 (只取第一行匹配项)
    result_row = result_rows.iloc[0]
    
    # 4. 确定 P 值列名 (Fallback 逻辑)
    pval_colname = None
    # 定义优先级：P>|z| > p_adj > pval
    pval_candidates = ['P>|z|', 'p_adj', 'pval']
    
    # 检查哪些候选列存在于当前的 DataFrame 中
    existing_cols = result_rows.columns  # 在 DataFrame (result_rows) 上检查 .columns 是正确的
    
    for col in pval_candidates:
        if col in existing_cols:
            pval_colname = col
            break
    
    # 5. 提取 P 值和显著性
    est_pval = result_row[pval_colname] if pval_colname else np.nan
    
    # 由于您的统计输出中已经有了 'significant' 列，我们优先使用它。
    # 如果没有 'significant' 列，则基于 P 值和 alpha 重新计算。
    if 'significant' in existing_cols:
        est_significant = result_row['significant']
    elif not np.isnan(est_pval):
        est_significant = (est_pval <= alpha)
    else:
        est_significant = False
    
    # Coef 列和 direction 列通常存在
    est_coef = next(
        (result_row[c] for c in existing_cols if "coef" in c.lower()),
        np.nan
    )
    
    est_direction = result_row['direction'] if 'direction' in existing_cols else 'None'
    
    return {
        'Est_Coef': est_coef,
        'Est_PValue': est_pval,
        'Est_Direction': est_direction,
        'Est_Significant': est_significant
    }


@logged
def evaluate_effect_size_scaling(
        scale_factors=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0],
        sim_func=None,  # 新增：模拟函数入口
        run_stats_func=None,
        formula="disease + C(tissue, Treatment(reference='nif'))",
        base_params=None,
        **kwargs
):
    """
    连续调整 effect_size 并收集性能指标的循环函数。
    支持自定义模拟器 sim_func。
    """
    
    
    # 1. 初始化模拟器和基础参数
    if sim_func is None:
        raise ValueError("Please provide a simulation function (sim_func).")
    
    if base_params is None:
        # 这是一个通用的基础模板，会根据 sim_func 的需求自动过滤
        base_params = {
            "n_donors": 20,
            "n_samples_per_donor": 4,
            "n_celltypes": 50,
            "baseline_alpha_scale": 51,  # DM 专用
            "baseline_mu_scale": 1.0,  # LN 专用
            "disease_effect_size": 0.5,
            "tissue_effect_size": 0.8,
            "interaction_effect_size": 0.5,
            "inflamed_cell_frac": 0.1,
            "disease_levels": ["HC", "BD", "CD", "Colitis", "UC"],
            "tissue_levels": ("nif", "if"),
            "random_state": 710
        }
    
    all_metrics = []
    print(f"Starting evaluation: Sim[{sim_func.__name__}] -> Stats[{run_stats_func.__name__}]")
    
    for k in tqdm(scale_factors):
        # 2. 整体缩放 effect_size
        current_params = base_params.copy()
        # 动态检测并缩放所有包含 'effect_size' 的键
        for key in current_params:
            if "effect_size" in key:
                current_params[key] *= k
        
        # 3. 生成模拟数据 (根据 sim_func 的签名自动过滤参数)
        sim_filtered_params = filter_kwargs_for_func(sim_func, current_params)
        df_sim, df_true_effect = sim_func(**sim_filtered_params)
        
        # 4. 运行统计检验 (同样自动过滤统计函数参数)
        # 这里合并了 base_params 和用户通过 **kwargs 传入的额外参数（如 coef_threshold）
        full_stats_params = {**current_params, **kwargs}
        stats_filtered_kwargs = filter_kwargs_for_func(run_stats_func, full_stats_params)
        
        print(f"\n[Scale {k}] Params for {run_stats_func.__name__}: {stats_filtered_kwargs}")
        
        results_df = _collect_simulation_results(
            df_sim=df_sim,
            df_true_effect=df_true_effect,
            run_stats_func=run_stats_func,
            formula=formula,
            **stats_filtered_kwargs
        )
        
        # 5. 计算性能指标
        metrics = _calculate_performance_metrics(results_df, alpha=0.05)
        
        # 6. 记录当前的倍数因子
        metrics['scale_factor'] = k
        all_metrics.append(metrics)
    
    # 合并所有结果
    final_df = pd.concat(all_metrics, ignore_index=True)
    return final_df


@logged
def evaluate_effect_size_scaling_with_raw(
        scale_factors,
        sim_func,
        run_stats_func,
        sim_params,
        stats_params,
        formula,
):
    """
    Evaluate statistical performance across effect size scaling factors.

    Returns
    -------
    summary_df : pd.DataFrame
        Power / FPR summary across scales
    raw_df : pd.DataFrame
        Per-gene / per-cell raw results for PPV analysis
    """
    
    all_summary_metrics = []
    all_raw_results = []
    
    print(f"Starting Effect Size Scaling Evaluation: {sim_func.__name__}")
    
    if not any("effect_size" in key for key in sim_params.keys()):
        raise ValueError("Base 'effect_size' must be provided.")
    
    for k in tqdm(scale_factors):
        # 1. 显式缩放 effect sizes
        current_params = sim_params.copy()
        for key in current_params:
            if "effect_size" in key:
                current_params[key] *= k
        
        # 2. 生成模拟数据
        df_sim, df_true_effect = sim_func(
            **current_params,
        )
        
        # 3. 运行统计模型
        results_df = _collect_simulation_results(
            df_sim=df_sim,
            df_true_effect=df_true_effect,
            run_stats_func=run_stats_func,
            formula=formula,
            **stats_params
        )
        
        results_df["scale_factor"] = k
        all_raw_results.append(results_df)
        
        # 4. 汇总性能指标
        metrics = _calculate_performance_metrics(
            results_df, alpha=0.05
        )
        metrics["scale_factor"] = k
        all_summary_metrics.append(metrics)
    
    final_summary_df = pd.concat(all_summary_metrics, ignore_index=True)
    final_raw_df = pd.concat(all_raw_results, ignore_index=True)
    
    return final_summary_df, final_raw_df










