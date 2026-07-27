# from typing import Dict, List, Tuple
# import re
# import warnings
from tqdm import tqdm

import inspect
import logging
import warnings

import numpy as np
import pandas as pd

from src.stats.support import *
from src.stats.evaluation.benchmark_metrics import calculate_binary_metrics, estimate_fdr_from_replicates
from src.stats.validation import parse_boolean_series, parse_boolean_value
from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


_TRUTH_SOURCE_COLUMNS = {
    "injected": "Is_Injected_Nonzero",
    "population": "Is_Population_Nonzero",
    "observed": "Is_Observed_Detectable",
}


def _truth_column_for_source(frame: pd.DataFrame, truth_source: str) -> str:
    if truth_source not in _TRUTH_SOURCE_COLUMNS:
        raise ValueError(
            "`truth_source` must be one of 'injected', 'population', or 'observed'."
        )
    column = _TRUTH_SOURCE_COLUMNS[truth_source]
    if column not in frame.columns:
        raise ValueError(f"Truth source {truth_source!r} requires column {column!r}.")
    return column


def _apply_fixed_nominal_decision(frame: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Apply a fixed, predeclared threshold without data-adaptive shrinkage."""
    result = frame.copy()
    existing = (
        parse_boolean_series(result["Est_Significant"])
        if "Est_Significant" in result.columns
        else pd.Series(pd.NA, index=result.index, dtype="boolean")
    )
    if "Est_PValue" in result.columns:
        pvalues = pd.to_numeric(result["Est_PValue"], errors="coerce")
        finite = pvalues.notna() & np.isfinite(pvalues)
        existing.loc[finite] = pvalues.loc[finite] <= alpha
    result["Est_Significant"] = existing
    result["nominal_alpha"] = alpha
    result["adaptive_alpha_enabled"] = False
    return result



def _calculate_performance_metrics(df_all_sims: pd.DataFrame,
                                   alpha: float = 0.05,
                                   truth_col: str = "True_Significant",
                                   decision_col: str | None = None,
                                   replicate_col: str = "simulation_replicate") -> pd.DataFrame:
    """计算定义正确的 Power/TPR、FPR、FDP、Precision 和 Specificity。

    Args:
        df_all_sims: 单次或多次模拟的估计结果表，至少包含 ``contrast_factor``、
            ``True_Significant``、``Est_Significant`` 或 ``Est_PValue``。
        alpha: 当 ``Est_Significant`` 不存在时，用该阈值从 p 值重新判断显著性。

    Returns:
        按 ``contrast_factor`` 汇总的性能表。missing/unavailable decision 不会被
        当作阴性；零分母指标返回 NaN 并附带 ``*_Reason``。

    Example:
        >>> metrics = _calculate_performance_metrics(results_df, alpha=0.05)
        >>> metrics[["contrast_factor", "Power", "FPR"]]
        # 用于比较不同统计方法在模拟真值下的表现。
    """
    output_columns = [
        "contrast_factor", "TP", "FP", "TN", "FN", "N_Evaluated", "N_Excluded",
        "Power", "TPR", "FPR", "FDP", "FDR_Estimate", "Precision", "Specificity",
        "Nominal_Alpha", "Empirical_Type_I_Error",
    ]
    if df_all_sims.empty:
        return pd.DataFrame(columns=output_columns)
    if "contrast_factor" not in df_all_sims.columns:
        raise ValueError("Missing required column: 'contrast_factor'.")
    if truth_col not in df_all_sims.columns:
        raise ValueError(f"Missing required truth column: {truth_col!r}.")

    frame = df_all_sims.copy()
    frame[truth_col] = parse_boolean_series(frame[truth_col])
    if decision_col is None:
        decision_col = "primary_decision" if "primary_decision" in frame.columns else "Est_Significant"
    if decision_col in frame.columns:
        frame["_benchmark_decision"] = parse_boolean_series(frame[decision_col])
    elif "Est_PValue" in frame.columns:
        pvalues = pd.to_numeric(frame["Est_PValue"], errors="coerce")
        frame["_benchmark_decision"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
        finite = pvalues.notna() & np.isfinite(pvalues)
        frame.loc[finite, "_benchmark_decision"] = pvalues.loc[finite] <= alpha
    else:
        raise ValueError(f"Missing decision column {decision_col!r} and fallback 'Est_PValue'.")

    if {"is_available", "is_valid"}.issubset(frame.columns):
        available = parse_boolean_series(frame["is_available"])
        valid = parse_boolean_series(frame["is_valid"])
        frame.loc[~(available.fillna(False) & valid.fillna(False)), "_benchmark_decision"] = pd.NA

    rows = []
    for contrast_factor, group in frame.groupby("contrast_factor", dropna=False, sort=False):
        row = {"contrast_factor": contrast_factor}
        row.update(calculate_binary_metrics(group[truth_col], group["_benchmark_decision"]))
        row["Nominal_Alpha"] = alpha
        row["Empirical_Type_I_Error"] = row["FPR"]
        rows.append(row)
    metrics = pd.DataFrame(rows)

    if replicate_col in frame.columns:
        fdr = estimate_fdr_from_replicates(
            frame,
            truth_col=truth_col,
            decision_col="_benchmark_decision",
            replicate_col=replicate_col,
            group_cols=("contrast_factor",),
        )
        metrics = metrics.merge(fdr, on="contrast_factor", how="left")
        if "FDR_Reason" not in metrics.columns:
            metrics["FDR_Reason"] = np.where(
                metrics["N_FDP_Replicates"].fillna(0).gt(0), None, "no_finite_replicate_fdp"
            )
    else:
        metrics["FDR_Estimate"] = np.nan
        metrics["N_FDP_Replicates"] = 0
        metrics["FDR_Reason"] = "replicate_id_unavailable"

    return metrics


def _collect_simulation_results(
        df_sim: pd.DataFrame,
        df_true_effect: pd.DataFrame,
        run_stats_func,  # 传入的统计运行函数，例如 run_Dirichlet_Wald
        formula: str,
        truth_source: str = "injected",
        **kwargs
) -> pd.DataFrame:
    """收集单个模拟数据集的统计结果并合并 ground truth。

    Args:
        df_sim: 模拟 count 长表。
        df_true_effect: 模拟真实效应表。
        run_stats_func: 实际运行统计分析的函数，例如 ``run_Dirichlet_Wald``。
        formula: 传递给统计函数的模型公式。
        **kwargs: 透传给统计函数的参数。

    Returns:
        包含所有 cell subtype/subpopulation、所有对比的真实效应和统计估计值。

    Example:
        >>> results_df = _collect_simulation_results(
        ...     df_sim,
        ...     df_truth,
        ...     run_stats_func=run_LMM,
        ...     formula="disease + tissue",
        ...     main_variable="disease",
        ... )
        >>> results_df[["True_Significant", "Est_PValue", "Est_Significant"]].head()
    """
    
    truth_column = _truth_column_for_source(df_true_effect, truth_source)

    # 获取唯一的细胞类型列表
    cell_types = df_sim['cell_type'].unique().tolist()
    
    # 真实效应表预处理: 确保 True_Significant 基于 alpha
    # 更新 refine 逻辑后不需要
    # df_true_effect['True_Significant'] = (df_true_effect['True_Effect'] != 0)
    
    # 存储所有对比结果
    all_results = []
    
    for ct_name in cell_types:
        try:
            # 1. 运行统计模型
            # 假设 run_stats_func(df_all, cell_type, formula) 返回结构化的结果
            stats_res = run_stats_func(df_all=df_sim, cell_type=ct_name, formula=formula, **kwargs)
            contrast_table = stats_res["contrast_table"]
        
        except Exception as e:
            print(f"[_collect_simulation_results] Warning! Stats failed for cell_type '{ct_name}'. Error: {e}")
            continue
        
        # 2. 提取该细胞类型的真实效应行
        df_true_ct = df_true_effect[df_true_effect['cell_type'] == ct_name].copy()
        
        # 3. 匹配真实效应和统计估计值
        for _, true_row in df_true_ct.iterrows():
            contrast_factor = true_row['contrast_factor']
            group_full = true_row['contrast_group']  # 例如 'UC x if'
            
            # 根据 Fallback 规则确定要查找的 'other' 组名称
            if contrast_factor == 'tissue':
                # Rule: contrast_factor=tissue 对应 other='if'
                target_other = group_full
                est_results = _extract_contrast_results(contrast_table, target_other)
            elif contrast_factor == 'disease':
                target_other = group_full
                est_results = _extract_contrast_results(contrast_table, target_other)
            elif contrast_factor in ('addition', 'interaction'):
                # Rule: disease/interaction 对应 other=疾病名称
                # 从 'UC x if' 或 'CD x if' 中提取疾病名称 'UC' 或 'CD'
                est_results = _extract_addition_results(contrast_table, group_full)
            
            # 组合结果
            result_record = {
                'cell_type': ct_name,
                'contrast_factor': contrast_factor,
                'contrast_group': true_row['contrast_group'],
                'contrast_ref': true_row['contrast_ref'],
                'True_Effect': true_row['True_Effect'],
                'True_Direction': true_row['True_Direction'],
                'True_Significant': true_row[truth_column],
                'Truth_Source': truth_source,
                'Injected_Effect': true_row.get('Injected_Effect', np.nan),
                'Population_Effect': true_row.get('Population_Effect', np.nan),
                'Observed_Effect': true_row.get('Observed_Effect', np.nan),
                **est_results
            }
            all_results.append(result_record)
    
    return pd.DataFrame(all_results)


def _extract_contrast_results(contrast_table: pd.DataFrame,
                              target_other: str,
                              alpha: float = 0.05
                              ) -> dict:
    """从 contrast_table 中提取指定 ``other`` 的估计结果。

    Args:
        contrast_table: stats engine 输出的对比表。
        target_other: 需要匹配的 ``other`` 标签。
        alpha: 当表中没有 ``significant`` 列时使用的阈值。

    Returns:
        包含 ``Est_Coef``、``Est_PValue``、``Est_Direction`` 和
        ``Est_Significant`` 的字典。

    Example:
        >>> est = _extract_contrast_results(res["contrast_table"], "CD")
        >>> est["Est_PValue"]
    """
    unavailable = {
        'Est_Coef': np.nan,
        'Est_PValue': np.nan,
        'Est_Direction': 'None',
        'Est_Significant': pd.NA,
        'contrast_status': 'unavailable',
        'failure_reason': 'unsupported_contrast',
    }
    if contrast_table is None or contrast_table.empty:
        return unavailable.copy()
    
    # 1. 重置索引以便按列名访问 'other'
    df_reset = contrast_table.reset_index()
    
    # 2. 使用布尔索引查找目标行
    if 'other' not in df_reset.columns:
        return unavailable.copy()
    result_rows = df_reset[df_reset['other'].astype(str) == str(target_other)]
    
    if result_rows.empty:
        return unavailable.copy()
    
    # 3. 提取结果 (只取第一行匹配项)
    result_row = result_rows.iloc[0]
    
    # 4. 确定 P 值列名 (Fallback 逻辑)
    pval_colname = None
    # 优先使用已经声明 family 的 adjusted p，随后才是 raw p。
    pval_candidates = ['pvalue_adjusted', 'p_adj', 'pvalue_raw', 'P>|z|', 'pval']
    
    # 检查哪些候选列存在于当前的 DataFrame 中
    existing_cols = result_rows.columns  # 在 DataFrame (result_rows) 上检查 .columns 是正确的
    
    for col in pval_candidates:
        if col in existing_cols:
            pval_colname = col
            break
    
    # 5. 提取 P 值和显著性
    est_pval = pd.to_numeric(pd.Series([result_row[pval_colname]]), errors='coerce').iloc[0] if pval_colname else np.nan
    
    # 由于您的统计输出中已经有了 'significant' 列，我们优先使用它。
    # 如果没有 'significant' 列，则基于 P 值和 alpha 重新计算。
    if 'significant' in existing_cols:
        est_significant = parse_boolean_value(result_row['significant'])
    elif pd.notna(est_pval) and np.isfinite(est_pval):
        est_significant = (est_pval <= alpha)
    else:
        est_significant = pd.NA
    
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
        'Est_Significant': est_significant,
        'contrast_status': result_row.get('contrast_status', 'success'),
        'failure_reason': result_row.get('failure_reason', None),
    }


def _extract_addition_results(contrast_table, group_name):
    """提取 addition/interaction 语义下的组合对比结果。

    Args:
        contrast_table: stats engine 输出的对比表。
        group_name: 组合标签，例如 ``"UC x if"``。

    Returns:
        估计结果字典。只接受模型已正式计算的组合/interaction 行；无法匹配时
        返回 unavailable，不从主效应 p 值拼接近似结果。

    Example:
        >>> _extract_addition_results(contrast_table, "UC x if")
        # 若没有正式组合项，返回 unavailable。
    """
    unavailable = {
        'Est_Coef': np.nan,
        'Est_PValue': np.nan,
        'Est_Significant': pd.NA,
        'Est_Direction': 'None',
        'contrast_status': 'unavailable',
        'failure_reason': 'unsupported_contrast',
    }
    if contrast_table is None or contrast_table.empty:
        return unavailable

    direct = _extract_contrast_results(contrast_table, group_name)
    if direct['contrast_status'] == 'success':
        return direct

    parts = group_name.split(' x ')
    if len(parts) == 2 and all(part in contrast_table.index for part in parts):
        unavailable['failure_reason'] = 'covariance_unavailable'
    return unavailable

@logged
def evaluate_effect_size_scaling(
        scale_factors=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0],
        sim_func=None,  # 新增：模拟函数入口
        run_stats_func=None,
        formula="disease + C(tissue, Treatment(reference='nif'))",
        base_params=None,
        truth_source="injected",
        **kwargs
):
    """按 effect size 缩放因子评估单个统计方法。

    Args:
        scale_factors: effect size 缩放倍数列表。
        sim_func: 模拟函数，需返回 ``(df_sim, df_true_effect)``。
        run_stats_func: 统计方法函数。
        formula: 传给统计方法的公式。
        base_params: 模拟基础参数；为 ``None`` 时使用内置默认值。
        **kwargs: 传给统计方法的额外参数。

    Returns:
        每个 ``scale_factor`` 下按 contrast factor 汇总的 Power/FPR 表。

    Example:
        >>> metrics = evaluate_effect_size_scaling(
        ...     scale_factors=[0.5, 1.0, 2.0],
        ...     sim_func=simulate_DM_data,
        ...     run_stats_func=run_Dirichlet_Multinomial_Wald,
        ...     formula="disease + tissue",
        ... )
        >>> metrics.head()
    """
    
    
    warnings.warn(
        "evaluate_effect_size_scaling is deprecated; use run_abundance_pipeline and evaluate_contrasts.",
        DeprecationWarning,
        stacklevel=2,
    )
    # 1. 初始化模拟器和基础参数
    if sim_func is None:
        raise ValueError("Please provide a simulation function via `sim_func`.")
    if run_stats_func is None:
        raise ValueError("Please provide a stats function via `run_stats_func`.")
    
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
    print(f"[evaluate_effect_size_scaling] Starting evaluation: Sim[{sim_func.__name__}] -> Stats[{run_stats_func.__name__}]")
    
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
        
        print(f"[evaluate_effect_size_scaling] Scale {k}. Params for {run_stats_func.__name__}: {stats_filtered_kwargs}")
        
        results_df = _collect_simulation_results(
            df_sim=df_sim,
            df_true_effect=df_true_effect,
            run_stats_func=run_stats_func,
            formula=formula,
            truth_source=truth_source,
            **stats_filtered_kwargs
        )
        
        if results_df.empty or "Est_PValue" not in results_df.columns:
            continue
        results_df = _apply_fixed_nominal_decision(results_df, alpha=0.05)
        
        # 5. 计算性能指标
        metrics = _calculate_performance_metrics(results_df, alpha=0.05)
        
        # 6. 记录当前的倍数因子
        metrics['scale_factor'] = k
        all_metrics.append(metrics)
    
    # 合并所有结果
    final_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    return final_df


@logged
def evaluate_effect_size_scaling_with_raw(
        scale_factors,
        sim_func,
        run_stats_func,
        sim_params,
        stats_params,
        formula,
        truth_source="injected",
):
    """按 effect size 缩放因子评估统计方法并保留原始结果。

    Args:
        scale_factors: effect size 缩放倍数。
        sim_func: 模拟函数。
        run_stats_func: 统计方法函数。
        sim_params: 模拟函数基础参数。
        stats_params: 统计函数参数。
        formula: 统计公式。

    Returns:
        ``(summary_df, raw_df)``。前者是 Power/FPR 汇总，后者保留每个
        cell subtype/subpopulation 的逐项估计，适合 PPV 分析。

    Example:
        >>> summary, raw = evaluate_effect_size_scaling_with_raw(
        ...     [0.5, 1.0],
        ...     simulate_LogisticNormal_hierarchical,
        ...     run_CLR_LMM,
        ...     sim_params,
        ...     stats_params,
        ...     "disease + tissue",
        ... )
    """
    
    warnings.warn(
        "evaluate_effect_size_scaling_with_raw is deprecated; use the canonical pipeline.",
        DeprecationWarning,
        stacklevel=2,
    )
    all_summary_metrics = []
    all_raw_results = []
    
    print(f"[evaluate_effect_size_scaling_with_raw] Starting effect size scaling evaluation: {sim_func.__name__}")
    
    if not any("effect_size" in key for key in sim_params.keys()):
        raise ValueError("At least one key containing 'effect_size' must be provided in `sim_params`.")
    
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
            truth_source=truth_source,
            **stats_params
        )
        
        results_df["scale_factor"] = k
        
        if results_df.empty or "Est_PValue" not in results_df.columns:
            continue
        results_df = _apply_fixed_nominal_decision(results_df, alpha=0.05)
        
        
        all_raw_results.append(results_df)
        
        # 4. 汇总性能指标
        metrics = _calculate_performance_metrics(
            results_df, alpha=0.05
        )
        metrics["scale_factor"] = k
        all_summary_metrics.append(metrics)
    
    final_summary_df = pd.concat(all_summary_metrics, ignore_index=True) if all_summary_metrics else pd.DataFrame()
    final_raw_df = pd.concat(all_raw_results, ignore_index=True) if all_raw_results else pd.DataFrame()
    
    return final_summary_df, final_raw_df


def _collect_simulation_meta_results(
        df_sim: pd.DataFrame,
        df_true_effect: pd.DataFrame,
        run_stats_func,
        formula: str,
        truth_source: str = "injected",
        **kwargs
):
    """收集 meta engine 及其子方法的模拟估计结果。

    Args:
        df_sim: 模拟 count 长表。
        df_true_effect: 模拟真实效应表。
        run_stats_func: meta engine 函数，通常为 ``run_Meta_Ensemble_adaptive``。
        formula: 传给 meta engine 的公式。
        **kwargs: 透传参数。

    Returns:
        ``{"meta": df, "dmw": df, "clr": df, "deseq2": df}``，每个 DataFrame
        都是统一的逐对比估计结果。

    Example:
        >>> storage = _collect_simulation_meta_results(
        ...     df_sim,
        ...     df_truth,
        ...     run_Meta_Ensemble_adaptive,
        ...     "disease + tissue",
        ... )
        >>> storage["meta"].head()
    """
    truth_column = _truth_column_for_source(df_true_effect, truth_source)
    cell_types = df_sim['cell_type'].unique().tolist()
    storage = {'meta': [], 'dmw': [], 'clr': [], 'deseq2': []}
    
    for ct_name in cell_types:
        tables_map = {}
        try:
            stats_res = run_stats_func(df_all=df_sim, cell_type=ct_name, formula=formula, **kwargs)
            # 此时 stats_res["contrast_table"] 是 Meta 的结果
            tables_map = {
                'meta': stats_res.get("contrast_table"),
                'dmw': stats_res.get("raw_results", {}).get("dmw", {}).get("contrast_table") if stats_res.get(
                    "raw_results") else None,
                'clr': stats_res.get("raw_results", {}).get("clr", {}).get("contrast_table") if stats_res.get(
                    "raw_results") else None,
                'deseq2': stats_res.get("raw_results", {}).get("deseq2", {}).get("contrast_table") if stats_res.get(
                    "raw_results") else None
            }
        except Exception as e:
            print(f"[_collect_simulation_meta_results] Warning! Meta-stats execution failed for cell_type '{ct_name}'. Error: {e}")
            continue
        
        df_true_ct = df_true_effect[df_true_effect['cell_type'] == ct_name].copy()
        
        for _, true_row in df_true_ct.iterrows():
            contrast_factor = true_row['contrast_factor']
            group_full = true_row['contrast_group']
            
            for key in storage.keys():
                current_table = tables_map.get(key)
                
                # --- 修正点 1: 增加空值判断，防止子方法失败时报错 ---
                if current_table is None or (isinstance(current_table, pd.DataFrame) and current_table.empty):
                    est_results = {
                        'Est_Coef': np.nan, 'Est_PValue': np.nan,
                        'Est_Direction': 'None', 'Est_Significant': pd.NA,
                        'contrast_status': 'unavailable',
                        'failure_reason': 'native_output_missing',
                    }
                else:
                    # --- 修正点 2: 严格使用 current_table 提取各方法独立的结果 ---
                    if contrast_factor == 'tissue':
                        est_results = _extract_contrast_results(current_table, group_full)
                    elif contrast_factor == 'disease':
                        est_results = _extract_contrast_results(current_table, group_full)
                    elif contrast_factor in ('addition', 'interaction'):
                        est_results = _extract_addition_results(current_table, group_full)
                
                record = {
                    'cell_type': ct_name,
                    'contrast_factor': contrast_factor,
                    'contrast_group': group_full,
                    'contrast_ref': true_row['contrast_ref'],
                    'True_Effect': true_row['True_Effect'],
                    'True_Direction': true_row['True_Direction'],
                    'True_Significant': true_row[truth_column],
                    'Truth_Source': truth_source,
                    'Injected_Effect': true_row.get('Injected_Effect', np.nan),
                    'Population_Effect': true_row.get('Population_Effect', np.nan),
                    'Observed_Effect': true_row.get('Observed_Effect', np.nan),
                    **est_results
                }
                storage[key].append(record)
    
    # 转换为 DataFrame
    final_storage = {}
    for key, records in storage.items():
        final_storage[key] = pd.DataFrame(records) if records else pd.DataFrame()
    
    return final_storage


@logged
def evaluate_effect_size_meta_scaling(
        scale_factors=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0],
        sim_func=None,  # 新增：模拟函数入口
        run_meta_func=None,
        formula="disease + C(tissue, Treatment(reference='nif'))",
        base_params=None,
        truth_source="injected",
        **kwargs
):
    """按 effect size 缩放因子评估 meta engine 及其子方法。

    Args:
        scale_factors: effect size 缩放倍数列表。
        sim_func: 模拟函数，需返回 ``(df_sim, df_true_effect)``。
        run_meta_func: meta engine 函数。
        formula: 传给 meta engine 的公式。
        base_params: 模拟基础参数；为 ``None`` 时使用默认模板。
        **kwargs: 传给 meta engine 的额外参数。

    Returns:
        字典，键为 ``meta``、``dmw``、``clr`` 和 ``deseq2``，值为对应方法在各
        scale factor 下的 Power/FPR 汇总表。

    Example:
        >>> out = evaluate_effect_size_meta_scaling(
        ...     scale_factors=[0.5, 1.0],
        ...     sim_func=simulate_DM_data,
        ...     run_meta_func=run_Meta_Ensemble_adaptive,
        ...     formula="disease + tissue",
        ... )
        >>> out["meta"].head()
    """
    
    warnings.warn(
        "evaluate_effect_size_meta_scaling is deprecated; use the canonical pipeline.",
        DeprecationWarning,
        stacklevel=2,
    )
    # 1. 初始化模拟器和基础参数
    if sim_func is None:
        raise ValueError("Please provide a simulation function via `sim_func`.")
    if run_meta_func is None:
        raise ValueError("Please provide a meta stats function via `run_meta_func`.")
    
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
    
    metrics_storage = {
        'meta': [],
        'dmw': [],
        'clr': [],
        'deseq2': []
    }
    print(f"[evaluate_effect_size_meta_scaling] Starting evaluation: Sim[{sim_func.__name__}] -> Stats[{run_meta_func.__name__}]")
    
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
        stats_filtered_kwargs = filter_kwargs_for_func(run_meta_func, full_stats_params)
        
        print(f"[evaluate_effect_size_meta_scaling] Scale {k}. Params for {run_meta_func.__name__}: {stats_filtered_kwargs}")
        
        results_df_dict = _collect_simulation_meta_results(
            df_sim=df_sim,
            df_true_effect=df_true_effect,
            run_stats_func=run_meta_func,
            formula=formula,
            truth_source=truth_source,
            **stats_filtered_kwargs
        )
        for key, value in results_df_dict.items():
            if value.empty:
                continue
            results_df_dict[key] = _apply_fixed_nominal_decision(value, alpha=0.05)
            
        # 5. 计算性能指标
        for key in results_df_dict.keys():
            if results_df_dict[key].empty:
                continue
            metrics = _calculate_performance_metrics(results_df_dict[key], alpha=0.05)
            # 6. 记录当前的倍数因子
            metrics['scale_factor'] = k
            metrics_storage[key].append(metrics)
    
    # 合并所有结果
    final_df_dict = {}
    for key in metrics_storage.keys():
        final_df = pd.concat(metrics_storage[key], ignore_index=True) if metrics_storage[key] else pd.DataFrame()
        final_df_dict.update({key: final_df})
    
    return final_df_dict


