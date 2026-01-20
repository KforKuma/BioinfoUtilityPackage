# from typing import Dict, List, Tuple
# import re
# import warnings
from tqdm import tqdm


import numpy as np
import pandas as pd


def combine_metrics(df):
    # 1. 按照 Method 和 scale_factor 分组，累加基础计数
    combined = df.groupby(['Method', 'scale_factor'])[['TP', 'FP', 'FN']].sum().reset_index()
    
    # 2. 重新计算整体指标
    # 防止分母为 0 的处理
    combined['Power'] = combined['TP'] / (combined['TP'] + combined['FN'])
    combined['FDR'] = combined['FP'] / (combined['TP'] + combined['FP'])
    
    # 填充 NaN (如果没有 call 出任何显著，FDR 定义为 0)
    combined['FDR'] = combined['FDR'].fillna(0)
    combined['Power'] = combined['Power'].fillna(0)
    
    # 3. 标记为 Combined 项
    combined['contrast_factor'] = 'Combined (Global)'
    
    return combined


def calculate_performance_metrics(df_all_sims: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    计算基于多次模拟结果的性能指标 (Power, FDR)。
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
    
    # 如果 Est_Significant 已经存在（由包装函数 collect_simulation_results 计算好了），则直接使用它
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
    metrics['FDR'] = metrics['FP'] / (metrics['TP'] + metrics['FP'])
    
    # 处理除以零的情况
    metrics['Power'] = metrics['Power'].fillna(0)
    metrics['FDR'] = metrics['FDR'].fillna(0)
    
    return metrics


def collect_simulation_results(
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
            contrast_table = stats_res["extra"]["contrast_table"]
        
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
            est_results = extract_contrast_results(contrast_table, target_other)
            
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


def extract_contrast_results(contrast_table: pd.DataFrame, target_other: str, alpha: float = 0.05) -> dict:
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


def merge_contrast_tables(tables_dict):
    """Merge multiple contrast_tables into one readable DataFrame."""

    merged = None
    for method, df in tables_dict.items():
        df_copy = df.copy()
        # 保留关键信息
        keep_cols = ["ref", "other", "mean_ref", "mean_other", "prop_diff",
                     "Coef", "p_adj", "significant", "direction"]
        for col in df_copy.columns:
            if col not in keep_cols:
                df_copy = df_copy.drop(columns=col)

        # 为列加方法前缀
        df_copy = df_copy.rename(columns={c: f"{method}_{c}" for c in df_copy.columns if c not in ["ref", "other"]})

        if merged is None:
            merged = df_copy
        else:
            merged = pd.merge(merged, df_copy, on=["ref", "other"], how="outer")

    return merged


import inspect
import pandas as pd
from tqdm import tqdm


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
    
    def filter_kwargs_for_func(func, params_dict, strict=False):
        """
        工具函数：过滤字典，只保留目标函数需要的参数
        """
        if func is None: return {}
        sig = inspect.signature(func)
        accepted = sig.parameters
        filtered = {k: v for k, v in params_dict.items() if k in accepted}
        return filtered
    
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
        
        results_df = collect_simulation_results(
            df_sim=df_sim,
            df_true_effect=df_true_effect,
            run_stats_func=run_stats_func,
            formula=formula,
            **stats_filtered_kwargs
        )
        
        # 5. 计算性能指标
        metrics = calculate_performance_metrics(results_df, alpha=0.05)
        
        # 6. 记录当前的倍数因子
        metrics['scale_factor'] = k
        all_metrics.append(metrics)
    
    # 合并所有结果
    final_df = pd.concat(all_metrics, ignore_index=True)
    return final_df


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
        Power / FDR summary across scales
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
        results_df = collect_simulation_results(
            df_sim=df_sim,
            df_true_effect=df_true_effect,
            run_stats_func=run_stats_func,
            formula=formula,
            **stats_params
        )
        
        results_df["scale_factor"] = k
        all_raw_results.append(results_df)
        
        # 4. 汇总性能指标
        metrics = calculate_performance_metrics(
            results_df, alpha=0.05
        )
        metrics["scale_factor"] = k
        all_summary_metrics.append(metrics)
    
    final_summary_df = pd.concat(all_summary_metrics, ignore_index=True)
    final_raw_df = pd.concat(all_raw_results, ignore_index=True)
    
    return final_summary_df, final_raw_df



def filter_kwargs_for_func(func, params_dict):
    if func is None: return {}
    sig = inspect.signature(func)
    return {k: v for k, v in params_dict.items() if k in sig.parameters}

def get_clean_core_data(df_real, min_detection_rate=0.9):
    # 计算每个细胞类型在多少比例的样本中出现过
    detection_stats = df_real.groupby('cell_type')['count'].apply(lambda x: (x > 0).mean())
    
    # 只保留在 >90% 的样本中都存在的细胞类型
    core_cts = detection_stats[detection_stats >= min_detection_rate].index.tolist()
    
    print(f"原始细胞类型数: {len(detection_stats)}, 核心细胞类型数: {len(core_cts)}")
    
    # 过滤数据
    df_clean = df_real[df_real['cell_type'].isin(core_cts)].copy()
    return df_clean, core_cts



from src.core.kdk_methodology import run_CLR_LMM_with_LFC
def collect_real_data_results(count_df, formula, coef_threshold=1.0):
    """
    运行统计模型并收集真实数据的显著性发现。
    """
    all_summary_list = []
    
    # 获取唯一的细胞类型
    cell_types = count_df["cell_type"].unique()
    print(f"Starting analysis for {len(cell_types)} cell types...")
    
    for ct_name in cell_types:
        try:
            # 1. 运行统计模型
            # 假设该函数返回包含 p-value, LFC, Std Error 等信息的字典
            stats_res = run_CLR_LMM_with_LFC(
                df_all=count_df,
                cell_type=ct_name,
                formula=formula,
                main_variable='disease',
                coef_threshold=coef_threshold
            )
            
            # 2. 提取 contrast 表（通常包含所有变量的回归系数）
            # 包含：Term (variable), Coefficient (LFC), p-value, SE 等
            contrast_table = stats_res["extra"]["contrast_table"].copy()
            
            # 3. 补充元信息以便后续汇总
            contrast_table['cell_type'] = ct_name
            
            # 判定显著性 (基于你预设的阈值)
            # 假设 contrast_table 已经过多重假设检验校正 (FDR)
            # 或者我们在这里手动标记
            if 'p_adj' not in contrast_table.columns:
                # 简单的示例：如果还没做校正，这里可以记录原始 p 值
                contrast_table['is_significant'] = (contrast_table['P>|z|'] < 0.05) & \
                                                   (contrast_table['Coef.'].abs() >= coef_threshold)
            
            all_summary_list.append(contrast_table)
        
        except Exception as e:
            print(f"Warning: Stats failed for {ct_name}. Error: {e}")
            continue
    
    if not all_summary_list:
        return pd.DataFrame()
    
    # 合并所有细胞类型的结果
    df_results = pd.concat(all_summary_list, ignore_index=False)
    
    # 整理列顺序，方便阅读
    cols = ['ref', 'Coef.', 'Std.Err.', 'z', 'P>|z|', '[0.025', '0.975]',
       'significant', 'direction']
    # 保留结果中存在的列
    existing_cols = [c for c in cols if c in df_results.columns]
    remaining_cols = [c for c in df_results.columns if c not in existing_cols]
    
    return df_results[existing_cols + remaining_cols]


def calculate_ppv_by_coef(df, bin_size=0.2):
    """
    根据估计的系数大小 (Est_Coef) 计算 PPV 表。
    """
    df = df.copy()
    
    # 1. 取估计系数的绝对值（因为 PPV 通常不分正负效应，只看强度）
    df['abs_est_coef'] = df['Est_Coef'].abs()
    
    # 2. 统计 TP 和所有的预测阳性 (Predicted Positives)
    # 注意：这里必须用全量数据，包含 True_Significant 为 False 的行
    df['is_tp'] = (df['True_Significant'] == True) & (df['Est_Significant'] == True)
    df['is_pred_pos'] = (df['Est_Significant'] == True)
    
    # 3. 对 Est_Coef 进行分箱
    max_coef = df['abs_est_coef'].max()
    bins = np.arange(0, max_coef + bin_size, bin_size)
    df['coef_bin'] = pd.cut(df['abs_est_coef'], bins=bins)
    
    # 4. 按分箱聚合
    ppv_table = df.groupby('coef_bin', observed=True).agg(
        tp_count=('is_tp', 'sum'),
        total_pred_pos=('is_pred_pos', 'sum'),
        avg_est_coef=('abs_est_coef', 'mean')
    ).reset_index()
    
    # 5. 计算 PPV
    ppv_table['PPV'] = ppv_table['tp_count'] / ppv_table['total_pred_pos']
    ppv_table['PPV'] = ppv_table['PPV'].fillna(0)
    
    return ppv_table