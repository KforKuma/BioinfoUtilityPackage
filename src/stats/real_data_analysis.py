import pandas as pd
import numpy as np

from src.stats.engine import *
from src.stats.support import *

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)

@logged
def collect_real_data_results(count_df, stats_func, **kwargs):
    """对真实数据中所有 cell subtype/subpopulation 运行统计方法。

    Args:
        count_df: 标准长表丰度数据，至少包含 ``cell_type``。
        stats_func: stats engine 函数，例如 ``run_LMM`` 或 ``run_CLR_LMM``。
        **kwargs: 透传给 ``stats_func`` 的参数。

    Returns:
        合并后的 contrast_table，额外包含 ``cell_type`` 列。若所有亚群失败，返回空表。

    Example:
        >>> df_res = collect_real_data_results(
        ...     count_df,
        ...     run_LMM,
        ...     formula="disease + tissue",
        ...     main_variable="disease",
        ... )
        >>> df_res[["cell_type", "Coef.", "P>|z|", "significant"]].head()
    """
    if "cell_type" not in count_df.columns:
        raise ValueError("Missing required column: 'cell_type'.")
    if stats_func is None:
        raise ValueError("Please provide a stats function via `stats_func`.")

    all_summary_list = []
    
    # 获取唯一的细胞类型
    cell_types = count_df["cell_type"].unique()
    print(f"[collect_real_data_results] Starting analysis for {len(cell_types)} cell types.")
    
    for ct_name in cell_types:
        try:
            # 1. 运行统计模型
            # 假设该函数返回包含 p-value, LFC, Std Error 等信息的字典
            base_param = {
                "df_all":count_df,
                "cell_type":ct_name
            }
            full_stats_params = {**base_param, **kwargs}
            sim_filtered_params = filter_kwargs_for_func(stats_func, full_stats_params)
            stats_res = stats_func(**sim_filtered_params)
            
            # 2. 提取 contrast 表（通常包含所有变量的回归系数）
            # 包含：Term (variable), Coefficient (LFC), p-value, SE 等
            if stats_res.get("contrast_table") is None or stats_res["contrast_table"].empty:
                continue
            contrast_table = stats_res["contrast_table"].copy()
            
            # 3. 补充元信息以便后续汇总
            contrast_table['cell_type'] = ct_name
            
            all_summary_list.append(contrast_table)
        
        except Exception as e:
            print(f"[collect_real_data_results] Warning! Stats failed for cell_type '{ct_name}'. Error: {e}")
            continue
    
    if not all_summary_list:
        return pd.DataFrame()
    
    # 合并所有细胞类型的结果
    df_results = pd.concat(all_summary_list, ignore_index=False)
    
    # 增加一个 posterior p 值校正逻辑
    pval_candidates = ['P>|z|', 'p_adj', 'pval']
    existing_cols = df_results.columns  # 在 DataFrame (result_rows) 上检查 .columns 是正确的
    pval_colname = None
    for col in pval_candidates:
        if col in existing_cols:
            pval_colname = col
            break
    if pval_colname is None:
        raise ValueError("No p-value column detected in the contrast table.")
    
    sig_ratio = sum(df_results[pval_colname] < 0.05) / df_results.shape[0]
    if sig_ratio < 0.4:
        alpha_adj = 0.05
    else:
        alpha_adj = 0.05 * (0.25 / sig_ratio) ** 2  # 自动收缩 alpha
    
    df_results["significant"] = (df_results["significant"].astype(bool) &
                                     (df_results[pval_colname] < alpha_adj))
    
    
    # 整理列顺序，方便阅读
    cols = ['cell_type','ref', 'Coef.', 'Std.Err.', 'z', 'P>|z|','p_adj', 'pval',
            '[0.025', '0.975]','method_agreement',
            'significant', 'direction']
    # 保留结果中存在的列
    existing_cols = [c for c in cols if c in df_results.columns]
    remaining_cols = [c for c in df_results.columns if c not in existing_cols]
    
    return df_results[existing_cols + remaining_cols]


def get_clean_core_data(df_real, min_detection_rate=0.9):
    """过滤出高检出率的核心 cell subtype/subpopulation。

    Args:
        df_real: 真实长表丰度数据，至少包含 ``cell_type`` 和 ``count``。
        min_detection_rate: 保留亚群所需的最小样本检出比例。

    Returns:
        ``(df_clean, core_cts)``。``df_clean`` 只包含核心亚群，``core_cts`` 是亚群名列表。

    Example:
        >>> df_clean, core_cts = get_clean_core_data(count_df, min_detection_rate=0.9)
        >>> len(core_cts)
        # 后续真实数据分析可只在这些稳定出现的亚群上运行。
    """
    required_cols = {"cell_type", "count"}
    missing_cols = required_cols - set(df_real.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")
    if not 0 <= min_detection_rate <= 1:
        raise ValueError("`min_detection_rate` must be between 0 and 1.")

    # 计算每个细胞类型在多少比例的样本中出现过
    detection_stats = df_real.groupby('cell_type')['count'].apply(lambda x: (x > 0).mean())
    
    # 只保留在 >90% 的样本中都存在的细胞类型
    core_cts = detection_stats[detection_stats >= min_detection_rate].index.tolist()
    
    print(f"[get_clean_core_data] Original cell types: {len(detection_stats)}, core cell types: {len(core_cts)}")
    
    # 过滤数据
    df_clean = df_real[df_real['cell_type'].isin(core_cts)].copy()
    return df_clean, core_cts
