import pandas as pd
import numpy as np

from src.stats.engine import *
from src.stats.support import *

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)

@logged
def collect_real_data_results(count_df, stats_func, **kwargs):
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
            base_param = {
                "df_all":count_df,
                "cell_type":ct_name
            }
            full_stats_params = {**base_param, **kwargs}
            sim_filtered_params = filter_kwargs_for_func(stats_func, full_stats_params)
            stats_res = stats_func(**sim_filtered_params)
            
            # 2. 提取 contrast 表（通常包含所有变量的回归系数）
            # 包含：Term (variable), Coefficient (LFC), p-value, SE 等
            contrast_table = stats_res["contrast_table"].copy()
            
            # 3. 补充元信息以便后续汇总
            contrast_table['cell_type'] = ct_name
            
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


def get_clean_core_data(df_real, min_detection_rate=0.9):
    # 计算每个细胞类型在多少比例的样本中出现过
    detection_stats = df_real.groupby('cell_type')['count'].apply(lambda x: (x > 0).mean())
    
    # 只保留在 >90% 的样本中都存在的细胞类型
    core_cts = detection_stats[detection_stats >= min_detection_rate].index.tolist()
    
    print(f"原始细胞类型数: {len(detection_stats)}, 核心细胞类型数: {len(core_cts)}")
    
    # 过滤数据
    df_clean = df_real[df_real['cell_type'].isin(core_cts)].copy()
    return df_clean, core_cts
