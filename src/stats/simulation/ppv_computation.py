import pandas as pd
import numpy as np

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)

@logged
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
