import numpy as np
import pandas as pd


def combine_metrics(df):
    # 1. 按照 Method 和 scale_factor 分组，累加基础计数
    combined = df.groupby(['Method', 'scale_factor'])[['TP', 'FP', 'FN']].sum().reset_index()
    
    # 2. 重新计算整体指标
    # 防止分母为 0 的处理
    combined['Power'] = combined['TP'] / (combined['TP'] + combined['FN'])
    combined['FPR'] = combined['FP'] / (combined['TP'] + combined['FP'])
    
    # 填充 NaN (如果没有 call 出任何显著，FPR 定义为 0)
    combined['FPR'] = combined['FPR'].fillna(0)
    combined['Power'] = combined['Power'].fillna(0)
    
    # 3. 标记为 Combined 项
    combined['contrast_factor'] = 'Combined (Global)'
    
    return combined


def compute_ratio_df(
        df,
        celltype_pair=("CD4 Tmem GZMK+", "CD4 Tmem"),
        sample_col="sample_id",
        disease_col="disease",
        celltype_col="cell_type",
        prop_col="prop",
        eps=1e-6
):
    A, B = celltype_pair
    
    df_sub = df[df[celltype_col].isin([A, B])]
    
    df_pivot = (
        df_sub
        .pivot_table(
            index=[sample_col, disease_col],
            columns=celltype_col,
            values=prop_col
        )
        .reset_index()
    )
    
    # 防止除零
    df_pivot[A] = df_pivot[A].fillna(0) + eps
    df_pivot[B] = df_pivot[B].fillna(0) + eps
    
    df_pivot["ratio"] = df_pivot[A] / df_pivot[B]
    df_pivot["log2_ratio"] = np.log2(df_pivot["ratio"])
    
    return df_pivot
