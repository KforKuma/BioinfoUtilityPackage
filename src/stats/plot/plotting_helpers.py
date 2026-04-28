import numpy as np
import pandas as pd


def combine_metrics(df):
    """合并不同 contrast factor 的模拟评估指标。

    Args:
        df: 包含 ``Method``、``scale_factor``、``TP``、``FP`` 和 ``FN`` 的结果表。

    Returns:
        按 ``Method`` 和 ``scale_factor`` 汇总后的整体 Power/FPR 表。

    Example:
        >>> combined = combine_metrics(metrics_df)
        >>> combined[["Method", "scale_factor", "Power", "FPR"]].head()
        # 用于绘制全局方法表现，而不是分 disease/tissue/interaction 展示。
    """
    required_cols = {"Method", "scale_factor", "TP", "FP", "FN"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

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
    """计算两个 cell subtype/subpopulation 比例的 ratio 和 log2 ratio。

    Args:
        df: 标准长表丰度数据。
        celltype_pair: ``(A, B)``，表示计算 ``A / B``。
        sample_col: 样本列名。
        disease_col: 分组列名。
        celltype_col: cell subtype/subpopulation 列名。
        prop_col: 比例列名。
        eps: 防止除零的伪计数。

    Returns:
        每个 sample 一行的 ratio 表，包含 ``ratio`` 和 ``log2_ratio``。

    Example:
        >>> ratio_df = compute_ratio_df(
        ...     count_df,
        ...     celltype_pair=("CD4 Tmem GZMK+", "CD4 Tmem"),
        ... )
        >>> ratio_df[["ratio", "log2_ratio"]].head()
        # 可传给 plot_ratio_scatter 可视化亚群比例变化。
    """
    required_cols = {sample_col, disease_col, celltype_col, prop_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

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
