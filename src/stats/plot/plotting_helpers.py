import numpy as np
import pandas as pd
import warnings


def plot_evaluation_summary(summary_table: pd.DataFrame, save_path) -> None:
    """Plot precomputed canonical evaluation metrics without recomputation or filtering."""
    import matplotlib.pyplot as plt

    required = {"method", "empirical_FDR", "mean_Power"}
    if missing := required - set(summary_table.columns):
        raise ValueError(f"Evaluation summary is missing columns: {sorted(missing)}")
    data = summary_table[["method", "empirical_FDR", "mean_Power"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    axes[0].bar(data["method"].astype(str), data["mean_Power"], color="#4472C4")
    axes[0].set(title="Power", ylabel="Mean replicate Power")
    axes[1].bar(data["method"].astype(str), data["empirical_FDR"], color="#C55A11")
    axes[1].set(title="Empirical FDR", ylabel="Mean FDP_for_FDR")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_real_data_summary(summary_table: pd.DataFrame, save_path) -> None:
    """Plot a precomputed real-data method summary without truth-dependent metrics."""
    import matplotlib.pyplot as plt

    required = {"method", "number_tested", "discoveries", "number_invalid", "number_unavailable"}
    if missing := required - set(summary_table.columns):
        raise ValueError(f"Real-data summary is missing columns: {sorted(missing)}")
    forbidden = {"Power", "TPR", "FPR", "FDP", "empirical_FDR"} & set(summary_table.columns)
    if forbidden:
        raise ValueError(f"Truth-dependent metrics are forbidden in real-data summaries: {sorted(forbidden)}")
    data = summary_table[[
        "method", "number_tested", "discoveries", "number_invalid", "number_unavailable"
    ]].copy()
    x = np.arange(len(data))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, column, color in (
        (-1.5, "number_tested", "#4472C4"),
        (-0.5, "discoveries", "#70AD47"),
        (0.5, "number_invalid", "#FFC000"),
        (1.5, "number_unavailable", "#C55A11"),
    ):
        ax.bar(x + offset * width, data[column], width, label=column, color=color)
    ax.set_xticks(x, data["method"].astype(str), rotation=30)
    ax.set_ylabel("Canonical result rows")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


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
    warnings.warn(
        "combine_metrics is a deprecated legacy compatibility helper; formal plotting "
        "must consume evaluate_contrasts summary tables.",
        DeprecationWarning,
        stacklevel=2,
    )
    required_cols = {"Method", "scale_factor", "TP", "FP", "FN"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    # Compatibility output keeps canonical definitions and never invents TN.
    count_columns = ['TP', 'FP', 'FN'] + (['TN'] if 'TN' in df.columns else [])
    combined = df.groupby(['Method', 'scale_factor'])[count_columns].sum().reset_index()

    power_denominator = combined['TP'] + combined['FN']
    combined['Power'] = combined['TP'].div(power_denominator.where(power_denominator.ne(0)))
    combined['Power_reason'] = np.where(
        power_denominator.eq(0), 'power_denominator_zero', None
    )

    discovery_denominator = combined['TP'] + combined['FP']
    combined['FDP_descriptive'] = combined['FP'].div(
        discovery_denominator.where(discovery_denominator.ne(0))
    )
    combined['FDP_descriptive_reason'] = np.where(
        discovery_denominator.eq(0), 'fdp_denominator_zero', None
    )
    if 'TN' in combined.columns:
        fpr_denominator = combined['FP'] + combined['TN']
        combined['FPR'] = combined['FP'].div(fpr_denominator.where(fpr_denominator.ne(0)))
        combined['FPR_reason'] = np.where(
            fpr_denominator.eq(0), 'fpr_denominator_zero', None
        )
    else:
        combined['FPR'] = np.nan
        combined['FPR_reason'] = 'tn_unavailable_in_legacy_input'
    
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
