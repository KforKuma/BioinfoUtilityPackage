import numpy as np
import pandas as pd


def refine_ground_truth_by_observation(df_long, df_true_effect, lfc_threshold=0.2):
    """根据模拟后的实际观测 LFC 修正 ground truth 可检测性。

    注入效应不一定能在最终 count/prop 中形成足够大的可观测差异，尤其是组成数据
    会受到其他 cell subtype/subpopulation 的挤压。因此本函数按模拟结果重新计算
    每个 truth 对比的观测 LFC，并生成 ``Is_Detectable_True`` 作为评估基准。

    Args:
        df_long: 模拟长表，至少包含 ``cell_type``、``disease``、``tissue`` 和 ``prop``。
        df_true_effect: 原始 ground truth 表，至少包含 ``cell_type``、
            ``contrast_group``、``contrast_ref``、``contrast_factor``、
            ``True_Effect`` 和 ``True_Significant``。
        lfc_threshold: 观测 ``abs(log2FC)`` 达到该阈值才视为可检测。

    Returns:
        增加 ``Observed_LFC`` 和 ``Is_Detectable_True`` 的 ground truth DataFrame。

    Example:
        >>> refined = refine_ground_truth_by_observation(df_sim, df_truth, lfc_threshold=0.2)
        >>> refined[["cell_type", "Observed_LFC", "Is_Detectable_True"]].head()
        # 评估统计方法时建议使用 Is_Detectable_True 作为更现实的真值标签。
    """
    required_long_cols = {"cell_type", "disease", "tissue", "prop"}
    missing_long_cols = required_long_cols - set(df_long.columns)
    if missing_long_cols:
        raise ValueError(f"Missing required columns in `df_long`: {sorted(missing_long_cols)}")
    required_truth_cols = {
        "cell_type", "contrast_group", "contrast_ref", "contrast_factor",
        "True_Effect", "True_Significant"
    }
    missing_truth_cols = required_truth_cols - set(df_true_effect.columns)
    if missing_truth_cols:
        raise ValueError(f"Missing required columns in `df_true_effect`: {sorted(missing_truth_cols)}")
    if lfc_threshold < 0:
        raise ValueError("`lfc_threshold` must be non-negative.")

    df_long = df_long.copy()
    df_refined = df_true_effect.copy()
    
    # 构造状态键，用于 interaction 对比的组合参照系。
    df_long['status_key'] = df_long['disease'].astype(str) + " x " + df_long['tissue'].astype(str)
    
    # 预计算各类中位数
    medians_inter = df_long.groupby(['cell_type', 'status_key'])['prop'].median().unstack('status_key')
    medians_disease = df_long.groupby(['cell_type', 'disease'])['prop'].median().unstack('disease')
    medians_tissue = df_long.groupby(['cell_type', 'tissue'])['prop'].median().unstack('tissue')
    
    observed_lfcs = []
    is_detectable_list = []
    
    for _, row in df_refined.iterrows():
        ct = row['cell_type']
        group = str(row['contrast_group'])
        ref = str(row['contrast_ref'])
        factor = row['contrast_factor']
        
        try:
            # 1. 精准提取观测值
            if factor == 'disease':
                val_g, val_r = medians_disease.loc[ct, group], medians_disease.loc[ct, ref]
            elif factor == 'tissue':
                val_g, val_r = medians_tissue.loc[ct, group], medians_tissue.loc[ct, ref]
            else:  # interaction
                val_g, val_r = medians_inter.loc[ct, group], medians_inter.loc[ct, ref]
            
            lfc = np.log2((val_g + 1e-6) / (val_r + 1e-6))
        except KeyError:
            lfc = 0.0
        
        # 2. 改进的判定逻辑
        injected_sig = bool(row['True_Significant'])
        
        if factor == 'interaction':
            # 对于交互项：由于受主效应挤压严重，只要观测到的绝对值足够大，
            # 且我们确实注入了信号，就认为它是“应该被检测”的基准。
            # 或者你可以移除方向检查，只看 abs(lfc)
            direction_consistent = True  # 交互项建议放宽，因为基准参照系（HC x nif）复杂
        else:
            # 对于主效应：方向必须一致，否则可能是被其他细胞挤压出来的假象
            direction_consistent = (row['True_Effect'] * lfc >= 0)
        
        detectable = injected_sig and (abs(lfc) >= lfc_threshold) and direction_consistent
        
        observed_lfcs.append(lfc)
        is_detectable_list.append(detectable)
    
    df_refined['Observed_LFC'] = observed_lfcs
    df_refined['Is_Detectable_True'] = is_detectable_list
    
    return df_refined
