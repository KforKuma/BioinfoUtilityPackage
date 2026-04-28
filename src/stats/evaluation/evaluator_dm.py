from __future__ import annotations
import warnings
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd



from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)

#######################

@logged
def collect_DM_results(
        df_count: pd.DataFrame,
        cell_types_list: List[str],
        run_DM_func,
        # 核心修改：默认公式改为包含交互项 (*)
        formula: str = "disease + C(tissue, Treatment(reference='nif'))",
        tissue_levels: Tuple[str, str] = ("nif", "if"),
        disease_levels: Tuple[str, str] = ("HC", "Colitis", "BD", "CD", "UC"),
) -> Dict[str, pd.DataFrame]:
    """收集 Dirichlet-Multinomial 方法在多个亚群上的系数结果。

    该函数用于从真实数据或种子数据中估计后续模拟所需的效应量。它逐个运行
    ``run_DM_func``，从 ``contrast_table`` 中识别 disease、tissue 和 interaction
    三类对比，并整理为长表。

    Args:
        df_count: 长表 count 数据，格式与 stats engine 输入一致。
        cell_types_list: 需要评估的 cell subtype/subpopulation 名称列表。
        run_DM_func: 兼容 ``run_Dirichlet_Multinomial_Wald`` 调用签名的函数。
        formula: 传给 ``run_DM_func`` 的右侧公式。
        tissue_levels: ``(ref_tissue, other_tissue)``，默认 ``("nif", "if")``。
        disease_levels: disease 水平，首个元素视为参考组。

    Returns:
        字典，包含 ``all_coefs`` 长表和 ``disease_levels``。``all_coefs`` 的列包括
        ``cell_type``、``factor``、``contrast_other``、``LogFC_Coef`` 和 ``PValue``。

    Example:
        >>> collected = collect_DM_results(
        ...     df_count=count_df,
        ...     cell_types_list=count_df["cell_type"].unique(),
        ...     run_DM_func=run_Dirichlet_Multinomial_Wald,
        ...     formula="disease + C(tissue, Treatment(reference='nif'))",
        ... )
        >>> collected["all_coefs"].head()
        # 用于 estimate_DM_parameters 推断模拟效应量。
    """
    required_cols = {"cell_type", "count"}
    missing_cols = required_cols - set(df_count.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    all_coefs = []
    ref_tissue, other_tissue = tissue_levels
    
    for cell_type in cell_types_list:
        try:
            res = run_DM_func(df_all=df_count, cell_type=cell_type, formula=formula, verbose=False)
            
            if 'error' in res.get('extra', {}):
                continue
            
            contrast_df: pd.DataFrame = res["contrast_table"]
            if contrast_df is None or contrast_df.empty:
                continue
            extra = res['extra']
            
            for other, row in contrast_df.iterrows():
                if other == row['ref']: continue
                
                coef = row["Coef."]
                pval = row["P>|z|"]
                if pd.isna(coef): continue
                
                # --- 改进的 Factor 识别逻辑 ---
                factor_type = ""
                
                # 1. 检查是否为交互项 (通常包含 : 或 x 或同时包含疾病名和组织名)
                is_inter = (":" in other) or (" x " in other) or \
                           (any(d in other for d in disease_levels[1:]) and other_tissue in other)
                
                if is_inter:
                    factor_type = 'interaction'
                elif other in extra.get('groups', []):
                    factor_type = 'disease'
                elif other == other_tissue:
                    factor_type = 'tissue'
                
                if not factor_type: continue
                
                all_coefs.append({
                    'cell_type': cell_type,
                    'factor': factor_type,
                    'contrast_other': other,
                    'LogFC_Coef': coef,
                    'PValue': pval
                })
        except Exception:
            continue
    
    return {'all_coefs': pd.DataFrame(all_coefs), 'disease_levels': disease_levels}


@logged
def estimate_DM_parameters(collected_results: Dict[str, pd.DataFrame], alpha=0.05) -> Dict[str, float]:
    """根据 DM 结果估计模拟函数的效应量参数。

    Args:
        collected_results: ``collect_DM_results`` 返回的字典。
        alpha: 判定显著信号的 p 值阈值。

    Returns:
        包含 ``disease_effect_size``、``tissue_effect_size``、
        ``interaction_effect_size`` 和 ``inflamed_cell_frac`` 的参数字典。

    Example:
        >>> params = estimate_DM_parameters(collected)
        >>> params["tissue_effect_size"], params["inflamed_cell_frac"]
        # 可作为 simulate_DM_data 或 simulate_LogisticNormal_hierarchical 的输入。
    """
    if "all_coefs" not in collected_results:
        raise ValueError("Missing required key in `collected_results`: 'all_coefs'.")
    df_coefs = collected_results['all_coefs']
    if df_coefs.empty:
        return {
            'disease_effect_size': 0.0,
            'tissue_effect_size': 0.0,
            'interaction_effect_size': 0.0,
            'inflamed_cell_frac': 0.05,
        }
    total_cell_types = df_coefs['cell_type'].nunique()
    if total_cell_types == 0:
        total_cell_types = 1
    
    # 过滤显著信号
    df_signal = df_coefs[df_coefs['PValue'] < alpha].copy()
    if df_signal.empty:  # 回退机制
        df_signal = df_coefs[df_coefs['LogFC_Coef'].abs() > 0.1].copy()
    
    df_signal['Abs_LogFC'] = df_signal['LogFC_Coef'].abs()
    
    params = {}
    
    # 分别计算三大效应：Disease, Tissue, Interaction
    for factor in ['disease', 'tissue', 'interaction']:
        df_f = df_signal[df_signal['factor'] == factor]
        
        # 1. 效应量估计：使用显著项的 75 分位数 (确保模拟出的信号具有代表性，不至于被平均值拉低)
        if not df_f.empty:
            params[f'{factor}_effect_size'] = df_f['Abs_LogFC'].quantile(0.75)
        else:
            params[f'{factor}_effect_size'] = 0.0
    
    # 2. 比例估计：
    # tissue_effect 覆盖比例
    affected_by_tissue = df_signal[df_signal['factor'] == 'tissue']['cell_type'].nunique()
    params['inflamed_cell_frac'] = max(0.05, affected_by_tissue / total_cell_types)
    
    # interaction 覆盖比例 (可选：用于精细控制交互项影响多少细胞)
    affected_by_inter = df_signal[df_signal['factor'] == 'interaction']['cell_type'].nunique()
    interaction_cell_frac = affected_by_inter / total_cell_types
    
    # 3. 最终安全性检查：如果 interaction_effect_size 还是太小（比如 < 0.1）
    # 在进行方法学评估（Power Test）时，通常需要一个手动注入的“最小可探测值”
    if interaction_cell_frac < 0.1:  # 如果原始数据基本没检测到交互项
        # 选择 A: 取消
        params['interaction_effect_size'] = 0.0
        print("[estimate_DM_parameters] Warning! No interaction signal detected in seed data. Skipping interaction simulation.")
    else:
        # 选择 B: 动态注入
        # 交互项强度不应低于主效应的某个比例，否则在数学上就被噪声盖过了
        min_detectable = params.get('disease_effect_size', 0.5) * 0.3  # 至少是主效应的 30%
        params['interaction_effect_size'] = max(params.get('interaction_effect_size', 0), min_detectable)
    
    return params


@logged
def filter_rare_celltypes(count_df, zero_threshold=0.25, verbose=True):
    """过滤在过多 sample 中缺失的 cell subtype/subpopulation。

    Args:
        count_df: 长表 count 数据，至少包含 ``sample_id``、``cell_type`` 和 ``count``。
        zero_threshold: 允许 count 为 0 的最大 sample 比例。
        verbose: 是否打印过滤摘要。打印内容使用英文，并带函数名前缀。

    Returns:
        ``(filtered_df, summary)``。``filtered_df`` 是过滤后的长表；``summary``
        是每个 cell subtype/subpopulation 的 zero count 统计。

    Example:
        >>> filtered_df, summary = filter_rare_celltypes(
        ...     count_df,
        ...     zero_threshold=0.25,
        ...     verbose=True,
        ... )
        >>> summary.loc["CT1", "zero_fraction"]
        # 用于判断该亚群是否适合进入模拟参数估计。
    """
    required_cols = {"sample_id", "cell_type", "count"}
    missing_cols = required_cols - set(count_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")
    if not 0 <= zero_threshold <= 1:
        raise ValueError("`zero_threshold` must be between 0 and 1.")
    
    # 构建矩阵
    mat = count_df.pivot_table(
        index="sample_id",
        columns="cell_type",
        values="count",
        fill_value=0
    )
    
    # 统计
    zero_counts = (mat == 0).sum()
    zero_prop = (mat == 0).mean()
    
    summary = pd.DataFrame({
        "n_zero_samples": zero_counts,
        "zero_fraction": zero_prop
    }).sort_values("zero_fraction", ascending=False)
    
    # 保留 cell types
    keep_celltypes = summary[summary["zero_fraction"] <= zero_threshold].index
    
    filtered_df = count_df[count_df["cell_type"].isin(keep_celltypes)].copy()
    
    if verbose:
        print(f"[filter_rare_celltypes] Cell type zero summary:\n{summary.head(20)}")
        print(f"[filter_rare_celltypes] Total cell types: {summary.shape[0]}")
        print(f"[filter_rare_celltypes] Remaining cell types: {len(keep_celltypes)}")
        print(f"[filter_rare_celltypes] Filtered out: {summary.shape[0] - len(keep_celltypes)}")

    return filtered_df, summary
