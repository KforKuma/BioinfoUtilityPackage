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
        formula: str = "disease * C(tissue, Treatment(reference='nif'))",
        tissue_levels: Tuple[str, str] = ("nif", "if"),
        disease_levels: Tuple[str, str] = ("HC", "Colitis", "BD", "CD", "UC"),
        debug=False
) -> Dict[str, pd.DataFrame]:
    all_coefs = []
    ref_tissue, other_tissue = tissue_levels
    
    for cell_type in cell_types_list:
        try:
            res = run_DM_func(df_all=df_count, cell_type=cell_type, formula=formula, verbose=False)
            
            if 'error' in res['extra']: continue
            
            contrast_df: pd.DataFrame = res["contrast_table"]
            extra = res['extra']
            
            for other, row in contrast_df.iterrows():
                if other == row['ref']: continue
                
                coef = row["Coef"]
                pval = row["P>|z|"]
                if pd.isna(coef): continue
                
                # --- 改进的 Factor 识别逻辑 ---
                factor_type = ""
                
                # 1. 检查是否为交互项 (通常包含 : 或 x 或同时包含疾病名和组织名)
                is_inter = (":" in other) or (" x " in other) or \
                           (any(d in other for d in disease_levels[1:]) and other_tissue in other)
                
                if is_inter:
                    factor_type = 'interaction'
                elif other in extra['groups']:
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
        except Exception as e:
            continue
    
    return {'all_coefs': pd.DataFrame(all_coefs), 'disease_levels': disease_levels}


@logged
def estimate_DM_parameters(collected_results: Dict[str, pd.DataFrame], alpha=0.05) -> Dict[str, float]:
    df_coefs = collected_results['all_coefs']
    total_cell_types = df_coefs['cell_type'].nunique()
    
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
        print("Warning: No interaction signal detected in seed data. Skipping interaction simulation.")
    else:
        # 选择 B: 动态注入
        # 交互项强度不应低于主效应的某个比例，否则在数学上就被噪声盖过了
        min_detectable = params.get('disease_effect_size', 0.5) * 0.3  # 至少是主效应的 30%
        params['interaction_effect_size'] = max(params.get('interaction_effect_size', 0), min_detectable)
    
    return params
