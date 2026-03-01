from typing import Dict, Any
import pandas as pd

from src.stats.engine import *
from src.utils.env_utils import call_with_compatible_args

def run_Meta_Ensemble(df_all: pd.DataFrame,
                      cell_type: str,
                      formula: str,
                      main_variable: str = "disease",
                      alpha: float = 0.05,
                      **kwargs) -> Dict[str, Any]:
    """
    标准化接口的 Meta 集成分析函数。
    内部自动调用 DMW, CLR-LMM 和 PyDESeq2 并进行逻辑集成。
    """
    
    # 1. 定义内部需要调用的方法列表
    # 注意：这里假设这些函数在当前命名空间可用
    sub_methods = {
        'dmw': run_Dirichlet_Multinomial_Wald,
        'clr': run_CLR_LMM,
        'deseq2': run_PyDESeq2
    }
    
    # 准备基础参数池
    base_kwargs = {
        'df_all': df_all,
        'cell_type': cell_type,
        'formula': formula,
        'main_variable': main_variable,
        'alpha': alpha,
        **kwargs
    }
    
    # 2. 内部化执行三种统计方法
    results = {}
    for name, func in sub_methods.items():
        try:
            # 利用你现有的兼容性包装器进行调用
            results[name] = call_with_compatible_args(func, **base_kwargs)
        except Exception as e:
            print(f"Warning: Sub-method {name} failed for cell {cell_type}: {e}")
            # 如果核心锚点 DMW 失败，Meta 方法无法进行，直接返回空结果或报错
            if name == 'dmw': return {'contrast_table': pd.DataFrame(), 'summary': 'DMW Failed'}
            results[name] = None
    
    # 3. 提取结果表 (处理可能存在的 None)
    res_dmw = results.get('dmw')
    res_clr = results.get('clr')
    res_dsq = results.get('deseq2')
    
    # 基础对齐检查
    if res_clr is None or res_dsq is None:
        return res_dmw if res_dmw else {'contrast_table': pd.DataFrame()}
    
    df_dmw = res_dmw['contrast_table'].copy()
    df_clr = res_clr['contrast_table'].copy()
    df_dsq = res_dsq['contrast_table'].copy()
    
    # 4. 索引对齐与逻辑集成 (复用之前的对齐逻辑)
    common_idx = df_dmw.index.intersection(df_clr.index).intersection(df_dsq.index)
    if len(common_idx) == 0:
        return {'contrast_table': pd.DataFrame(), 'summary': 'No common index'}
    
    df_dmw, df_clr, df_dsq = df_dmw.loc[common_idx], df_clr.loc[common_idx], df_dsq.loc[common_idx]
    
    # --- 关键修正：统一方向映射与类型转换 ---
    dir_map = {'other_greater': 1, 'ref_greater': -1}
    
    # 强制将 significant 转为布尔型，并处理可能存在的字符串干扰
    def to_bool(s):
        if isinstance(s, bool): return s
        return str(s).strip().lower() == 'true'
    
    sig1 = df_dmw['significant'].apply(to_bool)
    sig2 = df_clr['significant'].apply(to_bool)
    sig3 = df_dsq['significant'].apply(to_bool)
    
    d1 = df_dmw['direction'].map(dir_map).fillna(0)
    d2 = df_clr['direction'].map(dir_map).fillna(0)
    d3 = df_dsq['direction'].map(dir_map).fillna(0)
    
    # 多数表决方向
    dir_sum = d1 + d2 + d3
    rev_map = {1: 'other_greater', -1: 'ref_greater'}
    meta_direction = dir_sum.apply(lambda x: rev_map[1] if x > 0 else rev_map[-1] if x < 0 else "Ambiguous")
    
    # 显著性集成：DMW 锚定 + 至少一个同向验证
    # 显著性集成：DMW 锚定且必须同向 (d2 == d1)
    # 使用 .values 确保不受 pandas 对齐检查的二次干扰
    match_clr = (sig2.values) & (d2.values == d1.values)
    match_dsq = (sig3.values) & (d3.values == d1.values)
    
    # 最终 meta_significant
    # 这里的 & 两边现在绝对都是 boolean numpy arrays
    meta_significant = sig1.values & (match_clr | match_dsq)
    
    # 5. 构造最终输出表
    meta_table = pd.DataFrame({
        'ref': df_dmw['ref'],
        'Coef.': df_clr['Coef.'],  # 保持 CLR 系数的解释性
        'P>|z|': pd.concat([df_dmw['P>|z|'], df_clr['P>|z|']], axis=1).max(axis=1),
        'direction': meta_direction,
        'significant': meta_significant,
        'consistency_score': dir_sum.abs().astype(int),
        'method': 'Meta_Ensemble'
    }, index=common_idx)
    
    return {
        'contrast_table': meta_table,
        'summary': f"Meta-analysis complete. {meta_significant.sum()} hits found.",
        'raw_results': results  # 可选：保留原始结果以备查
    }