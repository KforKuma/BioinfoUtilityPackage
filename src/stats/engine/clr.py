from __future__ import annotations
import warnings
import logging
from typing import Dict, Any
import re

import numpy as np
import numpy.linalg as la
import pandas as pd

import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning  # optional

from src.stats.support import (
    make_result,
    remove_main_variable_from_formula,
    parse_formula_columns,
    split_C_terms,
)

from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)

# -----------------------
# Method 3: CLR + LMM √
# -----------------------
@logged
def run_CLR_LMM(df_all: pd.DataFrame,
                cell_type: str | tuple,
                formula: str = "disease",
                main_variable: str = None,
                ref_label: str = "HC",
                group_label: str = "sample_id",
                alpha: float = 0.05,
                use_reml: bool = True,
                pseudocount: float = 1.0) -> Dict[str, Any]:
    '''
    采用 中心对数变换（CLR）后，再进行 线性混合模型（LMM） 拟合。
    CLR = ln (细胞比例 xi / 所有细胞比例的几何平均值 gm)

    :param df_all:
    :param cell_type: 兼容两种输入。因为 CLR 方法对于细胞比例（log ratio diff）处理比较擅长，因此允许输入一个最多有两个值的元组。
    :param alpha:
    :param pseudocount:
    :return: 标准的 make_result 输出，需要详细介绍的是 extra 部分的格式，直接打印出来格式可能会崩掉；
             未来会写一个将关键信息整形的接口函数（TODO）
    extra["mixedlm_summary"]:
        Mixed Linear Model Regression Results
    ====================================================
    Model:            MixedLM Dependent Variable: y
    No. Observations: 6      Method:      REML 或 ML，默认 REML
    No. Groups:       3      Scale:             1.0000
    Min. group size:  2      Log-Likelihood:    -7.0711
    Max. group size:  2      Converged:         Yes
    Mean group size:  2.0
    ----------------------------------------------------
                     Coef.  Std.Err.   z    P>|z| [0.025 0.975]
    ----------------------------------------------------
    Intercept        0.000    0.000   0.000 1.000  0.000  0.000
    x                2.000    0.000    inf  0.000  2.000  2.000
    ====================================================

    '''
    
    # 输出准备
    extra = {}
    
    # 输入参数解析
    if isinstance(cell_type, (tuple, list)):
        if len(cell_type) != 2:
            raise ValueError("cell_type as tuple/list must have length 2: (A, B)")
        cell_A, cell_B = cell_type
        mode = "log_ratio"
        cell_type_label = f"{cell_A}_vs_{cell_B}"
    else:
        mode = "single"
        cell_A = cell_type
        cell_type_label = cell_type
    
    # 公式合法性判断
    if "+" in formula:
        if main_variable is None:
            raise KeyError("Main explanatory variable must be specified when formula contains more than one variable.")
    else:
        main_variable = formula
    
    formula_fixed = remove_main_variable_from_formula(formula, main_variable)
    
    if main_variable != formula:
        formula = f"clr_value ~ C({main_variable}, Treatment(reference=\"{ref_label}\")) + {formula_fixed}"
    else:
        formula = f"clr_value ~ C({main_variable}, Treatment(reference=\"{ref_label}\"))"
    
    # pivot：长表转宽表
    pivot = df_all.pivot_table(index=parse_formula_columns(formula) + [group_label],
                               columns="cell_type", values="count", aggfunc="sum", fill_value=0)
    # pivot columns are cell types
    counts = pivot.copy()
    
    try:
        # Step 1: CLR 变换
        # 加入 pseudocount
        counts_pc = counts + pseudocount
        # 计算 log
        log_counts = np.log(counts_pc)
        # 计算几何平均值
        gm = log_counts.mean(axis=1)
        # CLR 转换
        clr = log_counts.subtract(gm, axis=0)
        
        # 提取目的细胞亚群的 pd.Series (response)
        if mode == "single":
            clr_target = clr[cell_A].reset_index()
            clr_target = clr_target.rename(columns={cell_A: "clr_value"})
        else:
            # log(A / B) = CLR(A) - CLR(B)
            clr_target = (clr[cell_A] - clr[cell_B]).reset_index()
            clr_target = clr_target.rename(columns={0: "clr_value"})
        
        # Step 2: 拟合线性混合模型 LMM
        # 构建模型
        group = clr_target[group_label]  # 指定随机效应
        md = smf.mixedlm(formula, clr_target, groups=group)
        # 拟合
        mdf = md.fit(method="nm", maxiter=200, reml=use_reml)
        
        # 读取结果
        pval = mdf.pvalues.min()
        
        # 储存结果
        output = mdf.summary().tables
        extra["mixedlm_summary"] = output[0]
        extra["mixedlm_fixed_effect"] = output[1]
        
        contrast_table = extra["mixedlm_fixed_effect"].copy()
        contrast_table = contrast_table[1:-1]
        contrast_table["ref"], contrast_table["other"] = split_C_terms(pd.Series(contrast_table.index)).T.values
        
        contrast_table["P>|z|"] = contrast_table["P>|z|"].astype(float)
        contrast_table["significant"] = (contrast_table["P>|z|"] < alpha).astype(str)
        
        contrast_table["Coef."] = contrast_table["Coef."].astype(float)
        contrast_table["direction"] = contrast_table["Coef."].apply(lambda x: "ref_greater" if x < 0 else "other_greater")
        
        contrast_table = contrast_table[
            ['ref', 'other', 'Coef.', 'Std.Err.', 'z', 'P>|z|', '[0.025', '0.975]', 'significant', 'direction']]
        contrast_table = pd.DataFrame(contrast_table).set_index("other")
        
        if len(output) == 3:
            extra["mixedlm_random_effect"] = output[2]
                
        return make_result(method="CLR_LMM",
                           cell_type=cell_type_label,
                           p_val=pval if pval is not None else np.nan,p_type='Minimal',
                           contrast_table=contrast_table,
                           extra=extra,
                           alpha=alpha)
    
    except Exception as e:
        extra["error"] = str(e)
        print("CLR-LMM function failed.")
        return make_result(method="CLR_LMM",
                           cell_type=cell_type,
                           p_val=np.nan,
                           contrast_table=None,
                           extra=extra,
                           alpha=alpha)

@logged
def run_CLR_LMM_with_LFC(
        df_all: pd.DataFrame,
        cell_type: str,
        formula: str = "disease + tissue",
        main_variable: str = None,
        coef_threshold: float = 0.2,
        alpha: float = 0.05,
        **kwargs) -> Dict[str, Any]:
    """
    运行 CLR LMM 检验并应用 LFC 过滤。
    """
    print(f"Using Coef. Threshold of {coef_threshold}")
    res = run_CLR_LMM(df_all, cell_type, formula=formula, main_variable=main_variable, **kwargs)
    
    if res['contrast_table'] is not None:
        ct = res['contrast_table']
        ct['significant'] = (ct['P>|z|'] < alpha) & (ct['Coef.'].abs() > coef_threshold)
        
        res['significant'] = any(ct['significant'])
        res['contrast_table'] = ct
        
    return res


@logged
def run_pCLR_OLS(
        df_all: pd.DataFrame,
        cell_type: str,
        formula: str = "disease",
        random_effect: str = "1 | donor_id",
        n_samples: int = 32,
        alpha: float = 0.05,
        random_state: int = 42,
        disease_ref: str = "HC",
        tissue_ref: str = "nif"
) -> Dict[str, Any]:
    rng = np.random.default_rng(random_state)
    extra = {}
    
    try:
        # --- 核心修改 1: 显式处理分类变量及其参考级 ---
        # 确保 disease_ref 存在于数据中，否则报错
        for col, ref in [('disease', disease_ref), ('tissue', tissue_ref)]:
            if col in df_all.columns:
                unique_vals = df_all[col].unique()
                if ref not in unique_vals:
                    warnings.warn(f"Reference {ref} not found in {col}. Available: {unique_vals}")
                # 构造分类类型，将 reference 放在第一位，确保它成为 Intercept
                categories = [ref] + [v for v in unique_vals if v != ref]
                df_all[col] = pd.Categorical(df_all[col], categories=categories)
        
        # --- 核心修改 2: 防止 pivot_table 丢弃样本 ---
        metadata_cols = ['sample_id', 'donor_id', 'disease', 'tissue']
        # dropna=False 确保即使元数据有缺失，BD 样本也不会被删除
        df_wide = df_all.pivot_table(index=metadata_cols, columns='cell_type', values='count', fill_value=0,
                                     dropna=False)
        df_prop_wide = df_all.pivot_table(index=metadata_cols, columns='cell_type', values='prop', fill_value=0,
                                          dropna=False)
        
        cell_types = df_wide.columns.tolist()
        target_idx = cell_types.index(cell_type)
        counts_matrix = df_wide.values.astype(float)
        all_coefs_storage = []
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for s in range(n_samples):
                # Dirichlet 抽样增加微小先验防止 log(0)
                prob_samples = np.array([rng.dirichlet(row + 0.5) for row in counts_matrix])
                log_p = np.log(prob_samples)
                clr_matrix = log_p - np.mean(log_p, axis=1, keepdims=True)
                
                df_iter = df_wide.index.to_frame().reset_index(drop=True)
                df_iter['target_clr'] = clr_matrix[:, target_idx]
                
                try:
                    # 使用显式分类处理：ensure the formula handles the Categorical types
                    model = smf.ols(f"target_clr ~ {formula}", df_iter)
                    result = model.fit()
                    all_coefs_storage.append(result.summary2().tables[1].copy())
                except Exception:
                    continue
        
        if not all_coefs_storage:
            raise ValueError("All model iterations failed.")
        
        # 3. 汇总结果
        df_concat = pd.concat(all_coefs_storage).reset_index().rename(columns={'index': 'term'})
        for col in ['Coef.', 'Std.Err.', 't', 'P>|t|']:
            df_concat[col] = pd.to_numeric(df_concat[col], errors='coerce')
        
        summary = df_concat.groupby('term').agg(
            {'Coef.': 'median', 'Std.Err.': 'median', 't': 'median', 'P>|t|': 'median'}
        )
        
        rows = []
        for term, res in summary.iterrows():
            if term == 'Intercept' or ':' in term: continue
            
            # 改进正则：支持更复杂的 patsy 命名
            match = re.search(r"\[T\.(.*?)\]", term)
            if not match: continue
            
            other_val = match.group(1)
            level = 'disease' if 'disease' in term else 'tissue'
            ref_val = disease_ref if level == 'disease' else tissue_ref
            
            # --- 核心修改 3: 更稳健的均值计算 ---
            try:
                # 使用 groupby 而非 xs，防止 MultiIndex 层级不匹配
                m_ref = df_prop_wide.groupby(level)[cell_type].mean().get(ref_val, 0)
                m_other = df_prop_wide.groupby(level)[cell_type].mean().get(other_val, 0)
            except Exception:
                m_ref, m_other = 0, 0
            
            rows.append({
                'other': other_val,
                'ref': ref_val,
                'mean_ref': m_ref,
                'mean_other': m_other,
                'pval': res['P>|t|'],
                'significant': res['P>|t|'] < alpha,
                'prop_diff': m_other - m_ref,
                'Coef.': res['Coef.'],
                'Std.Err': res['Std.Err.'],
                'z': res['t'],
                'direction': 'other_greater' if res['Coef.'] > 0 else 'ref_greater'
            })
        
        contrast_table = pd.DataFrame(rows)
        if contrast_table.empty:
            raise ValueError(f"No valid contrasts found for {cell_type}")
        
        # 排序并返回
        contrast_table['_is_tissue'] = contrast_table['ref'] == tissue_ref
        contrast_table = contrast_table.sort_values(['_is_tissue', 'other']).drop(columns=['_is_tissue']).set_index(
            'other')
        
        return make_result(
            method="pCLR_LMM", cell_type=cell_type,
            p_val=contrast_table['pval'].min(), p_type='Minimal',
            contrast_table=contrast_table, extra=extra, alpha=alpha
        )
    
    except Exception as e:
        extra["error"] = str(e)
        return make_result(method="pCLR_LMM", cell_type=cell_type, p_val=None,
                           contrast_table=None, extra=extra, alpha=alpha)



@logged
def run_pCLR_LMM(
        df_all: pd.DataFrame,
        cell_type: str,
        formula: str = "disease",
        random_effect: str = "1 | donor_id",
        n_samples: int = 32,
        alpha: float = 0.05,
        random_state: int = 42,
        disease_ref: str = "HC",
        tissue_ref: str = "nif"
) -> Dict[str, Any]:
    rng = np.random.default_rng(random_state)
    extra = {}
    
    try:
        # 1. 显式分类变量转换与参考级固定 (防止 BD 被设为 Intercept)
        df_all = df_all.copy()
        for col, ref in [('disease', disease_ref), ('tissue', tissue_ref)]:
            if col in df_all.columns:
                unique_vals = df_all[col].unique()
                categories = [ref] + [v for v in unique_vals if v != ref]
                df_all[col] = pd.Categorical(df_all[col], categories=categories)
        
        # 2. 宽表准备 (dropna=False 保护小样本)
        metadata_cols = ['sample_id', 'donor_id', 'disease', 'tissue']
        df_wide = df_all.pivot_table(index=metadata_cols, columns='cell_type', values='count',
                                     fill_value=0, dropna=False)
        df_prop_wide = df_all.pivot_table(index=metadata_cols, columns='cell_type', values='prop',
                                          fill_value=0, dropna=False)
        
        target_idx = df_wide.columns.get_loc(cell_type)
        counts_matrix = df_wide.values.astype(float)
        re_group = random_effect.split('|')[1].strip()
        all_coefs_storage = []
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for s in range(n_samples):
                # Dirichlet 采样增加 0.5 先验稳定性
                prob_samples = np.array([rng.dirichlet(row + 0.5) for row in counts_matrix])
                log_p = np.log(prob_samples)
                clr_matrix = log_p - np.mean(log_p, axis=1, keepdims=True)
                
                df_iter = df_wide.index.to_frame().reset_index(drop=True)
                df_iter['target_clr'] = clr_matrix[:, target_idx]
                
                try:
                    # 使用 MixedLM 运行混合效应模型
                    # 注意：LMM 默认不包含截距项在 random_coefs 中，这里使用 groups 指定随机效应
                    model = smf.mixedlm(f"target_clr ~ {formula}", df_iter, groups=df_iter[re_group])
                    # reml=True 对于小样本方差估计更准确
                    result = model.fit(reml=True)
                    
                    # 混合模型的 summary 表结构与 OLS 不同，需提取前一部分（固定效应）
                    all_coefs_storage.append(result.summary().tables[1].copy())
                except Exception:
                    # LMM 容易因为 Singular matrix 或不收敛报错，跳过当前采样
                    continue
        
        if not all_coefs_storage:
            raise ValueError("All LMM iterations failed to converge.")
        
        # 3. 汇总中位数结果
        df_concat = pd.concat(all_coefs_storage).reset_index().rename(columns={'index': 'term'})
        # LMM 的列名通常是 Coef., Std.Err., z, P>|z|
        for col in ['Coef.', 'Std.Err.', 'z', 'P>|z|']:
            df_concat[col] = pd.to_numeric(df_concat[col], errors='coerce')
        
        summary = df_concat.groupby('term').agg(
            {'Coef.': 'median', 'Std.Err.': 'median', 'z': 'median', 'P>|z|': 'median'}
        )
        
        rows = []
        for term, res in summary.iterrows():
            if term == 'Intercept' or term == 'Group Var' or ':' in term: continue
            
            match = re.search(r"\[T\.(.*?)\]", term)
            if not match: continue
            
            other_val = match.group(1)
            level = 'disease' if 'disease' in term else 'tissue'
            ref_val = disease_ref if level == 'disease' else tissue_ref
            
            # 稳健的均值计算
            try:
                m_ref = df_prop_wide.groupby(level)[cell_type].mean().get(ref_val, 0)
                m_other = df_prop_wide.groupby(level)[cell_type].mean().get(other_val, 0)
            except Exception:
                m_ref, m_other = 0, 0
            
            rows.append({
                'other': other_val,
                'ref': ref_val,
                'mean_ref': m_ref,
                'mean_other': m_other,
                'pval': res['P>|z|'],
                'significant': res['P>|z|'] < alpha,
                'prop_diff': m_other - m_ref,
                'Coef.': res['Coef.'],
                'Std.Err': res['Std.Err.'],
                'z': res['z'],
                'direction': 'other_greater' if res['Coef.'] > 0 else 'ref_greater'
            })
        
        contrast_table = pd.DataFrame(rows)
        if contrast_table.empty:
            raise ValueError(f"No valid LMM contrasts found for {cell_type}")
        
        contrast_table['_is_tissue'] = contrast_table['ref'] == tissue_ref
        contrast_table = contrast_table.sort_values(['_is_tissue', 'other']).drop(columns=['_is_tissue']).set_index(
            'other')
        
        return make_result(
            method="pCLR_LMM", cell_type=cell_type,
            p_val=contrast_table['pval'].min(), p_type='Minimal',
            contrast_table=contrast_table, extra=extra, alpha=alpha
        )
    
    except Exception as e:
        extra["error"] = str(e)
        return make_result(method="pCLR_LMM", cell_type=cell_type, p_val=None,
                           contrast_table=None, extra=extra, alpha=alpha)