from typing import Dict, Any
import re

import numpy as np
import pandas as pd




def make_input(adata_obs, **kwargs):
    '''
    :param adata:
    :param meta_file: 包含样本制作信息的表格，兼容 csv 和 xlsx，默认 header=True index=False
    :param batch_key:
    :param type_key:
    :return: pd.DataFrame，包含:
             sample_id, donor_id, disease,
             tissue, presort,cell_type,
             sampling_depth,
             prop, count, total_count
    '''
    # ⚡ 保证关键列都是字符串
    
    from src.core.kdk_ops import kdk_prepare
    
    default_params = {
        "sample_id": "orig.ident",
        "donor_id": "Patient",
        "disease": "disease",
        "tissue": "tissue-type",
        "presort": "presorted",
        "cell_type": "Subset_Identity",
        "prop": "percent",
        "count": "count",
        "total_count": "total_count"
    }
    default_params.update(kwargs)
    
    remap = {value: key for key, value in default_params.items()}
    
    count_df = kdk_prepare(adata_obs,
                           batch_key="orig.ident", type_key="Subset_Identity")
    
    meta_tmp = count_df.rename(columns=remap)
    meta = meta_tmp[list(default_params.keys())]
    
    return meta


def make_result(method: str, cell_type: str, p_value: float, adj_p_value=None, effect_size=None, extra=None,
                 alpha=0.05) -> Dict[str, Any]:
    if extra is None:
        extra = {}
    res = {
        "method": method,
        "cell_type": cell_type,
        "p_value": float(p_value) if p_value is not None else None,
        "adj_p_value": adj_p_value,
        "significant": bool(p_value is not None and p_value < alpha),
        "effect_size": effect_size,
        "extra": extra
    }
    return res



def remove_main_variable_from_formula(formula: str, main_variable: str) -> str:
    """
    Remove terms containing main_variable (or C(main_variable)) from a mixedlm-style formula.

    Parameters
    ----------
    formula : str
        Original formula, such as "y ~ disease + tissue + C(batch)".
    main_variable : str
        Variable name to remove.

    Returns
    -------
    str
        Cleaned formula with main_variable removed.
    """
    
    # 标准化字符串
    formula = formula.strip()
    
    # --- 1) 拆分左右部分 ---
    if "~" in formula:
        left, right = [x.strip() for x in formula.split("~", 1)]
    else:
        # 没有 ~，整个 formula 都是右侧
        left, right = "", formula
    
    # --- 2) 拆右侧 terms ---
    terms = [t.strip() for t in right.split("+")]
    
    # --- 3) 需要去除的模式 ---
    # 精确匹配 term，例如
    #   disease
    #   C(disease)
    #   C(disease, <other args>)
    #
    # 注意：不删除 disease_stage 这种包含但不同的变量
    pattern_exact = re.compile(rf"^{re.escape(main_variable)}$")
    pattern_C = re.compile(rf"^C\(\s*{re.escape(main_variable)}\s*(,.*)?\)$")
    
    cleaned_terms = [
        t for t in terms
        if not (pattern_exact.match(t) or pattern_C.match(t))
    ]
    
    # 如果右边全空了，默认放一个 1（类似 R 的 intercept）
    if not cleaned_terms:
        cleaned_right = "1"
    else:
        cleaned_right = " + ".join(cleaned_terms)
    
    # --- 4) 重新组装 ---
    if left:
        return f"{left} ~ {cleaned_right}"
    else:
        return cleaned_right


def parse_formula_columns(formula: str):
    """
    Parse RHS variable names from a formula like 'y ~ a + b + C(c) + I(d**2)'
    Returns: list of column names ['a', 'b', 'c', 'd']
    """
    # 1. 拆分左右式
    if '~' not in formula:
        raise ValueError("Formula must contain '~'")
    rhs = formula.split('~', 1)[1].strip()
    
    # 2. 用 '+' 拆分 term
    terms = [t.strip() for t in rhs.split('+')]
    
    cols = []
    for t in terms:
        # C(variable, ...) → variable
        m = re.match(r'C\(([^,]+)', t)
        if m:
            cols.append(m.group(1).strip())
            continue
        
        # I(expression) → expression 内的变量
        m = re.match(r'I\((.+)\)', t)
        if m:
            expr = m.group(1)
            # 抽取字母开头的变量名称（排除数字/函数）
            cols.extend(re.findall(r'[A-Za-z_]\w*', expr))
            continue
        
        # 普通变量
        # 排除 "."（允许用户写 y ~ .）
        if t != '.':
            cols.append(t)
    
    # 去重 preserve order
    seen = set()
    final = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            final.append(c)
    return final


def split_C_terms(series):
    """
    Input: pd.Series of patsy-style terms, e.g.
        'C(tissue, Treatment(reference="nif"))[T.if]'
    Output: pd.DataFrame with columns ['baseline', 'category']
    """
    
    def _split_term(term):
        if pd.isna(term):
            return pd.Series([None, None])
        # baseline
        m_ref = re.search(r'reference\s*=\s*["\']([^"\']+)["\']', term)
        baseline = m_ref.group(1) if m_ref else None
        # category
        m_cat = re.search(r'\[T\.([^\]]+)\]', term)
        category = m_cat.group(1) if m_cat else None
        return pd.Series([baseline, category])
    
    return series.apply(_split_term).rename(columns={0: 'baseline', 1: 'category'})


def collapse_dunn_matrix(dunn_p_matrix, group_means, ref="HC", alpha=0.05):
    """
    将 Dunn's test 全矩阵压缩成与 LMM/Dirichlet 对齐的结果格式。
    """
    results = []
    ref_mean = group_means[ref]
    
    for other in dunn_p_matrix.index:
        if other == ref:
            continue
        
        p = dunn_p_matrix.loc[other, ref]
        
        # direction
        other_mean = group_means[other]
        direction = "other_greater" if other_mean > ref_mean else "ref_greater"
        
        results.append({
            "ref": ref,
            "other": other,
            "mean_ref": ref_mean,
            "mean_other": other_mean,
            "prop_diff": other_mean - ref_mean,
            "P>|z|": p,
            "significant": p < alpha,
            "direction": direction
        })
    
    return pd.DataFrame(results).set_index("other")


def extract_contrast(ref_label, means, tukey_res):
    """Extract contrast results relative to a specified reference level."""
    
    # Normalize means into a dict-like structure
    if isinstance(means, dict):
        mean_lookup = means
    else:
        mean_lookup = dict(zip(means.index, means.values))
    
    table_data = tukey_res._results_table.data[1:]  # skip header
    contrast_rows = []
    
    for row in table_data:
        g1, g2, meandiff, p_adj, lo, hi, reject = row
        
        # skip unrelated comparisons
        if ref_label not in (g1, g2):
            continue
        
        # determine ref / other
        if g1 == ref_label:
            ref = g1
            other = g2
            coef = meandiff  # g1 - g2 = ref - other
        else:
            ref = g2
            other = g1
            coef = -meandiff  # reverse sign => (ref - other)
            
            # CIs also need to flip sign
            lo, hi = -hi, -lo
        
        # determine direction
        direction = "ref_greater" if coef > 0 else "other_greater"
        
        contrast_rows.append({
            "ref": ref,
            "other": other,
            "mean_ref": mean_lookup[ref],
            "mean_other": mean_lookup[other],
            "prop_diff": mean_lookup[other] - mean_lookup[ref],
            "Coef": coef,
            "p_adj": p_adj,
            "ci_low": lo,
            "ci_high": hi,
            "significant": bool(reject),
            "direction": direction,
        })
    
    return contrast_rows
