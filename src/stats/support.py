from __future__ import annotations

from typing import Any
import inspect
import re

import numpy as np
import pandas as pd

from src.stats.plot.plot import *

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)

@logged
def _from_adata_make_count(adata_obs, meta, batch_key="orig.ident", type_key="Subset_Identity"):
    """从 AnnData obs 风格表中生成 cell subtype/subpopulation count 长表。

    Args:
        adata_obs: ``adata.obs`` 或同结构 DataFrame。
        meta: 样本级元数据，至少包含 ``batch_key``。
        batch_key: 样本/批次列名。
        type_key: cell subtype/subpopulation 列名。

    Returns:
        包含 ``batch_key``、``type_key``、``count``、``percent``、``log_count``、
        ``logit_percent`` 和 ``total_count`` 的长表。

    Example:
        >>> count_df = _from_adata_make_count(
        ...     adata.obs,
        ...     meta,
        ...     batch_key="orig.ident",
        ...     type_key="Subset_Identity",
        ... )
        >>> count_df[["orig.ident", "Subset_Identity", "count", "percent"]].head()
        # 后续 make_input 会再把这些列重命名成 stats engine 统一列名。
    """
    count_dataframe = (
        adata_obs[[batch_key, type_key]]
        .groupby([batch_key, type_key])
        .size()
        .reset_index(name='count')
    )
    merge_df = pd.merge(count_dataframe, meta, how='inner', on=batch_key)
    count_group_df = merge_df
    
    count_group_df["log_count"] = np.log1p(count_group_df["count"])
    count_group_df["percent"] = count_group_df["count"] / count_group_df.groupby(batch_key)["count"].transform("sum")
    count_group_df["logit_percent"] = np.log(count_group_df["percent"] + 1e-5 / (1 - count_group_df["percent"] + 1e-5))
    count_group_df["total_count"] = count_group_df.groupby(batch_key)["count"].transform("sum")
    
    return count_group_df

@logged
def _from_adata_make_meta(adata_obs, group_key="orig.ident"):
    """从 obs 中推断样本级 metadata。

    Args:
        adata_obs: ``adata.obs`` 或同结构 DataFrame。
        group_key: 样本/批次列名。

    Returns:
        每个 ``group_key`` 一行的 metadata。若某个字符串列在同一样本内有多个值，
        该样本对应值会被置为 ``None``，避免误用不唯一元数据。

    Example:
        >>> meta = _from_adata_make_meta(adata.obs, group_key="orig.ident")
        >>> meta.head()
        # 可作为 _from_adata_make_count 的 meta 输入。
    """
    # 选出字符串列（object 或 string）
    if adata_obs.empty:
        raise ValueError("`adata_obs` must not be empty.")
    string_cols = [c for c in adata_obs.columns if isinstance(adata_obs[c].iloc[0], str)]
    
    # 确保 group_key 也在结果里
    if group_key not in string_cols:
        string_cols.append(group_key)
    
    def unique_or_none(x):
        vals = x.dropna().unique()
        if len(vals) == 1:
            return vals[0]
        else:
            return None  # 多值或空值用 None
    
    # 聚合
    df_grouped = adata_obs[string_cols].groupby(group_key).agg(unique_or_none).reset_index()
    
    # 去除全 None 列
    cols_remain = [c for c in df_grouped.columns if not df_grouped[c].isna().all()]
    df_grouped = df_grouped[cols_remain]
    
    return df_grouped


@logged
def input_prepare(adata_obs, meta_file=None, batch_key="orig.ident", type_key="Subset_Identity"):
    """准备丰度统计输入长表。

    Args:
        adata_obs: ``adata.obs`` 或同结构 DataFrame。
        meta_file: 可选样本 metadata 文件，支持 ``.csv`` 和 ``.xlsx``。
        batch_key: 样本/批次列名。
        type_key: cell subtype/subpopulation 列名。

    Returns:
        合并 metadata 后的 count 长表。

    Example:
        >>> count_df = input_prepare(
        ...     adata.obs,
        ...     meta_file="sample_meta.csv",
        ...     batch_key="orig.ident",
        ...     type_key="Subset_Identity",
        ... )
        >>> count_df.columns
        # 包含 count、percent 和 total_count 等下游字段。
    """
    # 读取 meta 信息
    if meta_file is None:
        meta = _from_adata_make_meta(adata_obs, batch_key)
    else:
        meta_file = meta_file.strip()
        if meta_file.lower().endswith("csv"):
            meta = pd.read_csv(meta_file)
        elif meta_file.lower().endswith("xlsx"):
            meta = pd.read_excel(meta_file)
        else:
            raise ValueError("`meta_file` must end with 'csv' or 'xlsx'.")
    
    # ⚡ 保证关键列都是字符串
    for key in [batch_key, type_key]:
        if key in meta.columns:
            meta[key] = meta[key].astype(str)
        if key in adata_obs.columns:
            adata_obs[key] = adata_obs[key].astype(str)
    
    # 准备丰度分析所需矩阵
    count_df = _from_adata_make_count(adata_obs, meta, batch_key=batch_key, type_key=type_key)
    
    return count_df

@logged
def make_input(adata_obs, **kwargs):
    """将 obs 风格数据转换为 stats engine 标准输入。

    Args:
        adata_obs: ``adata.obs`` 或同结构 DataFrame。
        **kwargs: 原始列名映射。默认会把 ``orig.ident`` 映射为 ``sample_id``，
            ``Subset_Identity`` 映射为 ``cell_type``。

    Returns:
        标准长表，包含 ``sample_id``、``donor_id``、``disease``、``tissue``、
        ``presort``、``cell_type``、``prop``、``count`` 和 ``total_count``。

    Example:
        >>> meta = make_input(
        ...     adata.obs,
        ...     donor_id="Patient",
        ...     tissue="tissue-type",
        ...     cell_type="Subset_Identity",
        ... )
        >>> meta[["sample_id", "cell_type", "count", "prop"]].head()
        # 可直接传入 run_LMM、run_Dirichlet_Wald 等函数。
    """
    
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
    
    count_df = input_prepare(adata_obs,
                           batch_key="orig.ident", type_key="Subset_Identity")
    
    meta_tmp = count_df.rename(columns=remap)
    meta = meta_tmp[list(default_params.keys())]
    
    return meta

@logged
def make_result(method: str,
                cell_type: str,
                p_val: float | None,
                p_type: str | None,
                contrast_table: pd.DataFrame = None,
                extra: dict | None = None,
                alpha: float = 0.05) -> dict[str, Any]:
    """构造 stats engine 的统一返回结构。

    Args:
        method: 方法名。
        cell_type: 目标 cell subtype/subpopulation。
        p_val: 主 p 值。
        p_type: p 值语义，例如 ``"Global"`` 或 ``"Minimal"``。
        contrast_table: 对比表。
        extra: 额外诊断信息。
        alpha: 显著性阈值。

    Returns:
        标准结果字典。

    Example:
        >>> make_result("LMM", "CD4_Tcm", 0.01, "Minimal", alpha=0.05)["significant"]
        True
    """
    if extra is None:
        extra = {}
    return {
        "method": method,
        "cell_type": cell_type,
        "p_val": float(p_val) if p_val is not None else None,
        "p_type": p_type,
        'contrast_table': contrast_table,
        "significant": bool(p_val is not None and p_val < alpha),
        "extra": extra
    }


def remove_main_variable_from_formula(formula: str, main_variable: str) -> str:
    """从公式右侧移除主要解释变量。

    Args:
        formula: 原始公式，例如 ``"y ~ disease + tissue + C(batch)"``。
        main_variable: 需要移除的变量名。

    Returns:
        移除主要变量后的公式右侧；若右侧为空则返回 ``"1"``。

    Example:
        >>> remove_main_variable_from_formula("disease + tissue", "disease")
        'tissue'
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
    """解析公式右侧引用的列名。

    Args:
        formula: 包含 ``~`` 的 patsy/statsmodels 公式。

    Returns:
        去重后的列名列表。

    Example:
        >>> parse_formula_columns("y ~ disease + C(tissue) + I(age**2)")
        ['disease', 'tissue', 'age']
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
    """拆解 patsy 分类变量项中的参考组和类别。

    Args:
        series: patsy 风格 term，例如
            ``C(tissue, Treatment(reference="nif"))[T.if]``。

    Returns:
        包含 ``baseline`` 和 ``category`` 两列的 DataFrame。

    Example:
        >>> split_C_terms(pd.Series(['C(tissue, Treatment(reference="nif"))[T.if]']))
        # baseline='nif', category='if'
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
    """将 Dunn's test 全矩阵压缩为统一 contrast_table。

    Args:
        dunn_p_matrix: Dunn post-hoc 的两两 p 值矩阵。
        group_means: 各组平均丰度。
        ref: 参考组。
        alpha: 显著性阈值。

    Returns:
        以 ``other`` 为索引的对比表。

    Example:
        >>> contrast = collapse_dunn_matrix(dunn, group_means, ref="HC")
        >>> contrast[["P>|z|", "direction"]]
        # 与 LMM/Dirichlet 输出字段对齐。
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
    """从 Tukey HSD 结果中提取参考组相关对比。

    Args:
        ref_label: 参考组标签。
        means: 各组平均比例。
        tukey_res: ``pairwise_tukeyhsd`` 返回对象。

    Returns:
        contrast row 字典列表。

    Example:
        >>> rows = extract_contrast("HC", means, tukey)
        >>> pd.DataFrame(rows).set_index("other")
        # 只包含 HC vs other 的 post-hoc 结果。
    """
    
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

def filter_kwargs_for_func(func, params_dict):
    """过滤参数字典，只保留目标函数接受的参数。

    Args:
        func: 目标函数。
        params_dict: 候选参数字典。

    Returns:
        过滤后的参数字典。

    Example:
        >>> filter_kwargs_for_func(run_LMM, {"df_all": df, "unused": 1})
        # {'df_all': df}
    """
    if func is None: return {}
    sig = inspect.signature(func)
    accepted = sig.parameters
    filtered = {k: v for k, v in params_dict.items() if k in accepted}
    return filtered

def merge_contrast_tables(tables_dict):
    """合并多个方法的 contrast_table。

    Args:
        tables_dict: ``{method_name: contrast_table}`` 字典。

    Returns:
        合并后的 DataFrame，非 ``ref``/``other`` 字段会加方法名前缀。

    Example:
        >>> merged = merge_contrast_tables({"LMM": lmm_ct, "CLR": clr_ct})
        >>> merged.head()
        # 用于横向比较多个 engine 的同一组对比。
    """

    merged = None
    for method, df in tables_dict.items():
        df_copy = df.copy()
        # 保留关键信息
        keep_cols = ["ref", "other", "mean_ref", "mean_other", "prop_diff",
                     "Coef", "p_adj", "significant", "direction"]
        for col in df_copy.columns:
            if col not in keep_cols:
                df_copy = df_copy.drop(columns=col)

        # 为列加方法前缀
        df_copy = df_copy.rename(columns={c: f"{method}_{c}" for c in df_copy.columns if c not in ["ref", "other"]})

        if merged is None:
            merged = df_copy
        else:
            merged = pd.merge(merged, df_copy, on=["ref", "other"], how="outer")

    return merged
