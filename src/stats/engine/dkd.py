from __future__ import annotations
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from src.utils.hier_logger import logged
from src.utils.env_utils import ensure_package
from src.stats.support import (
    make_result,
    remove_main_variable_from_formula,
    collapse_dunn_matrix,
)

logger = logging.getLogger(__name__)

try:
    import scikit_posthocs as sp
except ImportError:
    sp = None

# -----------------------
# Method 1: DKD pipeline (basic implementation) √
# -----------------------
@logged
def run_DKD(df_all: pd.DataFrame,
            cell_type: str,
            formula: str = "disease",
            main_variable: str = None,
            ref_label: str = "HC",
            alpha: float = 0.05,
            group_label="sample_id",
            use_reml: bool = True) -> Dict[str, Any]:
    """运行 deconfounded Kruskal-Dunn 丰度检验。

    基本流程是：先用 LMM/OLS 从目标 cell subtype/subpopulation 的 ``prop`` 中去掉
    协变量或分组均值影响，再对残差按主要变量做 Kruskal-Wallis 全局检验；当
    ``scikit_posthocs`` 可用且主要变量超过两个水平时，进一步执行 Dunn post-hoc。

    Args:
        df_all: 长表丰度数据，至少包含 ``cell_type``、``prop``、``group_label``、
            ``disease`` 以及公式中引用的列。
        cell_type: 目标 cell subtype/subpopulation。
        formula: 右侧公式，例如 ``"disease"`` 或 ``"disease + tissue"``。
        main_variable: 主要解释变量。多因素公式中必须指定。
        ref_label: Dunn 对比的参考组标签。
        alpha: 显著性阈值。
        group_label: LMM 随机截距分组列；每组样本过少时会回退到 OLS。
        use_reml: LMM 是否使用 REML。

    Returns:
        标准 ``make_result`` 字典。``extra["kw_disease_p"]`` 保存全局 K-W p 值；
        ``extra["dunn_p_matrix"]`` 保存 Dunn 原始矩阵；如果回退到 OLS，会在
        ``extra["model_type"]`` 中标记为 ``"OLS"``。

    Example:
        >>> res = run_DKD(
        ...     df_all=abundance_df,
        ...     cell_type="NK_CD56bright",
        ...     formula="disease + tissue",
        ...     main_variable="disease",
        ...     ref_label="HC",
        ...     group_label="donor_id",
        ... )
        >>> res["p_type"]
        'Global'
        >>> res["extra"].get("dunn_p_matrix")  # 多组 disease 时查看两两 post-hoc
        # Square matrix of adjusted Dunn p-values.
    """
    extra = {}
    contrast_table = pd.DataFrame()
    required_cols = {"cell_type", "prop", "disease", group_label}
    missing_cols = required_cols - set(df_all.columns)
    if missing_cols:
        return make_result(method="DKD",
                           cell_type=cell_type,
                           p_val=np.nan, p_type='Global',
                           contrast_table=contrast_table,
                           extra={"error": f"Missing required columns: {sorted(missing_cols)}"},
                           alpha=alpha)
    df = df_all[df_all.cell_type == cell_type]
    if df.empty:
        return make_result(method="DKD",
                           cell_type=cell_type,
                           p_val=np.nan, p_type='Global',
                           contrast_table=contrast_table,
                           extra={"error": f"No rows for cell_type: '{cell_type}'"},
                           alpha=alpha)
    
    if "+" in formula:
        if main_variable is None:
            raise KeyError("Main explanatory variable must be specified when `formula` contains more than one variable.")
    else:
        main_variable = formula
    
    formula_fixed = remove_main_variable_from_formula(formula, main_variable)
    group_means = df.groupby("disease")["prop"].mean()
    
    # Step 1: 跳过原有的第一步检验，直接拟合 `mixedlm` 去除混杂因素（Deconfound）
    # fit mixedlm intercept only + random intercept
    try:
        # 去除无法拟合的行；后面会把残差重新对齐到原 index，保留可追踪性。
        df_fit = df.dropna(subset=["prop", group_label]).copy()
        if df_fit.empty:
            raise ValueError("No valid rows after dropping missing `prop` or `group_label`.")
        group_means = df_fit.groupby("disease")["prop"].mean()
        
        formula_lmm = f"prop ~ {formula_fixed}"
        group_sizes = df_fit[group_label].value_counts()
        
        # 每个随机效应组只有一条记录时 MixedLM 容易奇异，回退 OLS 更稳。
        if group_sizes.min() < 2:
            use_mixedlm = False
        else:
            use_mixedlm = True
        
        if not use_mixedlm:
            model = smf.ols(formula_lmm, df_fit).fit()
            fitted = model.fittedvalues
            extra["model_type"] = "OLS"
            extra["ols_summary"] = model.summary()
        else:
            md = smf.mixedlm(formula_lmm, df_fit, groups=df_fit[group_label])
            mdf = md.fit(method="nm", maxiter=200, reml=use_reml)
            fitted = mdf.fittedvalues
            extra["model_type"] = "MixedLM"
            output = mdf.summary().tables
            extra["mixedlm_summary"] = output[0]
            extra["mixedlm_fixed_effect"] = output[1]
            if len(output) == 3:
                extra["mixedlm_random_effect"] = output[2]
        
        resid = df_fit["prop"] - fitted
        # 拟合前丢掉的行保留为 NaN，避免残差与原始行错位。
        residuals = pd.Series(index=df.index, dtype=float)
        residuals.loc[df_fit.index] = resid
    except Exception as e:
        residuals = pd.Series(index=df.index, data=df["prop"].values, dtype=float)
        extra["mixedlm_error"] = str(e)
        extra["model_type"] = "raw_prop_fallback"
    
    # Step 3: KW 检验
    p_disease = None
    try:
        if "disease" in df.columns:
            groups = []
            for _, g in df.groupby("disease"):
                grp_vals = residuals.loc[g.index].dropna().values
                if len(grp_vals) > 0:
                    groups.append(grp_vals)
            if len(groups) > 1:
                p_disease = float(stats.kruskal(*groups, nan_policy="omit").pvalue)
                extra["kw_disease_p"] = p_disease
    except Exception as e:
        extra["kw_disease_error"] = str(e)
    
    # Step 4: Dunn's test (if >2 groups and scikit_posthocs available)
    if sp is not None and "disease" in df.columns and df["disease"].nunique() > 2:
        try:
            # scikit-posthocs expects a DataFrame with values and group labels
            dunn = sp.posthoc_dunn(df.assign(resid=residuals),
                                   val_col="resid", group_col="disease",
                                   p_adjust="holm")
            extra["dunn_p_matrix"] = dunn
            contrast_table = collapse_dunn_matrix(
                dunn,
                group_means,
                ref=ref_label,
                alpha=alpha
            )
        except Exception as e:
            extra["dunn_error"] = str(e)
    elif sp is None:
        extra["dunn_warning"] = "Warning! scikit_posthocs is not installed; Dunn post-hoc was skipped."
    
    return make_result(method="DKD",
                       cell_type=cell_type,
                       p_val=p_disease if p_disease is not None else np.nan, p_type='Global',
                       contrast_table=contrast_table,
                       extra=extra,
                       alpha=alpha)
