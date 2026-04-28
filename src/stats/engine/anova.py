from __future__ import annotations
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from src.stats.support import *
from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


@logged
def run_ANOVA_naive(
        df_all: pd.DataFrame,
        cell_type: str,
        formula: str = "disease",
        ref_label: str = "HC",
        alpha: float = 0.05
) -> Dict[str, Any]:
    """对单个 cell subtype/subpopulation 的比例直接执行 ANOVA。

    该方法直接以 ``prop`` 作为响应变量，不处理组成性约束，也不加入随机效应。
    当分组变量存在多个水平时，会额外执行 Tukey HSD，并提取 ``ref_label``
    与其他水平的 post-hoc 对比结果。该方法适合快速探索，不适合作为复杂实验
    设计的最终统计结论。

    Args:
        df_all: 长表格式的丰度数据，至少包含 ``cell_type``、``prop`` 以及
            ``formula`` 右侧引用的列。
        cell_type: 需要检验的 cell subtype/subpopulation 名称。
        formula: statsmodels 公式。可传 ``"disease"``，函数会自动补成
            ``"prop ~ disease"``；也可直接传完整公式。
        ref_label: post-hoc 对比中作为参考组的标签，通常为 ``"HC"``。
        alpha: 显著性阈值，同时传递给 Tukey HSD 和 ``make_result``。

    Returns:
        标准 ``make_result`` 字典。``extra["anova_table"]`` 保存 ANOVA 表，
        ``extra["means"]`` 保存各组平均比例；无法拟合时 ``extra["error"]``
        会记录英文错误信息。

    Example:
        >>> res = run_ANOVA_naive(
        ...     df_all=abundance_df,
        ...     cell_type="CD4_Tcm",
        ...     formula="disease",
        ...     ref_label="HC",
        ...     alpha=0.05,
        ... )
        >>> res["method"]
        'ANOVA_naive'
        >>> res["contrast_table"]  # 查看 HC 与其他 disease 水平的 Tukey 对比
        # DataFrame indexed by other labels, containing p_adj and direction.
    """
    if '~' not in formula:
        formula = f"prop ~ {formula}"
    extra = {}
    required_cols = {"cell_type", "prop"}
    missing_cols = required_cols - set(df_all.columns)
    if missing_cols:
        return make_result("ANOVA_naive", cell_type, np.nan, "Global",
                           contrast_table=None,
                           extra={"error": f"Missing required columns: {sorted(missing_cols)}"},
                           alpha=alpha)

    # 只保留目标 cell subtype/subpopulation，避免其他亚群影响当前模型。
    df_sub = df_all[df_all["cell_type"] == cell_type].copy()
    if df_sub.empty:
        return make_result("ANOVA_naive", cell_type, np.nan, "Global",
                           contrast_table=None,
                           extra={"error": f"No rows for cell_type: '{cell_type}'"},
                           alpha=alpha)
    
    try:
        mod = smf.ols(formula, data=df_sub).fit()
        anova_table = sm.stats.anova_lm(mod, typ=2)
        # 取第一个非 Residual 项作为全局检验主效应，保留原先“通常为 disease”的语义。
        main_terms = [idx for idx in anova_table.index if idx != "Residual"]
        if not main_terms:
            raise ValueError("No non-residual ANOVA term found.")
        main_factor = main_terms[0]
        factor_col = main_factor
        if main_factor.startswith("C(") and ")" in main_factor:
            factor_col = main_factor.split("(", 1)[1].split(",", 1)[0].split(")", 1)[0].strip()
        p_main = anova_table.loc[main_factor, "PR(>F)"]

        means = df_sub.groupby(factor_col)["prop"].mean().to_dict()

        contrast_table = None
        if df_sub[factor_col].nunique(dropna=True) > 1:
            tukey = pairwise_tukeyhsd(
                endog=df_sub["prop"],
                groups=df_sub[factor_col],
                alpha=alpha
            )
            contrast_rows = extract_contrast(ref_label, means, tukey)
            contrast_table = pd.DataFrame(contrast_rows).set_index("other") if contrast_rows else None

        extra.update({"anova_table": anova_table, "means": means})
    except Exception as e:
        p_main = np.nan
        contrast_table = None
        extra["error"] = str(e)

    return make_result(method="ANOVA_naive",
                       cell_type=cell_type,
                       p_val=p_main, p_type='Global',
                       contrast_table=contrast_table,
                       extra=extra,
                       alpha=alpha)

@logged
def run_ANOVA_transformed(
        df_all: pd.DataFrame,
        cell_type: str,
        formula: str = "disease",
        ref_label: str = "HC",
        alpha: float = 0.05
) -> Dict[str, Any]:
    """对比例进行 arcsin-sqrt 变换后执行 ANOVA。

    arcsin-sqrt 变换用于缓解比例数据在 0 和 1 附近的方差不稳定问题。
    函数会使用变换后的 ``prop_trans`` 拟合模型，但 ``extra["means"]`` 仍保留
    原始 ``prop`` 尺度上的均值，便于解释 cell subtype/subpopulation 丰度差异。

    Args:
        df_all: 长表格式的丰度数据，至少包含 ``cell_type`` 和 ``prop``。
        cell_type: 需要检验的 cell subtype/subpopulation 名称。
        formula: statsmodels 公式。可传 ``"disease + tissue"``，函数会自动补成
            ``"prop_trans ~ disease + tissue"``。
        ref_label: Tukey post-hoc 中的参考组标签。
        alpha: 显著性阈值。

    Returns:
        标准 ``make_result`` 字典。``contrast_table`` 为参考组对其他组的
        Tukey 对比；拟合失败时返回 ``p_val=np.nan`` 并在 ``extra`` 中记录错误。

    Example:
        >>> res = run_ANOVA_transformed(
        ...     df_all=abundance_df,
        ...     cell_type="B_memory",
        ...     formula="disease + tissue",
        ...     ref_label="HC",
        ... )
        >>> res["extra"]["means"]  # 原始 prop 尺度，方便报告丰度方向
        # {'HC': 0.12, 'BD': 0.18, ...}
    """
    if '~' not in formula:
        formula = f"prop_trans ~ {formula}"
    extra = {}
    required_cols = {"cell_type", "prop"}
    missing_cols = required_cols - set(df_all.columns)
    if missing_cols:
        return make_result("ANOVA_transformed", cell_type, np.nan, "Global",
                           contrast_table=None,
                           extra={"error": f"Missing required columns: {sorted(missing_cols)}"},
                           alpha=alpha)

    df_sub = df_all[df_all["cell_type"] == cell_type].copy()
    if df_sub.empty:
        return make_result("ANOVA_transformed", cell_type, np.nan, "Global",
                           contrast_table=None,
                           extra={"error": f"No rows for cell_type: '{cell_type}'"},
                           alpha=alpha)

    try:
        # clip 是为了兜底浮点误差或上游比例异常，避免 sqrt 收到 0-1 以外的值。
        df_sub["prop_trans"] = np.arcsin(np.sqrt(df_sub["prop"].clip(0, 1)))

        mod = smf.ols(formula, data=df_sub).fit()
        anova_table = sm.stats.anova_lm(mod, typ=2)

        main_terms = [idx for idx in anova_table.index if idx != "Residual"]
        if not main_terms:
            raise ValueError("No non-residual ANOVA term found.")
        main_factor = main_terms[0]
        factor_col = main_factor
        if main_factor.startswith("C(") and ")" in main_factor:
            factor_col = main_factor.split("(", 1)[1].split(",", 1)[0].split(")", 1)[0].strip()
        p_main = anova_table.loc[main_factor, "PR(>F)"]

        means = df_sub.groupby(factor_col)["prop"].mean().to_dict()

        contrast_table = None
        if df_sub[factor_col].nunique(dropna=True) > 1:
            tukey = pairwise_tukeyhsd(
                endog=df_sub["prop_trans"],
                groups=df_sub[factor_col],
                alpha=alpha
            )
            contrast_rows = extract_contrast(ref_label, means, tukey)
            contrast_table = pd.DataFrame(contrast_rows).set_index("other") if contrast_rows else None

        extra.update({"anova_table": anova_table, "means": means})
    except Exception as e:
        p_main = np.nan
        contrast_table = None
        extra["error"] = str(e)

    return make_result(method="ANOVA_transformed",
                       cell_type=cell_type,
                       p_val=p_main, p_type='Global',
                       contrast_table=contrast_table,
                       extra=extra,
                       alpha=alpha
                       )
