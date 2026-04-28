from __future__ import annotations
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

import statsmodels.formula.api as smf

from src.stats.support import (
    make_result,
    remove_main_variable_from_formula,
    split_C_terms,
)

from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


# -----------------------
# Method 2: LMM direct √
# -----------------------
@logged
def run_LMM(df_all: pd.DataFrame,
            cell_type: str,
            formula: str = "disease",
            main_variable: str = None,
            ref_label: str = "HC",
            group_label: str = "sample_id",
            use_reml: bool = True,
            alpha: float = 0.05) -> Dict[str, Any]:
    """直接使用线性混合模型检验单个 cell subtype/subpopulation 丰度。

    函数会先按 ``cell_type`` 过滤长表，然后拟合 ``prop`` 的 LMM。默认把
    ``group_label`` 作为随机截距分组，用来吸收 sample/donor 层面的重复测量结构。
    当 ``formula`` 包含多个解释变量时，需要显式指定 ``main_variable``，这样函数
    才能为主要变量设置 ``ref_label`` 并正确解析对比表。

    Args:
        df_all: 长表丰度数据，至少包含 ``cell_type``、``prop``、``group_label``，
            以及 ``formula`` 右侧用到的列。
        cell_type: 目标 cell subtype/subpopulation。
        formula: 右侧公式，例如 ``"disease"`` 或 ``"disease + tissue"``。
        main_variable: 主要解释变量。多因素公式中必须传入，例如 ``"disease"``。
        ref_label: ``main_variable`` 的参考水平，通常为 ``"HC"``。
        group_label: 随机截距分组列，默认 ``"sample_id"``。
        use_reml: 是否使用 REML 拟合，默认适合小样本方差估计。
        alpha: 显著性阈值。

    Returns:
        标准 ``make_result`` 字典。``extra`` 中可能包含 ``mixedlm_summary``、
        ``mixedlm_fixed_effect``、``mixedlm_random_effect`` 和 ``contrast_table``。
        拟合失败时返回 ``p_val=np.nan``，并在 ``extra["error"]`` 保存英文错误。

    Example:
        >>> res = run_LMM(
        ...     df_all=abundance_df,
        ...     cell_type="CD8_Teff",
        ...     formula="disease + tissue",
        ...     main_variable="disease",
        ...     ref_label="HC",
        ...     group_label="donor_id",
        ... )
        >>> res["contrast_table"]  # disease 各水平相对 HC 的固定效应估计
        # DataFrame indexed by other labels, containing Coef., P>|z| and direction.
    """
    extra = {}
    required_cols = {"cell_type", "prop", group_label}
    missing_cols = required_cols - set(df_all.columns)
    if missing_cols:
        return make_result(method="LMM",
                           cell_type=cell_type,
                           p_val=np.nan, p_type=None,
                           contrast_table=None,
                           extra={"error": f"Missing required columns: {sorted(missing_cols)}"},
                           alpha=alpha)

    # 只对目标 cell subtype/subpopulation 建模，避免不同亚群混入同一个响应变量。
    df = df_all[df_all["cell_type"] == cell_type].copy()
    if df.empty:
        return make_result(method="LMM",
                           cell_type=cell_type,
                           p_val=np.nan, p_type=None,
                           contrast_table=None,
                           extra={"error": f"No rows for cell_type: '{cell_type}'"},
                           alpha=alpha)
    
    if "+" in formula:
        if main_variable is None:
            raise KeyError("Main explanatory variable must be specified when `formula` contains more than one variable.")
    else:
        main_variable = formula
    
    formula_fixed = remove_main_variable_from_formula(formula, main_variable)
    
    if main_variable != formula:
        formula = f"prop ~ C({main_variable}, Treatment(reference=\"{ref_label}\")) + {formula_fixed}"
    else:
        formula = f"prop ~ C({main_variable}, Treatment(reference=\"{ref_label}\"))"
    
    try:
        group = df[group_label]
        md = smf.mixedlm(formula, df, groups=group)
        mdf = md.fit(method="nm", maxiter=200, reml=use_reml)
        # MixedLM 在奇异拟合时可能给出 NaN p 值，因此统一走 min(skipna)。
        pval = mdf.pvalues.min()
        
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
        extra["contrast_table"] = contrast_table
        if len(output) == 3:
            extra["mixedlm_random_effect"] = output[2]
        return make_result(method="LMM",
                           cell_type=cell_type,
                           p_val=pval if pval is not None else np.nan,p_type='Minimal',
                           contrast_table=contrast_table,
                           extra=extra,
                           alpha=alpha)
    except Exception as e:
        extra["error"] = str(e)
        return make_result(method="LMM", 
                           cell_type=cell_type, 
                           p_val=np.nan, p_type=None,
                           contrast_table=None, 
                           extra=extra, 
                           alpha=alpha)
