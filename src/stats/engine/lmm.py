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
    '''
    直接使用线性混合模型（LMM）拟合。

    :param group_label:
    :param use_reml:
    :param df_all:
    :param cell_type:
    :param alpha:
    :param formula:
    :return: 标准的 make_result 输出，需要详细介绍的是 extra 部分的格式，直接打印出来格式可能会崩掉
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
    
    # 参数处理
    if "+" in formula:
        if main_variable is None:
            raise KeyError("Main explanatory variable must be specified when formula contains more than one variable.")
    else:
        main_variable = formula
    
    formula_fixed = remove_main_variable_from_formula(formula, main_variable)
    
    if main_variable != formula:
        formula = f"prop ~ C({main_variable}, Treatment(reference=\"{ref_label}\")) + {formula_fixed}"
    else:
        formula = f"prop ~ C({main_variable}, Treatment(reference=\"{ref_label}\"))"
    
    try:
        group = df_all[group_label]
        md = smf.mixedlm(formula, df_all, groups=group)
        mdf = md.fit(method="nm", maxiter=200, reml=use_reml)
        # p-values for fixed effects (note: sometimes p-values may be NaN)
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
