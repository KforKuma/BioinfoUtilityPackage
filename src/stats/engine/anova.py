from __future__ import annotations
import logging

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
        df_all,
        cell_type,
        formula="disease",
        ref_label="HC",
        alpha=0.05
):
    """
    Naive ANOVA: 直接对 prop 做 ANOVA，不考虑成分性或随机效应。
    在显著时执行 TukeyHSD 并提取 ref vs other 的对比。
    """
    if '~' not in formula:
        formula = f"prop ~ {formula}"
    # 过滤 cell_type
    df_sub = df_all[df_all["cell_type"] == cell_type].copy()
    if df_sub.empty:
        return {"error": f"No rows for {cell_type}"}
    
    # ---- 模型拟合 ----
    mod = smf.ols(formula, data=df_sub).fit()
    anova_table = sm.stats.anova_lm(mod, typ=2)
    # 提取主效应变量的名字（一般为 disease）
    main_factor = [idx for idx in anova_table.index if idx != "Residual"][0]
    p_main = anova_table.loc[main_factor, "PR(>F)"]
    
    # ---- 均值 ----
    factor_name = main_factor
    means = df_sub.groupby(factor_name)["prop"].mean().to_dict()
    
    # ---- Post-hoc: Tukey ----
    tukey = pairwise_tukeyhsd(
        endog=df_sub["prop"],
        groups=df_sub[factor_name],
        alpha=alpha
    )
    contrast_rows = extract_contrast(ref_label, means, tukey)
    contrast_table = pd.DataFrame(contrast_rows).set_index("other") if contrast_rows else None
    # ---- 输出 ----
    return make_result(method="ANOVA_naive",
                       cell_type=cell_type,
                       p_val=p_main,p_type='Global',
                       contrast_table=contrast_table,
                       extra={"anova_table": anova_table,"means": means},
                       alpha=alpha
                       )

@logged
def run_ANOVA_transformed(
        df_all,
        cell_type,
        formula="disease",
        ref_label="HC",
        alpha=0.05
):
    """
    arcsin-sqrt transform 后再做 ANOVA，兼容多因素。
    """
    if '~' not in formula:
        formula = f"prop_trans ~ {formula}"
    
    df_sub = df_all[df_all["cell_type"] == cell_type].copy()
    if df_sub.empty:
        return {"error": f"No rows for {cell_type}"}
    
    # ---- arcsin sqrt transform ----
    df_sub["prop_trans"] = np.arcsin(np.sqrt(df_sub["prop"]))
    
    mod = smf.ols(formula, data=df_sub).fit()
    anova_table = sm.stats.anova_lm(mod, typ=2)
    
    main_factor = [idx for idx in anova_table.index if idx != "Residual"][0]
    p_main = anova_table.loc[main_factor, "PR(>F)"]
    
    # ---- 均值 (原 scale) ----
    means = df_sub.groupby(main_factor)["prop"].mean().to_dict()
    
    tukey = pairwise_tukeyhsd(
        endog=df_sub["prop_trans"],
        groups=df_sub[main_factor],
        alpha=alpha
    )
    contrast_rows = extract_contrast(ref_label, means, tukey)
    contrast_table = pd.DataFrame(contrast_rows).set_index("other") if contrast_rows else None
    return make_result(method="ANOVA_transformed",
                       cell_type=cell_type,
                       p_val=p_main,p_type='Global',
                       contrast_table=contrast_table,
                       extra = {"anova_table": anova_table,
                                "means": means},
                       alpha=alpha
                       )
