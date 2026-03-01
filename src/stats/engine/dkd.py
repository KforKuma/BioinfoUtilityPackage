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
    """
    基本步骤:
      1. Fit mixedlm prop ~ 1 + (1|group_key) and take residuals
      2. KW test on residuals: residual ~ disease
      3. Dunn post-hoc (if scikit_posthocs installed) between disease groups (if >2 groups)

    :return: 标准的 make_result 输出，需要详细介绍的是 extra 部分的格式
             如果第一步 K-W 成功：extra["kw_sampling_p"] = p_sampling
             失败，则：extra["kw_sampling_error"] = str(e)
             第二步 拟合线性模型成功：extra["mixedlm_summary"] = summary
             失败，则：extra["mixedlm_error"] = str(e)
             第三步 K-W 成功：extra["kw_disease_p"] = p_disease
             失败，则：extra["kw_disease_error"] = p_disease
             第四步 Dunn post-hoc 成功：extra["dunn_p_matrix"] = p_matrix
             失败，则：extra["dunn_error"] = str(e)
             Dunn post-hoc 的 p_matrix 格式为两两一组的 p 值矩阵，无任何修饰：
                                        1             2             3
             1  1.000000e+00  2.047087e-14  1.536598e-07
             2  2.047087e-14  1.000000e+00  1.580934e-02
             3  1.536598e-07  1.580934e-02  1.000000e+00



    """
    df = df_all[df_all.cell_type == cell_type]
    
    extra = {}
    
    if "+" in formula:
        if main_variable is None:
            raise KeyError("Main explanatory variable must be specified when formula contains more than one variable.")
    else:
        main_variable = formula
    
    formula_fixed = remove_main_variable_from_formula(formula, main_variable)
    
    # Step 1: 跳过原有的第一步检验，直接拟合 `mixedlm` 去除混杂因素（Deconfound）
    # fit mixedlm intercept only + random intercept
    try:
        # ensure no NaN in prop
        df_fit = df.dropna(subset=["prop", group_label]).copy()
        group_means = df_fit.groupby("disease")["prop"].mean()
        
        formula_lmm = f"prop ~ {formula_fixed}"
        group_sizes = df_fit[group_label].value_counts()
        
        if group_sizes.min() < 2:
            # fallback: no random effect
            use_mixedlm = False
        else:
            use_mixedlm = True
        
        if not use_mixedlm:
            model = smf.ols(formula_lmm, df_fit).fit()
            fitted = model.fittedvalues
        else:
            md = smf.mixedlm(formula_lmm, df_fit, groups=df_fit[group_label])
            mdf = md.fit(...)
            fitted = mdf.fittedvalues
        
        # residuals aligned to df_fit index
        resid = df_fit["prop"] - fitted
        # create residuals column for full df (NaN where dropped)
        residuals = pd.Series(index=df.index, dtype=float)
        residuals.loc[df_fit.index] = resid
        # 储存结果
        output = mdf.summary().tables
        extra["mixedlm_summary"] = output[0]
        extra["mixedlm_fixed_effect"] = output[1]
        if len(output) == 3:
            extra["mixedlm_random_effect"] = output[2]
    except Exception as e:
        extra["mixedlm_error"] = str(e)
    
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
                ref=ref_label
            )
        except Exception as e:
            contrast_table=None
            extra["dunn_error"] = str(e)
    
    # assemble result: p_disease is main
    return make_result(method="KDKD",
                       cell_type=cell_type,
                       p_val=p_disease if p_disease is not None else np.nan,p_type='Global',
                       contrast_table=contrast_table,
                       extra=extra,
                       alpha=alpha)
