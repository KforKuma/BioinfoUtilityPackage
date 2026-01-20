import re
import warnings
from typing import Dict, List, Tuple

from tqdm import tqdm

import numpy as np
import pandas as pd

from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammaln, psi, softmax   # psi = digamma
from scipy.stats import (
    norm,
    median_abs_deviation,
    dirichlet,
    multinomial,
)


import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from patsy import dmatrix

from src.core.kdk_support import *

##################################
# optional: posthoc dunn
try:
    import scikit_posthocs as sp
except Exception:
    sp = None
    warnings.warn("scikit_posthocs not installed; Dunn's post-hoc will be skipped if requested.")


##################################

# -----------------------
# Method 1: DKD pipeline (basic implementation) √
# -----------------------
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
    
    # Step 1: 跳过第一步检验，拟合 `mixedlm` 去除混杂因素（Deconfound）
    # fit mixedlm intercept only + random intercept
    try:
        # ensure no NaN in prop
        df_fit = df.dropna(subset=["prop", group_label]).copy()
        group_means = df_fit.groupby("disease")["prop"].mean()
        
        formula_lmm = f"prop ~ {formula_fixed}"
        md = smf.mixedlm(formula_lmm, df_fit, groups=df_fit[group_label])
        mdf = md.fit(method="nm", maxiter=200, reml=use_reml)
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
            collapse_df = collapse_dunn_matrix(
                dunn,
                group_means,
                ref=ref_label
            )
            extra["contrast_table"] = collapse_df
        except Exception as e:
            extra["dunn_error"] = str(e)
    
    # assemble result: p_disease is main
    return make_result("KDKD", cell_type, p_disease if p_disease is not None else np.nan,
                        effect_size=None, extra=extra, alpha=alpha)


# -----------------------
# Method 2: LMM direct √
# -----------------------
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
        pval = float(mdf.pvalues.get("disease", None)) if "disease" in mdf.pvalues else None
        eff = float(mdf.params.get("disease", None)) if "disease" in mdf.params else None
        # 储存结果
        output = mdf.summary().tables
        extra["mixedlm_summary"] = output[0]
        extra["mixedlm_fixed_effect"] = output[1]
        
        df_new = extra["mixedlm_fixed_effect"].copy()
        df_new = df_new[1:-1]
        df_new["ref"], df_new["other"] = split_C_terms(pd.Series(df_new.index)).T.values
        
        df_new["P>|z|"] = df_new["P>|z|"].astype(float)
        df_new["significant"] = (df_new["P>|z|"] < alpha).astype(str)
        
        df_new["Coef."] = df_new["Coef."].astype(float)
        df_new["direction"] = df_new["Coef."].apply(lambda x: "ref_greater" if x < 0 else "other_greater")
        
        df_new = df_new[
            ['ref', 'other', 'Coef.', 'Std.Err.', 'z', 'P>|z|', '[0.025', '0.975]', 'significant', 'direction']]
        df_new = pd.DataFrame(df_new).set_index("other")
        extra["contrast_table"] = df_new
        if len(output) == 3:
            extra["mixedlm_random_effect"] = output[2]
        return make_result("LMM", cell_type, pval if pval is not None else np.nan, effect_size=eff, extra=extra,
                            alpha=alpha)
    except Exception as e:
        extra["error"] = str(e)
        return make_result("LMM", cell_type, np.nan, effect_size=None, extra=extra, alpha=alpha)


# -----------------------
# Method 3: CLR + LMM √
# -----------------------
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
        pval = float(mdf.pvalues.get("disease", np.nan)) if "disease" in mdf.pvalues else None
        eff = float(mdf.params.get("disease", np.nan)) if "disease" in mdf.params else None
        # 储存结果
        output = mdf.summary().tables
        extra["mixedlm_summary"] = output[0]
        extra["mixedlm_fixed_effect"] = output[1]
        
        df_new = extra["mixedlm_fixed_effect"].copy()
        df_new = df_new[1:-1]
        df_new["ref"], df_new["other"] = split_C_terms(pd.Series(df_new.index)).T.values
        
        df_new["P>|z|"] = df_new["P>|z|"].astype(float)
        df_new["significant"] = (df_new["P>|z|"] < alpha).astype(str)
        
        df_new["Coef."] = df_new["Coef."].astype(float)
        df_new["direction"] = df_new["Coef."].apply(lambda x: "ref_greater" if x < 0 else "other_greater")
        
        df_new = df_new[
            ['ref', 'other', 'Coef.', 'Std.Err.', 'z', 'P>|z|', '[0.025', '0.975]', 'significant', 'direction']]
        df_new = pd.DataFrame(df_new).set_index("other")
        extra["contrast_table"] = df_new
        
        if len(output) == 3:
            extra["mixedlm_random_effect"] = output[2]
        
        # print("CLR-LMM function run successfully.")
        
        return make_result("CLR_LMM", cell_type_label,
                            pval if pval is not None else np.nan,
                            effect_size=eff, extra=extra, alpha=alpha)
    
    except Exception as e:
        extra["error"] = str(e)
        print("CLR-LMM function failed.")
        return make_result("CLR_LMM", cell_type, np.nan, effect_size=None, extra=extra, alpha=alpha)


def run_CLR_LMM_with_LFC(
        df_all: pd.DataFrame,
        cell_type: str,
        formula: str = "disease + tissue",
        main_variable: str = None,
        coef_threshold: float = 0.2,
        **kwargs
) -> Dict[str, Any]:
    """
    运行 CLR LMM 检验并应用 LFC 过滤。
    """
    res = run_CLR_LMM(df_all, cell_type, formula=formula,main_variable=main_variable, **kwargs)
    
    if res['extra'] and res['extra'].get('contrast_table') is not None:
        ct = res['extra']['contrast_table']
        ct['significant'] = (ct['P>|z|'] < 0.05) & (ct['Coef.'].abs() > coef_threshold)
        
        res['significant'] = any(ct['significant'])
        res['P>|z|'] = ct.loc[ct['significant'], 'P>|z|'].min() if res['significant'] else 1.0
    
    return res
# -----------------------
# Method 4: Dirichlet regression (placeholder)
# -----------------------
def _dirichlet_loglik(params, Y, X):
    """
    params: β flattened (k celltypes × p covariates), shape = k*p
    Y: proportions (n × k)
    X: design (n × p)
    """
    n, k = Y.shape
    p = X.shape[1]
    B = params.reshape((k, p))  # k × p
    
    # linear predictor for α
    eta = X @ B.T  # n × k
    alpha = np.exp(eta)  # n × k, each row > 0
    
    # LL = Σ_i [ ln Γ(Σ_j α_ij) - Σ_j ln Γ(α_ij) + Σ_j (α_ij - 1)*ln Y_ij ]
    ll = np.sum(
        gammaln(np.sum(alpha, axis=1))
        - np.sum(gammaln(alpha), axis=1)
        + np.sum((alpha - 1) * np.log(Y + 1e-12), axis=1)
    )
    return -ll  # minimize negative log-likelihood


def _neg_loglik_and_grad(flat_params, Y, X, K, P):
    """
    纯 Dirichlet 模型的负对数似然和梯度 (激进版核心)。
    假设数据完全符合 Dirichlet 分布，不考虑过度离散。
    """
    n = Y.shape[0]
    B = flat_params.reshape((K - 1, P))
    
    eta = X @ B.T
    eta_full = np.hstack([eta, np.zeros((n, 1))])
    alpha = np.exp(eta_full)
    alpha0 = alpha.sum(axis=1)
    
    Y_safe = np.clip(Y, 1e-12, None)
    
    ll_terms = gammaln(alpha0) - np.sum(gammaln(alpha), axis=1) + np.sum((alpha - 1) * np.log(Y_safe), axis=1)
    neg_ll = -ll_terms.sum()
    
    # Gradient
    digamma_alpha0 = psi(alpha0)
    digamma_alpha = psi(alpha)
    F = (digamma_alpha0[:, None] - digamma_alpha + np.log(Y_safe)) * alpha
    
    grad = np.zeros((K - 1, P))
    for k in range(K - 1):
        grad[k, :] = - (X.T @ F[:, k])
    
    return neg_ll, grad.ravel()


def _neg_loglik_and_grad_DM(flat_params: np.ndarray, Y: np.ndarray, X: np.ndarray, K: int, P: int, C: np.ndarray,
                            N: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Dirichlet-Multinomial (DM) 模型的负对数似然和梯度 (稳健版核心)。
    引入 alpha_sum 参数来模拟数据的过度离散 (Overdispersion)。
    """
    n = Y.shape[0]
    
    # 1. 提取参数
    B = flat_params[:-1].reshape((K - 1, P))
    log_alpha_sum = flat_params[-1]
    alpha_sum = np.exp(log_alpha_sum)
    
    # 2. 估计 Dirichlet 均值 E[P]
    eta = X @ B.T
    eta_full = np.hstack([eta, np.zeros((n, 1))])
    exp_eta_full = np.exp(eta_full)
    P_hat = exp_eta_full / exp_eta_full.sum(axis=1)[:, None]
    
    # 3. 计算 alpha 向量 (Mean * Precision)
    alpha = P_hat * alpha_sum
    
    # 4. DM 负对数似然
    ll_terms = (
            gammaln(alpha_sum)
            - gammaln(alpha_sum + N)
            + np.sum(gammaln(alpha + C), axis=1)
            - np.sum(gammaln(alpha), axis=1)
    )
    neg_ll = -ll_terms.sum()
    
    # 5. 梯度计算
    digamma_diff = psi(alpha + C) - psi(alpha)
    digamma_alpha_sum_diff = psi(alpha_sum + N) - psi(alpha_sum)
    
    G = digamma_diff - digamma_alpha_sum_diff[:, None]
    H = G - np.sum(P_hat * G, axis=1)[:, None]
    
    # Grad Beta
    grad_beta = np.zeros((K - 1, P))
    for k in range(K - 1):
        grad_beta[k, :] = - (X.T @ H[:, k])
    
    # Grad Log(alpha_sum)
    term_alpha_sum = (psi(alpha_sum) - psi(alpha_sum + N)) + np.sum(P_hat * digamma_diff, axis=1)
    grad_log_alpha_sum = - alpha_sum * term_alpha_sum.sum()
    
    grad_flat = np.concatenate([grad_beta.ravel(), np.array([grad_log_alpha_sum])])
    return neg_ll, grad_flat


# ==============================================================================
# 2. 激进版主函数: run_Dirichlet_Wald
# ==============================================================================

def run_Dirichlet_Wald(df_all: pd.DataFrame,
                       cell_type: str,
                       formula: str = "disease",
                       ref_label: str = "HC",
                       group_label="sample_id",
                       maxiter: int = 1000,
                       alpha: float = 0.05,
                       verbose: bool = False) -> Dict[str, Any]:
    """
    激进版：假设无过度离散。通常会产生更显著的 P 值。
    """
    method_name = "Dirichlet_Standard_Wald"  # 更新名称以示区别
    
    # 1) Pivot Data
    wide = df_all.pivot_table(index=group_label, columns="cell_type", values="count",
                              aggfunc="sum", fill_value=0)
    celltypes = list(wide.columns)
    n_samples, K = wide.shape
    
    if cell_type not in celltypes:
        return make_result(method_name, cell_type, None, effect_size=None,
                           extra={"error": f"target cell_type '{cell_type}' not found"})
    
    # Ensure reference celltype is last
    if celltypes[-1] == cell_type:
        if K < 2:
            return make_result(method_name, cell_type, None, effect_size=None,
                               extra={"error": "Need >=2 cell types"})
        cols = celltypes[:-2] + [celltypes[-1], celltypes[-2]]
        wide = wide[cols]
        celltypes = list(wide.columns)
    
    # 2) Metadata & Design Matrix (CRITICAL: Run BEFORE defining 'counts'/'C')
    #    这样做是为了防止局部变量 C 覆盖 patsy.C
    meta = df_all.drop_duplicates(subset=[group_label]).set_index(group_label)
    meta = meta.reindex(wide.index)
    try:
        X_df = dmatrix("1 + " + formula, meta, return_type="dataframe")
    except Exception as e:
        return make_result(method_name, cell_type, None, effect_size=None,
                           extra={"error": f"patsy dmatrix error: {e}"})
    X = np.asarray(X_df)
    colnames = X_df.design_info.column_names
    P = X.shape[1]
    
    # 3) Proportions Y (Run AFTER dmatrix)
    counts = wide.values.astype(float)
    row_sums = counts.sum(axis=1)
    zero_rows = row_sums == 0
    if zero_rows.any():
        counts[zero_rows, :] = 1.0 / K
        row_sums = counts.sum(axis=1)
    Y = counts / row_sums[:, None]
    
    # 4) Init Params & Fit (Standard Dirichlet)
    init = np.zeros((K - 1) * P, dtype=float)
    try:
        def fun_and_jac(params):
            val, grad = _neg_loglik_and_grad(params, Y, X, K, P)
            return val, grad
        
        res = minimize(fun_and_jac, x0=init, method="BFGS", jac=True,
                       options={"maxiter": maxiter, "disp": verbose})
    except Exception as e:
        return make_result(method_name, cell_type, None, effect_size=None,
                           extra={"error": f"optimizer error: {e}"})
    
    extra = {"message": res.message} if res.success else {"warning": "optimizer did not converge",
                                                          "message": res.message}
    params = res.x.reshape((K - 1, P))
    
    k_index = celltypes.index(cell_type)
    if k_index == K - 1:
        return make_result(method_name, cell_type, None, effect_size=None,
                           extra={"error": "target cell_type became reference"})
    
    # 5) Post-hoc Analysis
    if "disease" not in meta.columns:
        return make_result(method_name, cell_type, None, effect_size=None, extra={"error": "missing 'disease' column"})
    
    all_groups = list(meta["disease"].astype(str).unique())
    groups = [ref_label] + [g for g in all_groups if g != ref_label] if ref_label in all_groups else all_groups
    meta["disease"] = pd.Categorical(meta["disease"].astype(str), categories=groups, ordered=True)
    disease_series = meta["disease"].astype(str)
    
    mean_X_by_group = {}
    for g in groups:
        idx = np.where(disease_series.values == g)[0]
        mean_X_by_group[g] = X[idx, :].mean(axis=0) if len(idx) > 0 else np.zeros((P,))
    
    mean_props = {}
    for g in groups:
        eta_kminus1 = mean_X_by_group[g] @ params.T
        eta_full = np.concatenate([eta_kminus1, [0.0]])
        alpha_dir = np.exp(eta_full)
        mean_props[g] = float(alpha_dir[k_index] / alpha_dir.sum())
    
    # Wald Tests
    nparam = (K - 1) * P
    try:
        hess_inv = res.hess_inv
        if hasattr(hess_inv, "todense"): hess_inv = np.asarray(hess_inv.todense())
        
        # Fixed Effects
        fe_rows = []
        for j, term in enumerate(colnames):
            coef = float(params[k_index, j])
            idx = int(k_index * P + j)
            var_j = float(hess_inv[idx, idx])
            se_j = float(np.sqrt(abs(var_j)))
            z_j = coef / (se_j + 1e-12)
            p_j = float(2.0 * (1.0 - norm.cdf(abs(z_j))))
            fe_rows.append({"term": term, "Coef": coef, "Std.Err": se_j, "z": z_j, "P>|z|": p_j,
                            "2.5%": coef - 1.96 * se_j, "97.5%": coef + 1.96 * se_j})
        fixed_effect_df = pd.DataFrame(fe_rows).set_index("term")
        
        # Contrasts
        mean_X_ref = mean_X_by_group.get(ref_label, np.zeros(P))
        contrast_rows = []
        for g in groups:
            if g == ref_label:
                contrast_rows.append({"ref": ref_label, "other": g, "mean_ref": mean_props[ref_label],
                                      "mean_other": mean_props[ref_label], "P>|z|": np.nan, "significant": False})
                continue
            
            mean_X_g = mean_X_by_group[g]
            delta = float((mean_X_g - mean_X_ref) @ params[k_index, :])
            
            c = np.zeros((nparam,), dtype=float)
            c[k_index * P: (k_index + 1) * P] = (mean_X_g - mean_X_ref)
            
            var = float(c @ (hess_inv @ c))
            se = float(np.sqrt(abs(var)))
            z = delta / (se + 1e-12)
            pval = float(2.0 * (1.0 - norm.cdf(abs(z))))
            
            # Prop Diff Calculation
            eta_ref = np.zeros(K);
            eta_g = np.zeros(K)
            for kk in range(K - 1):
                eta_ref[kk] = mean_X_ref @ params[kk, :];
                eta_g[kk] = mean_X_g @ params[kk, :]
            
            pred_prop_ref = np.exp(eta_ref)[k_index] / np.exp(eta_ref).sum()
            pred_prop_g = np.exp(eta_g)[k_index] / np.exp(eta_g).sum()
            
            contrast_rows.append({
                "ref": ref_label, "other": g, "mean_ref": mean_props[ref_label], "mean_other": mean_props[g],
                "prop_diff": pred_prop_g - pred_prop_ref, "Coef": delta, "Std.Err": se, "z": z, "P>|z|": pval,
                "direction": "ref_greater" if (pred_prop_g - pred_prop_ref) < 0 else "other_greater",
                "significant": bool(pval < alpha)
            })
        
        # Append fixed effects (tissue etc) to contrasts
        for term, row in fixed_effect_df.iterrows():
            if term.startswith('C('):
                name_split = split_C_terms(pd.Series(term))  # Dependency check
                contrast_rows.append({
                    "ref": name_split.iloc[0, 0], "other": name_split.iloc[0, 1],
                    "Coef": row["Coef"], "Std.Err": row["Std.Err"], "z": row["z"], "P>|z|": row["P>|z|"],
                    "direction": "ref_greater" if row["Coef"] < 0 else "other_greater",
                    "significant": row["P>|z|"] < 0.05
                })
        
        extra.update({"contrast_table": pd.DataFrame(contrast_rows).set_index("other"),
                      "fixed_effect": fixed_effect_df, "groups": groups})
    
    except Exception as e:
        extra.update({"hess_inv_error": str(e), "groups": groups})
    
    return make_result(method_name, cell_type, None, effect_size=None, extra=extra)


def run_Dirichlet_Wald_with_LFC(
        df_all: pd.DataFrame,
        cell_type: str,
        formula: str = "disease",
        coef_threshold: float = 0.2,  # 新增 LFC 阈值参数
        **kwargs
) -> Dict[str, Any]:
    """
    运行 Dirichlet Wald 检验并应用 LFC 过滤。
    """
    # 假设 run_Dirichlet_Wald 是你现有的原始函数
    res = run_Dirichlet_Wald(df_all, cell_type, formula=formula, **kwargs)
    
    # 获取 contrast_table 进行二次加工
    if res['extra'] and res['extra'].get('contrast_table') is not None:
        ct = res['extra']['contrast_table']
        # 重新定义显著性：P值达标且效应量足够大
        # 注意：这里的 Coef 在 CLR/Dirichlet 空间通常对应 Log 尺度的变化
        ct['significant'] = (ct['P>|z|'] < 0.05) & (ct['Coef'].abs() > coef_threshold)
        
        # 同步更新顶级指标
        res['significant'] = any(ct['significant'])
        res['P>|z|'] = ct.loc[ct['significant'], 'P>|z|'].min() if res['significant'] else 1.0
    
    return res
# ==============================================================================
# 3. 稳健版主函数: run_Dirichlet_Multinomial_Wald
# ==============================================================================

def run_Dirichlet_Multinomial_Wald(df_all: pd.DataFrame,
                                   cell_type: str,
                                   formula: str = "disease",
                                   ref_label: str = "HC",
                                   group_label: str = "sample_id",
                                   maxiter: int = 1000,
                                   alpha: float = 0.05,
                                   verbose: bool = False) -> Dict[str, Any]:
    """
    稳健版：估计 alpha_sum (Overdispersion)。通常 P 值较保守，但更真实。
    """
    method_name = "Dirichlet_Multinomial_Wald"  # 更新名称
    
    # 1) Pivot Data
    wide = df_all.pivot_table(index=group_label, columns="cell_type", values="count",
                              aggfunc="sum", fill_value=0)
    celltypes = list(wide.columns)
    n_samples, K = wide.shape
    
    if cell_type not in celltypes:
        return make_result(method_name, cell_type, None, effect_size=None,
                           extra={"error": f"target cell_type '{cell_type}' not found"})
    
    # Target Swap Logic
    target_idx = celltypes.index(cell_type)
    if target_idx == K - 1:
        cols_to_swap = celltypes.copy()
        cols_to_swap[target_idx], cols_to_swap[-2] = cols_to_swap[-2], cols_to_swap[target_idx]
        wide = wide[cols_to_swap]
        celltypes = list(wide.columns)
        target_idx = celltypes.index(cell_type)
    
    # 2) Metadata & Design Matrix (CRITICAL: Run BEFORE defining 'C')
    meta = df_all.drop_duplicates(subset=[group_label]).set_index(group_label)
    meta = meta.reindex(wide.index)
    try:
        X_df = dmatrix("1 + " + formula, meta, return_type="dataframe")
    except Exception as e:
        return make_result(method_name, cell_type, None, effect_size=None,
                           extra={"error": f"patsy dmatrix error: {e}"})
    X = np.asarray(X_df)
    colnames = X_df.design_info.column_names
    P = X.shape[1]
    
    # 3) Counts C, Total N, Props Y (Run AFTER dmatrix)
    counts = wide.values.astype(float)
    row_sums = counts.sum(axis=1)
    if (row_sums == 0).any():
        counts[row_sums == 0, :] = 1.0 / K
        row_sums = counts.sum(axis=1)
    
    C = counts
    N = row_sums
    Y = C / N[:, None]
    
    # 4) Init Params & Fit (DM with alpha_sum)
    init_beta = np.zeros((K - 1) * P, dtype=float)
    init_alpha_sum_log = np.log(160.0)  # Empirical Init
    init = np.concatenate([init_beta, [init_alpha_sum_log]])
    nparam_total = (K - 1) * P + 1
    
    try:
        def fun_and_jac_DM(params):
            val, grad = _neg_loglik_and_grad_DM(params, Y, X, K, P, C, N)
            return val, grad
        
        res = minimize(fun_and_jac_DM, x0=init, method="BFGS", jac=True,
                       options={"maxiter": maxiter, "disp": verbose})
    except Exception as e:
        return make_result(method_name, cell_type, None, effect_size=None,
                           extra={"error": f"optimizer error: {e}"})
    
    extra = {"message": res.message} if res.success else {"warning": "optimizer did not converge",
                                                          "message": res.message}
    
    # Extract Params
    params_full = res.x
    params = params_full[:-1].reshape((K - 1, P))
    alpha_sum_est = np.exp(params_full[-1])
    
    k_index = celltypes.index(cell_type)
    
    # 5) Post-hoc Analysis
    if "disease" not in meta.columns:
        return make_result(method_name, cell_type, None, effect_size=None, extra={"error": "missing 'disease' column"})
    
    all_groups = list(meta["disease"].astype(str).unique())
    groups = [ref_label] + [g for g in all_groups if g != ref_label] if ref_label in all_groups else all_groups
    meta["disease"] = pd.Categorical(meta["disease"].astype(str), categories=groups, ordered=True)
    disease_series = meta["disease"].astype(str)
    
    mean_X_by_group = {}
    for g in groups:
        idx = np.where(disease_series.values == g)[0]
        mean_X_by_group[g] = X[idx, :].mean(axis=0) if len(idx) > 0 else np.zeros((P,))
    
    mean_props = {}
    for g in groups:
        eta_kminus1 = mean_X_by_group[g] @ params.T
        eta_full = np.concatenate([eta_kminus1, [0.0]])
        alpha_dir = np.exp(eta_full)
        mean_props[g] = float(alpha_dir[k_index] / alpha_dir.sum())
    
    # Wald Tests
    try:
        hess_inv = res.hess_inv
        if hasattr(hess_inv, "todense"): hess_inv = np.asarray(hess_inv.todense())
        
        # Fixed Effects
        fe_rows = []
        for j, term in enumerate(colnames):
            coef = float(params[k_index, j])
            idx = int(k_index * P + j)
            var_j = float(hess_inv[idx, idx])
            se_j = float(np.sqrt(abs(var_j)))
            z_j = coef / (se_j + 1e-12)
            p_j = float(2.0 * (1.0 - norm.cdf(abs(z_j))))
            fe_rows.append({"term": term, "Coef": coef, "Std.Err": se_j, "z": z_j, "P>|z|": p_j,
                            "2.5%": coef - 1.96 * se_j, "97.5%": coef + 1.96 * se_j})
        
        # Append Alpha Sum Stats
        idx_alpha = nparam_total - 1
        se_alpha = float(np.sqrt(abs(hess_inv[idx_alpha, idx_alpha])))
        fe_rows.append({"term": "Log_Alpha_Sum", "Coef": params_full[-1], "Std.Err": se_alpha,
                        "z": np.nan, "P>|z|": np.nan})
        fixed_effect_df = pd.DataFrame(fe_rows).set_index("term")
        
        # Contrasts
        mean_X_ref = mean_X_by_group.get(ref_label, np.zeros(P))
        contrast_rows = []
        for g in groups:
            if g == ref_label:
                contrast_rows.append({"ref": ref_label, "other": g, "mean_ref": mean_props[ref_label],
                                      "mean_other": mean_props[ref_label], "P>|z|": np.nan, "significant": False})
                continue
            
            mean_X_g = mean_X_by_group[g]
            delta = float((mean_X_g - mean_X_ref) @ params[k_index, :])
            
            # Contrast vector (Last element for alpha_sum is 0)
            c = np.zeros((nparam_total,), dtype=float)
            c[k_index * P: (k_index + 1) * P] = (mean_X_g - mean_X_ref)
            
            var = float(c @ (hess_inv @ c))
            se = float(np.sqrt(abs(var)))
            z = delta / (se + 1e-12)
            pval = float(2.0 * (1.0 - norm.cdf(abs(z))))
            
            # Prop Diff
            eta_ref = np.zeros(K);
            eta_g = np.zeros(K)
            for kk in range(K - 1):
                eta_ref[kk] = mean_X_ref @ params[kk, :];
                eta_g[kk] = mean_X_g @ params[kk, :]
            
            pred_prop_ref = np.exp(eta_ref)[k_index] / np.exp(eta_ref).sum()
            pred_prop_g = np.exp(eta_g)[k_index] / np.exp(eta_g).sum()
            
            contrast_rows.append({
                "ref": ref_label, "other": g, "mean_ref": mean_props[ref_label], "mean_other": mean_props[g],
                "prop_diff": pred_prop_g - pred_prop_ref, "Coef": delta, "Std.Err": se, "z": z, "P>|z|": pval,
                "direction": "ref_greater" if (pred_prop_g - pred_prop_ref) < 0 else "other_greater",
                "significant": bool(pval < alpha)
            })
        
        # Append fixed effects
        for term, row in fixed_effect_df.iterrows():
            if term.startswith('C('):
                name_split = split_C_terms(pd.Series(term))
                contrast_rows.append({
                    "ref": name_split.iloc[0, 0], "other": name_split.iloc[0, 1],
                    "Coef": row["Coef"], "Std.Err": row["Std.Err"], "z": row["z"], "P>|z|": row["P>|z|"],
                    "direction": "ref_greater" if row["Coef"] < 0 else "other_greater",
                    "significant": row["P>|z|"] < 0.05
                })
        
        extra.update({"contrast_table": pd.DataFrame(contrast_rows).set_index("other"),
                      "fixed_effect": fixed_effect_df, "groups": groups,
                      "estimated_alpha_sum": float(alpha_sum_est)})
    
    except Exception as e:
        extra.update({"hess_inv_error": str(e), "groups": groups, "estimated_alpha_sum": float(alpha_sum_est)})
    
    return make_result(method_name, cell_type, None, effect_size=None, extra=extra)
    
# -----------------------
# Method 5: Permutation-based mixed test (block-permutation by donor)
# -----------------------
def _pairwise_perm_vs_ref(df: pd.DataFrame,
                          cell_type: str,
                          formula_fixed: str,
                          main_variable: str,
                          use_reml: bool,
                          ref_label: str,
                          group_label: str,
                          pairwise_level: str,
                          n_perm: int = 2000,
                          alpha: float = 0.05,
                          seed: int = 0):
    '''
    返回成对排列检验的 DataFrame，将 ref_label 与每个其他疾病类别进行比较。
    :param df:
    :param cell_type:
    :param ref_label:
    :param pairwise_level: 进行检验的单位，sample_id, donor_id 或某种 group_id
    :param n_perm:
    :param alpha:
    :param seed:
    :return:
    '''
    
    # 输入处理
    rng = np.random.default_rng(seed)
    df = df[df["cell_type"] == cell_type].copy()
    if df.empty:
        return pd.DataFrame()
    
    formula = f"prop ~ {formula_fixed}"
    
    # Step 1: 按 target cell_type 取残差（和 run_PermMixed 一样，用 mixedlm 去掉 donor/sample 平均）
    if group_label is not None:
        try:
            md = smf.mixedlm(formula, df, groups=df[group_label])
            mdf = md.fit(method="nm", maxiter=200, reml=use_reml)
            resid = df["prop"] - mdf.fittedvalues.reindex(df.index)
            df = df.assign(resid=resid)
        except Exception:
            df = df.assign(resid=df["prop"].copy())
    else:
        df = df.assign(resid=df["prop"].copy())
    
    # donor → disease label
    donor_col = pairwise_level
    donor_map = df.groupby(pairwise_level)[main_variable].first().to_dict()
    
    labels = list(set(donor_map.values()))
    other_labels = [lab for lab in labels if lab != ref_label]
    other_labels.sort() # 按照字母序
    results = []
    
    # ----------------------------------------------------
    #          For each "other vs ref": run test
    # ----------------------------------------------------
    for lab in other_labels:
        # donor filtering
        donor_items = [(d, l) for d, l in donor_map.items() if l in (ref_label, lab)]
        if len(donor_items) < 2:
            continue
        
        donor_ids = np.array([d for d, _ in donor_items])
        donor_labels = np.array([l for _, l in donor_items])
        
        df_sub = df[df[donor_col].isin(donor_ids)].copy()
        
        # observed stat (Mann–Whitney)
        groups_obs = [
            df_sub.loc[df_sub[main_variable] == ref_label, "resid"].dropna().values,
            df_sub.loc[df_sub[main_variable] == lab, "resid"].dropna().values
        ]
        if len(groups_obs[0]) == 0 or len(groups_obs[1]) == 0:
            continue
        
        try:
            obs_stat = stats.mannwhitneyu(
                groups_obs[0], groups_obs[1], alternative="two-sided"
            ).statistic
        except Exception:
            obs_stat = stats.kruskal(*groups_obs, nan_policy="omit").statistic
        
        # perm stats
        perm_stats = []
        for _ in range(n_perm):
            perm = donor_labels.copy()
            rng.shuffle(perm)
            perm_map = dict(zip(donor_ids, perm))
            df_sub[f"{main_variable}_perm"] = df_sub[donor_col].map(perm_map)
            g0 = df_sub.loc[df_sub[f"{main_variable}_perm"] == ref_label, "resid"].values
            g1 = df_sub.loc[df_sub[f"{main_variable}_perm"] == lab, "resid"].values
            
            try:
                stat = stats.mannwhitneyu(g0, g1, alternative="two-sided").statistic
            except Exception:
                stat = stats.kruskal(g0, g1, nan_policy="omit").statistic
            
            perm_stats.append(stat)
        
        perm_stats = np.array(perm_stats)
        pval = (np.sum(perm_stats >= obs_stat) + 1) / (len(perm_stats) + 1)
        
        # effect size: Cliff's delta 非参数检验
        x, y = groups_obs
        nx, ny = len(x), len(y)
        nxy = sum((xi > y).sum() for xi in x) / (nx * ny)
        nyx = sum((xi < y).sum() for xi in x) / (nx * ny)
        cliffs = nxy - nyx
        direction = "ref_greater" if cliffs > 0 else "other_greater"
        
        results.append({
            "ref": ref_label,
            "other": lab,
            "H stats": float(obs_stat),
            "perm_mean H": float(perm_stats.mean()),
            "pval": float(pval),
            "cliffs_delta": float(cliffs),
            "direction": direction,
            "n_donors": len(donor_ids),
        })
    
    if len(results) == 0:
        return pd.DataFrame()
    
    df_res = pd.DataFrame(results)
    # FDR
    rej, p_adj, _, _ = multipletests(df_res["pval"], alpha=alpha, method="fdr_bh")
    df_res["p_adj"] = p_adj
    df_res["significant"] = rej
    
    df_res = df_res[["ref","other","n_donors","H stats","perm_mean H","cliffs_delta","pval","p_adj","significant","direction"]]
    df_res = pd.DataFrame(df_res).set_index("other")
    return df_res


def run_Perm_Mixed(df_all: pd.DataFrame,
                   cell_type: str,
                   formula: str = "disease",
                   main_variable: str = None,
                   n_perm: int = 2000,
                   ref_label: str = "HC",
                   group_label: str = "sample_id",
                   pairwise_level="donor_id",
                   use_reml: bool = True,
                   alpha: float = 0.05,
                   seed: int = 0):
    '''
    
    :param df_all:
    :param cell_type:
    :param formula: 仅支持 + 的处理
    :param main_variable: 主要解释变量；默认为 None 作为 fool-proofing
        当 formula 只是单一变量时（省略了左侧的 props，自动补全为 props ~ disease），不需要填写 main_variable；
        当 formula 包含多个元素时，必须指定 main_variable。
        因为函数 _pairwise_perm_vs_ref 会通过剔除 main_variable 的 formula 构建 mixedlm，取残差进行检验。
    :param n_perm:
    :param ref_label:
    :param ref_label:
    :param pairwise_level:
    :param alpha:
    :param seed:
    :return:
    '''
    
    # 输入处理
    if "+" in formula:
        if main_variable is None:
            raise KeyError("Main explanatory variable must be specified when formula contains more than one variable.")
    else:
        main_variable = formula
    
    formula_fixed = remove_main_variable_from_formula(formula, main_variable)
    
    if pairwise_level not in df_all.columns:
        raise ValueError("Need column for pairwise_level.")
    
    rng = np.random.default_rng(seed)
    
    df = df_all[df_all["cell_type"] == cell_type].copy()
    if df.empty:
        raise ValueError(f"No rows for {cell_type}.")
    
    #  Step 1: 通过打乱 disease 标签，观察是否还存在同样水平的差异性
    #  K-W 法检测总体差异性
    groups = [x["prop"].dropna().values for _, x in df.groupby("disease")]
    obs_stat = stats.kruskal(*groups, nan_policy="omit").statistic
    
    # 生成 donor 水平的 permutation（混杂）
    donor_map = df.groupby(pairwise_level)["disease"].first().to_dict()
    donor_ids = np.array(list(donor_map.keys()))
    donor_labels = np.array(list(donor_map.values()))
    
    # 检验是否混杂后无显著差异
    perm_stats = []
    for _ in range(n_perm):
        perm = donor_labels.copy()
        rng.shuffle(perm)
        perm_map = dict(zip(donor_ids, perm))
        
        df["disease_perm"] = df["donor_id"].map(perm_map)
        groups_perm = [
            x["prop"].dropna().values
            for _, x in df.groupby("disease_perm")
        ]
        stat = stats.kruskal(*groups_perm, nan_policy="omit").statistic
        perm_stats.append(stat)
    
    perm_stats = np.array(perm_stats)
    pval = (np.sum(perm_stats >= obs_stat) + 1) / (n_perm + 1)
    
    # Step 2: 进行一对一的差异检验
    pairwise_df = _pairwise_perm_vs_ref(
        df=df,
        cell_type=cell_type,
        formula_fixed=formula_fixed,
        main_variable=main_variable,
        use_reml=use_reml,
        ref_label=ref_label,
        group_label=group_label,
        pairwise_level=pairwise_level,
        n_perm=n_perm,
        alpha=alpha,
        seed=seed
    )
    
    # Step 3: 结果包装
    extra = {
        "contrast_table": pairwise_df if not pairwise_df.empty else pd.DataFrame()
    }
    
    return make_result(
        method="PERM_MIXED",
        cell_type=cell_type,
        p_value=pval,
        adj_p_value=None,
        effect_size=None,
        extra=extra
    )

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

    # ---- 输出 ----
    return {
        "method": "ANOVA_naive",
        "cell_type": cell_type,
        "p_main": p_main,
        "extra": {
            "anova_table": anova_table,
            "means": means,
            "contrast_table": pd.DataFrame(contrast_rows)
                             .set_index("other") if contrast_rows else None
        }
    }


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

    return {
        "method": "ANOVA_transformed",
        "cell_type": cell_type,
        "p_main": p_main,
        "extra": {
            "anova_table": anova_table,
            "means": means,
            "contrast_table": pd.DataFrame(contrast_rows)
                             .set_index("other") if contrast_rows else None
        }
    }


def run_pCLR_LMM(
        df_all: pd.DataFrame,
        cell_type: str,
        formula: str = "disease * C(tissue, Treatment(reference='nif'))",
        random_effect: str = "1 | donor_id",
        n_samples: int = 32,
        alpha: float = 0.05,
        random_state: int = 42,
        disease_ref: str = "HC",
        tissue_ref: str = "nif"
) -> Dict[str, Any]:
    rng = np.random.default_rng(random_state)
    extra = {"mixedlm_error": None, "contrast_table": None}
    
    warnings.filterwarnings('ignore', message=".*Maximum Likelihood optimization failed.*")
    warnings.filterwarnings('ignore', message=".*Random effects covariance is singular.*")
    
    try:
        # 1. 准备宽表和均值计算 (用于 mean_ref, mean_other)
        metadata_cols = ['sample_id', 'donor_id', 'disease', 'tissue']
        df_wide = df_all.pivot_table(index=metadata_cols, columns='cell_type', values='count', fill_value=0)
        df_prop_wide = df_all.pivot_table(index=metadata_cols, columns='cell_type', values='prop', fill_value=0)
        
        cell_types = df_wide.columns.tolist()
        target_idx = cell_types.index(cell_type)
        counts_matrix = df_wide.values.astype(float)
        re_group = random_effect.split('|')[1].strip()
        all_coefs_storage = []
        
        # 2. 蒙特卡洛采样循环
        for s in range(n_samples):
            prob_samples = np.array([rng.dirichlet(row + 0.5) for row in counts_matrix])
            log_p = np.log(prob_samples)
            clr_matrix = log_p - np.mean(log_p, axis=1, keepdims=True)
            
            df_iter = df_wide.index.to_frame().reset_index(drop=True)
            df_iter['target_clr'] = clr_matrix[:, target_idx]
            
            try:
                model = smf.mixedlm(f"target_clr ~ {formula}", df_iter, groups=df_iter[re_group])
                result = model.fit(method=['lbfgs'], maxiter=1000, ignore_constrained_optim_warnings=True)
                if result.converged:
                    all_coefs_storage.append(result.summary().tables[1].copy())
            except:
                continue
        
        if not all_coefs_storage:
            return make_result("pCLR_LMM", cell_type, None, extra={"error": "Convergence failed"})
        
        # 3. 汇总中位数
        df_all = pd.concat(all_coefs_storage).reset_index().rename(columns={'index': 'term'})
        for col in ['Coef.', 'Std.Err.', 'z', 'P>|z|']:
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
        
        summary = df_all.groupby('term').agg(
            {'Coef.': 'median', 'Std.Err.': 'median', 'z': 'median', 'P>|z|': 'median'})
        
        # 4. 解析 Term 并构建 contrast_table
        rows = []
        
        # A. 插入参考组占位行 (HC-HC)
        mean_hc_nif = df_prop_wide.xs((disease_ref, tissue_ref), level=('disease', 'tissue'))[cell_type].mean()
        rows.append({
            'other': disease_ref, 'ref': disease_ref, 'mean_ref': mean_hc_nif, 'mean_other': mean_hc_nif,
            'pval': np.nan, 'significant': False, 'prop_diff': np.nan, 'Coef': np.nan, 'Std.Err': np.nan, 'z': np.nan,
            'direction': np.nan
        })
        
        # B. 解析模型项
        for term, res in summary.iterrows():
            if term == 'Intercept': continue
            
            # 解析 disease (如 "disease[T.BD]") 或 tissue (如 "C(tissue...)[T.if]")
            other_val = disease_ref
            ref_val = disease_ref
            
            if 'disease' in term and ':' not in term:
                other_val = re.findall(r"\[T\.(.*?)\]", term)[0]
                ref_val = disease_ref
            elif 'tissue' in term and ':' not in term:
                other_val = re.findall(r"\[T\.(.*?)\]", term)[0]
                ref_val = tissue_ref
            else:  # 忽略交互项或特殊项用于此表格对齐，或根据需要扩展
                continue
            
            # 计算原始比例均值
            m_ref = df_prop_wide.xs(ref_val, level=('disease' if ref_val == disease_ref else 'tissue'))[
                cell_type].mean()
            m_other = df_prop_wide.xs(other_val, level=('disease' if ref_val == disease_ref else 'tissue'))[
                cell_type].mean()
            
            rows.append({
                'other': other_val, 'ref': ref_val, 'mean_ref': m_ref, 'mean_other': m_other,
                'pval': res['P>|z|'], 'significant': res['P>|z|'] < alpha,
                'prop_diff': m_other - m_ref, 'Coef': res['Coef.'], 'Std.Err': res['Std.Err.'], 'z': res['z'],
                'direction': 'other_greater' if res['Coef.'] > 0 else 'ref_greater'
            })
        
        ct_df = pd.DataFrame(rows)
        
        # 5. 排序逻辑: 疾病在前(字母序)，组织在后
        ct_df['_is_if_nif'] = ct_df['ref'] == tissue_ref
        ct_df["_is_HC_HC"] = (ct_df["ref"] == "HC") & (ct_df["other"] == "HC")
        
        ct_df = ct_df.sort_values(
            by=["_is_HC_HC", "_is_if_nif", "other"],
            ascending=[False, True, True]
        )
        ct_df = (
            ct_df.drop(columns=["_is_HC_HC", "_is_if_nif"])
            .set_index("other")
        )
        
        extra["contrast_table"] = ct_df
        return make_result("pCLR_LMM", cell_type, ct_df['pval'].min(), effect_size=ct_df['Coef'].abs().mean(),
                           extra=extra)
    
    except Exception as e:
        extra["mixedlm_error"] = str(e)
        return make_result("pCLR_LMM", cell_type, None, extra=extra)


# -----------------------
# 生成模拟数据：Dirichlet-Multinomial 模拟
# 有利于 Dirichlet 回归
# -----------------------


def simulate_DM_data(
        n_donors=8,
        n_samples_per_donor=4,
        cell_types=50,
        baseline_alpha_scale=30,
        disease_effect_size=0.5,  # <-- 新增参数
        tissue_effect_size=0.6,
        interaction_effect_size=1.0,
        inflamed_cell_frac=0.15,
        sampling_bias_strength=0.0,
        disease_levels=("HC", "CD", "UC"),  # 适配多疾病
        tissue_levels=("nif", "if"),
        sample_size_range=(5000, 20000),
        donor_noise_sd=0.3,
        random_state=1234
):
    rng = np.random.default_rng(random_state)
    n_celltypes = cell_types
    cell_type_names = [f"CT{i + 1}" for i in range(n_celltypes)]
    
    # ---------------------------
    # Step 1: baseline α₀
    # ---------------------------
    baseline = rng.uniform(0.5, 2.0, n_celltypes)
    baseline = baseline / baseline.sum() * baseline_alpha_scale
    
    # ---------------------------
    # Step 1.5: 构建效应向量和 True Effect Table (新增)
    # ---------------------------
    disease_main_effects_dict, tissue_effect_vec, interaction_effects_dict, df_true_effect = build_DM_effects_with_main_effect(
        cell_type_names, disease_levels, tissue_levels,
        disease_effect_size, tissue_effect_size, interaction_effect_size,
        inflamed_cell_frac, rng
    )
    
    records = []
    donors = [f"D{i + 1}" for i in range(n_donors)]
    
    # ---------------------------
    # Step 2: donor-level baseline (不变)
    # ---------------------------
    donor_info = {}
    ref_disease = disease_levels[0]
    
    for donor in donors:
        disease = rng.choice(disease_levels)
        alpha_d = baseline.copy()
        donor_noise = rng.normal(0, donor_noise_sd, n_celltypes)
        alpha_d *= np.exp(donor_noise)
        
        donor_info[donor] = {
            "disease": disease,
            "alpha": alpha_d
        }
    
    # ---------------------------
    # Step 3: sample-level generation (关键修改：应用主效应)
    # ---------------------------
    ref_tissue = tissue_levels[0]
    
    # 提前计算采样偏差 latent_axis (不变)
    if sampling_bias_strength > 0:
        latent_axis = rng.normal(0, 1, n_celltypes)
        latent_axis = latent_axis / np.linalg.norm(latent_axis)
    
    for donor in donors:
        for sample_id in range(n_samples_per_donor):
            
            tissue = rng.choice(tissue_levels)
            disease = donor_info[donor]["disease"]
            
            alpha = donor_info[donor]["alpha"].copy()
            
            # ---- 1. Disease Main Effect (新增) ----
            if disease != ref_disease:
                # 获取当前疾病对应的 Disease 主效应向量
                disease_main_effect_vec = disease_main_effects_dict[disease]
                alpha *= np.exp(disease_main_effect_vec)
            
            # ---- 2. Tissue Main Effect ----
            if tissue != ref_tissue:
                # 对 if 施加 tissue main effect
                alpha *= np.exp(tissue_effect_vec)
                
                # ---- 3. Disease x Tissue Interaction ----
                if disease != ref_disease:
                    # 获取当前疾病对应的交互作用效应向量
                    disease_inter_effect_vec = interaction_effects_dict[disease]
                    alpha *= np.exp(disease_inter_effect_vec)
            
            # ---- 4. technical sampling bias ----
            if sampling_bias_strength > 0:
                bias_scalar = rng.normal(0, sampling_bias_strength)
                bias = bias_scalar * latent_axis
                alpha *= np.exp(bias)
            
            # 所有效应累积后得到样本特异性的最终 alpha 向量。
            alpha = np.maximum(alpha, 1e-6)
            
            N = rng.integers(*sample_size_range)
            
            # 使用 sps.dirichlet/multinomial (假设已正确导入)
            p = dirichlet.rvs(alpha, size=1, random_state=rng).ravel()
            counts = multinomial.rvs(n=N, p=p, size=1, random_state=rng).ravel()
            
            record = {
                "donor_id": donor,
                "disease": disease,
                "tissue": tissue,
                "sample_id": f"{donor}_S{sample_id}"
            }
            
            for i in range(n_celltypes):
                record[f"CT{i + 1}"] = counts[i]
            
            records.append(record)
    
    df = pd.DataFrame(records)
    
    # ---------------------------
    # Step 4: format output (不变)
    # ---------------------------
    
    df_long = df.melt(
        id_vars=["donor_id", "sample_id", "disease", "tissue"],
        var_name="cell_type",
        value_name="count"
    )
    df_long['total_count'] = df_long.groupby('sample_id')['count'].transform('sum')
    df_long['prop'] = df_long['count'] / df_long['total_count']
    
    return df_long, df_true_effect


def build_DM_effects_with_main_effect(
        cell_types, disease_levels, tissue_levels,
        disease_effect_size, tissue_effect_size, interaction_effect_size,
        inflamed_cell_frac, rng
):
    """
    DM 模型的效应生成函数，现在包含独立的 Disease Main Effect 和双向效应（增加或减少）。
    同时，使用全局基线 HC x nif 修正了 True Effect Table 中的交互作用参照组。
    """
    n_celltypes = len(cell_types)
    ref_disease = disease_levels[0]  #  HC
    ref_tissue = tissue_levels[0]  #  nif
    other_tissue = tissue_levels[1]  #  if
    
    # ------------------------------------
    # Step 1: 确定受影响的细胞集和方向
    # ------------------------------------
    
    # 疾病主效应细胞集 (Disease Main Effect Cells)
    n_disease_main_cts = max(1, int(n_celltypes * 0.1))
    disease_main_cts_indices = rng.choice(n_celltypes, size=n_disease_main_cts, replace=False)
    # NEW: 随机分配方向 (+1 或 -1)
    disease_signs = rng.choice([-1, 1], size=n_disease_main_cts)
    
    # 组织/交互作用效应细胞集 (Tissue/Interaction Effect Cells)
    n_inflamed_cts = max(1, int(n_celltypes * inflamed_cell_frac))
    inflamed_cts_indices = rng.choice(n_celltypes, size=n_inflamed_cts, replace=False)
    # NEW: 随机分配方向 (+1 或 -1)
    inflamed_signs = rng.choice([-1, 1], size=n_inflamed_cts)
    
    # --- 2. Disease Main Effects (字典存储) ---
    disease_main_effects_dict = {}
    for other_disease in disease_levels[1:]:
        effect_vec = np.zeros(n_celltypes)
        # 清理逻辑：移除对 len(disease_levels) > 2 的判断
        random_multiplier = rng.uniform(0.8, 1.2)
        
        # NEW: 应用双向效应
        effect_values = disease_effect_size * random_multiplier * disease_signs
        effect_vec[disease_main_cts_indices] = effect_values
        
        disease_main_effects_dict[other_disease] = effect_vec
    
    # --- 3. Tissue Main Effect ---
    tissue_effect_vec = np.zeros(n_celltypes)
    random_multiplier = rng.uniform(0.8, 1.2)  # 同样增加随机性
    
    # NEW: 应用双向效应
    tissue_effect_values = tissue_effect_size * random_multiplier * inflamed_signs
    tissue_effect_vec[inflamed_cts_indices] = tissue_effect_values
    
    # --- 4. Disease x Tissue Interaction Effects (字典存储) ---
    interaction_effects_dict = {}
    for other_disease in disease_levels[1:]:
        effect_vec = np.zeros(n_celltypes)
        random_multiplier = rng.uniform(0.5, 1.5)
        
        # NEW: 应用双向效应 (使用与 Tissue Main 效应相同的受影响细胞集和方向，但大小独立)
        interaction_effect_values = interaction_effect_size * random_multiplier * inflamed_signs
        effect_vec[inflamed_cts_indices] = interaction_effect_values
        
        interaction_effects_dict[other_disease] = effect_vec
    
    # --------------------
    # Step 5: 构建 True Effect Table (保持先前修正的参照组和方向判断逻辑)
    # --------------------
    true_effects = []
    
    # 1. Disease Main Effect (Disease vs HC)
    for other_disease, E_vec in disease_main_effects_dict.items():
        for i, ct_name in enumerate(cell_types):
            E_disease = E_vec[i]
            true_effects.append({
                'cell_type': ct_name,
                'contrast_factor': 'disease',
                'contrast_group': other_disease,
                'contrast_ref': ref_disease,
                'True_Effect': E_disease,
                # NEW: E_disease < 0 时为 ref_greater
                'True_Direction': 'other_greater' if E_disease > 0 else ('ref_greater' if E_disease < 0 else 'None'),
                'True_Significant': True if E_disease != 0 else False
            })
    
    # 2. Tissue Main Effect (if vs nif)
    for i, ct_name in enumerate(cell_types):
        E_tissue = tissue_effect_vec[i]
        true_effects.append({
            'cell_type': ct_name,
            'contrast_factor': 'tissue',
            'contrast_group': other_tissue,
            'contrast_ref': ref_tissue,
            'True_Effect': E_tissue,
            # NEW: E_tissue < 0 时为 ref_greater
            'True_Direction': 'other_greater' if E_tissue > 0 else ('ref_greater' if E_tissue < 0 else 'None'),
            'True_Significant': True if E_tissue != 0 else False
        })
    
    # 3. Disease x Tissue Interaction
    for other_disease, E_inter_vec in interaction_effects_dict.items():
        for i, ct_name in enumerate(cell_types):
            E_interaction = E_inter_vec[i]
            
            true_effects.append({
                'cell_type': ct_name,
                'contrast_factor': 'interaction',
                'contrast_group': f'{other_disease} x {other_tissue}',
                'contrast_ref': f'{ref_disease} x {ref_tissue}',
                'True_Effect': E_interaction,
                # NEW: E_interaction < 0 时为 ref_greater
                'True_Direction': 'other_greater' if E_interaction > 0 else (
                    'ref_greater' if E_interaction < 0 else 'None'),
                'True_Significant': True if E_interaction != 0 else False
            })
    
    return disease_main_effects_dict, tissue_effect_vec, interaction_effects_dict, pd.DataFrame(true_effects)

# 生成模拟数据：Logistic-Normal Multinomial 模拟
# 有利于 LMM/CLR
# -----------------------


def simulate_LogisticNormal_hierarchical(
        n_donors=8,
        n_samples_per_donor=4,
        n_celltypes=50,
        baseline_mu_scale=1.0,
        disease_effect_size=0.5,
        tissue_effect_size=0.8,
        interaction_effect_size=0.5,
        inflamed_cell_frac=0.1,  # 比例，tissue=if受影响的细胞类型
        latent_sd=0.5,
        total_count_mean=5e4,
        total_count_sd=2e4,
        min_count=1000,
        disease_levels=("HC", "CD", "UC"),  # 假设多疾病状态，例如：HC, CD, UC
        tissue_levels=("nif", "if"),
        random_state=1234
):
    """
    Logit-Normal 层次化模拟器，现在支持每个疾病 (如 CD, UC) 具有独立的双向效应向量。
    返回 (模拟数据, 真实效应查找表)。
    """
    rng = np.random.default_rng(random_state)
    
    ref_disease = disease_levels[0]
    ref_tissue = tissue_levels[0]
    
    # ------------------------------------------------------------------
    # 步骤 1: 构建 donor × sample metadata (不变)
    # ------------------------------------------------------------------
    donors = [f"D{i + 1}" for i in range(n_donors)]
    records = []
    disease_choices = disease_levels
    
    for donor in donors:
        disease = rng.choice(disease_choices)
        for sample_id in range(n_samples_per_donor):
            tissue = rng.choice(tissue_levels)
            records.append({
                "donor_id": donor,
                "disease": disease,
                "tissue": tissue,
                "sample_id": f"{donor}_S{sample_id}"
            })
    df_meta = pd.DataFrame(records)
    n_samples = len(df_meta)
    cell_types = [f"CT{i + 1}" for i in range(n_celltypes)]
    
    # ------------------------------------------------------------------
    # 步骤 2-5: 定义独立的效应向量 (关键修改：引入双向随机符号)
    # ------------------------------------------------------------------
    
    # 2) baseline mu
    baseline_mu = rng.normal(0, baseline_mu_scale, n_celltypes)
    
    # --- 3) donor-level disease effects (字典存储) ---
    disease_effects = {}
    
    # 确定受疾病主效应影响的细胞集和随机符号
    n_disease_main_cts = max(1, int(n_celltypes * 0.1))
    disease_main_cts_indices = rng.choice(n_celltypes, size=n_disease_main_cts, replace=False)
    # NEW: 疾病效应随机方向 (+1 或 -1)
    disease_signs = rng.choice([-1, 1], size=n_disease_main_cts)
    
    for other_disease in disease_levels[1:]:
        effect_vec = np.zeros(n_celltypes)
        
        # 乘上随机乘数 (0.8 ~ 1.2) 和随机符号
        random_multiplier = rng.uniform(0.8, 1.2)
        effect_values = disease_effect_size * random_multiplier * disease_signs
        
        effect_vec[disease_main_cts_indices] = effect_values
        disease_effects[other_disease] = effect_vec
    
    # --- 4) sample-level tissue effect (不变) ---
    tissue_effect = np.zeros(n_celltypes)
    
    # 确定受组织效应/交互作用影响的细胞集和随机符号
    n_inflamed_cts = max(1, int(n_celltypes * inflamed_cell_frac))
    inflamed_cts_indices = rng.choice(n_celltypes, size=n_inflamed_cts, replace=False)
    # NEW: 组织效应随机方向 (+1 或 -1)
    tissue_signs = rng.choice([-1, 1], size=n_inflamed_cts)
    
    # 组织效应的赋值 (使用 tissue_signs)
    random_multiplier = rng.uniform(0.8, 1.2)
    tissue_effect_values = tissue_effect_size * random_multiplier * tissue_signs
    
    tissue_effect[inflamed_cts_indices] = tissue_effect_values
    
    # --- 5) disease × tissue interaction effects (字典存储) ---
    interaction_effects = {}
    
    # 交互作用也应该针对每个疾病和 tissue 组合独立定义
    for other_disease in disease_levels[1:]:
        effect_vec = np.zeros(n_celltypes)
        
        # 乘上随机乘数 (0.5 ~ 1.5) 和随机符号 (使用 tissue_signs)
        random_multiplier = rng.uniform(0.5, 1.5)
        interaction_effect_values = interaction_effect_size * random_multiplier * tissue_signs
        
        effect_vec[inflamed_cts_indices] = interaction_effect_values
        interaction_effects[other_disease] = effect_vec
    
    # ------------------------------------------------------------------
    # 步骤 5.5: 构建真实效应查找表 (需要确保 build_true_effect_table 能够处理双向变化)
    # ------------------------------------------------------------------
    
    # 假设 build_true_effect_table 能够根据效应值的符号正确设置 True_Direction
    
    # NOTE: Since the user did not provide the implementation of build_true_effect_table
    # within this function, we assume a correct version is defined elsewhere.
    # For a self-contained answer, I will provide a minimal mock-up here
    # to show the correct data structure and direction logic.
    
    def build_true_effect_table(cell_types, ref_disease, ref_tissue, disease_effects, tissue_effect,
                                interaction_effects, other_tissue):
        true_effects = []
        # 1. Disease Main Effect
        for other_disease, E_vec in disease_effects.items():
            for i, ct_name in enumerate(cell_types):
                E_disease = E_vec[i]
                true_effects.append({
                    'cell_type': ct_name, 'contrast_factor': 'disease', 'contrast_group': other_disease,
                    'contrast_ref': ref_disease,
                    'True_Effect': E_disease, 'True_Direction': 'other_greater' if E_disease > 0 else (
                        'ref_greater' if E_disease < 0 else 'None'),
                    'True_Significant': True if E_disease != 0 else False
                })
        # 2. Tissue Main Effect
        for i, ct_name in enumerate(cell_types):
            E_tissue = tissue_effect[i]
            true_effects.append({
                'cell_type': ct_name, 'contrast_factor': 'tissue', 'contrast_group': other_tissue,
                'contrast_ref': ref_tissue,
                'True_Effect': E_tissue,
                'True_Direction': 'other_greater' if E_tissue > 0 else ('ref_greater' if E_tissue < 0 else 'None'),
                'True_Significant': True if E_tissue != 0 else False
            })
        # 3. Disease x Tissue Interaction
        for other_disease, E_inter_vec in interaction_effects.items():
            for i, ct_name in enumerate(cell_types):
                E_interaction = E_inter_vec[i]
                true_effects.append({
                    'cell_type': ct_name, 'contrast_factor': 'interaction',
                    'contrast_group': f'{other_disease} x {other_tissue}',
                    'contrast_ref': f'{ref_disease} x {ref_tissue}',
                    'True_Effect': E_interaction, 'True_Direction': 'other_greater' if E_interaction > 0 else (
                        'ref_greater' if E_interaction < 0 else 'None'),
                    'True_Significant': True if E_interaction != 0 else False
                })
        return pd.DataFrame(true_effects)
    
    df_true_effect = build_true_effect_table(
        cell_types, ref_disease, ref_tissue,
        disease_effects, tissue_effect, interaction_effects, tissue_levels[1]
    )
    
    # ------------------------------------------------------------------
    # 步骤 6: 构建 logits (保持不变)
    # ------------------------------------------------------------------
    
    logits = np.zeros((n_samples, n_celltypes))
    
    for i, row in df_meta.iterrows():
        mu = baseline_mu.copy()
        current_disease = row["disease"]
        current_tissue = row["tissue"]
        
        # 查找 donor-level disease effect
        if current_disease != ref_disease:
            mu += disease_effects[current_disease]
        
        # sample-level tissue effect
        if current_tissue != ref_tissue:
            mu += tissue_effect
        
        # disease × tissue interaction
        if current_disease != ref_disease and current_tissue != ref_tissue:
            mu += interaction_effects[current_disease]
        
        # latent sample-level variation
        mu += rng.normal(0, latent_sd, n_celltypes)
        
        # logistic-normal sample
        # Note: The original code uses rng.multivariate_normal with cov=I*0.5,
        # which is correct for Logit-Normal
        logits[i] = rng.multivariate_normal(mean=mu, cov=np.eye(n_celltypes) * 0.5)
    
    # ------------------------------------------------------------------
    # 步骤 7-9: 转换到 proportions 并采样 (不变)
    # ------------------------------------------------------------------
    
    proportions = softmax(logits, axis=1)
    
    total_counts = np.maximum(
        rng.normal(total_count_mean, total_count_sd, n_samples).astype(int), min_count
    )
    
    epsilon = 1e-12
    proportions = np.clip(proportions, epsilon, 1 - epsilon)
    row_sums = proportions.sum(axis=1, keepdims=True)
    proportions = proportions / row_sums
    
    counts = np.vstack([
        rng.multinomial(n=total_counts[i], pvals=proportions[i])
        for i in range(n_samples)
    ])
    
    df = df_meta.copy()
    for ct_idx in range(n_celltypes):
        df[f"CT{ct_idx + 1}"] = counts[:, ct_idx]
    
    df_long = df.melt(
        id_vars=["donor_id", "sample_id", "disease", "tissue"],
        var_name="cell_type",
        value_name="count"
    )
    df_long['total_count'] = df_long.groupby('sample_id')['count'].transform('sum')
    df_long['prop'] = df_long['count'] / df_long['total_count']
    
    return df_long, df_true_effect


# -----------------------
# 生成模拟数据：“真实数据 resampling” 模拟
# 相对最公正
# -----------------------
def simulate_CLR_resample_data(
        count_df,
        n_donors=20,  # 新增：供体数量
        n_samples_per_donor=4,  # 新增：每个供体的样本数
        disease_effect_size=0.5,
        tissue_effect_size=0.8,
        interaction_effect_size=0.5,
        inflamed_cell_frac=0.1,
        donor_noise_sd=0.2,  # 新增：供体随机效应 (Logit 空间)
        sample_noise_sd=0.1,  # 对应原 latent_axis_sd，建议设小一点
        disease_levels=("HC", "CD", "UC"),
        tissue_levels=("nif", "if"),
        pseudocount=1.0,
        random_state=1234
):
    rng = np.random.default_rng(random_state)
    n_sim_samples = n_donors * n_samples_per_donor
    
    # ---------------------------
    # Step 1 & 2: 数据准备、宽化和 CLR 转换 (保持不变)
    # ---------------------------
    metadata_cols = ['sample_id', 'donor_id', 'disease', 'tissue']
    # 确保原始 count_df 有总计数
    count_df = count_df.copy()
    count_df['total_count'] = count_df.groupby('sample_id')['count'].transform('sum')
    
    df_wide = count_df.pivot_table(
        index=metadata_cols + ['total_count'], columns='cell_type', values='count', fill_value=0
    ).reset_index()
    
    cell_types_original = df_wide.columns[len(metadata_cols) + 1:].tolist()
    n_celltypes = len(cell_types_original)
    ct_map = {original_name: f"CT{i + 1}" for i, original_name in enumerate(cell_types_original)}
    
    # 获取基线样本池 (HC + nif)
    ref_disease = disease_levels[0]
    ref_tissue = tissue_levels[0]
    df_baseline = df_wide[(df_wide['disease'] == ref_disease) & (df_wide['tissue'] == ref_tissue)].copy()
    
    if df_baseline.empty:
        raise ValueError(f"基线样本池为空。请确保数据中包含 {ref_disease} & {ref_tissue}。")
    
    counts_baseline = df_baseline[cell_types_original].values + pseudocount
    log_counts = np.log(counts_baseline)
    clr_logits_baseline = log_counts - np.mean(log_counts, axis=1, keepdims=True)
    
    # ---------------------------
    # Step 3: 设计效应向量 (使用你的辅助函数)
    # ---------------------------
    disease_main_effects_dict, tissue_effect, interaction_effects_dict, df_true_effect = build_CLR_effects_and_table(
        cell_types=[f"CT{i + 1}" for i in range(n_celltypes)],
        disease_levels=disease_levels,
        tissue_levels=tissue_levels,
        disease_effect_size=disease_effect_size,
        interaction_effect_size=interaction_effect_size,
        tissue_effect_size=tissue_effect_size,
        inflamed_cell_frac=inflamed_cell_frac,
        rng=rng
    )
    
    # ---------------------------
    # Step 4: 层次化模拟 (Donor -> Sample)
    # ---------------------------
    sim_records = []
    
    for d_idx in range(n_donors):
        donor_id = f"D{d_idx + 1:02d}"
        # 疾病状态通常固定在 Donor 级别
        disease = rng.choice(disease_levels)
        # 供体随机效应：模拟该 Donor 整体比例的偏好偏移
        donor_shift = rng.normal(0, donor_noise_sd, n_celltypes)
        
        for s_idx in range(n_samples_per_donor):
            sample_id = f"{donor_id}_S{s_idx + 1}"
            # 组织状态（if/nif）通常在同一 Donor 的不同样本间变化
            tissue = rng.choice(tissue_levels)
            
            # 从真实基线池中重采样一个 Logit 背景
            idx_resample = rng.integers(0, len(clr_logits_baseline))
            clr_logit_sim = clr_logits_baseline[idx_resample].copy()
            
            # 1. 注入供体随机效应
            clr_logit_sim += donor_shift
            
            # 2. 注入疾病主效应
            if disease != ref_disease:
                clr_logit_sim += disease_main_effects_dict[disease]
            
            # 3. 注入组织主效应
            if tissue != ref_tissue:
                clr_logit_sim += tissue_effect
            
            # 4. 注入交互作用
            if disease != ref_disease and tissue != ref_tissue:
                clr_logit_sim += interaction_effects_dict[disease]
            
            # 5. 注入样本级残差噪声
            clr_logit_sim += rng.normal(0, sample_noise_sd, n_celltypes)
            
            sim_records.append({
                "donor_id": donor_id,
                "sample_id": sample_id,
                "disease": disease,
                "tissue": tissue,
                "clr_logit_sim": clr_logit_sim
            })
    
    df_sim_meta = pd.DataFrame(sim_records)
    
    # ---------------------------
    # Step 5 & 6: 生成 Count (反向转换)
    # ---------------------------
    logits_matrix = np.vstack(df_sim_meta['clr_logit_sim'].values)
    # 限制 logit 范围防止溢出
    logits_matrix = np.clip(logits_matrix, -700, 700)
    
    # Softmax 转换为比例
    exp_logits = np.exp(logits_matrix)
    proportions = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # 采样真实数据的总深度分布
    N_real_pool = df_wide['total_count'].values
    total_counts = rng.choice(N_real_pool, size=n_sim_samples, replace=True)
    
    # 多项分布采样
    counts_matrix = np.vstack([
        rng.multinomial(n=total_counts[i], pvals=proportions[i])
        for i in range(n_sim_samples)
    ])
    
    # 构建最终表格
    df_sim = df_sim_meta[['donor_id', 'sample_id', 'disease', 'tissue']].copy()
    for ct_idx, ct_name in enumerate(cell_types_original):
        df_sim[ct_name] = counts_matrix[:, ct_idx]
    
    # 转长表并映射 CT 编号
    df_sim_long = df_sim.melt(
        id_vars=['donor_id', 'sample_id', 'disease', 'tissue'],
        var_name='cell_type', value_name='count'
    )
    df_sim_long['cell_type'] = df_sim_long['cell_type'].map(ct_map)
    
    # 补充辅助列
    df_sim_long['total_count'] = df_sim_long.groupby('sample_id')['count'].transform('sum')
    df_sim_long['prop'] = df_sim_long['count'] / df_sim_long['total_count']
    
    return df_sim_long, df_true_effect


def build_CLR_effects_and_table(
        cell_types, disease_levels, tissue_levels,
        disease_effect_size, tissue_effect_size, interaction_effect_size,
        inflamed_cell_frac, rng
):
    """
    CLR 模型的效应生成函数，现在支持每个疾病具有独立效应和双向变化（增加或减少）。
    """
    n_celltypes = len(cell_types)
    ref_disease = disease_levels[0]  # HC
    ref_tissue = tissue_levels[0]  # nif
    other_tissue = tissue_levels[1]  # if
    
    # ------------------------------------
    # Step 1: 确定受疾病主效应影响的细胞集和方向
    # ------------------------------------
    n_disease_main_cts = max(1, int(n_celltypes * 0.1))
    disease_main_cts_indices = rng.choice(n_celltypes, size=n_disease_main_cts, replace=False)
    # NEW: 疾病效应随机方向 (+1 或 -1)
    disease_signs = rng.choice([-1, 1], size=n_disease_main_cts)
    
    # --- 1. 疾病效应 (Donor-level Logit Effects, 字典存储) ---
    disease_main_effects_dict = {}
    
    for other_disease in disease_levels[1:]:
        effect_vec = np.zeros(n_celltypes)
        
        # 乘上随机乘数 (0.8 ~ 1.2) 和随机符号
        random_multiplier = rng.uniform(0.8, 1.2)
        effect_values = disease_effect_size * random_multiplier * disease_signs
        
        effect_vec[disease_main_cts_indices] = effect_values
        disease_main_effects_dict[other_disease] = effect_vec
    
    # ------------------------------------
    # Step 2: 确定受组织/交互作用影响的细胞集和方向
    # ------------------------------------
    n_inflamed_cts = max(1, int(n_celltypes * inflamed_cell_frac))
    inflamed_cts_indices = rng.choice(n_celltypes, size=n_inflamed_cts, replace=False)
    # NEW: 组织/交互作用效应随机方向 (+1 或 -1)
    inflamed_signs = rng.choice([-1, 1], size=n_inflamed_cts)
    
    # --- 2. 组织效应 (Sample-level Logit Effect, 向量存储) ---
    tissue_effect = np.zeros(n_celltypes)
    
    # 组织效应的赋值 (使用 inflamed_signs)
    random_multiplier = rng.uniform(0.8, 1.2)
    tissue_effect_values = tissue_effect_size * random_multiplier * inflamed_signs
    
    tissue_effect[inflamed_cts_indices] = tissue_effect_values
    
    # --- 3. 交互作用效应 (Sample-level Logit Effects, 字典存储) ---
    interaction_effects_dict = {}
    
    for other_disease in disease_levels[1:]:
        effect_vec = np.zeros(n_celltypes)
        
        # 乘上随机乘数 (0.5 ~ 1.5) 和随机符号 (使用 inflamed_signs)
        random_multiplier = rng.uniform(0.5, 1.5)
        interaction_effect_values = interaction_effect_size * random_multiplier * inflamed_signs
        
        effect_vec[inflamed_cts_indices] = interaction_effect_values
        interaction_effects_dict[other_disease] = effect_vec
    
    # --------------------
    # 构建 True Effect Table (保持先前修正的参照组和方向判断逻辑)
    # --------------------
    true_effects = []
    
    # 1. Disease Main Effect (Disease vs HC)
    for other_disease, E_vec in disease_main_effects_dict.items():
        for i, ct_name in enumerate(cell_types):
            E_disease = E_vec[i]
            true_effects.append({
                'cell_type': ct_name,
                'contrast_factor': 'disease',
                'contrast_group': other_disease,
                'contrast_ref': ref_disease,
                'True_Effect': E_disease,  # Logit Coef
                # NEW: E_disease < 0 时为 ref_greater
                'True_Direction': 'other_greater' if E_disease > 0 else ('ref_greater' if E_disease < 0 else 'None'),
                'True_Significant': True if E_disease != 0 else False
            })
    
    # 2. Tissue Main Effect (if vs nif)
    for i, ct_name in enumerate(cell_types):
        E_tissue = tissue_effect[i]
        true_effects.append({
            'cell_type': ct_name,
            'contrast_factor': 'tissue',
            'contrast_group': other_tissue,
            'contrast_ref': ref_tissue,
            'True_Effect': E_tissue,
            # NEW: E_tissue < 0 时为 ref_greater
            'True_Direction': 'other_greater' if E_tissue > 0 else ('ref_greater' if E_tissue < 0 else 'None'),
            'True_Significant': True if E_tissue != 0 else False
        })
    
    # 3. Disease x Tissue Interaction
    for other_disease, E_inter_vec in interaction_effects_dict.items():
        for i, ct_name in enumerate(cell_types):
            E_interaction = E_inter_vec[i]
            
            true_effects.append({
                'cell_type': ct_name,
                'contrast_factor': 'interaction',
                'contrast_group': f'{other_disease} x {other_tissue}',
                'contrast_ref': f'{ref_disease} x {ref_tissue}',
                'True_Effect': E_interaction,
                # NEW: E_interaction < 0 时为 ref_greater
                'True_Direction': 'other_greater' if E_interaction > 0 else (
                    'ref_greater' if E_interaction < 0 else 'None'),
                'True_Significant': True if E_interaction != 0 else False
            })
    
    return disease_main_effects_dict, tissue_effect, interaction_effects_dict, pd.DataFrame(true_effects)




###############################
# 辅助函数，用于获取真实数据对于拟合数据函数的相关参数
###############################
def collect_DM_results(
        df_count: pd.DataFrame,
        cell_types_list: List[str],
        run_DM_func,
        formula: str = "disease + C(tissue, Treatment(reference=\"nif\"))",
        tissue_levels: Tuple[str, str] = ("nif", "if")
) -> Dict[str, pd.DataFrame]:
    """
    遍历所有细胞类型，运行 DM-Wald 检验，并收集 LogFC 系数和 alpha_sum。

    Args:
        df_count: 原始的细胞计数长格式 DataFrame。
        cell_types_list: 要处理的细胞类型列表。
        run_DM_func: 您的 run_Dirichlet_Wald 函数。
        formula: DM 模型公式。
        tissue_levels: 组织水平，用于识别 tissue 对比的 reference 和 other。

    Returns:
        一个包含 'all_coefs' DataFrame 和 'alpha_sum_df' DataFrame 的字典。
    """
    
    all_coefs = []
    alpha_sum_estimates = []
    
    ref_tissue, other_tissue = tissue_levels
    
    for cell_type in cell_types_list:
        try:
            # 运行 DM-Wald 拟合
            res = run_DM_func(df_all=df_count, cell_type=cell_type, formula=formula, verbose=False)
            
            # 检查是否有错误
            if 'error' in res['extra']:
                print(f"Skipping {cell_type} due to error: {res['extra']['error']}")
                continue
            
            extra = res['extra']
            
            # --- 1. 提取对比表 (Contrast Table) ---
            contrast_df: pd.DataFrame = extra["contrast_table"]
            
            # --- 2. 提取 LogFC 系数 ---
            
            # 遍历 contrast_df 中的每一行
            for other, row in contrast_df.iterrows():
                # 忽略参考行 (如 HC vs HC)
                if other == row['ref']:
                    continue
                
                # Coef 是 LogFC 估计值
                coef = row["Coef"]
                pval = row["P>|z|"]
                
                if pd.isna(coef):
                    continue
                
                # 确定效应类型 (Factor)
                factor_type = ""
                if other in extra['groups']:  # 其他 disease 组
                    factor_type = 'disease'
                elif other == other_tissue and row['ref'] == ref_tissue:  # 组织对比
                    factor_type = 'tissue'
                # NOTE: interaction 项通常在 fixed_effect_df 中，但我们在 contrast_table 中已覆盖 main effects。
                
                # 如果无法识别 factor_type，则跳过
                if not factor_type:
                    continue
                
                all_coefs.append({
                    'cell_type': cell_type,
                    'factor': factor_type,
                    'contrast_other': other,
                    'contrast_ref': row['ref'],
                    'LogFC_Coef': coef,
                    'PValue': pval
                })
            
            # --- 3. 提取 alpha_sum ---
            alpha_sum = extra.get("estimated_alpha_sum")
            if alpha_sum is not None:
                alpha_sum_estimates.append({
                    'cell_type': cell_type,
                    'alpha_sum': alpha_sum
                })
        
        except Exception as e:
            print(f"Failed to process {cell_type}: {e}")
            continue
    
    df_coefs = pd.DataFrame(all_coefs)
    df_alpha_sum = pd.DataFrame(alpha_sum_estimates)
    
    return {
        'all_coefs': df_coefs,
        'alpha_sum_df': df_alpha_sum
    }


def summarize_DM_parameters(collected_results: Dict[str, pd.DataFrame],
                            alpha=0.05) -> Dict[str, float]:
    """
    从收集的结果中计算最终的模拟参数值。
    """
    df_coefs = collected_results['all_coefs']
    df_alpha_sum = collected_results['alpha_sum_df']
    
    if df_alpha_sum.empty:
        raise ValueError("No alpha_sum estimates available.")
    
    # 1. baseline_alpha_scale (alpha_sum)
    # 使用所有细胞类型估计值的**中位数**作为基线
    baseline_alpha_scale = df_alpha_sum['alpha_sum'].median()
    
    # 2. Effect Sizes (LogFC_Coef)
    # 仅使用 PValue < 0.05 的显著效应，并计算其绝对值的中位数作为典型幅度。
    df_signal = df_coefs[df_coefs['PValue'] < alpha].copy()
    df_signal['Abs_LogFC'] = df_signal['LogFC_Coef'].abs()
    
    # 如果显著信号太少，可以使用一个更宽的 PValue 阈值，或仅使用 LogFC > 0.1
    if df_signal.empty and not df_coefs.empty:
        df_signal = df_coefs[df_coefs['LogFC_Coef'].abs() > 0.1].copy()
        df_signal['Abs_LogFC'] = df_signal['LogFC_Coef'].abs()
    
    effect_params = {}
    total_cell_types = df_coefs['cell_type'].nunique()
    
    for factor in ['disease', 'tissue']:
        df_factor = df_signal[df_signal['factor'] == factor]
        
        # 效应大小：显著效应的绝对中位数
        median_effect = df_factor['Abs_LogFC'].median()
        effect_params[f'{factor}_effect_size'] = median_effect if not pd.isna(median_effect) else 0.0
        
        # 受影响比例：用于 inflamed_cell_frac
        if factor == 'tissue':
            unique_affected_cts = df_factor['cell_type'].nunique()
            # 针对 tissue effect，计算受影响细胞类型占总数的比例
            inflamed_cell_frac = unique_affected_cts / total_cell_types
            effect_params['inflamed_cell_frac'] = inflamed_cell_frac
        
        # NOTE: 交互作用效应 (interaction_effect_size) 需要额外运行 formula="disease * tissue" 并提取交互项。
        # 由于您的公式是 "disease + C(tissue, ...)"，我们在这里只估计主效应。
        effect_params['interaction_effect_size'] = 0.0  # 默认为 0，除非您运行包含交互项的模型
    
    return {
        'baseline_alpha_scale': baseline_alpha_scale,
        **effect_params
    }


def estimate_simulation_parameters(
        df_real: pd.DataFrame,
        dm_results: Dict[str, pd.DataFrame],
        ref_disease: str = "HC",
        ref_tissue: str = "nif",
        group_label: str = "sample_id",
        alpha: float = 0.05,
        min_effect_size: float = 0.1  # 设定一个最小效应值，避免估计出0
) -> Dict[str, Any]:
    """
    从真实数据和 DM 统计结果中估计模拟所需的参数。

    Args:
        df_real: 原始的长格式计数数据 (columns: sample_id, cell_type, count, disease, tissue)。
        dm_results: 由 collect_DM_results 函数返回的字典 (包含 'all_coefs' 和 'alpha_sum_df')。
        ref_disease: 参考疾病组 (用于计算 baseline_mu)。
        ref_tissue: 参考组织 (用于计算 baseline_mu)。
        group_label: 样本ID列名。
        min_effect_size: 如果计算出的中位数为0或不存在显著结果，使用的默认最小效应值。

    Returns:
        Dict: 包含 simulate_LogisticNormal_hierarchical 所需参数的字典。
    """
    
    print("--- Estimating Simulation Parameters from Real Data ---")
    params = {}
    
    # ==========================================================================
    # 1. 估计测序深度 (Sequencing Depth)
    # ==========================================================================
    # 计算每个样本的总 count
    sample_depths = df_real.groupby(group_label)['count'].sum()
    params['total_count_mean'] = int(sample_depths.mean())
    params['total_count_sd'] = int(sample_depths.std())
    print(f"Depth: Mean={params['total_count_mean']}, SD={params['total_count_sd']}")
    
    # ==========================================================================
    # 2. 估计 Baseline Mu Scale (细胞类型间丰度的差异)
    # ==========================================================================
    # 逻辑：
    # 1. 筛选出参考组样本 (Reference Condition)
    # 2. 计算各细胞类型的平均比例
    # 3. 进行 CLR (Centered Log-Ratio) 变换以转换到 Logit 空间
    # 4. 计算这些值的标准差 (Standard Deviation)
    
    ref_df = df_real[(df_real['disease'] == ref_disease) & (df_real['tissue'] == ref_tissue)]
    if ref_df.empty:
        print(
            f"Warning: No samples found for Ref Disease '{ref_disease}' and Ref Tissue '{ref_tissue}'. Using all data.")
        ref_df = df_real
    
    # 聚合得到每个细胞类型的总计数
    ct_counts = ref_df.groupby('cell_type')['count'].sum()
    # 添加伪计数防止 log(0)
    ct_counts += 1
    ct_props = ct_counts / ct_counts.sum()
    
    # CLR 变换: log(p) - mean(log(p))
    log_props = np.log(ct_props)
    clr_values = log_props - log_props.mean()
    
    # 估计 scale
    params['baseline_mu_scale'] = float(clr_values.std())
    print(f"Baseline Mu Scale: {params['baseline_mu_scale']:.4f}")
    
    # ==========================================================================
    # 3. 估计 Effect Sizes (从统计结果中)
    # ==========================================================================
    df_coefs = dm_results['all_coefs']
    
    # 辅助函数：计算显著效应的绝对值中位数
    def get_median_effect(factor_name, interaction=False):
        if df_coefs.empty:
            return 0.0
        
        if interaction:
            # 简单的关键词匹配来寻找交互项 (假设 row['factor'] 或 contrast_other 中包含交互标识)
            # 注意：之前的 collect_DM_results 可能需要微调才能明确标记 'interaction'
            # 这里假设如果 contrast_other 包含 ":" 或 "interaction" 字样
            subset = df_coefs[df_coefs['factor'] == 'interaction']
        else:
            subset = df_coefs[df_coefs['factor'] == factor_name]
        
        # 筛选显著结果 (P < 0.05)
        sig_subset = subset[subset['PValue'] < alpha]
        
        if sig_subset.empty:
            print(f"  Note: No significant effects found for {factor_name}. Using default 0.0.")
            return 0.0
        
        median_val = sig_subset['LogFC_Coef'].abs().median()
        return float(median_val) if not np.isnan(median_val) else 0.0
    
    # 3.1 Disease Effect Size
    disease_eff = get_median_effect('disease')
    params['disease_effect_size'] = max(disease_eff, min_effect_size) if disease_eff > 0 else 0.0
    
    # 3.2 Tissue Effect Size
    tissue_eff = get_median_effect('tissue')
    params['tissue_effect_size'] = max(tissue_eff, min_effect_size) if tissue_eff > 0 else 0.0
    
    # 3.3 Interaction Effect Size
    # 只有当统计模型包含交互项时才能估计，否则默认为 0
    # 检查 collect_DM_results 是否捕获了交互项
    interaction_eff = get_median_effect('interaction', interaction=True)
    if interaction_eff == 0.0:
        # 如果没检测到显著交互，或者没运行交互模型，这里可以给一个较小的值或者0
        # 为了模拟真实性，通常交互作用比主效应弱，这里设为0或保留提取值
        params['interaction_effect_size'] = 0.0
    else:
        params['interaction_effect_size'] = interaction_eff
    
    # 3.4 Inflamed Cell Fraction (受组织炎症影响的细胞比例)
    # 计算有多少比例的细胞类型在 Tissue 对比中显著
    total_cells = df_real['cell_type'].nunique()
    if not df_coefs.empty:
        sig_tissue_cells = df_coefs[
            (df_coefs['factor'] == 'tissue') & (df_coefs['PValue'] < alpha)
            ]['cell_type'].nunique()
        params['inflamed_cell_frac'] = sig_tissue_cells / total_cells
    else:
        params['inflamed_cell_frac'] = 0.1  # Default
    
    print(
        f"Effects: Disease={params['disease_effect_size']:.3f}, Tissue={params['tissue_effect_size']:.3f}, Interaction={params['interaction_effect_size']:.3f}")
    print(f"Inflamed Fraction: {params['inflamed_cell_frac']:.2%}")
    
    return params


def estimate_CLR_params_hierarchical(
        df_real: pd.DataFrame,
        collected_results: dict,
        disease_ref: str = "HC",
        tissue_ref: str = "nif",
        alpha: float = 0.05,
        min_abundance: float = 0.01,  # 过滤掉占比小于 1% 的细胞，减少采样噪声干扰
        min_effect_floor: float = 0.1,
        pseudocount: float = 1.0
) -> dict:
    """
    从真实数据和分析结果中层级化地估计仿真参数。
    将噪声分解为 donor_noise_sd 和 sample_noise_sd。
    """
    params = {}
    df_coefs = collected_results.get('all_coefs', pd.DataFrame())
    
    # ==========================================================================
    # 1. 准备数据与丰度过滤
    # ==========================================================================
    # 仅使用高丰度细胞类型来估计噪声，因为低丰度细胞的波动主要是由多项分布采样（Shot Noise）引起的
    total_counts_per_ct = df_real.groupby('cell_type')['count'].sum()
    rel_abundance = total_counts_per_ct / total_counts_per_ct.sum()
    reliable_cts = rel_abundance[rel_abundance > min_abundance].index.tolist()
    
    if len(reliable_cts) < 3:
        reliable_cts = rel_abundance.nlargest(5).index.tolist()
    
    # 提取基线样本 (Baseline: HC + nif)
    df_baseline = df_real[
        (df_real['disease'] == disease_ref) & (df_real['tissue'] == tissue_ref)
        ].copy()
    
    # 如果基线样本太少，则使用全体样本进行方差分解（虽然会包含疾病效应，但作为噪声估计仍比随机给值好）
    use_full_for_noise = len(df_baseline['sample_id'].unique()) < 10
    df_noise_source = df_real.copy() if use_full_for_noise else df_baseline
    
    # 宽表化
    wide_noise = df_noise_source.pivot_table(
        index=['donor_id', 'sample_id'],
        columns='cell_type',
        values='count',
        fill_value=0
    )[reliable_cts]
    
    # CLR 转换
    clr_data = np.log(wide_noise.values + pseudocount)
    clr_data -= clr_data.mean(axis=1, keepdims=True)
    df_clr = pd.DataFrame(clr_data, index=wide_noise.index, columns=reliable_cts)
    
    # ==========================================================================
    # 2. 噪声分解 (Variance Component Analysis 简化版)
    # ==========================================================================
    # A. 计算供体内方差 (Within-donor variance -> sample_noise)
    # 计算每个 Donor 内部各样本间的标准差，然后取中位数
    within_donor_sd = df_clr.groupby('donor_id').std(ddof=1)
    # 过滤掉只有一个样本的 donor (std 为 NaN)
    valid_within_sd = within_donor_sd.dropna()
    
    if not valid_within_sd.empty:
        # 使用中位数以抵抗异常值，乘以 0.8 修正系数（去除残余采样噪声）
        params['sample_noise_sd'] = float(valid_within_sd.median().median()) * 0.8
    else:
        params['sample_noise_sd'] = 0.1  # 默认保底
    
    # B. 计算供体间方差 (Between-donor variance -> donor_noise)
    # 先计算每个 Donor 的平均 Logit 表现
    donor_means = df_clr.groupby('donor_id').mean()
    if len(donor_means) > 1:
        # Donor 均值的标准差反映了供体间的系统性偏移
        params['donor_noise_sd'] = float(donor_means.std().median()) * 0.9
    else:
        params['donor_noise_sd'] = 0.2  # 默认保底
    
    # ==========================================================================
    # 3. 估计效应量 (Effect Sizes)
    # ==========================================================================
    def get_stat_summary(factor_name):
        if df_coefs.empty or factor_name not in df_coefs['factor'].values:
            return min_effect_floor, 0.1
        
        subset = df_coefs[df_coefs['factor'] == factor_name]
        sig_subset = subset[subset['PValue'] < alpha]
        
        if sig_subset.empty:
            return min_effect_floor, 0.05
        
        # 效应强度 = 显著项 LogFC 绝对值的中位数 (CLR 空间)
        effect_size = float(sig_subset['LogFC_Coef'].abs().median())
        # 影响比例 = 显著细胞类型数 / 总细胞类型数
        frac = len(sig_subset) / len(subset)
        
        return max(effect_size, min_effect_floor), frac
    
    params['disease_effect_size'], _ = get_stat_summary('disease')
    params['tissue_effect_size'], t_frac = get_stat_summary('tissue')
    params['inflamed_cell_frac'] = max(t_frac, 0.05)
    
    # 交互作用估计
    i_eff, _ = get_stat_summary('interaction')
    params['interaction_effect_size'] = i_eff if i_eff > min_effect_floor else 0.0
    
    # 打印结果供参考
    print("\n" + "=" * 40)
    print("   ESTIMATED HIERARCHICAL PARAMETERS")
    print("=" * 40)
    for k, v in params.items():
        print(f"{k:25s}: {v:.4f}")
    print("=" * 40)
    
    return params

###############################################################
import hashlib
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


class PyDESeq2Manager:
    def __init__(self):
        self.last_data_hash = None
        self.cached_results = {}  # 存储 {other_label: results_df}
        self.current_cell_type = None
        self.ref_label = "HC"
        self.method_name = "PyDESeq2"
    
    def _get_data_hash(self, df_all, formula):
        # 结合数据特征和公式生成唯一标识
        # 注意：df_all.values 的哈希可能较慢，这里用 shape 和采样
        content_hash = hashlib.md5(pd.util.hash_pandas_object(df_all).values).hexdigest()
        return f"{content_hash}_{formula}"
    
    def __call__(self, df_all: pd.DataFrame, cell_type: str, formula: str = "disease",
                 main_variable: str = None, ref_label: str = "HC",
                 group_label: str = "sample_id", alpha: float = 0.05, **kwargs) -> Dict[str, Any]:
        
        self.ref_label = ref_label
        current_hash = self._get_data_hash(df_all, formula)
        
        # 如果是新数据，触发全量计算
        if current_hash != self.last_data_hash:
            self.cached_results = self._run_full_deseq2(
                df_all, formula, main_variable or formula, ref_label, group_label, alpha
            )
            self.last_data_hash = current_hash
        
        return self._extract_result(cell_type, alpha)
    
    def _run_full_deseq2(self, df_all, formula, main_variable, ref_label, group_label, alpha):
        # 1. 准备数据
        design_cols = parse_formula_columns(f"y ~ {formula}")
        
        # 聚合数据，确保每个 sample_id 只有一行（避免 MultiIndex 冲突）
        # 先提取元数据映射表 (sample_id -> disease/tissue 等)
        meta_map = df_all[[group_label] + design_cols].drop_duplicates().set_index(group_label)
        
        # 生成 Count 宽表：只用 group_label 做 index，避免产生 MultiIndex
        pivot = df_all.pivot_table(
            index=group_label,
            columns="cell_type",
            values="count",
            aggfunc="sum",
            fill_value=0
        )
        
        # 严格对齐 Metadata 和 Counts
        counts_df = pivot.astype(int)  # DESeq2 需要整数
        metadata = meta_map.loc[counts_df.index].copy()
        
        # --- 关键修复：确保 metadata 的所有列都是简单的 object 或 category 类型 ---
        for col in metadata.columns:
            metadata[col] = metadata[col].astype(str)
        
        # 2. 全量拟合
        # quiet=True 停止打印所有细胞类型的进度
        design_factors = design_cols
        dds = DeseqDataSet(
            counts=counts_df,
            metadata=metadata,
            design_factors=design_factors,
            refit_cooks=True,
            quiet=True
        )
        dds.deseq2()
        print(dds.varm["LFC"].columns)
        
        # 3. 提取所有对比组
        # full_cache = {k: {} for k in design_cols} # TODO: 改的兼容一点
        full_cache = {"disease": {}, "tissue": {}}
        
        # --- 提取 Disease 对比 (对比各 labels vs HC) ---
        clean_main = main_variable  # 假设传入的是干净的 "disease"
        unique_disease = [l for l in metadata[clean_main].unique() if l != ref_label]
        for other_label in unique_disease:
            stat_res = DeseqStats(dds, contrast=[clean_main, other_label, ref_label], quiet=True)
            stat_res.summary()
            full_cache["disease"][other_label] = stat_res.results_df
        
        # --- 提取 Tissue 对比 (if vs nif) ---
        # 假设你的 tissue 列名叫 'tissue'，对照组叫 'nif'
        if 'tissue' in metadata.columns:
            # 自动寻找非 nif 的 label (通常是 'if')
            tissue_labels = [l for l in metadata['tissue'].unique() if l != 'nif']
            for t_label in tissue_labels:
                stat_res_t = DeseqStats(dds, contrast=['tissue', t_label, 'nif'], quiet=True)
                stat_res_t.summary()
                full_cache["tissue"][t_label] = stat_res_t.results_df
        
        return full_cache
    
    def _extract_result(self, cell_type: str, alpha: float) -> Dict[str, Any]:
        contrast_rows = []
        
        # 遍历所有已缓存的对比组结果
        for other_label, res_df in self.cached_results["disease"].items():
            if cell_type in res_df.index:
                row = res_df.loc[cell_type]
                
                # PyDESeq2 使用 log2，为了和 CLR (ln) 对应，建议转换：
                # ln(x) = log2(x) * ln(2)
                coef_ln = row["log2FoldChange"] * np.log(2)
                
                contrast_rows.append({
                    "other": other_label,
                    "ref": self.ref_label,
                    "Coef.": coef_ln,
                    "Std.Err.": row["lfcSE"] * np.log(2),
                    "z": row["stat"],
                    "P>|z|": row["pvalue"],
                    "significant": row["pvalue"] < alpha,
                    "direction": "other_greater" if coef_ln > 0 else "ref_greater"
                })
        
        # 提取 Tissue 的表格 (新增)
        tissue_rows = []
        for t_label, res_df in self.cached_results.get("tissue", {}).items():
            if cell_type in res_df.index:
                row = res_df.loc[cell_type]
                lfc_ln = row["log2FoldChange"] * np.log(2)
                tissue_rows.append({
                    "other": t_label, "ref": "nif", "Coef.": lfc_ln,
                    "P>|z|": row["pvalue"], "significant": row["pvalue"] < alpha,
                    "direction": "other_greater" if lfc_ln > 0 else "ref_greater"
                })
        
        if not contrast_rows:
            return make_result(self.method_name, cell_type, np.nan)
        
        # 构建最终的 contrast_table
        df_contrast = pd.DataFrame(contrast_rows+tissue_rows).set_index("other")
        
        # 选出一个代表性的 P 值和效应值（通常是第一个对比组）
        main_p = df_contrast["P>|z|"].iloc[0]
        main_eff = df_contrast["Coef."].iloc[0]
        
        # 调用你定义的标准 make_result
        return make_result(
            method=self.method_name,
            cell_type=cell_type,
            p_value=main_p,
            effect_size=main_eff,
            extra={"contrast_table": df_contrast},
            alpha=alpha
        )


#