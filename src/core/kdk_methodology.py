import warnings
from typing import Dict, Any
import re

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.special import gammaln, psi  # psi = digamma
from scipy.stats import norm
import scipy.stats as sps

from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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
                cell_type: str,
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
    :param cell_type:
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
    
    # 参数处理
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
    
    parse_formula_columns("clr_value ~ disease + tissue")
    
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
        # 提取目的细胞亚群的 pd.Series
        clr_target = clr[cell_type].reset_index()  # has sample_id, donor_id, disease
        clr_target = clr_target.rename(columns={cell_type: "clr_value"})
        
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
        
        print("CLR-LMM function run successfully.")
        
        return make_result("CLR_LMM", cell_type,
                            pval if pval is not None else np.nan,
                            effect_size=eff, extra=extra, alpha=alpha)
    
    except Exception as e:
        extra["error"] = str(e)
        print("CLR-LMM function failed.")
        return make_result("CLR_LMM", cell_type, np.nan, effect_size=None, extra=extra, alpha=alpha)


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
    flat_params: (K-1)*P
    Y: n x K proportions (no zeros ideally; small epsilon added)
    X: n x P design matrix
    returns: (neg_loglik, grad_flat)
    """
    n = Y.shape[0]
    # flat_params 是一维向量，长度是 (K−1)×P
    # 通过 reshap 重构成一个矩阵，理解为 P 个 （K-1) 向量
    # 即 每一类细胞（共 P 类）一个 β 向量
    B = flat_params.reshape((K - 1, P))  # (K-1) x P
    
    # 线性预测模型： η=XB⊤
    # 对每个元素，有 eta(i,k) = X(i) × β(k)
    # 因此将 B 转置为 P × (K-1)，和 n × P 的 X 矩阵对齐
    eta = X @ B.T  # n x (K-1)
    # 加入额外的一列，即第 K 列，代表被前面 K-1 个自由度决定的 reference cell_type
    # 显然其回归系数向量（一竖列）都是 0
    eta_full = np.hstack([eta, np.zeros((n, 1))])  # n x K
    # 取完整的矩阵，因为 α 参数必须为正，因此直接取对数
    alpha = np.exp(eta_full)  # n x K
    # 对每个样本的 Dirichlet 参数 α0(i) 为其一列上 α 参数的和
    # 即：α0(i) = α(i,1) + ... + α(i,K)
    alpha0 = alpha.sum(axis=1)  # n
    
    # 边缘处理，避免 log(0)
    Y_safe = np.clip(Y, 1e-12, None)
    
    # 这是 Dirichlet 的核心部分
    # 我们首先假设了数据符合 Dirichlet 分布，因此通过对数似然法来得到这一假设 Dirichlet 分布的参数 α
    # 根据 Dirichlet 函数的概率密度得到对数似然（ℓ）公式如下：
    ll_terms = gammaln(alpha0) - np.sum(gammaln(alpha), axis=1) + np.sum((alpha - 1) * np.log(Y_safe), axis=1)
    # 通过取负值，以方便地使用 minimize() 函数最大化似然 → 最大化似然的“点”就是我们最有可能获取到真实参数的点
    neg_ll = -ll_terms.sum()
    
    # 计算梯度：
    # 我们的最终目的是用似然函数（ℓ）对某个特定细胞的回归系数（β）求导
    # 根据链式法则，有：∂ℓ/∂β = sum_i [ ∂ℓ/∂α_i · ∂α_i/∂β ]
    # 对于等式右边的左边部分，缩放系数 F 或 ∂ℓ/∂α(i)，相当于求解 似然函数对每个参数的偏导，或每个参数的贡献度、在每个参数上的敏感度
    # F(ik) = digamma( sum_j [ alpha_ij ] ) - digamma( alpha_ik ) + log( y_ik )
    # digamma（双伽玛）函数是 gammaln（伽玛函数对数）的导数，用 psi 表示
    digamma_alpha0 = psi(alpha0)  # n
    digamma_alpha = psi(alpha)  # n x K
    F = (digamma_alpha0[:, None] - digamma_alpha + np.log(Y_safe)) * alpha  # n x K
    
    # 对于右边部分，∂α(i)/∂β = ∂[exp(Xi · β)]/∂β
    
    # 取任意 k，令 f(β_k) = exp(Xi · β_k), g(u) = exp(u), h(v) = Xi·v, f(β_k) = g(h(β_k))
    # 而 g'(u) = (e^u)' = e^u, h'(v) = Xi
    # 显然 f'(β_k) = ∂[exp(Xi · β_k)] / ∂β = exp(Xi · β_k) · Xi = alpha_ik * X_i
    
    # 对这个 Dirichlet 分布的整体，少不了将这个偏导数组合成一个向量（共 K 列），如下
    # ∂α_i/∂β = [(X_i)^T alpha_i1, (X_i)^T alpha_i2, ..., (X_i)^T * alpha_iK]
    # 等效为乘 alpha 的对角矩阵（其余元素都是 0），即 (X_i)^T * diag(alpha)
    
    # 梯度实际上也是 N 元函数在 N 个变量上的偏导的向量，在本例中是 K-1 维向量，取值从 0 到 K-2
    # ∇ B_k = - sum_i( X_i * F_ik ) （因为使用 minimize 函数而取负值）
    grad = np.zeros((K - 1, P))
    for k in range(K - 1):
        # F[:, k] 为 n 行 × 1 列；乘以 X 的每一行（X 经过转置），即 P 行 × n 列，最终结果为一 1 行 × P 列
        grad[k, :] = - (X.T @ F[:, k])
    # 将之扁平化并返回
    grad_flat = grad.ravel()
    return neg_ll, grad_flat


def run_Dirichlet_Wald(df_all: pd.DataFrame,
                       cell_type: str,
                       formula: str = "disease",
                       ref_label: str = "HC",
                       group_label="sample_id",
                       maxiter: int = 1000,
                       alpha: float = 0.05,
                       verbose: bool = False) -> Dict[str, Any]:
    method_name = "Dirichlet_FixedEffect"

    # 1) pivot counts to wide format
    wide = df_all.pivot_table(index=group_label, columns="cell_type", values="count",
                              aggfunc="sum", fill_value=0)
    celltypes = list(wide.columns)
    n_samples, K = wide.shape

    if cell_type not in celltypes:
        return make_result(method_name, cell_type, None, effect_size=None,
                            extra={"error": f"target cell_type '{cell_type}' not found"})

    # ensure reference celltype is last (so params shape is (K-1, P))
    if celltypes[-1] == cell_type:
        if K < 2:
            return make_result(method_name, cell_type, None, effect_size=None,
                                extra={"error": "Need >=2 cell types"})
        cols = celltypes[:-2] + [celltypes[-1], celltypes[-2]]
        wide = wide[cols]
        celltypes = list(wide.columns)

    # 2) proportions Y
    counts = wide.values.astype(float)
    row_sums = counts.sum(axis=1)
    zero_rows = row_sums == 0
    if zero_rows.any():
        counts[zero_rows, :] = 1.0 / K
        row_sums = counts.sum(axis=1)
    Y = counts / row_sums[:, None]

    # 3) metadata and design matrix
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

    # 4) init params & fit
    init = np.zeros((K - 1) * P, dtype=float)
    try:
        def fun_and_jac(params):
            val, grad = _neg_loglik_and_grad(params, Y, X, K, P)
            return val, grad

        res = minimize(fun_and_jac,
                       x0=init,
                       method="BFGS",
                       jac=True,
                       options={"maxiter": maxiter, "disp": verbose})
    except Exception as e:
        return make_result(method_name, cell_type, None, effect_size=None,
                            extra={"error": f"optimizer error: {e}"})

    extra = {"message": res.message} if res.success else {"warning": "optimizer did not converge", "message": res.message}

    params = res.x.reshape((K - 1, P))  # shape (K-1, P)
    # check k_index w.r.t current celltypes ordering
    k_index = celltypes.index(cell_type)
    if k_index == K - 1:
        return make_result(method_name, cell_type, None, effect_size=None,
                            extra={"error": "target cell_type became reference; cannot compute direct coef"})

    # ---- compute mean design-row per disease group ----
    if "disease" not in meta.columns:
        return make_result(method_name, cell_type, None, effect_size=None,
                            extra={"error": "metadata missing 'disease' column"})

    # prepare ordered groups with ref_label first (but do NOT change celltypes)
    all_groups = list(meta["disease"].astype(str).unique())
    # preserve observed order but ensure ref_label is first if present
    if ref_label in all_groups:
        groups = [ref_label] + [g for g in all_groups if g != ref_label]
    else:
        groups = all_groups

    meta["disease"] = pd.Categorical(meta["disease"].astype(str), categories=groups, ordered=True)
    disease_series = meta["disease"].astype(str)

    # mean X per group
    mean_X_by_group = {}
    for g in groups:
        idx = np.where(disease_series.values == g)[0]
        if len(idx) == 0:
            mean_X_by_group[g] = np.zeros((P,))
        else:
            mean_X_by_group[g] = X[idx, :].mean(axis=0)

    # predicted mean proportions per group (via alpha = exp(eta))
    mean_props = {}
    for g in groups:
        mean_X = mean_X_by_group[g]
        # eta for K-1 categories
        eta_kminus1 = mean_X @ params.T  # shape (K-1,)
        # append zero for reference celltype
        eta_full = np.concatenate([eta_kminus1, np.array([0.0])])
        alpha_dir = np.exp(eta_full)
        mean_props[g] = float(alpha_dir[k_index] / alpha_dir.sum())

    # ---- prepare for Wald tests: get hess_inv if possible ----
    nparam = (K - 1) * P
    group_pvals = {g: None for g in groups}
    group_z = {g: None for g in groups}
    group_se = {g: None for g in groups}
    fixed_effect_df = None  # will fill below if we can

    # try obtain dense hess_inv
    try:
        hess_inv = res.hess_inv
        if hasattr(hess_inv, "todense"):
            hess_inv = np.asarray(hess_inv.todense())
        else:
            hess_inv = np.asarray(hess_inv)
        if hess_inv.shape != (nparam, nparam):
            raise ValueError("hess_inv has unexpected shape")

        # ---- 1) Build LMM-style fixed effect table for all design columns ----
        fe_rows = []
        for j, term in enumerate(colnames):
            # coef for this design column for the target celltype
            coef = float(params[k_index, j])
            idx = int(k_index * P + j)  # parameter index in flat vector
            var_j = float(hess_inv[idx, idx])
            se_j = float(np.sqrt(abs(var_j)))
            z_j = coef / (se_j + 1e-12)
            p_j = float(2.0 * (1.0 - norm.cdf(abs(z_j))))
            ci_low = coef - 1.96 * se_j
            ci_high = coef + 1.96 * se_j
            fe_rows.append({
                "term": term,
                "Coef": coef,
                "Std.Err": se_j,
                "z": z_j,
                "P>|z|": p_j,
                "2.5%": ci_low,
                "97.5%": ci_high
            })
        fixed_effect_df = pd.DataFrame(fe_rows).set_index("term")

        # ---- 2) group contrasts (mean-based delta vs ref_label) ----
        mean_X_ref = mean_X_by_group[ref_label] if ref_label in mean_X_by_group else np.zeros(P)
        contrast_rows = []
        for g in groups:
            mean_props_g = mean_props[g]
            if g == ref_label:
                contrast_rows.append({
                    "ref": ref_label,
                    "other": g,
                    "mean_ref": mean_props[ref_label],
                    "mean_other": mean_props_g,
                    "prop_diff": np.nan,
                    "Coef": np.nan,
                    "Std.Err": np.nan,
                    "z": np.nan,
                    "P>|z|": np.nan,
                    "direction": None,
                    "significant": False
                })
                continue

            mean_X_g = mean_X_by_group[g]
            # delta on linear predictor for this target celltype
            delta = float((mean_X_g - mean_X_ref) @ params[k_index, :])

            # contrast vector c (length nparam) with entries only in block for k_index
            c = np.zeros((nparam,), dtype=float)
            start = k_index * P
            end = start + P
            c[start:end] = (mean_X_g - mean_X_ref)

            var = float(c @ (hess_inv @ c))
            se = float(np.sqrt(abs(var)))
            z = delta / (se + 1e-12)
            pval = float(2.0 * (1.0 - norm.cdf(abs(z))))

            # predict proportions via softmax (handle K-1 params)
            eta_ref = np.zeros(K)
            eta_g = np.zeros(K)
            for kk in range(K - 1):
                eta_ref[kk] = mean_X_ref @ params[kk, :]
                eta_g[kk] = mean_X_g @ params[kk, :]
            # last reference category has eta = 0
            eta_ref[K - 1] = 0.0
            eta_g[K - 1] = 0.0
            alpha_ref = np.exp(eta_ref)
            alpha_g = np.exp(eta_g)
            pred_prop_ref = float(alpha_ref[k_index] / alpha_ref.sum())
            pred_prop_g = float(alpha_g[k_index] / alpha_g.sum())
            prop_diff = pred_prop_g - pred_prop_ref

            direction = "ref_greater" if prop_diff < 0 else "other_greater"
            significant = bool(pval < alpha)

            contrast_rows.append({
                "ref": ref_label,
                "other": g,
                "mean_ref": mean_props[ref_label],
                "mean_other": mean_props_g,
                "prop_diff": float(prop_diff),
                "Coef": float(delta),
                "Std.Err": float(se),
                "z": float(z),
                "P>|z|": float(pval),
                "direction": direction,
                "significant": significant
            })
        
        for term, row in fixed_effect_df.iterrows():
            if term == "Intercept":
                continue
            
            # disease contrast 已经独立输出，不重复
            if term.startswith("disease["):
                continue
            
            # 处理 C(tissue, Treatment(reference="nif"))[T.if] 这样的项
            if term.startswith('C('):
                # 解析 tissue 对比
                # term 例子: C(tissue, Treatment(reference="nif"))[T.if]
                name_split = split_C_terms(pd.Series(term))
                other = name_split.iloc[0,1]
                ref = name_split.iloc[0,0]
                
                coef = row["Coef"]
                direction = "ref_greater" if coef < 0 else "other_greater"
                significant = row["P>|z|"] < 0.05  # 或 bool() 按你逻辑
                # group_means = df_fit.groupby("disease")["prop"].mean()
                
                contrast_rows.append({
                    "ref": ref,
                    "other": other,
                    "mean_ref": None,
                    "mean_other": None,
                    "prop_diff": None,
                    "Coef": coef,
                    "Std.Err": row["Std.Err"],
                    "z": row["z"],
                    "P>|z|": row["P>|z|"],
                    "direction": direction,
                    "significant": significant
                })
        
        contrast_df = pd.DataFrame(contrast_rows).set_index("other")

        extra.update({
            "baseline_disease": ref_label,
            "groups": groups,
            "contrast_table": contrast_df,
            "fixed_effect": fixed_effect_df,
            "design_colnames": colnames
        })

    except Exception as e:
        # fallback: return means and note that Hessian-based stats failed
        extra.update({
            "hess_inv_error": str(e),
            "baseline_disease": ref_label,
            "groups": groups,
            "mean_props": mean_props,
            "design_colnames": colnames
        })

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
    df,
    cell_type,
    formula="prop ~ disease",
    ref_label="HC",
    alpha=0.05
):
    """
    Naive ANOVA: 直接对 prop 做 ANOVA，不考虑成分性或随机效应。
    在显著时执行 TukeyHSD 并提取 ref vs other 的对比。
    """

    # 过滤 cell_type
    df_sub = df[df["cell_type"] == cell_type].copy()
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
    if p_main < alpha:
        tukey = pairwise_tukeyhsd(
            endog=df_sub["prop"],
            groups=df_sub[factor_name],
            alpha=alpha
        )
        contrast_rows = extract_contrast(ref_label, means, tukey)
    else:
        contrast_rows = []

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
    df,
    cell_type,
    formula="prop_trans ~ disease",
    ref_label="HC",
    alpha=0.05
):
    """
    arcsin-sqrt transform 后再做 ANOVA，兼容多因素。
    """

    df_sub = df[df["cell_type"] == cell_type].copy()
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

    # ---- Tukey ----
    if p_main < alpha:
        tukey = pairwise_tukeyhsd(
            endog=df_sub["prop_trans"],
            groups=df_sub[main_factor],
            alpha=alpha
        )
        contrast_rows = extract_contrast(ref_label, means, tukey)
    else:
        contrast_rows = []

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

# -----------------------
# 生成模拟数据：Dirichlet-Multinomial 模拟
# 有利于 Dirichlet 回归
# -----------------------
from scipy.stats import dirichlet_multinomial
from scipy.stats import dirichlet, multinomial

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
    DM 模型的效应生成函数，现在包含独立的 Disease Main Effect。
    同时，根据 HC ≡ nif 的约束，修正了 True Effect Table 中的参照组 (contrast_ref)。
    """
    n_celltypes = len(cell_types)
    ref_disease = disease_levels[0]  # 例如 HC
    ref_tissue = tissue_levels[0]  # 例如 nif
    other_tissue = tissue_levels[1]  # 例如 if
    
    # ... (效应向量生成逻辑保持不变，确保随机性) ...
    
    # 随机选择受影响的细胞集
    disease_main_cts = rng.choice(
        n_celltypes,
        size=max(1, int(n_celltypes * 0.1)),
        replace=False
    )
    inflamed_cts = rng.choice(
        n_celltypes,
        size=max(1, int(n_celltypes * inflamed_cell_frac)),
        replace=False
    )
    
    # --- 1. Disease Main Effects (字典存储) ---
    disease_main_effects_dict = {}
    for other_disease in disease_levels[1:]:
        effect_vec = np.zeros(n_celltypes)
        random_multiplier = rng.uniform(0.8, 1.2) if len(disease_levels) > 2 else 1.0
        effect_vec[disease_main_cts] = disease_effect_size * random_multiplier
        disease_main_effects_dict[other_disease] = effect_vec
    
    # --- 2. Tissue Main Effect ---
    tissue_effect_vec = np.zeros(n_celltypes)
    tissue_effect_vec[inflamed_cts] = tissue_effect_size
    
    # --- 3. Disease x Tissue Interaction Effects (字典存储) ---
    interaction_effects_dict = {}
    for other_disease in disease_levels[1:]:
        effect_vec = np.zeros(n_celltypes)
        random_multiplier = rng.uniform(0.5, 1.5) if len(disease_levels) > 2 else 1.0
        effect_vec[inflamed_cts] = interaction_effect_size * random_multiplier
        interaction_effects_dict[other_disease] = effect_vec
    
    # --------------------
    # 构建 True Effect Table (关键修正点在此)
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
                'contrast_ref': ref_disease,  # HC (隐含 HC ≡ nif)
                'True_Effect': E_disease,
                'True_Direction': 'other_greater' if E_disease > 0 else ('ref_greater' if E_disease < 0 else 'None'),
                'True_Significant': True if E_disease != 0 else False
            })
    
    # 2. Tissue Main Effect (if vs nif)
    # 注意：这个主效应在 HC 组中不发生。它表示的是 "disease" 组中 if vs nif 的平均 LogFC。
    for i, ct_name in enumerate(cell_types):
        E_tissue = tissue_effect_vec[i]
        true_effects.append({
            'cell_type': ct_name,
            'contrast_factor': 'tissue',
            'contrast_group': other_tissue,
            'contrast_ref': ref_tissue,  # nif
            'True_Effect': E_tissue,
            'True_Direction': 'other_greater' if E_tissue > 0 else ('ref_greater' if E_tissue < 0 else 'None'),
            'True_Significant': True if E_tissue != 0 else False
        })
    
    # 3. Disease x Tissue Interaction
    for other_disease, E_inter_vec in interaction_effects_dict.items():
        for i, ct_name in enumerate(cell_types):
            E_interaction = E_inter_vec[i]
            
            # 修正 contrast_ref:
            # 交互作用通常被定义为： (Disease_if - Disease_nif) - (HC_if - HC_nif)
            # 由于 HC_if 不存在，这简化为 (Disease_if - Disease_nif) - (0 - 0)
            # 施加的 E_inter 实际上是该交互作用项的系数。
            # 为了评估功效，最简单、最准确的参照组是唯一的全局基线 HC x nif。
            true_effects.append({
                'cell_type': ct_name,
                'contrast_factor': 'interaction',
                'contrast_group': f'{other_disease} x {other_tissue}',
                # *** 修正点 ***: 使用最简洁的全局参照组标记
                'contrast_ref': f'{ref_disease} x {ref_tissue}',
                'True_Effect': E_interaction,  # LogFC
                'True_Direction': 'other_greater' if E_interaction > 0 else (
                    'ref_greater' if E_interaction < 0 else 'None'),
                'True_Significant': True if E_interaction != 0 else False
            })
    
    return disease_main_effects_dict, tissue_effect_vec, interaction_effects_dict, pd.DataFrame(true_effects)


# 生成模拟数据：Logistic-Normal Multinomial 模拟
# 有利于 LMM/CLR
# -----------------------

from scipy.special import softmax


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
        disease_levels=("HC", "CD", "UC"),  # 假设多疾病状态，例如：HC, CD, UC
        tissue_levels=("nif", "if"),
        random_state=1234
):
    """
    Logit-Normal 层次化模拟器，现在支持每个疾病 (如 CD, UC) 具有独立的效应向量。
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
    # 确保疾病状态是均匀随机分配的，以确保每个疾病组都有样本
    disease_choices = disease_levels
    
    for donor in donors:
        # 假设 donor-level disease 随机分配
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
    # 步骤 2-5: 定义独立的效应向量
    # ------------------------------------------------------------------
    
    # 2) baseline mu
    baseline_mu = rng.normal(0, baseline_mu_scale, n_celltypes)
    
    # --- 3) donor-level disease effects (字典存储) ---
    disease_effects = {}
    
    # 遍历所有非参照疾病组 (CD, UC, ...)
    for other_disease in disease_levels[1:]:
        effect_vec = np.zeros(n_celltypes)
        # 随机选择受影响的细胞类型 (为每个疾病独立选择，或基于原设计)
        disease_cts = rng.choice(
            n_celltypes,
            size=max(1, int(n_celltypes * 0.1)),
            replace=False
        )
        # 为了创造不同的分布，我们为不同疾病的效应大小添加一个随机乘数
        random_multiplier = rng.uniform(0.8, 1.2)
        
        effect_vec[disease_cts] = disease_effect_size * random_multiplier
        disease_effects[other_disease] = effect_vec
    
    # --- 4) sample-level tissue effect (不变) ---
    tissue_effect = np.zeros(n_celltypes)
    inflamed_cts = rng.choice(
        n_celltypes,
        size=max(1, int(n_celltypes * inflamed_cell_frac)),
        replace=False
    )
    tissue_effect[inflamed_cts] = tissue_effect_size
    
    # --- 5) disease × tissue interaction effects (字典存储) ---
    interaction_effects = {}
    
    # 交互作用也应该针对每个疾病和 tissue 组合独立定义
    for other_disease in disease_levels[1:]:
        effect_vec = np.zeros(n_celltypes)
        interaction_cts = rng.choice(
            n_celltypes,
            size=max(1, int(n_celltypes * inflamed_cell_frac)),
            replace=False
        )
        # 为了区分，交互作用大小也随机变化
        random_multiplier = rng.uniform(0.5, 1.5)
        
        effect_vec[interaction_cts] = interaction_effect_size * random_multiplier
        interaction_effects[other_disease] = effect_vec
    
    # ------------------------------------------------------------------
    # 步骤 5.5: 构建真实效应查找表 (关键修改：从字典中读取效应)
    # ------------------------------------------------------------------
    
    df_true_effect = build_true_effect_table(
        cell_types, ref_disease, ref_tissue,
        disease_effects, tissue_effect, interaction_effects, tissue_levels[1]
    )
    
    # ------------------------------------------------------------------
    # 步骤 6: 构建 logits (关键修改：根据 row["disease"] 查找对应的效应向量)
    # ------------------------------------------------------------------
    
    logits = np.zeros((n_samples, n_celltypes))
    
    for i, row in df_meta.iterrows():
        mu = baseline_mu.copy()
        current_disease = row["disease"]
        current_tissue = row["tissue"]
        
        # 查找 donor-level disease effect
        if current_disease != ref_disease:
            mu += disease_effects[current_disease]  # **使用对应疾病的效应向量**
        
        # sample-level tissue effect
        if current_tissue != ref_tissue:
            mu += tissue_effect
        
        # disease × tissue interaction
        if current_disease != ref_disease and current_tissue != ref_tissue:
            mu += interaction_effects[current_disease]  # **使用对应疾病的交互作用向量**
        
        # latent sample-level variation
        mu += rng.normal(0, latent_sd, n_celltypes)
        
        # logistic-normal sample
        logits[i] = rng.multivariate_normal(mean=mu, cov=np.eye(n_celltypes) * 0.5)
    
    # ------------------------------------------------------------------
    # 步骤 7-9: 转换到 proportions 并采样 (不变)
    # ------------------------------------------------------------------
    
    proportions = softmax(logits, axis=1)
    
    total_counts = np.maximum(
        rng.normal(total_count_mean, total_count_sd, n_samples).astype(int), 1000
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


# ------------------------------------------------------------------
# 辅助函数: 构建真实效应查找表 (分离出来，便于清晰度)
# ------------------------------------------------------------------

def build_true_effect_table(
        cell_types, ref_disease, ref_tissue,
        disease_effects, tissue_effect, interaction_effects, other_tissue
):
    """
    根据效应字典和向量构建真实效应查找表。
    已修正：根据 HC ≡ nif 的约束，修正了交互作用的参照组 (contrast_ref)。
    """
    
    true_effects = []
    n_celltypes = len(cell_types)
    
    # 1. 疾病效应 (Disease vs Ref_Disease)
    for other_disease, E_vec in disease_effects.items():
        for i, ct_name in enumerate(cell_types):
            E_disease = E_vec[i]
            true_effects.append({
                'cell_type': ct_name,
                'contrast_factor': 'disease',
                'contrast_group': other_disease,
                'contrast_ref': ref_disease,
                'True_Effect': E_disease,
                'True_Direction': 'other_greater' if E_disease > 0 else ('ref_greater' if E_disease < 0 else 'None'),
                'True_Significant': True if E_disease != 0 else False
            })
    
    # 2. 组织效应 (Tissue vs Ref_Tissue)
    for i, ct_name in enumerate(cell_types):
        E_tissue = tissue_effect[i]
        true_effects.append({
            'cell_type': ct_name,
            'contrast_factor': 'tissue',
            'contrast_group': other_tissue,
            'contrast_ref': ref_tissue,
            'True_Effect': E_tissue,
            'True_Direction': 'other_greater' if E_tissue > 0 else ('ref_greater' if E_tissue < 0 else 'None'),
            'True_Significant': True if E_tissue != 0 else False
        })
    
    # 3. 交互作用效应 (Disease_Group * Tissue_Group)
    for other_disease, E_inter_vec in interaction_effects.items():
        for i, ct_name in enumerate(cell_types):
            E_interaction = E_inter_vec[i]
            true_effects.append({
                'cell_type': ct_name,
                'contrast_factor': 'interaction',
                'contrast_group': f'{other_disease} x {other_tissue}',
                # *** 修正点 ***: 简化交互作用的参照组为唯一的全局基线 HC x nif
                'contrast_ref': f'{ref_disease} x {ref_tissue}',
                'True_Effect': E_interaction,
                'True_Direction': 'other_greater' if E_interaction > 0 else (
                    'ref_greater' if E_interaction < 0 else 'None'),
                'True_Significant': True if E_interaction != 0 else False
            })
    
    return pd.DataFrame(true_effects)


# -----------------------
# 生成模拟数据：“真实数据 resampling” 模拟
# 相对最公正
# -----------------------
from scipy.special import softmax  # 用于 Logit 到 Proportion 的转换


def simulate_CLR_resample_data(
        count_df,
        n_sim_samples=100,
        disease_effect_size=0.5,
        tissue_effect_size=0.8,
        interaction_effect_size=0.5,
        inflamed_cell_frac=0.1,
        latent_axis_sd=0.5,
        disease_levels=("HC", "CD", "UC"),  # 适配多疾病
        tissue_levels=("nif", "if"),
        pseudocount=1.0,
        random_state=1234
):
    rng = np.random.default_rng(random_state)
    
    # ---------------------------
    # Step 1 & 2: 数据准备、宽化和 CLR 转换 (保持不变)
    # ---------------------------
    metadata_cols = ['sample_id', 'donor_id', 'disease', 'tissue']
    count_df['total_count'] = count_df.groupby('sample_id')['count'].transform('sum')
    
    df_wide = count_df.pivot_table(
        index=metadata_cols + ['total_count'], columns='cell_type', values='count', fill_value=0
    ).reset_index()
    
    cell_types_original = df_wide.columns[len(metadata_cols) + 1:].tolist()
    n_celltypes = len(cell_types_original)
    ct_map = {original_name: f"CT{i + 1}" for i, original_name in enumerate(cell_types_original)}
    
    baseline_level = (disease_levels[0], tissue_levels[0])
    df_baseline = df_wide[(df_wide['disease'] == baseline_level[0]) & (df_wide['tissue'] == baseline_level[1])].copy()
    
    if df_baseline.empty:
        raise ValueError(
            f"基线样本池为空。请确保数据中存在 {baseline_level[0]} (disease) 和 {baseline_level[1]} (tissue) 的样本组合。")
    
    counts_baseline = df_baseline[cell_types_original].values + pseudocount
    log_counts = np.log(counts_baseline)
    log_g_mean = np.mean(log_counts, axis=1, keepdims=True)
    clr_logits_baseline = log_counts - log_g_mean
    
    # ---------------------------
    # Step 3: 设计效应向量和 True Effect Table (使用辅助函数)
    # ---------------------------
    ref_disease = disease_levels[0]
    ref_tissue = tissue_levels[0]
    
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
    # Step 4: 模拟元数据和 Logit 注入 (关键修改：动态查找效应向量)
    # ---------------------------
    
    sim_records = []
    donors = [f"D{i // n_sim_samples}" for i in range(n_sim_samples)]
    
    for i in range(n_sim_samples):
        disease = rng.choice(disease_levels)
        tissue = rng.choice(tissue_levels)
        
        idx_resample = rng.integers(0, len(clr_logits_baseline))
        clr_logit_base = clr_logits_baseline[idx_resample].copy()
        
        clr_logit_sim = clr_logit_base
        
        # 注入疾病主效应
        if disease != ref_disease:
            # *** 修正点: 查找当前疾病对应的独立效应向量 ***
            E_disease = disease_main_effects_dict[disease]
            clr_logit_sim += E_disease
            
            # 注入组织主效应 (tissue_effect 是一个单一向量)
        if tissue != ref_tissue:
            clr_logit_sim += tissue_effect
        
        # 注入交互作用效应
        if disease != ref_disease and tissue != ref_tissue:
            # *** 修正点: 查找当前疾病对应的独立交互作用向量 ***
            E_interaction = interaction_effects_dict[disease]
            clr_logit_sim += E_interaction
            
            # 施加潜在扰动
        clr_logit_sim += rng.normal(0, latent_axis_sd, n_celltypes)
        
        sim_records.append({
            "donor_id": donors[i],
            "sample_id": f"{donors[i]}_S{i}",
            "disease": disease,
            "tissue": tissue,
            "clr_logit_sim": clr_logit_sim
        })
    
    df_sim_meta = pd.DataFrame(sim_records)
    
    # ---------------------------
    # Step 5 & 6: 反向转换和最终输出 (不变，但确保使用映射后的 cell_type)
    # ---------------------------
    
    logits_matrix = np.vstack(df_sim_meta['clr_logit_sim'].values)
    
    # ... (安全检查和 softmax/归一化代码, 与原函数一致) ...
    MAX_LOGIT = 700
    logits_matrix = np.clip(logits_matrix, -MAX_LOGIT, MAX_LOGIT)
    
    proportions = softmax(logits_matrix, axis=1)
    epsilon = 1e-12
    proportions = np.clip(proportions, epsilon, 1 - epsilon)
    row_sums = proportions.sum(axis=1, keepdims=True)
    proportions = proportions / row_sums
    
    N_real = count_df['total_count'].unique()
    total_counts = rng.choice(N_real, size=n_sim_samples, replace=True)
    
    counts = np.vstack([
        rng.multinomial(n=total_counts[i], pvals=proportions[i])
        for i in range(n_sim_samples)
    ])
    
    df_sim = df_sim_meta[['donor_id', 'sample_id', 'disease', 'tissue']].copy()
    
    for ct_idx, ct_name in enumerate(cell_types_original):
        # 注意: 这里使用原始 cell_type 名称作为列名
        df_sim[ct_name] = counts[:, ct_idx]
    
    df_sim_long = df_sim.melt(
        id_vars=['donor_id', 'sample_id', 'disease', 'tissue'],
        var_name='cell_type',
        value_name='count'
    )
    
    df_sim_long['total_count'] = df_sim_long.groupby('sample_id')['count'].transform('sum')
    df_sim_long['prop'] = df_sim_long['count'] / df_sim_long['total_count']
    
    # *** 关键点: 将 cell_type 列的值映射为 CT1, CT2... 编号 ***
    df_sim_long['cell_type'] = df_sim_long['cell_type'].map(ct_map)
    
    return df_sim_long, df_true_effect


def build_CLR_effects_and_table(
        cell_types, disease_levels, tissue_levels,
        disease_effect_size, tissue_effect_size, interaction_effect_size,
        inflamed_cell_frac, rng
):
    """
    CLR 模型的效应生成函数，现在支持每个疾病具有独立效应。
    """
    n_celltypes = len(cell_types)
    ref_disease = disease_levels[0]  # HC
    ref_tissue = tissue_levels[0]  # nif
    other_tissue = tissue_levels[1]  # if
    
    # --- 1. 疾病效应 (Donor-level Logit Effects, 字典存储) ---
    disease_main_effects_dict = {}
    
    for other_disease in disease_levels[1:]:
        effect_vec = np.zeros(n_celltypes)
        # 随机选择受影响的细胞类型 (为每个疾病独立选择)
        disease_cts = rng.choice(
            n_celltypes,
            size=max(1, int(n_celltypes * 0.1)),
            replace=False
        )
        # 为了创造不同的分布，添加一个随机乘数
        random_multiplier = rng.uniform(0.8, 1.2)
        
        effect_vec[disease_cts] = disease_effect_size * random_multiplier
        disease_main_effects_dict[other_disease] = effect_vec
        
        # --- 2. 组织效应 (Sample-level Logit Effect, 向量存储) ---
    inflamed_cts = rng.choice(n_celltypes, size=max(1, int(n_celltypes * inflamed_cell_frac)), replace=False)
    tissue_effect = np.zeros(n_celltypes)
    tissue_effect[inflamed_cts] = tissue_effect_size
    
    # --- 3. 交互作用效应 (Sample-level Logit Effects, 字典存储) ---
    interaction_effects_dict = {}
    
    for other_disease in disease_levels[1:]:
        effect_vec = np.zeros(n_celltypes)
        # 交互作用影响的细胞集和大小也可以独立
        interaction_cts = rng.choice(
            n_celltypes,
            size=max(1, int(n_celltypes * inflamed_cell_frac)),
            replace=False
        )
        random_multiplier = rng.uniform(0.5, 1.5)
        
        effect_vec[interaction_cts] = interaction_effect_size * random_multiplier
        interaction_effects_dict[other_disease] = effect_vec
    
    # --------------------
    # 构建 True Effect Table (保持先前修正的参照组逻辑)
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
                'True_Direction': 'other_greater' if E_interaction > 0 else (
                    'ref_greater' if E_interaction < 0 else 'None'),
                'True_Significant': True if E_interaction != 0 else False
            })
    
    return disease_main_effects_dict, tissue_effect, interaction_effects_dict, pd.DataFrame(true_effects)
