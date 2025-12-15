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
        contrast_rows = _extract_contrast(ref_label, means, tukey)
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
        contrast_rows = _extract_contrast(ref_label, means, tukey)
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

def simulate_DM_data(
        n_donors=8,
        n_samples_per_donor=4,
        cell_types=50,
        baseline_alpha_scale=30,  # 基础组成（越大越低离散）
        disease_effect_size=0.0,  # log-fold-change on α
        sampling_bias_strength=0.0,  # 采样偏差大小
        disease_levels:list = ["HC","disease"],
        tissue_levels:list =["nif", "if"],
        # presort_levels:list =["none", "CD45+", "EpCAM+"],
        sample_size_range=(5000, 20000),  # 每个样本总细胞数
        random_state=1234
):
    """
    返回一个与你的 KDKD pipeline 兼容的宽格式 dataframe：
    每行 = sample
    每列 = 50 个 cell types 的计数
    以及 meta 信息：donor / disease / tissue / presort
    """
    
    rng = np.random.default_rng(random_state)
    
    # ---------------------------
    # Step 1: baseline α 向量
    # ---------------------------
    baseline = rng.uniform(0.5, 2.0, cell_types)
    baseline = baseline / baseline.sum() * baseline_alpha_scale  # scale to overdispersion
    
    # ---------------------------
    # Step 2: 构造 donor × sample 信息
    # ---------------------------
    records = []
    
    donors = [f"D{i + 1}" for i in range(n_donors)]
    
    for donor in donors:
        for sample_id in range(n_samples_per_donor):
            
            disease = rng.choice(disease_levels)
            tissue = rng.choice(tissue_levels)
            
            # --------------------------
            # Step 3: 构造 α_i
            # --------------------------
            alpha = baseline.copy()
            
            # ---- (1) disease effect ----
            if disease == "disease":
                disease_effect = np.zeros(cell_types)
                disease_effect[:5] = disease_effect_size  # 假设前5类细胞受影响
                alpha *= np.exp(disease_effect)
            
            # ---- (2) sampling bias ----
            # e.g. deep biopsy enriches stromal/epithelial
            if sampling_bias_strength > 0:
                bias = np.zeros(cell_types)
                # 前 10 类作为 “深层细胞”
                bias[:10] = sampling_bias_strength
                # 后 10 类作为 “浅层细胞”
                bias[-10:] = -sampling_bias_strength
                alpha *= np.exp(bias)
            
            # Normalize α
            alpha = np.maximum(alpha, 1e-6)
            
            # --------------------------
            # Step 4: 总细胞数
            # --------------------------
            N = rng.integers(*sample_size_range)
            
            # --------------------------
            # Step 5: DM 采样
            # --------------------------
            # counts = dirichlet_multinomial.rvs(N, alpha)
            import scipy.stats as sps
            def dirichlet_multinomial_sample(alpha, n):
                p = sps.dirichlet.rvs(alpha=alpha, size=1).ravel()
                return sps.multinomial.rvs(n=n, p=p)
            
            counts = dirichlet_multinomial_sample(alpha, N)
            
            record = {
                "donor_id": donor,
                "disease": disease,
                "tissue": tissue,
            }
            
            # 加入 cell type count
            for i in range(cell_types):
                record[f"CT{i + 1}"] = counts[i]
            
            records.append(record)
    
    df = pd.DataFrame(records)
    
    df_long = df.melt(
        id_vars=["donor_id", "disease", "tissue"],
        var_name="cell_type",
        value_name="count"
    )
    
    return df_long


# -----------------------
# 生成模拟数据：Logistic-Normal Multinomial 模拟
# 有利于 LMM/CLR
# -----------------------

from scipy.special import softmax


def simulate_LogisticNormal_data(
        n_samples=200,
        n_celltypes=50,
        meta_factors={"disease": ["control", "case"]},
        mean_shift=None,
        total_count_mean=5e4,
        total_count_sd=2e4,
        random_state=0
):
    """
    Logistic-Normal compositional simulator.
    """
    rng = np.random.default_rng(random_state)
    
    # ----- 1) 随机生成 metadata -----
    metadata = {}
    for k, levels in meta_factors.items():
        metadata[k] = rng.choice(levels, n_samples)
    metadata = pd.DataFrame(metadata)
    
    # ----- 2) 生成 baseline mean vector -----
    baseline_mu = rng.normal(0, 1, n_celltypes)
    
    # ----- 3) 加入 meta 因素 effect -----
    if mean_shift is None:
        # 每个 celltype 给 disease 一个随机 effect
        mean_shift = {
            "disease": rng.normal(0, 0.6, n_celltypes)  # effect size
        }
    
    # 保存真实 effect
    true_effect = mean_shift
    
    # ----- 4) 生成 logistic-normal proportions -----
    logits = np.zeros((n_samples, n_celltypes))
    
    for i in range(n_samples):
        mu = baseline_mu.copy()
        
        # 每个 meta factor 施加 effect
        for factor, effect in mean_shift.items():
            level = metadata.loc[i, factor]
            # 假设 effect 施加于（level != 第一个 level）
            if level != meta_factors[factor][0]:
                mu += effect
        
        # logistic-normal 随机误差
        logits[i] = rng.multivariate_normal(mean=mu, cov=np.eye(n_celltypes) * 0.5)
    
    proportions = softmax(logits, axis=1)
    
    # ----- 5) 生成 total counts + multinomial -----
    total_counts = np.maximum(
        rng.normal(total_count_mean, total_count_sd, n_samples).astype(int),
        1000
    )
    
    counts = np.vstack([
        rng.multinomial(n=total_counts[i], pvals=proportions[i])
        for i in range(n_samples)
    ])
    
    return counts, metadata, true_effect


# -----------------------
# 生成模拟数据：“真实数据 resampling” 模拟
# 相对最公正
# -----------------------

def simulate_real_resampling_data(
        real_counts: pd.DataFrame,
        real_metadata: pd.DataFrame,
        n_samples=200,
        disease_effect_strength=1.5,
        disease_levels=("control", "case"),
        effect_n_celltypes=5,
        random_state=0
):
    """
    模拟真实 resampling：
    - 从真实单细胞计数表中 bootstrap 抽样
    - 随机赋予疾病标签
    - 在疾病组中施加 Dirichlet-like composition 变化

    返回：
        new_counts (np.ndarray, shape = n_samples × n_celltypes)
        new_metadata (pd.DataFrame)
    """
    rng = np.random.default_rng(random_state)
    
    m_real, n_celltypes = real_counts.shape
    
    # ----- 1) bootstrap sample -----
    sampled_idx = rng.integers(0, m_real, n_samples)
    base_counts = real_counts.iloc[sampled_idx].reset_index(drop=True)
    metadata = real_metadata.iloc[sampled_idx].reset_index(drop=True)
    
    # ----- 2) random disease assignment -----
    metadata["disease"] = rng.choice(disease_levels, n_samples)
    
    # ----- 3) construct disease effect vector -----
    effect_vector = np.ones(n_celltypes)
    effect_vector[:effect_n_celltypes] = disease_effect_strength
    
    # ----- 4) allocate new count matrix -----
    new_counts = np.zeros((n_samples, n_celltypes), dtype=int)
    
    # ----- 5) apply effect and resample -----
    for i in range(n_samples):
        row = base_counts.iloc[i].to_numpy()
        total = row.sum()
        
        if total == 0:
            new_counts[i] = 0
            continue
        
        probs = row / total
        
        if metadata.loc[i, "disease"] == disease_levels[1]:
            probs = probs * effect_vector
            probs = probs / probs.sum()
        
        new_counts[i] = rng.multinomial(total, probs)
    
    return new_counts, metadata


