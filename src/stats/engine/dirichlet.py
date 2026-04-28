from __future__ import annotations
import logging
import warnings
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd

from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammaln, psi
from scipy.stats import norm

import statsmodels.formula.api as smf

from patsy import dmatrix

from src.stats.support import *
from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)

# -----------------------
# Method 4: Dirichlet regression (placeholder)
# -----------------------

def _neg_loglik_and_grad(flat_params, Y, X, K, P, offset=None, lambda_=1e-4):
    """计算 Dirichlet 回归负对数似然和梯度。

    Args:
        flat_params: 展平后的 beta 参数，形状为 ``(K - 1) * P``。
        Y: sample x cell_type 的组成比例矩阵。
        X: 设计矩阵。
        K: cell subtype/subpopulation 数量。
        P: 设计矩阵列数。
        offset: 可选 offset 矩阵，用于 donor BLUP 校正。
        lambda_: ridge 惩罚强度。

    Returns:
        ``(neg_ll, grad)``，分别为负对数似然和展平梯度。
    """
    n = Y.shape[0]
    B = flat_params.reshape((K - 1, P))
    
    if offset is None:
        eta = X @ B.T
    else:
        eta = X @ B.T + offset[:, :K - 1]
    
    eta_full = np.hstack([eta, np.zeros((n, 1))])
    alpha = np.exp(eta_full)
    alpha0 = alpha.sum(axis=1)
    
    Y_safe = np.clip(Y, 1e-12, None)
    
    ll_terms = (
            gammaln(alpha0)
            - np.sum(gammaln(alpha), axis=1)
            + np.sum((alpha - 1) * np.log(Y_safe), axis=1)
    )
    neg_ll = -ll_terms.sum()
    
    # ----- Gradient -----
    digamma_alpha0 = psi(alpha0)
    digamma_alpha = psi(alpha)
    F = (digamma_alpha0[:, None] - digamma_alpha + np.log(Y_safe)) * alpha
    
    grad = np.zeros((K - 1, P))
    for k in range(K - 1):
        grad[k, :] = - (X.T @ F[:, k])
    
    # ===== Ridge penalty =====
    if lambda_ > 0:
        neg_ll += 0.5 * lambda_ * np.sum(flat_params ** 2)
        grad = grad.ravel() + lambda_ * flat_params
    else:
        grad = grad.ravel()
    
    return neg_ll, grad


def _neg_loglik_and_grad_DM(
        flat_params, Y, X, K, P, C, N, offset=None,
        lambda_ridge=1e-4
):
    """计算 Dirichlet-Multinomial 回归负对数似然和梯度。

    Args:
        flat_params: 展平参数，最后一个元素是 ``log(alpha_sum)``。
        Y: sample x cell_type 的组成比例矩阵。
        X: 设计矩阵。
        K: cell subtype/subpopulation 数量。
        P: 设计矩阵列数。
        C: sample x cell_type 的 count 矩阵。
        N: 每个样本的总 count。
        offset: 可选 offset 矩阵。
        lambda_ridge: 只加在 beta 参数上的 ridge 惩罚强度。

    Returns:
        ``(neg_ll, grad_flat)``，分别为负对数似然和展平梯度。
    """
    n = Y.shape[0]
    
    # 参数拆解
    beta_flat = flat_params[:-1]
    B = beta_flat.reshape((K - 1, P))
    log_alpha_sum = flat_params[-1]
    alpha_sum = np.exp(log_alpha_sum)
    
    # ---- 原始 likelihood（完全不动）----
    if offset is None:
        eta = X @ B.T
    else:
        eta = X @ B.T + offset[:, :K - 1]
    
    eta_full = np.hstack([eta, np.zeros((n, 1))])
    exp_eta_full = np.exp(eta_full)
    P_hat = exp_eta_full / exp_eta_full.sum(axis=1)[:, None]
    alpha = P_hat * alpha_sum
    
    ll_terms = (
            gammaln(alpha_sum)
            - gammaln(alpha_sum + N)
            + np.sum(gammaln(alpha + C), axis=1)
            - np.sum(gammaln(alpha), axis=1)
    )
    neg_ll = -ll_terms.sum()
    
    # ---- 梯度 ----
    digamma_diff = psi(alpha + C) - psi(alpha)
    digamma_alpha_sum_diff = psi(alpha_sum + N) - psi(alpha_sum)
    
    G = digamma_diff - digamma_alpha_sum_diff[:, None]
    H = G - np.sum(P_hat * G, axis=1)[:, None]
    
    grad_beta = np.zeros((K - 1, P))
    for k in range(K - 1):
        grad_beta[k, :] = -(X.T @ H[:, k])
    
    term_alpha_sum = (
            psi(alpha_sum) - psi(alpha_sum + N)
            + np.sum(P_hat * digamma_diff, axis=1)
    )
    grad_log_alpha_sum = -alpha_sum * term_alpha_sum.sum()
    
    # ---- Ridge penalty（只加在 beta 上）----
    neg_ll += 0.5 * lambda_ridge * np.sum(beta_flat ** 2)
    grad_beta += lambda_ridge * B
    
    grad_flat = np.concatenate([
        grad_beta.ravel(),
        np.array([grad_log_alpha_sum])
    ])
    
    return neg_ll, grad_flat


def _compute_donor_blups(df_all, celltypes, epsilon=1e-6, clip_val=5.0):
    """计算 donor 层面的 BLUP offset。

    Args:
        df_all: 长表丰度数据，需包含 ``cell_type``、``prop`` 和 ``donor_id``。
        celltypes: 需要计算 offset 的 cell subtype/subpopulation 列表。
        epsilon: logit 变换前的比例裁剪值，避免 ``logit(0)`` 和 ``logit(1)``。
        clip_val: offset 绝对值上限。

    Returns:
        行为 donor、列为 cell subtype/subpopulation 的 offset DataFrame。

    Example:
        >>> blup_df = _compute_donor_blups(abundance_df, ["CD4_Tcm", "CD8_Teff"])
        >>> blup_df.loc["donor_01", "CD4_Tcm"]
        # 用作 Dirichlet 回归的 donor-level offset，并被限制在 [-5, 5]。
    """
    blup_dict = {}
    
    for ct in celltypes:
        df_ct = df_all[df_all["cell_type"] == ct].copy()
        if df_ct["prop"].notna().sum() < 2:
            continue
        
        # logit transform
        p = df_ct["prop"].clip(epsilon, 1 - epsilon)
        df_ct["logit_prop"] = np.log(p / (1 - p))
        
        try:
            md = smf.mixedlm("logit_prop ~ 1", df_ct, groups=df_ct["donor_id"])
            mdf = md.fit(reml=True, method="nm", maxiter=200)
            
            re = mdf.random_effects
            blup_dict[ct] = {d: float(v[0]) for d, v in re.items()}
        
        except Exception:
            continue
    
    # donor x cell_type offset matrix
    offset_df = pd.DataFrame(blup_dict).fillna(0.0)
    
    # 限制 offset 的范围，是为了防止小样本 donor 的极端 BLUP 主导优化。
    offset_df = offset_df.clip(lower=-clip_val, upper=clip_val)
    
    return offset_df


def _smart_init(Y, X, K, P):
    """用 ALR 近似解初始化 Dirichlet 回归参数。

    Args:
        Y: sample x cell_type 的组成比例矩阵。
        X: 设计矩阵。
        K: cell subtype/subpopulation 数量。
        P: 设计矩阵列数。

    Returns:
        展平后的 beta 初始值。若最小二乘失败，则返回全 0 初始化。
    """
    # Log-Ratio transformation (ALR) with respect to last column
    # y_ij = log(Y_ij / Y_iK)
    Y_ref = Y[:, -1][:, None]
    Y_alr = np.log((Y[:, :-1] + 1e-6) / (Y_ref + 1e-6))  # Simple pseudocount
    
    beta_init = np.zeros((K - 1, P))
    # Solve OLS: X * beta.T = Y_alr
    # beta.T = inv(X.T X) X.T Y_alr
    try:
        # Use simple least squares
        beta_init_T, _, _, _ = np.linalg.lstsq(X, Y_alr, rcond=None)
        beta_init = beta_init_T.T
    except Exception:
        pass  # 最小二乘失败时保留零初始化，保证优化器还能继续尝试。
    return beta_init.ravel()
# ==============================================================================
# 2. 激进版主函数: run_Dirichlet_Wald
# ==============================================================================
@logged
def run_Dirichlet_Wald(df_all: pd.DataFrame,
                       cell_type: str,
                       formula: str = "disease",
                       ref_label: str = "HC",
                       group_label="sample_id",
                       maxiter: int = 1000,
                       alpha: float = 0.05,
                       verbose: bool = False) -> Dict[str, Any]:
    """运行 Dirichlet 回归 Wald 检验。

    该实现面向 cell subtype/subpopulation 组成数据：先把 count 长表转为
    sample x cell_type 宽表，再用 Dirichlet 回归拟合 ``formula`` 对组成比例的影响。
    为了兼容小样本和零计数，函数保留三个稳健性设计：失败时返回空对比表、可选
    donor BLUP offset、以及参数和梯度 clipping。

    Args:
        df_all: 长表丰度数据，至少包含 ``group_label``、``cell_type``、``count``、
            ``prop``、``donor_id`` 和公式中的列。
        cell_type: 目标 cell subtype/subpopulation。
        formula: patsy 右侧公式，例如 ``"disease + tissue"``。
        ref_label: disease 对比的参考组。
        group_label: 样本标识列。
        maxiter: BFGS 最大迭代次数。
        alpha: 显著性阈值。
        verbose: 是否显示优化器输出。

    Returns:
        标准 ``make_result`` 字典。即使失败，也尽量返回空 DataFrame 作为
        ``contrast_table``，减少下游 AttributeError。

    Example:
        >>> res = run_Dirichlet_Wald(
        ...     df_all=abundance_df,
        ...     cell_type="CD8_Teff",
        ...     formula="disease + tissue",
        ...     ref_label="HC",
        ...     group_label="sample_id",
        ...     maxiter=1000,
        ... )
        >>> res["extra"].get("success")
        # 查看优化器是否报告收敛；contrast_table 保存 disease/tissue 对比。
    """
    method_name = "Dirichlet_Wald"
    extra = {}
    
    # --- 关键：定义标准空表模板，防止 downstream AttributeError ---
    empty_contrast = pd.DataFrame(columns=[
        'ref', 'mean_ref', 'mean_other', 'prop_diff',
        'Coef.', 'Std.Err', 'z', 'P>|z|', 'direction', 'significant'
    ])
    empty_contrast.index.name = 'other'
    required_cols = {group_label, "cell_type", "count", "prop", "donor_id"}
    missing_cols = required_cols - set(df_all.columns)
    if missing_cols:
        return make_result(method=method_name, cell_type=cell_type,
                           p_val=1.0, p_type='Minimal',
                           contrast_table=empty_contrast.copy(),
                           extra={"error": f"Missing required columns: {sorted(missing_cols)}"},
                           alpha=alpha)
    
    # 1) Pivot Data
    wide = df_all.pivot_table(index=group_label, columns="cell_type", values="count",
                              aggfunc="sum", fill_value=0)
    celltypes = list(wide.columns)
    n_samples, K = wide.shape
    
    # 验证目标细胞是否存在
    if cell_type not in celltypes:
        return make_result(method=method_name, cell_type=cell_type,
                           p_val=1.0, p_type='Minimal',
                           contrast_table=empty_contrast.copy(),
                           extra={"error": f"target cell_type '{cell_type}' not found"},
                           alpha=alpha)
    
    
    best_ref = find_stable_reference(wide)
    if best_ref == cell_type and K >= 2:
        # ALR 的最后一列是隐式参考类，目标亚群不能放在最后，否则没有对应参数行。
        non_target = [ct for ct in celltypes if ct != cell_type]
        best_ref = non_target[0]
    if celltypes[-1] != best_ref:
        if K < 2:
            return make_result(method=method_name, cell_type=cell_type,
                               p_val=1.0, p_type='Minimal',
                               contrast_table=empty_contrast.copy(),
                               extra={"error": "Need >=2 cell types"},
                               alpha=alpha)
        # 交换最后两个，确保 target 不在末尾
        cols = [c for c in celltypes if c != best_ref] + [best_ref]
        wide = wide[cols]
        celltypes = list(wide.columns)
    
    # 2) Metadata & Design Matrix
    meta = df_all.drop_duplicates(subset=[group_label]).set_index(group_label).reindex(wide.index)
    
    # 获取 BLUPs
    try:
        blup_df = _compute_donor_blups(df_all, celltypes)
        offset = np.zeros_like(wide.values)
        for i, sample_id in enumerate(wide.index):
            donor = meta.loc[sample_id, "donor_id"]
            for k, ct in enumerate(celltypes):
                if donor in blup_df.index and ct in blup_df.columns:
                    offset[i, k] = blup_df.loc[donor, ct]
    except Exception as e:
        offset = np.zeros_like(wide.values)
        extra["blup_warning"] = f"BLUP calculation failed: {e}"
    
    try:
        X_df = dmatrix("1 + " + formula, meta, return_type="dataframe")
    except Exception as e:
        return make_result(method=method_name, cell_type=cell_type,
                           p_val=1.0, p_type='Minimal',
                           contrast_table=empty_contrast.copy(),
                           extra={"error": f"patsy error: {e}"},
                           alpha=alpha)
    
    X = np.asarray(X_df)
    colnames = X_df.design_info.column_names
    P = X.shape[1]
    
    # 3) Proportions Y (加上伪计数保证 log 空间安全)
    counts = wide.values.astype(float)
    row_sums = counts.sum(axis=1)
    counts[row_sums == 0, :] = 1.0 / K
    Y = (counts + 1e-6) / (counts.sum(axis=1)[:, None] + K * 1e-6)
    
    # 4) Init & Optimization with Numerical Guard
    init = _smart_init(Y, X, K, P)
    
    try:
        def fun_and_jac(params):
            # 关键：限制参数范围防止 exp 溢出导致 NaN
            clipped_params = np.clip(params, -50, 50)
            val, grad = _neg_loglik_and_grad(clipped_params, Y, X, K, P, offset)
            # 裁剪梯度防止 BFGS 步长爆炸
            grad = np.clip(grad, -1e6, 1e6)
            return val, grad
        
        res = minimize(fun_and_jac, x0=init, method="BFGS", jac=True,
                       options={"maxiter": maxiter, "disp": verbose})
        params = res.x.reshape((K - 1, P))
        extra["message"] = res.message
        extra["success"] = res.success
    except Exception as e:
        return make_result(method=method_name, cell_type=cell_type,
                           p_val=1.0, p_type='Minimal',
                           contrast_table=empty_contrast.copy(),
                           extra={"error": f"optimizer error: {e}"},
                           alpha=alpha)
    
    # 5) Post-hoc Stability helper
    def safe_exp_prop(X_vec, params_mat, k_idx):
        eta = np.clip(X_vec @ params_mat.T, -40, 40)
        eta_full = np.concatenate([eta, [0.0]])
        alpha_vec = np.exp(eta_full)
        return alpha_vec[k_idx] / np.sum(alpha_vec)
    
    # 6) Wald Tests & Contrast Table
    k_index = celltypes.index(cell_type)
    all_groups = list(meta["disease"].unique())
    groups = [ref_label] + [g for g in all_groups if g != ref_label] if ref_label in all_groups else all_groups
    
    mean_X_by_group = {g: X[meta["disease"] == g, :].mean(axis=0) for g in groups}
    
    try:
        if hasattr(res, 'hess_inv'):
            hess_inv = res.hess_inv.todense() if hasattr(res.hess_inv, 'todense') else res.hess_inv
        else:
            raise ValueError("Hessian matrix not available")
        
        mean_X_ref = mean_X_by_group.get(ref_label, np.zeros(P))
        contrast_rows = []
        for g in groups:
            if g == ref_label: continue
            
            mean_X_g = mean_X_by_group[g]
            delta = float((mean_X_g - mean_X_ref) @ params[k_index, :])
            
            c_vec = np.zeros(((K - 1) * P,))
            c_vec[k_index * P: (k_index + 1) * P] = (mean_X_g - mean_X_ref)
            
            var_val = c_vec @ (hess_inv @ c_vec)
            se = np.sqrt(max(abs(var_val), 1e-12))
            z = delta / se
            pval = float(2.0 * (1.0 - norm.cdf(abs(z))))
            
            p_ref = safe_exp_prop(mean_X_ref, params, k_index)
            p_g = safe_exp_prop(mean_X_g, params, k_index)
            
            contrast_rows.append({
                "ref": ref_label, "other": g, "mean_ref": p_ref, "mean_other": p_g,
                "prop_diff": p_g - p_ref, "Coef.": delta, "Std.Err": se, "z": z, "P>|z|": pval,
                "direction": "other_greater" if (p_g - p_ref) > 0 else "ref_greater",
                "significant": pval < alpha
            })
        
        # 处理其他协变量 (tissue等)
        for j, term in enumerate(colnames):
            if term == 'Intercept' or term.startswith('disease'):
                continue  # 已经在前面的 groups 循环中处理过了
            
            coef = float(params[k_index, j])
            idx = k_index * P + j
            se_j = np.sqrt(max(abs(hess_inv[idx, idx]), 1e-12))
            z_j = coef / se_j  # 计算 z 值
            pval_j = float(2.0 * (1.0 - norm.cdf(abs(z_j))))
            
            # 使用与 DMW 版本一致的解析逻辑
            if term.startswith('C('):
                # 调用解析函数（确保该函数已在 support.py 中定义并导入）
                name_split = split_C_terms(pd.Series(term))
                ref_name = name_split.iloc[0, 0]
                other_name = name_split.iloc[0, 1]
            else:
                ref_name = "base"
                other_name = term
            
            contrast_rows.append({
                "ref": ref_name,
                "other": other_name,
                "mean_ref": np.nan, "mean_other": np.nan, "prop_diff": np.nan,  # 保持列对齐
                "Coef.": coef, "Std.Err": se_j, "z": z_j,
                "P>|z|": pval_j, "significant": pval_j < alpha,
                "direction": "other_greater" if coef > 0 else "ref_greater"
            })
        
        contrast_table = pd.DataFrame(contrast_rows).set_index("other")
        extra.update({"groups": groups})
    
    except Exception as e:
        contrast_table = empty_contrast.copy()  # 即使 Wald 失败也返回空表模板
        extra.update({"error": f"Post-hoc failed: {str(e)}"})
    
    return make_result(method=method_name,
                       cell_type=cell_type,
                       p_val=contrast_table['P>|z|'].min() if not contrast_table.empty else 1.0,
                       p_type='Minimal',
                       contrast_table=contrast_table,
                       extra=extra,
                       alpha=alpha)

@logged
def run_Dirichlet_Wald_with_LFC(
        df_all: pd.DataFrame,
        cell_type: str,
        formula: str = "disease",
        coef_threshold: float = 0.2,  # 新增 LFC 阈值参数
        **kwargs
) -> Dict[str, Any]:
    """运行 Dirichlet Wald 检验并应用效应量阈值过滤。

    Args:
        df_all: 长表丰度数据。
        cell_type: 目标 cell subtype/subpopulation。
        formula: patsy 右侧公式。
        coef_threshold: ``Coef.`` 的绝对值阈值。
        **kwargs: 透传给 ``run_Dirichlet_Wald`` 的参数，例如 ``ref_label``、
            ``group_label`` 和 ``alpha``。

    Returns:
        标准 ``make_result`` 字典，显著性需同时满足 p 值和效应量阈值。

    Example:
        >>> res = run_Dirichlet_Wald_with_LFC(
        ...     abundance_df,
        ...     cell_type="Treg",
        ...     formula="disease + tissue",
        ...     ref_label="HC",
        ...     coef_threshold=0.2,
        ... )
        >>> res["contrast_table"]["significant"]
        # True 表示该对比达到统计和效应量双阈值。
    """
    # 假设 run_Dirichlet_Wald 是你现有的原始函数
    res = run_Dirichlet_Wald(df_all, cell_type, formula=formula, **kwargs)
    
    # 获取 contrast_table 进行二次加工
    if res['contrast_table'] is not None:
        ct = res['contrast_table']
        # 重新定义显著性：P值达标且效应量足够大
        # 注意：这里的 Coef 在 CLR/Dirichlet 空间通常对应 Log 尺度的变化
        ct['significant'] = (ct['P>|z|'] < res['alpha']) & (ct['Coef.'].abs() > coef_threshold)
        
        # 同步更新顶级指标
        res['significant'] = any(ct['significant'])
        res['contrast_table'] = ct
    
    return res


# ==============================================================================
# 3. 稳健版主函数: run_Dirichlet_Multinomial_Wald
# ==============================================================================
@logged
def run_Dirichlet_Multinomial_Wald(df_all: pd.DataFrame,
                                   cell_type: str,
                                   formula: str = "disease",
                                   ref_label: str = "HC",
                                   group_label: str = "sample_id",
                                   maxiter: int = 1000,
                                   alpha: float = 0.05,
                                   verbose: bool = False) -> Dict[str, Any]:
    """运行 Dirichlet-Multinomial Wald 检验。

    该版本额外估计 ``alpha_sum`` 以表达 overdispersion，因此 p 值通常比普通
    Dirichlet Wald 更保守，但在 count 波动明显时更贴近真实不确定性。

    Args:
        df_all: 长表丰度数据。
        cell_type: 目标 cell subtype/subpopulation。
        formula: patsy 右侧公式。
        ref_label: disease 参考组。
        group_label: 样本标识列。
        maxiter: BFGS 最大迭代次数。
        alpha: 显著性阈值。
        verbose: 是否显示优化器输出。

    Returns:
        标准 ``make_result`` 字典。``extra["estimated_alpha_sum"]`` 保存估计的
        overdispersion 总浓度参数。

    Example:
        >>> res = run_Dirichlet_Multinomial_Wald(
        ...     abundance_df,
        ...     cell_type="Mono_CD16",
        ...     formula="disease + tissue",
        ...     ref_label="HC",
        ... )
        >>> res["extra"].get("estimated_alpha_sum")
        # 数值越大通常表示组成比例越接近 multinomial 的低过度离散情形。
    """
    method_name = "Dirichlet_Multinomial_Wald"  # 更新名称
    required_cols = {group_label, "cell_type", "count", "prop", "donor_id"}
    missing_cols = required_cols - set(df_all.columns)
    if missing_cols:
        return make_result(method_name, cell_type, None, None,
                           contrast_table=None,
                           extra={"error": f"Missing required columns: {sorted(missing_cols)}"},
                           alpha=alpha)
    
    # 1) Pivot Data
    wide = df_all.pivot_table(index=group_label, columns="cell_type", values="count",
                              aggfunc="sum", fill_value=0)
    celltypes = list(wide.columns)
    n_samples, K = wide.shape
    
    if cell_type not in celltypes:
        return make_result(method_name, cell_type, None,None,
                           contrast_table=None,
                           extra={"error": f"target cell_type '{cell_type}' not found"})
    if K < 2:
        return make_result(method_name, cell_type, None, None,
                           contrast_table=None,
                           extra={"error": "Need at least two cell_type columns for Dirichlet-Multinomial regression."})
    
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
    
    # 计算 BLUPs
    blup_df = _compute_donor_blups(df_all, celltypes)
    
    # 构建 offset 矩阵，与 Y 对齐
    offset = np.zeros_like(wide.values)  # shape: (n_samples, K)
    
    for i, sample_id in enumerate(wide.index):
        donor = meta.loc[sample_id, "donor_id"]
        for k, ct in enumerate(celltypes):
            if donor in blup_df.index and ct in blup_df.columns:
                offset[i, k] = blup_df.loc[donor, ct]
    
    try:
        X_df = dmatrix("1 + " + formula, meta, return_type="dataframe")
    except Exception as e:
        return make_result(method_name, cell_type, None,None,
                           contrast_table=None,
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
    init_beta = _smart_init(Y, X, K, P)
    init_alpha_sum_log = np.log(160.0)  # Empirical Init
    init = np.concatenate([init_beta, [init_alpha_sum_log]])
    nparam_total = (K - 1) * P + 1
    
    try:
        def fun_and_jac_DM(params):
            val, grad = _neg_loglik_and_grad_DM(params, Y, X, K, P, C, N,offset)
            return val, grad
        
        res = minimize(fun_and_jac_DM, x0=init, method="BFGS", jac=True,
                       options={"maxiter": maxiter, "disp": verbose})
    except Exception as e:
        return make_result(method_name, cell_type, None,None,
                           contrast_table=None,
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
        return make_result(method_name, cell_type, None,None,
                           contrast_table=None,
                           extra={"error": "missing 'disease' column"})
    
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
        if hasattr(res, 'hess_inv'):
            cov_matrix = res.hess_inv
            if hasattr(cov_matrix, 'todense'): cov_matrix = cov_matrix.todense()
        else:
            # Fallback 或报错
            raise ValueError("Optimizer did not return hess_inv")
        
        hess_inv = cov_matrix
        
        # Fixed Effects
        fe_rows = []
        for j, term in enumerate(colnames):
            coef = float(params[k_index, j])
            idx = int(k_index * P + j)
            var_j = float(hess_inv[idx, idx])
            se_j = float(np.sqrt(abs(var_j)))
            z_j = coef / (se_j + 1e-12)
            p_j = float(2.0 * (1.0 - norm.cdf(abs(z_j))))
            fe_rows.append({"term": term, "Coef.": coef, "Std.Err": se_j, "z": z_j, "P>|z|": p_j,
                            "2.5%": coef - 1.96 * se_j, "97.5%": coef + 1.96 * se_j})
        
        # Append Alpha Sum Stats
        idx_alpha = nparam_total - 1
        se_alpha = float(np.sqrt(abs(hess_inv[idx_alpha, idx_alpha])))
        fe_rows.append({"term": "Log_Alpha_Sum", "Coef.": params_full[-1], "Std.Err": se_alpha,
                        "z": np.nan, "P>|z|": np.nan})
        fixed_effect_df = pd.DataFrame(fe_rows).set_index("term")
        
        # Contrasts
        mean_X_ref = mean_X_by_group.get(ref_label, np.zeros(P))
        contrast_rows = []
        for g in groups:
            if g == ref_label:
                # contrast_rows.append({"ref": ref_label, "other": g, "mean_ref": mean_props[ref_label],
                #                       "mean_other": mean_props[ref_label], "P>|z|": np.nan, "significant": False})
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
                "prop_diff": pred_prop_g - pred_prop_ref, "Coef.": delta, "Std.Err": se, "z": z, "P>|z|": pval,
                "direction": "ref_greater" if (pred_prop_g - pred_prop_ref) < 0 else "other_greater",
                "significant": bool(pval < alpha)
            })
        
        # Append fixed effects
        for term, row in fixed_effect_df.iterrows():
            if term.startswith('C('):
                name_split = split_C_terms(pd.Series(term))
                contrast_rows.append({
                    "ref": name_split.iloc[0, 0], "other": name_split.iloc[0, 1],
                    "Coef.": row["Coef."], "Std.Err": row["Std.Err"], "z": row["z"], "P>|z|": row["P>|z|"],
                    "direction": "ref_greater" if row["Coef."] < 0 else "other_greater",
                    "significant": row["P>|z|"] < 0.05
                })
        contrast_table = pd.DataFrame(contrast_rows).set_index("other")
        extra.update({"fixed_effect": fixed_effect_df,
                      "groups": groups,
                      "estimated_alpha_sum": float(alpha_sum_est)})
    
    except Exception as e:
        contrast_table = None
        extra.update({"error": str(e), "groups": groups, "estimated_alpha_sum": float(alpha_sum_est)})
    
    p_val = contrast_table['P>|z|'].min() if contrast_table is not None and not contrast_table.empty else None
    return make_result(method=method_name,
                       cell_type=cell_type,
                       p_val=p_val, p_type='Minimal',
                       contrast_table=contrast_table,
                       extra=extra,
                       alpha=alpha)


def find_stable_reference(wide_df):
    """寻找适合作为 ALR 基准的 cell subtype/subpopulation。

    选择逻辑优先保证非零出现率，其次在全样本存在的亚群里选择变异系数较低者。
    这样做是为了让隐式参考列更稳定，减少 Dirichlet 优化时的数值波动。

    Args:
        wide_df: sample x cell_type count 宽表。

    Returns:
        推荐作为参考列的 cell subtype/subpopulation 名称。

    Example:
        >>> ref_ct = find_stable_reference(count_wide)
        >>> ref_ct
        # 通常是出现率高且跨样本波动较小的亚群。
    """
    presence = (wide_df > 0).sum(axis=0)
    # 优先选在所有样本中都存在的细胞
    full_presence_cts = presence[presence == len(wide_df)].index
    
    if len(full_presence_cts) > 0:
        # 在全存在的细胞中选变异系数最小的
        cv = wide_df[full_presence_cts].std() / wide_df[full_presence_cts].mean()
        return cv.idxmin()
    else:
        # 如果没有全存在的，选出现频率最高的 5 个中丰度最大的
        return presence.idxmax()
