from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from scipy import stats

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from src.stats.support import *

from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)

# -----------------------
# Method 5: Permutation-based mixed test (block-permutation by donor)
# -----------------------
@logged
def _pairwise_perm_vs_ref(df: pd.DataFrame,
                          cell_type: str,
                          formula_fixed: str,
                          main_variable: str,
                          use_reml: bool,
                          ref_label: str,
                          group_label: str,
                          pairwise_level: str,
                          n_perm: int = 500,
                          alpha: float = 0.05,
                          seed: int = 0):
    """执行参考组对其他组的成对 block permutation 检验。

    Args:
        df: 长表丰度数据。
        cell_type: 目标 cell subtype/subpopulation。
        formula_fixed: 去掉主要解释变量后的协变量公式，用于残差化。
        main_variable: 主要解释变量，例如 ``"disease"``。
        use_reml: 混合模型是否使用 REML。
        ref_label: 参考组标签。
        group_label: 残差化模型中的随机截距分组列。
        pairwise_level: 打乱标签的 block 单位，例如 ``"donor_id"``。
        n_perm: 每个 pairwise 对比的排列次数。
        alpha: FDR 阈值。
        seed: 随机种子。

    Returns:
        以 ``other`` 为索引的 DataFrame；若没有可比较分组，返回空 DataFrame。
    """
    
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
    
    # block 单位到主要变量标签的映射，保证同一 donor/sample 的标签一起被置换。
    donor_col = pairwise_level
    donor_map = df.groupby(pairwise_level)[main_variable].first().to_dict()
    
    labels = list(set(donor_map.values()))
    other_labels = [lab for lab in labels if lab != ref_label]
    other_labels.sort()
    results = []
    
    # ----------------------------------------------------
    #          For each "other vs ref": run test
    # ----------------------------------------------------
    for lab in other_labels:
        donor_items = [(d, l) for d, l in donor_map.items() if l in (ref_label, lab)]
        if len(donor_items) < 2:
            continue
        
        donor_ids = np.array([d for d, _ in donor_items])
        donor_labels = np.array([l for _, l in donor_items])
        
        df_sub = df[df[donor_col].isin(donor_ids)].copy()
        
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
        
        # Cliff's delta 作为方向性效应量，便于与参数模型的 direction 字段对齐。
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
    
    df_res = df_res[
        ["ref", "other", "n_donors", "H stats", "perm_mean H", "cliffs_delta", "pval", "p_adj", "significant",
         "direction"]]
    df_res = pd.DataFrame(df_res).set_index("other")
    return df_res


@logged
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
    """运行基于 block permutation 的混合丰度检验。

    函数先对目标 cell subtype/subpopulation 的 ``prop`` 做全局 Kruskal-Wallis
    统计量，然后在 ``pairwise_level`` 层面打乱主要变量标签，得到全局置换 p 值。
    随后调用 ``_pairwise_perm_vs_ref``，将 ``ref_label`` 与每个其他水平逐一比较。

    Args:
        df_all: 长表丰度数据，至少包含 ``cell_type``、``prop``、``main_variable``、
            ``group_label`` 和 ``pairwise_level``。
        cell_type: 目标 cell subtype/subpopulation。
        formula: 右侧公式。目前主要支持 ``+`` 拼接的公式，例如
            ``"disease + tissue"``。
        main_variable: 主要解释变量。单变量公式可省略；多变量公式必须指定。
        n_perm: 全局检验和 pairwise 检验的排列次数。
        ref_label: 参考组标签，通常为 ``"HC"``。
        group_label: 残差化模型中的随机截距分组列。
        pairwise_level: 置换标签的 block 层级，例如 ``"donor_id"``。
        use_reml: MixedLM 残差化时是否使用 REML。
        alpha: FDR 阈值。
        seed: 随机种子，保证 HPC 上批量任务可复现。

    Returns:
        标准 ``make_result`` 字典。``p_val`` 是全局 permutation p 值，
        ``contrast_table`` 是参考组对其他组的 pairwise 结果。

    Example:
        >>> res = run_Perm_Mixed(
        ...     df_all=abundance_df,
        ...     cell_type="Mono_CD14",
        ...     formula="disease + tissue",
        ...     main_variable="disease",
        ...     n_perm=1000,
        ...     ref_label="HC",
        ...     group_label="sample_id",
        ...     pairwise_level="donor_id",
        ...     seed=2024,
        ... )
        >>> res["contrast_table"][["p_adj", "direction"]]
        # 查看各 disease 水平相对 HC 的校正 p 值和丰度方向。
    """
    extra = {}
    pval=None
    # 输入处理
    if "+" in formula:
        if main_variable is None:
            raise KeyError("Main explanatory variable must be specified when `formula` contains more than one variable.")
    else:
        main_variable = formula
    
    formula_fixed = remove_main_variable_from_formula(formula, main_variable)
    
    if pairwise_level not in df_all.columns:
        raise ValueError(f"Missing required column for `pairwise_level`: '{pairwise_level}'.")
    if main_variable not in df_all.columns:
        raise ValueError(f"Missing required column for `main_variable`: '{main_variable}'.")
    required_cols = {"cell_type", "prop", group_label}
    missing_cols = required_cols - set(df_all.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")
    
    rng = np.random.default_rng(seed)
    
    df = df_all[df_all["cell_type"] == cell_type].copy()
    if df.empty:
        raise ValueError(f"No rows for cell_type: '{cell_type}'.")
    
    try:
        #  Step 1: 通过打乱 disease 标签，观察是否还存在同样水平的差异性
        #  K-W 法检测总体差异性
        groups = [x["prop"].dropna().values for _, x in df.groupby(main_variable)]
        obs_stat = stats.kruskal(*groups, nan_policy="omit").statistic
        
        # 生成 block 水平的 permutation，保护同一 donor/sample 内的相关结构。
        donor_map = df.groupby(pairwise_level)[main_variable].first().to_dict()
        donor_ids = np.array(list(donor_map.keys()))
        donor_labels = np.array(list(donor_map.values()))
        
        # 检验是否混杂后无显著差异
        perm_stats = []
        for _ in range(n_perm):
            perm = donor_labels.copy()
            rng.shuffle(perm)
            perm_map = dict(zip(donor_ids, perm))
            
            # 使用 pairwise_level 而不是硬编码 donor_id，兼容 sample_id 或自定义 block。
            df["disease_perm"] = df[pairwise_level].map(perm_map)
            groups_perm = [
                x["prop"].dropna().values
                for _, x in df.groupby("disease_perm")
            ]
            stat = stats.kruskal(*groups_perm, nan_policy="omit").statistic
            perm_stats.append(stat)
        
        perm_stats = np.array(perm_stats)
        pval = (np.sum(perm_stats >= obs_stat) + 1) / (n_perm + 1)
        
        # Step 2: 进行一对一的差异检验
        contrast_table = _pairwise_perm_vs_ref(
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
    except Exception as e:
        contrast_table=None
        extra["error"] = str(e)
        
    return make_result(
        method="PERM_MIXED",
        cell_type=cell_type,
        p_val=pval if pval is not None else np.nan, p_type='Global',
        contrast_table=contrast_table,
        extra=extra,
        alpha=alpha
    )
