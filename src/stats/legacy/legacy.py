import os
import logging

import numpy as np
import pandas as pd
import scipy.stats as stats

import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from scikit_posthocs import posthoc_dunn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.stats.plot import *
from src.stats.support import *
from src.utils.warnings import deprecated
from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


@deprecated(alternative="run_DKD")
@logged
def run_kdk(count_df, type_key, group_key, save_addr, batch_key, sample_key=None, method="Dunn"):
    subset_list = count_df[type_key].unique().tolist()
    
    dfs = []
    
    for subset in subset_list:
        subset_df = count_df[count_df[type_key] == subset]
        all_zeros = (subset_df['count'] == 0).all()
        if all_zeros:
            logger.info(f"{subset} contains all zero count.")
            continue
        df, posthoc_df = kdk_analyze(subset_df,
                                     subset=subset, group_key=group_key,
                                     batch_key=batch_key, sample_key=sample_key,
                                     save_addr=save_addr,
                                     method=method,
                                     do_return=True)
        dfs.append(df)
        if posthoc_df.empty is False:
            posthoc_df["subset"] = subset
    
    summary_stats = pd.concat(dfs, ignore_index=True)
    return summary_stats

@deprecated
@logged
def auto_choose_k(X, k_min=2, k_max=10):
    """
    自动选择最优 k 值，宁多毋少，返回 silhouette score 最优的向上取整的 k。
    """
    
    # best_k = k_min
    best_score = -1
    
    scores = []
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
        labels = kmeans.labels_
        score = silhouette_score(X, labels)
        scores.append((k, score))
        
        if score > best_score:
            # best_k = k
            best_score = score
    
    # 如果“宁多毋少”，就选得分相近的最大 k（±1% 差距）
    max_k = max(scores, key=lambda x: x[1])[0]
    threshold = best_score * 0.99  # 宽容一点
    candidates = [k for k, s in scores if s >= threshold]
    return int(np.ceil(max(candidates))) + 1

@deprecated(alternative="run_DKD")
@logged
def kdk_analyze(df, subset, group_key, batch_key, sample_key=None, save_addr=None, method="Combined", do_return=False):
    '''
    分析某个细胞亚群在不同采样组和疾病组间的差异。
    sample_key 和 batch_key 的前提假设是，同一个 batch 采取同一个 sample 方法，但是不同
    数据集之间可能相同，也可能不同。

    这是单细胞数据的常见情况，举例而言，如研究某一疾病条件下的单细胞环境，得到以下分组：
    Group   Batch   Sample
    Healthy 001     sorted CD45+
    Cancer  002     sorted CD45+
    Healthy 003     all PBMC
    Cancer  004     all PBMC
    如果只研究 001, 002 或 003, 004，则不需要填写 sample_key，这种情况下默认策略是单随机效应模型
    如果同时考虑，情况则略微有所变化。

    :param df: 包含列 [unit_key, group_key, 'count', 'percent', 'logit_percent','total_count']，以及其他需要控制的参数
    :param subset: 当前分析的细胞亚群名
    :param save_addr: 输出图像的文件夹路径
    :return:
    '''
    
    if save_addr is None:
        save_addr = os.getcwd()
    
    # Step 1: 检查不同采样方式是否显著影响该细胞亚群丰度
    if sample_key is not None:
        groups = [g["percent"].values for _, g in df.groupby(sample_key)]
        stat, p = stats.kruskal(*groups)
        logger.info(f"{subset} Kruskal-Wallis test across sample_key: H={stat:.3f}, p={p:.3g}")
    else:
        groups = [g["percent"].values for _, g in df.groupby(batch_key)]
        stat, p = stats.kruskal(*groups)
        logger.info(f"{subset} Kruskal-Wallis test across batch_key: H={stat:.3f}, p={p:.3g}")
    
    # Step 2: 采样效应建模并提取残差
    # 因为每个 batch 内只有一个 sample_key，所以这里建模只能用 batch
    if p < 0.01:
        model = smf.mixedlm("logit_percent ~ 1", data=df, groups=df[batch_key])
        result = model.fit()
        df["residual"] = result.resid
        plot_residual_boxplot(df, subset, group_key, sample_key, save_addr)
    else:
        logger.info(f"{subset} Skip residual analysis: no significant sampling effect (p={p:.3g})")
        return pd.DataFrame(), pd.DataFrame()  # 不进入后续分析
    
    # Step 3: 检查疾病组之间残差是否有显著差异
    groups = [g["residual"].values for _, g in df.groupby(group_key)]
    stat, p = stats.kruskal(*groups)
    logger.info(f"{subset}] Residual-based Kruskal-Wallis across groups: H={stat:.3f}, p={p:.3g}")
    
    # Step 4: Tukey HSD 多组比较
    if p < 0.05:
        if method == "Tukey" or method == "Combined":
            tukey = pairwise_tukeyhsd(df["residual"], df[group_key])
            logger.info(f"Tukey’s HSD\n")
            logger.info(tukey.summary())
            
            tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
            # 修正 Tukey 输出方向：确保 meandiff > 0 表示 group1 > group2
            # 参 https://github.com/statsmodels/statsmodels/issues/8458
            tukey_df["meandiff"] = -tukey_df["meandiff"].astype(float)
            lower = -tukey_df["lower"].astype(float)
            upper = -tukey_df["upper"].astype(float)
            tukey_df["lower"] = upper
            tukey_df["upper"] = lower
            
            tukey_df["reject"] = tukey_df["reject"].astype(str)
            if tukey_df is not None and method != "Combined":
                plot_confidence_interval(tukey_df, subset, save_addr, method)
                plot_better_residual(df, tukey_df, group_key, subset, save_addr)
                if do_return:
                    return df, tukey_df
        if method == "Dunn" or method == "Combined":
            # Dunn’s posthoc test with Holm correction
            dunn_res = posthoc_dunn(df, val_col="residual", group_col=group_key, p_adjust="holm")
            logger.info(f"Dunn’s posthoc (Holm correction):\n{dunn_res}\n")
            
            # 先计算残差的秩
            df_ranks = df.copy()
            df_ranks["residual_rank"] = df_ranks["residual"].rank()
            
            # 构建一个类似 Tukey_df 的表格方便复用绘图函数
            pairs = []
            for i, g1 in enumerate(dunn_res.index):
                for j, g2 in enumerate(dunn_res.columns):
                    if j > i:
                        pval = dunn_res.loc[g1, g2]
                        
                        # 使用秩均值计算差值
                        g1_rank_mean = df_ranks.loc[df_ranks[group_key] == g1, "residual_rank"].mean()
                        g2_rank_mean = df_ranks.loc[df_ranks[group_key] == g2, "residual_rank"].mean()
                        meandiff = g1_rank_mean - g2_rank_mean  # >0 表示 g1 > g2
                        
                        pairs.append({
                            "group1": g1,
                            "group2": g2,
                            "meandiff": meandiff,
                            "lower": np.nan,
                            "upper": np.nan,
                            "reject": str(pval < 0.05),
                            "p-adj": pval
                        })
            dunn_df = pd.DataFrame(pairs).sort_values("meandiff", ascending=False)
            if dunn_df is not None and method != "Combined":
                plot_confidence_interval(dunn_df, subset, save_addr, method)
                plot_better_residual(df, dunn_df, group_key, subset, save_addr)
                if do_return:
                    return df, dunn_df
        
        if method == "Combined":
            # 先选择需要合并的列
            dunn_merge = dunn_df[["group1", "group2", "reject"]].rename(columns={"reject": "dunn_reject"})
            
            # 按前两列合并
            tukey_df = tukey_df.merge(dunn_merge, on=["group1", "group2"], how="left")
            
            # 如果有些组合在 Dunn 中不存在，dunn_reject 会是 NaN，可以填 False
            tukey_df["dunn_reject"] = tukey_df["dunn_reject"].fillna(False)
            
            if tukey_df is not None:
                plot_confidence_interval(tukey_df, subset, save_addr, "Combined")
                plot_better_residual(df, tukey_df, group_key, subset, save_addr)
                if do_return:
                    return df, tukey_df
    
    else:
        logger.info(
            f"{subset} Skip posthoc analysis: no significant Kruskal-Wallis across groups (p={p:.3g})")
        if do_return:
            return df, pd.DataFrame()

@deprecated()
@logged
def make_a_meta(adata, meta_file, batch_key="orig.ident"):
    '''
    生成一个可读的 meta 文件，也可以手动在上面修改

    :param adata:
    :param meta_file:
    :param batch_key:
    :return:
    '''
    meta = _kdk_make_meta(adata, batch_key)
    meta.to_csv(meta_file)

@deprecated()
@logged
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

@deprecated()
@logged
def estimate_resample_params_by_DM(
        df_real: pd.DataFrame,
        collected_results: dict,
        disease_ref: str = "HC",
        tissue_ref: str = "nif",
        alpha: float = 0.05,
        min_abundance: float = 0.01,  # 过滤掉占比小于 1% 的细胞，减少采样噪声干扰
        min_effect_floor: float = 0.1,
        pseudocount: float = 0.1
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
    # use global abundance to define reliable cell types,
    # assuming technical noise dominates over condition-specific expansion
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
    valid_within_sd = within_donor_sd.dropna(how="all")
    
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
