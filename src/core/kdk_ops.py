import anndata
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf
from scikit_posthocs import posthoc_dunn
from scipy.spatial import ConvexHull
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from src.core.kdk_vis import *

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)

def _kdk_data_prepare(adata, meta, batch_key="orig.ident", type_key="Subset_Identity"):
    '''

    :param adata:
    :param meta: 至少包含 unit_key 列的 pd.DataFrame，储存了分组和采样的详细信息
    :param batch_key:
    :param type_key:
    :return: count_group_df，一个包含至少 unit_key, type_key, count 的 pd.DataFrame，其他列来自 meta 表格的合并
    '''
    count_dataframe = (
        adata.obs[[batch_key, type_key]]
        .groupby([batch_key, type_key])
        .size()
        .reset_index(name='count')
    )
    merge_df = pd.merge(count_dataframe, meta, how='inner', on=batch_key)
    count_group_df = merge_df
    
    count_group_df["log_count"] = np.log1p(count_group_df["count"])
    count_group_df["percent"] = count_group_df["count"] / count_group_df.groupby(batch_key)["count"].transform("sum")
    count_group_df["logit_percent"] = np.log(count_group_df["percent"] + 1e-5 / (1 - count_group_df["percent"] + 1e-5))
    count_group_df["total_count"] = count_group_df.groupby(batch_key)["count"].transform("sum")
    
    return count_group_df

def _kdk_make_meta(adata, group_key="orig.ident"):
    '''
    生成一个用来进行下游分析 meta 文件，包含必要控制的变量。

    :param adata:
    :return:
    '''
    # 选出字符串列（object 或 string）
    string_cols = [c for c in adata.obs.columns if type(adata.obs[c][0]) == str]
    
    # 确保 group_key 也在结果里
    if group_key not in string_cols:
        string_cols.append(group_key)
    
    def unique_or_none(x):
        vals = x.dropna().unique()
        if len(vals) == 1:
            return vals[0]
        else:
            return None  # 多值或空值用 None
    
    # 聚合
    df_grouped = adata.obs[string_cols].groupby(group_key).agg(unique_or_none).reset_index()
    
    # 去除全 None 列
    cols_remain = [c for c in df_grouped.columns if not df_grouped[c].isna().all()]
    df_grouped = df_grouped[cols_remain]
    
    return df_grouped

@logged
def kdk_analyze(df, subset, group_key, batch_key, sample_key=None, save_addr=None, method="Combined",do_return=False):
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
        return  # 不进入后续分析
    
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
            lower= -tukey_df["lower"].astype(float)
            upper = -tukey_df["upper"].astype(float)
            tukey_df["lower"] = upper
            tukey_df["upper"] = lower
            
            tukey_df["reject"] = tukey_df["reject"].astype(str)
            if tukey_df is not None and method != "Combined":
                plot_confidence_interval(tukey_df, subset, save_addr,method)
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
                            "p-adj":pval
                        })
            dunn_df = pd.DataFrame(pairs).sort_values("meandiff", ascending=False)
            if dunn_df is not None and method != "Combined":
                plot_confidence_interval(dunn_df, subset, save_addr,method)
                plot_better_residual(df, dunn_df,group_key, subset, save_addr)
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
                plot_confidence_interval(tukey_df , subset, save_addr,"Combined")
                plot_better_residual(df, tukey_df ,group_key, subset, save_addr)
                if do_return:
                    return df, tukey_df
        
    else:
        logger.info(
            f"{subset} Skip posthoc analysis: no significant Kruskal-Wallis across groups (p={p:.3g})")
        if do_return:
            return df, pd.DataFrame()

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

@logged
def kdk_prepare(adata, meta_file=None, batch_key="orig.ident", type_key="Subset_Identity"):
    '''

    :param adata:
    :param meta_file: 包含样本制作信息的表格，兼容 csv 和 xlsx，默认 header=True index=False
    :param batch_key:
    :param type_key:
    :return:
    '''
    # 读取 meta 信息
    if meta_file is None:
        meta = _kdk_make_meta(adata, batch_key)
    else:
        meta_file = meta_file.strip()
        if meta_file.lower().endswith("csv"):
            meta = pd.read_csv(meta_file)
        elif meta_file.lower().endswith("xlsx"):
            meta = pd.read_excel(meta_file)
        else:
            raise ValueError("[kdk_prepare] Meta file must ends with 'csv' or 'xlsx'.")
    
    # 准备 KW 分析所需矩阵
    count_df = _kdk_data_prepare(adata, meta, batch_key=batch_key, type_key=type_key)
    
    return count_df

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
                                     subset=subset,group_key=group_key,
                                     batch_key=batch_key, sample_key=sample_key,
                                     save_addr=save_addr,
                                     method=method,
                                     do_return=True)
        if posthoc_df.empty is False:
            posthoc_df["subset"] = subset
    
    summary_stats = pd.concat(dfs, ignore_index=True)
    return summary_stats
    

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
