import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import os
import os


def check_counts_layer(adata, layer="counts"):
    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' 不存在于 adata.layers 中。")
    
    counts = adata.layers[layer]
    
    # 数据类型检查
    if isinstance(counts, np.ndarray):
        dtype = counts.dtype
    elif sp.isspmatrix(counts):
        dtype = counts.dtype
    else:
        raise TypeError(f"{layer} 既不是 ndarray 也不是稀疏矩阵")
    print(f"{layer}.dtype = {dtype}")
    if not np.issubdtype(dtype, np.integer):
        raise TypeError(f"{layer} 不是整数类型！")
    
    # 非负性检查
    if isinstance(counts, np.ndarray):
        min_val = counts.min()
    else:
        min_val = counts.min()
    print(f"{layer} 最小值 = {min_val}")
    if min_val < 0:
        raise ValueError(f"{layer} 存在负值 {min_val}！")
    
    # NaN 与 Inf 检查
    if isinstance(counts, np.ndarray):
        has_nan = np.isnan(counts).any()
        has_inf = np.isinf(counts).any()
    else:
        has_nan = np.isnan(counts.data).any()
        has_inf = np.isinf(counts.data).any()
    print(f"{layer} 含 NaN？{has_nan}, 含 Inf？{has_inf}")
    if has_nan:
        raise ValueError(f"{layer} 中存在 NaN！")
    if has_inf:
        raise ValueError(f"{layer} 中存在 Inf！")
    
    print(f"{layer} 检查通过：未归一化、整数、无 NaN/Inf。")


###########################################################
# 饼图绘制 piechart drawing

def get_cluster_counts(adata,
                       cluster_key="cluster_final",
                       sample_key="replicate",
                       # combined=False,
                       drop_values=None):
    # Step.1 计算每个cluster在每个样本中的大小
    sizes = adata.obs.groupby([sample_key,cluster_key]).size().unstack(fill_value=0)
    # Step.2 如果指定了需要删除的值，进行过滤
    if drop_values is not None:
        sizes = sizes.drop(index=drop_values, errors='ignore')
        # 使用 errors='ignore' 以防止因不存在的索引导致错误。
    return sizes
#####################################################################################
#####################################################################################



def plot_cluster_counts(cluster_counts,
                        cluster_palette=None,
                        xlabel_rotation=0):
    # 检查cluster_counts是否为空
    if cluster_counts.empty:
        raise ValueError("The cluster_counts DataFrame is empty.")
    
    # 创建绘图
    fig, ax = plt.subplots(dpi=300)
    fig.patch.set_facecolor("white")
    
    # 如果指定了调色板，则生成对应的颜色映射
    if cluster_palette is not None:
        cmap = sns.color_palette(cluster_palette, n_colors=len(cluster_counts.columns))
    else:
        cmap = sns.color_palette("tab10", n_colors=len(cluster_counts.columns))  # 默认调色板
    
    # 绘制叠加柱状图
    cluster_counts.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        legend=None,
        color=cmap
    )
    
    # 调整图例
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, title="Cluster")
    
    # 美化图形
    sns.despine(fig, ax)
    ax.tick_params(axis="x", rotation=xlabel_rotation)
    ax.set_xlabel(cluster_counts.index.name.capitalize() if cluster_counts.index.name else "Samples")
    ax.set_ylabel("Counts")
    ax.grid(False)
    # 调整图像布局
    fig.tight_layout()
    
    return fig
#####################################################################################
#####################################################################################

def get_cluster_proportions(adata,
                            cluster_key="cluster_final",
                            sample_key="replicate",
                            drop_values=None):
    # 计算每个cluster在每个样本中的大小
    sizes = adata.obs.groupby([sample_key,cluster_key]).size().unstack(fill_value=0)
    
    # 计算每个样本中的cluster比例
    proportions = sizes.div(sizes.sum(axis=1), axis=0) * 100
    
    # 如果指定了需要删除的值，进行过滤
    if drop_values is not None:
        proportions = proportions.drop(index=drop_values, errors='ignore')
    
    return proportions
#####################################################################################
#####################################################################################

def plot_cluster_proportions(cluster_props,
                             cluster_palette=None,
                             xlabel_rotation=0):
    fig, ax = plt.subplots(dpi=300)
    fig.patch.set_facecolor("white")
    
    # 设置颜色映射
    cmap = cluster_palette if cluster_palette is not None else None
    
    # 绘制堆叠条形图
    cluster_props.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        legend=None,
        colormap=cmap
    )
    
    # 设置图例位置
    ax.legend(bbox_to_anchor=(1.01, 1), frameon=False, title="Cluster")
    
    # 美化图形
    sns.despine(fig, ax)
    ax.tick_params(axis="x", rotation=xlabel_rotation)
    ax.set_xlabel(cluster_props.index.name.capitalize() if cluster_props.index.name else "Sample")
    ax.set_ylabel("Proportion (%)")
    ax.grid(False)
    # 调整布局
    fig.tight_layout()
    
    return fig


#####################################################################################
#####################################################################################

def easy_new_ident(anno_list, adata_sub_obs_key, adata_par_obs_key, adata_sub, adata_par):
    # 创建子集标注的映射字典
    cl_annotation = {str(i): anno for i, anno in enumerate(anno_list)}
    print("Annotation Mapping:", cl_annotation)
    # 更新子集的标注
    adata_sub.obs[adata_par_obs_key] = adata_sub.obs[adata_sub_obs_key].map(cl_annotation)
    # 如果父集数据中没有目标 key，则创建并初始化
    if adata_par_obs_key not in adata_par.obs:
        print(f"The parental adata doesn't have the key '{adata_par_obs_key}', now creating.")
        adata_par.obs[adata_par_obs_key] = adata_par.obs.get("Subset_Identity", pd.Series([None] * len(adata_par),
                                                                                          index=adata_par.obs.index))
    # 更新父集数据中的标注信息
    for label in adata_sub.obs[adata_par_obs_key].unique():
        print(f"Processing label: {label}")
        index = adata_sub.obs_names[adata_sub.obs[adata_par_obs_key] == label]
        adata_par.obs.loc[index, adata_par_obs_key] = label
        print(f"Updated {len(adata_par[adata_par.obs[adata_par_obs_key] == label])} entries for label '{label}'.")
#####################################################################################
#####################################################################################

def invert_dict(d,key_list=False):
    inverse = dict()
    for key in d:
        # Go through the list that is saved in the dict:
        for item in d[key]:
            item = str(item)
            # Check if in the inverted dict the key exists
            if key_list == True:
                if item not in inverse:
                    # If not create a new list
                    inverse[item] = [key]
                else:
                    inverse[item].append(key)
            else:
                inverse[item] = key
    return inverse

import itertools

def dict_2_annodict(dict,adata,remap_obs):
    flat_list = list(itertools.chain(*dict.values()))
    if len(set(flat_list)) != len(flat_list):
        print(set([x for x in flat_list if flat_list.count(x) > 1]))
    elif len(set(flat_list)) != len(adata.obs[remap_obs].unique().tolist()):
        print("Missing values...")
        print(set([x for x in adata.obs[remap_obs].unique().tolist() if int(x) not in flat_list]))
    else:
        print("Checked.")
    rev_anno_dict = invert_dict(dict)
    return rev_anno_dict


def BasicQCPlot(adata,suffix):
    sc.pl.umap(adata,color=["orig.project"],save = "Origin_UMAP_"+suffix+".png")
    sc.pl.umap(adata,color=["DF_hi.lo", "pct_counts_ribo", "pct_counts_mt", "phase"],save="QC_Info_"+suffix+".png")
    sc.pl.umap(adata,color=["n_genes_by_counts", "total_counts", "n_genes"],save="QC_Info2_"+suffix+".png")



def remove_genes(adata, ribo_gene=True, mito_gene=True, sex_chr=True):
    # 定义染色体注释文件的路径
    chr_annot_path = "/data/HeLab/bio/biosoftware/customised/HSA_chromsome_annot.csv"
    # 加载或下载染色体注释
    if not os.path.exists(chr_annot_path):
        try:
            annot = sc.queries.biomart_annotations(
                "hsapiens",
                ["ensembl_gene_id", "external_gene_name", "start_position", "end_position", "chromosome_name"]
            ).set_index("external_gene_name")
            annot.to_csv(chr_annot_path)
            print(f"Annotations downloaded and saved to {chr_annot_path}.")
        except Exception as e:
            print(f"Failed to download annotations: {e}")
            return adata
    else:
        annot = pd.read_csv(chr_annot_path, index_col="external_gene_name")
    # 获取性染色体基因
    chrY_genes = adata.var_names.intersection(annot.index[annot.chromosome_name == "Y"])
    chrX_genes = adata.var_names.intersection(annot.index[annot.chromosome_name == "X"])
    sex_genes = chrY_genes.union(chrX_genes)
    # 获取线粒体和核糖体基因
    mt_genes = adata.var_names[adata.var_names.str.startswith('MT-')].tolist()
    rb_genes = adata.var_names[adata.var_names.str.startswith(('RPS', 'RPL', 'RPLP', 'RPSA'))].tolist()
    # 构建需要移除的基因列表
    remove_genes = []
    if mito_gene:
        remove_genes += mt_genes
    if ribo_gene:
        remove_genes += rb_genes
    if sex_chr:
        remove_genes += sex_genes.tolist()
    # 输出基因数
    print(f"Number of genes to remove: {len(remove_genes)}")
    # 保留不在移除列表中的基因
    keep_genes = adata.var_names.difference(remove_genes)
    print(f"Number of genes to keep: {len(keep_genes)}")
    # 更新数据
    adata = adata[:, keep_genes].copy()
    return adata


###########################################################
# 基于COSG的快速HVG富集

def cosg_rankplot(adata,groupby,csv_name,plot_name,top_n=5):
    import cosg as cosg
    import pandas as pd
    import importlib
    cosg.cosg(adata, key_added='cosg', mu=1, n_genes_user=100, groupby=groupby)
    colnames = ['names', 'scores']
    test = [pd.DataFrame(adata.uns["cosg"][c]) for c in colnames]
    test = pd.concat(test, axis=1, names=[None, 'group'], keys=colnames)
    test.to_csv(csv_name)
    markers = {}
    cats = adata.obs[groupby].cat.categories
    for i, c in enumerate(cats):
        cell_type_df = test.loc[:, 'names'][c]
        scores_df = test.loc[:, 'scores'][c]
        markers[c] = cell_type_df.values.tolist()[:top_n]
    sc.pl.dotplot(adata, var_names=markers,
                  groupby=groupby,
                  cmap='Spectral_r', use_raw=False, standard_scale='var',
                  save=plot_name)
    return(adata)


# 打分函数
# def plot_cluster_counts(marker_dict,adata_subset):
#     for key in marker_dict:
#         sc.tl.score_genes(adata = adata_subset,gene_list = marker_dict[key],score_name = str(key),use_raw = False)
#     var_df = pd.DataFrame(data = list(marker_dict.keys()), index = list(marker_dict.keys()),columns=['features'])
#     obs_remained = ['orig.ident','disease',"Subset",'Subset_Identity','Celltype_Identity'] + [x for x in adata_subset.obs.columns.tolist() if x.startswith('leiden')]  + [x for x in adata_subset.obs.columns.tolist() if x.startswith('expl')]
#     adata_score = anndata.AnnData(X = adata_subset.obs[list(marker_dict.keys())],
#                                   obs = adata_subset.obs[obs_remained],
#                                   var = var_df)
#     return adata_score

# downsample

# This function at least subsamples all classes in an obs column to the same number of cells. Would be straightforward to modify to what you probably think of.

##################
def plot_piechart(subset_count,general_count,colormaplist = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00"]):
    plt.clf()  # clear current figure
    plt.title(i, fontsize=10)
    plt.pie(x=general_count,
            colors=colormaplist,
            radius=0.8,  # 半径
            pctdistance=0.765,
            autopct='%3.1f%%',
            # labels=val_counts3.index.tolist(),
            textprops=dict(color="w"),
            wedgeprops=dict(width=0.3, edgecolor='w'))
    plt.pie(x=subset_count,
            autopct="%3.1f%%",  # 百分比显示格式
            radius=1,  # 半径
            pctdistance=0.85,  # 百分比文本距离圆心距离
            colors=colormaplist,  # 颜色
            textprops=dict(color="w"),  # 文本设置
            labels=subset_count.index.tolist(),  # 各类别标签
            wedgeprops=dict(width=0.3, edgecolor='w'))  # 饼图内外边格式设置
    plt.legend(loc='upper center', bbox_to_anchor=(1, 0.5))
    plt.rcParams['figure.figsize'] = [6, 6]
    return plt

##########################################################################################
def compute_celltype_proportion(adata, cell_subtype, celltype_adjust=True):
    import pandas as pd
    import re
    # 缓存常用列
    obs = adata.obs
    
    # 获取总细胞数按 disease 分组
    disease_total = obs["tissue-disease"].value_counts()
    
    # 子集掩码
    subset_mask = obs["Subset_Identity"] == cell_subtype
    subset_disease = obs.loc[subset_mask, "tissue-disease"]
    subset_counts = subset_disease.value_counts()
    
    # 计算全局比例
    adj_all = subset_mask.sum() / obs.shape[0]
    proportions_all = (subset_counts / disease_total) * 100
    proportions_all_adj = proportions_all / (100 * adj_all)
    
    frame = {
        "Among All(%)": proportions_all,
        "Among All (Normed)": proportions_all_adj,
        "Counts": subset_counts
    }
    
    if celltype_adjust:
        celltype_auto = cell_subtype.split("_")[0]
        celltype_mask = obs["Celltype"] == celltype_auto
        celltype_disease = obs.loc[celltype_mask, "tissue-disease"]
        celltype_counts = celltype_disease.value_counts()
        
        # 如果分母为 0，需要处理
        adj = subset_mask.sum() / celltype_mask.sum() if celltype_mask.sum() > 0 else float("nan")
        
        proportions = (subset_counts / celltype_counts) * 100
        proportions_adj = proportions / (100 * adj)
        
        frame.update({
            "Among Celltype": proportions,
            "Among Celltype (Normed)": proportions_adj
        })
    
    result = pd.DataFrame(frame).fillna(0).sort_index()
    print(result)
    return result


cell_subtype = 'T Cell_CD8.Trm.KLRC2+'
def conservative_celltype_proportion(adata, cell_subtype, groupby="disease"):
    celltype = cell_subtype.split("_")[0]
    if celltype in [ 'T Cell', 'Plasma', 'Myeloid', 'B Cell']:
        is_immune = True
    elif celltype in ['Epi','Endo', 'Fibroblast']:
        is_immune = False
    else:
        print("Might be mitotitc or celltype cannot be recognized, skipped.")
        return
    
    if is_immune:
        cell_mask = adata.obs["presorted"].isin(["CD45+", "intact"])
    else:
        cell_mask = adata.obs["presorted"].isin(["CD45-", "intact"])
    
    cell_df = adata.obs[cell_mask]
    
    # 每个 Patient 的 Subset_Identity 比例表
    cell_props = (
        cell_df.groupby(["orig.ident", "Subset_Identity"])
        .size()
        .div(cell_df.groupby("orig.ident").size(), level=0)
        .unstack(fill_value=0)
    )
    cell_props = cell_props.dropna()
    
    # 将 disease 信息补上（从原始 obs 中抓取）
    cell_props["disease"] = adata.obs.groupby("orig.ident")["disease"].first()
    
    # 计算每个 groupby 中各细胞类型的均值与标准差
    cell_stats = cell_props.groupby(groupby).agg(["mean", "std"])


# ## 检查是否差异显著
# from scipy.stats import kruskal
#
# groups = [adata.obs.loc[adata.obs["presorted"] == g, "Subset_Identity"]
#           for g in adata.obs["presorted"].unique()]
#
# stat, p_value = kruskal(*groups)
# print(f"Kruskal-Wallis H-test: H = {stat:.2f}, p = {p_value:.4e}")
#
# ## 检查是否可以合并
# from itertools import combinations
#
# group_names = adata.obs["presorted"].dropna().unique()
# results = []
#
# for g1, g2 in combinations(group_names, 2):
#     sub_df = adata.obs[adata.obs["presorted"].isin([g1, g2])]
#     contingency = pd.crosstab(sub_df["presorted"], sub_df["Subset_Identity"])
#
#     if contingency.shape[0] < 2 or contingency.shape[1] < 2:
#         p = float('nan')
#     else:
#         _, p, _, _ = chi2_contingency(contingency)
#
#     results.append({"group1": g1, "group2": g2, "p_value": p})
#
# # 输出结果
# pairwise_df = pd.DataFrame(results).sort_values("p_value")
# print(pairwise_df)


##########################################################################################
def count_element_list_occurrence(list_of_lists):
    '''
    跨列表元素统计，常用于多个列表之间的基因出现的频率统计
    :param list_of_lists: 多个列表的列表
    :return: 返回计数字典
    '''
    from collections import defaultdict
    counter = defaultdict(int)
    for unique_list in list_of_lists:
        for item in set(unique_list):  # 用 set() 保证列表内唯一
            counter[item] += 1
    return dict(counter)


def map_element_to_list_indices(list_of_lists):
    '''
    和count_element_list_occurence类似，但增加了出现的列表的的信息
    :param list_of_lists:
    :return:
    '''
    from collections import defaultdict
    element_map = defaultdict(set)
    for i, group in enumerate(list_of_lists):
        for item in set(group):
            element_map[item].add(i)
    return {k: sorted(v) for k, v in element_map.items()}

##################################################
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_celltype_residuals(df, subset, output_dir):
    """
    分析某个细胞亚群在不同采样组和疾病组间的差异。
    
    参数：
    - df: 包含列 ['sample', 'disease_group', 'sampling_group', 'celltype', 'count', 'percent', 'logit_percent']
    - subset: 当前分析的细胞亚群名
    - output_dir: 输出图像的文件夹路径
    """
    
    # Step 2: 检查不同采样方式是否显著影响该细胞亚群丰度
    groups = [g["percent"].values for _, g in df.groupby("sampling_group")]
    stat, p = stats.kruskal(*groups)
    print(f"[{subset}] Kruskal-Wallis test across sampling_group: H={stat:.3f}, p={p:.3g}")
    
    # Step 3: 采样效应建模并提取残差
    if p < 0.01:
        model = smf.mixedlm("logit_percent ~ 1", data=df, groups=df["sampling_group"])
        result = model.fit()
        df["residual"] = result.resid
        plot_residual_boxplot(df, subset, output_dir)
    else:
        print(f"[{subset}] Skip residual analysis: no significant sampling effect (p={p:.3g})")
        return  # 不进入后续分析
    
    # Step 4: 检查疾病组之间残差是否有显著差异
    groups = [g["residual"].values for _, g in df.groupby("disease_group")]
    stat, p = stats.kruskal(*groups)
    print(f"[{subset}] Residual-based Kruskal-Wallis across disease_group: H={stat:.3f}, p={p:.3g}")
    
    # Step 5: Tukey HSD 多组比较
    tukey = None
    tukey_df = None
    if p < 0.05:
        tukey = pairwise_tukeyhsd(df["residual"], df["disease_group"])
        print(tukey.summary())
        
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        tukey_df["meandiff"] = tukey_df["meandiff"].astype(float)
        tukey_df["lower"] = tukey_df["lower"].astype(float)
        tukey_df["upper"] = tukey_df["upper"].astype(float)
        tukey_df["reject"] = tukey_df["reject"].astype(str)
        if tukey_df is not None:
            plot_tukey_ci(tukey_df, subset, output_dir)
            plot_better_residual(df, tukey_df, subset,output_dir)
            plot_average_percentage(df,subset,output_dir)
    else:
        print(f"[{subset}] Skip tukey analysis: no significant Kruskal-Wallis across disease_group (p={p:.3g})")
        return  # 不进入后续分析
    

def plot_residual_boxplot(df, subset, output_dir):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="disease_group", y="residual", hue="sampling_group")
    plt.title(f"Residuals of {subset} after correcting for sampling_group")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{subset}_residual(boxplot).png", bbox_inches='tight')
    plt.close()


def plot_tukey_ci(tukey_df, subset, output_dir):
    fig, ax = plt.subplots(figsize=(8, len(tukey_df) * 0.5))
    for i, row in tukey_df.iterrows():
        ax.plot([row["lower"], row["upper"]], [i, i], color="black")
        ax.plot(row["meandiff"], i, "o", color="red" if row["reject"] == "True" else "gray")
    
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_yticks(range(len(tukey_df)))
    ax.set_yticklabels([f"{a} vs {b}" for a, b in zip(tukey_df['group1'], tukey_df['group2'])])
    ax.set_xlabel("Mean difference (95% CI)")
    ax.set_title("Tukey HSD pairwise comparisons")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{subset}_tukey_ci.png", bbox_inches='tight')
    plt.close()


def plot_better_residual(df, tukey_df, subset, output_dir):
    print("Plotting beteer residual barplot.")
    # 计算每组的平均残差
    grouped = df.groupby("disease_group")["residual"].mean().reset_index()
    grouped = grouped.sort_values("residual")  # 按 residual 从小到大排序
    
    # 创建索引映射，便于 Tukey 连线定位
    group_order = grouped["disease_group"].tolist()
    group_to_x = {group: i for i, group in enumerate(group_order)}
    
    #
    plt.figure(figsize=(8, 6))
    sns.barplot(x="disease_group", y="residual", data=grouped,
                order=group_order, palette="viridis")
    
    plt.ylabel("Residual")
    plt.xlabel("Disease Group")
    
    # 过滤出显著的比较
    significant = tukey_df[tukey_df["reject"] == "True"]
    significant = significant.reset_index()
    
    # 连线高度的基础高度（在柱子顶部稍上方）
    y_max = grouped["residual"].max()
    height_step = (grouped["residual"].max() - grouped["residual"].min()) * 0.2
    base_height = y_max + 0.05  # 可调
    
    # 为每个显著组添加线和星号
    for i, row in significant.iterrows():
        g1, g2 = row["group1"], row["group2"]
        x1, x2 = group_to_x[g1], group_to_x[g2]
        x_middle = (x1 + x2) / 2
        h = base_height + i * height_step
        # 连线
        plt.plot([x1, x1, x2, x2], [h - 0.01, h, h, h - 0.01], lw=1.5, c='black')
        # 星号
        pval = row["p-adj"]
        if pval <= 0.001:
            star = "***"
        elif pval <= 0.01:
            star = "**"
        elif pval <= 0.05:
            star = "*"
        else:
            star = "ns"
        # 把星号加在图上
        plt.text(x_middle, h, star, ha='center', va='bottom', fontsize=16)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{subset}_residual(barplot).png", bbox_inches='tight')


def plot_average_percentage(df, subset, output_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from scipy.stats import f_oneway
    
    print("Plotting average percentage")
    
    # 计算 mean 和 sem
    summary = df.groupby("disease_group")["percent"].agg(['mean', 'sem']).reset_index()
    sorted_summary = summary.sort_values("mean").reset_index(drop=True)
    group_order = sorted_summary["disease_group"].tolist()
    
    # ANOVA 检验
    groups = [df[df["disease_group"] == group]["percent"] for group in group_order]
    f_stat, p_value = f_oneway(*groups)
    anova_text = f"(ANOVA p = {p_value:.3e})"
    
    # Tukey 检验
    tukey = pairwise_tukeyhsd(endog=df["percent"], groups=df["disease_group"], alpha=0.05)
    tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
    
    # 开始绘图
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=sorted_summary, x="disease_group", y="mean", yerr=sorted_summary["sem"])
    
    # 添加具体数值
    for i, row in sorted_summary.iterrows():
        plt.text(i, row["mean"] + 0.001, f"{row['mean']:.3f}", ha='center', va='bottom', fontsize=9)
    
    # 添加显著性标记（星号）
    current_height = sorted_summary["mean"].max() + sorted_summary["sem"].max() + 0.01
    height_step = 0.01
    
    for i, row in tukey_df.iterrows():
        if row["reject"]:
            g1, g2 = row["group1"], row["group2"]
            if g1 in group_order and g2 in group_order:
                x1 = group_order.index(g1)
                x2 = group_order.index(g2)
                x1, x2 = sorted([x1, x2])
                y = current_height
                plt.plot([x1, x1, x2, x2], [y, y + height_step, y + height_step, y], lw=1.2, c='k')
                plt.text((x1 + x2) / 2, y + height_step + 0.001, "*", ha='center', va='bottom', color='k', fontsize=14)
                current_height += height_step * 2
    
    plt.ylabel("Fraction of cells in sample")
    plt.xlabel("Disease group")
    plt.title(f"{subset} relative abundance across disease groups\n{anova_text}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{subset}_average_percentage.png", bbox_inches='tight')
    plt.close()


def auto_choose_k(X, k_min=2, k_max=10):
    """
    自动选择最优 k 值，宁多毋少，返回 silhouette score 最优的向上取整的 k。
    """
    from sklearn.metrics import silhouette_score
    
    best_k = k_min
    best_score = -1
    
    scores = []
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
        labels = kmeans.labels_
        score = silhouette_score(X, labels)
        scores.append((k, score))
        
        if score > best_score:
            best_k = k
            best_score = score
    
    # 如果“宁多毋少”，就选得分相近的最大 k（±1% 差距）
    max_k = max(scores, key=lambda x: x[1])[0]
    threshold = best_score * 0.99  # 宽容一点
    candidates = [k for k, s in scores if s >= threshold]
    return int(np.ceil(max(candidates)))+1


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull

def perform_pca_clustering_on_residuals(df, output_path, auto_choose_k_func):
    """
    对 residual 进行标准化、PCA、聚类、绘图。
    参数:
        df: 包含 'Subset_Identity', 'disease_group', 'residual' 列的 DataFrame
        output_path: 保存图像的基础路径
        auto_choose_k_func: 接收 DataFrame 并返回最佳 k 的函数
    返回:
        pca_df: 包含 PC1, PC2, cluster 的 DataFrame，索引为 Subset_Identity
        resid_scaled_row: 归一化后的 pivot 表（含 residual 值）
        subset_to_cluster: dict，subset → cluster label
        explained_var: PCA 每个主成分的解释度
    """
    
    # pivot + 缺失填补
    resid_pivot = df.groupby(["Subset_Identity", "disease_group"])["residual"].mean().unstack().fillna(0)
    
    # 标准化（按 subset 行）
    scaler = StandardScaler()
    resid_scaled_row = pd.DataFrame(
        scaler.fit_transform(resid_pivot.T).T,
        index=resid_pivot.index,
        columns=resid_pivot.columns
    )
    
    # PCA
    pca = PCA(n_components=2)
    resid_pca = pca.fit_transform(resid_scaled_row)
    explained_var = pca.explained_variance_ratio_ * 100
    
    # PCA dataframe
    pca_df = pd.DataFrame(
        resid_pca, columns=["PC1", "PC2"], index=resid_scaled_row.index
    )
    
    # 自动聚类
    auto_k = auto_choose_k_func(pca_df)
    kmeans = KMeans(n_clusters=auto_k, random_state=0, n_init=10).fit(pca_df)
    pca_df["cluster"] = kmeans.labels_
    
    # 建立映射
    subset_to_cluster = pca_df["cluster"].to_dict()
    
    # 绘图
    plt.figure(figsize=(6, 5))
    for cluster in pca_df["cluster"].unique():
        cluster_data = pca_df[pca_df["cluster"] == cluster]
        plt.scatter(cluster_data["PC1"], cluster_data["PC2"], label=f"Cluster {cluster}", s=40)
        
        if len(cluster_data) >= 3:
            points = cluster_data[["PC1", "PC2"]].values
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k--', linewidth=1)
        
        for subset in cluster_data.index:
            plt.text(pca_df.loc[subset, "PC1"], pca_df.loc[subset, "PC2"], subset, fontsize=8)
    
    plt.xlabel(f"PC1 ({explained_var[0]:.1f}%)")
    plt.ylabel(f"PC2 ({explained_var[1]:.1f}%)")
    plt.title("PCA + Clustering on Residuals")
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{output_path}/All_subset_pca_clusters.png")
    
    return pca_df, resid_scaled_row, subset_to_cluster, explained_var


def plot_residual_heatmap(resid_scaled_row, subset_to_cluster, output_path):
    """
    根据 residual 矩阵和 cluster 信息绘制热图。
    参数:
        resid_scaled_row: 标准化后的 residual pivot 表
        subset_to_cluster: subset → cluster 的映射 dict
        output_path: 输出图像路径
    """
    
    df = resid_scaled_row.copy()
    df["cluster"] = df.index.map(subset_to_cluster)
    
    df = df.sort_values(by=["cluster", df.index.name or "index"])
    heatmap_data = df.drop(columns=["cluster"])
    
    cluster_labels = df["cluster"].values
    cluster_change_locs = np.where(np.diff(cluster_labels) != 0)[0] + 1
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        yticklabels=True,
        linewidths=0.5,
        linecolor='grey',
        cbar_kws={"label": "Residual"}
    )
    
    for y in cluster_change_locs:
        plt.axhline(y=y, color="black", linewidth=2)
    
    plt.title("Mean Residuals by Subset and Disease Group (Cluster-separated)")
    plt.xlabel("Disease Group")
    plt.ylabel("Subset (Cluster Sorted)")
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{output_path}/All_subset_residual_heatmap.png")
