"""
Step07d_Stats(Corr_and_Dist).py
Author: John Hsiung
Update date: 2025-08-29
Description:
    - 绘制细胞亚群身份相关性图，和细胞亚群 × 疾病分布图
Notes:
    - 依赖环境: conda activate scvpy10
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list

# 假设每行是细胞，每列是基因
# 使用平均表达作为聚类基础
grouped = adata.to_df().groupby(adata.obs["Subset_Identity"]).mean()

# 计算皮尔逊相关性矩阵
corr_matrix = grouped.T.corr()

# 计算距离矩阵（1 - correlation）
dist_mat = 1 - corr_matrix
linkage_mat = linkage(squareform(dist_mat), method='average')
ordered_idx = leaves_list(linkage_mat)

# 重新排序
ordered_labels = corr_matrix.index[ordered_idx]
corr_matrix_sorted = corr_matrix.loc[ordered_labels, ordered_labels]

# 使用 clustermap 更快实现，同时满足对称聚类、颜色编码等
from matplotlib.colors import LinearSegmentedColormap
bl_yel_red = LinearSegmentedColormap.from_list("bl_yel_red", ["navy", "lightyellow", "maroon"])
g = sns.clustermap(
    corr_matrix,
    row_linkage=linkage_mat,
    col_linkage=linkage_mat,
    cmap=bl_yel_red,vmax=1,vmin=0,
    linewidths=0.5, figsize=(20, 18)
)
g.ax_heatmap.grid(False)
g.fig.suptitle("Subset Identity Correlation", y=1.02)
g.savefig("/data/HeLab/bio/IBD_analysis/output/Step07/Step07b_Characeterization/Correlation_heatmap.png",bbox_inches="tight")

# 转为 long format
corr_long = corr_matrix_sorted.reset_index().melt(id_vars='Subset_Identity')
corr_long.columns = ['x', 'y', 'value']
plt.figure(figsize=(16, 14))
sns.scatterplot(data=corr_long, x="x", y="y", size="value", hue="value", sizes=(20, 200), palette="coolwarm")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title("Correlation Dot Plot by Cell Type")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(False)
plt.savefig("/data/HeLab/bio/IBD_analysis/output/Step07/Step07b_Characeterization/Correlation_dotplot.png")


adata_imm_solo = adata[adata.obs["Celltype"].isin(['B Cell','Myeloid', 'Plasma','T Cell'])]

# 统计每个样本的细胞数
sample_counts = adata_imm_solo.obs["orig.ident"].value_counts()
# 选出细胞数 >=500 的样本
valid_samples = sample_counts[sample_counts >= 500].index
# 用布尔索引筛选adata
gc.collect()
all_samples = set(adata_imm_solo.obs["orig.ident"])
invalid_samples = all_samples - set(valid_samples)
adata_filtered = adata_imm_solo[~adata_imm_solo.obs["orig.ident"].isin(invalid_samples)].copy()
adata_filtered.write_h5ad("/data/HeLab/bio/IBD_analysis/output/Step07/Step07b_Characeterization/Step07b_statistics_imm.h5ad")


adata_filtered.obs["disease_tissue"] = (
    adata_filtered.obs["disease"].astype(str) + "_" + adata_filtered.obs["tissue-type"].astype(str)
)
value_list = adata_filtered.obs["orig.ident"].value_counts()
value_list = value_list.sort_index()
with pd.option_context('display.max_rows', None):
    print(value_list)

############################
# 按样本统计每种细胞类型的数量
subset_counts = (
    adata_filtered.obs.groupby(["orig.ident", "Subset_Identity"])
    .size()
    .unstack(fill_value=0)
)

# 计算每个样本中每种细胞类型的比例
subset_props = subset_counts.div(subset_counts.sum(axis=1), axis=0)
############################
# 建立 orig.ident 到 disease 的映射
ident2disease = adata_filtered.obs[["orig.ident", "disease_tissue"]].drop_duplicates().set_index("orig.ident")

# 为每个样本添加 disease 列
subset_props["disease_tissue"] = subset_props.index.map(ident2disease["disease_tissue"])
############################
# 分组平均（每组样本内先算平均比例）
mean_props = subset_props.groupby("disease_tissue").mean().T  # 行：细胞类群；列：疾病组
############################
import matplotlib.pyplot as plt

mean_props.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="tab20")
plt.ylabel("Mean Proportion")
plt.title("Mean Proportion of Subset_Identity per Disease")
plt.legend(title="Disease", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.grid(False)
plt.savefig("/data/HeLab/bio/IBD_analysis/output/Step07/Step07b_Characeterization/Proportion_barplot.png")
########################
# Step 1：归一化到每行之和为 1
prop_norm = mean_props.div(mean_props.sum(axis=1), axis=0)

# Step 2：转为 long-form，便于绘图
plot_df = prop_norm.reset_index().melt(id_vars="Subset_Identity",
                                       var_name="disease", value_name="proportion")

# Step 3：绘制并列柱形图（横向）
plt.figure(figsize=(10, 20))
sns.barplot(data=plot_df,
            x="proportion", y="Subset_Identity", hue="disease",
            orient="h", dodge=True)  # dodge=True 是并列的关键参数

plt.xlabel("Proportion (normalized)")
plt.ylabel("Subset Identity")
plt.title("Normalized Cell Subset Proportions by Disease")
plt.legend(title="Disease", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(False)
plt.savefig("/data/HeLab/bio/IBD_analysis/output/Step07/Step07b_Characeterization/Proportion_barplot(norm).png")
##########################################################
# 假设 prop_df 行是细胞类型，列是疾病组
# 每一项都除以对应行的 HC 值
prop_relative_to_hc = mean_props.div(mean_props["HC_normal"], axis=0)

# 可选：去掉 HC 本身列（全是 1），只绘制 CD 和 UC 相对于 HC 的变化
prop_relative_to_hc = prop_relative_to_hc.drop(columns=["HC_normal"])

# 转换为适用于 seaborn 的 long-form 数据格式
plot_df = prop_relative_to_hc.reset_index().melt(id_vars="Subset_Identity",
                                                 var_name="disease", value_name="fold_change_vs_HC")

# 画图：横向并列柱状图
plt.figure(figsize=(10, 18))
sns.barplot(data=plot_df,
            x="fold_change_vs_HC", y="Subset_Identity", hue="disease",
            orient="h", dodge=True)

plt.axvline(1, color="grey", linestyle="--")  # 参考线：与 HC 持平
plt.xlabel("Relative Proportion (vs. HC)")
plt.ylabel("Subset Identity")
plt.title("Fold Change of Cell Type Proportions Relative to HC")
plt.legend(title="Disease", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(False)
plt.savefig("/data/HeLab/bio/IBD_analysis/output/Step07/Step07b_Characeterization/Proportion_barplot(relate_2_hc).png")
##################################################################
# prop_df: index = Subset_Identity, columns = disease
# Step 1: 对行做聚类排序（用 linkage + leaves_list）
# from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib import gridspec

from scipy.cluster.hierarchy import dendrogram, linkage

# 聚类
linkage_mat = linkage(prop_relative_to_hc.values, method='average')
dendro = dendrogram(linkage_mat, labels=prop_relative_to_hc.index, orientation='left', no_plot=True)

# 重新排序 DataFrame
ordered = prop_relative_to_hc.iloc[dendro['leaves'][::-1]]

# 绘图布局
fig = plt.figure(figsize=(12, 16))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4])

# 子图1: dendrogram
ax0 = plt.subplot(gs[0])
dendrogram(linkage_mat, labels=prop_relative_to_hc.index, orientation='left', ax=ax0)
ax0.set_yticks([])

# 子图2: barplot
ax1 = plt.subplot(gs[1])
ordered.plot(kind='barh', ax=ax1)
ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")         # 图例放右边
plt.tight_layout()
plt.grid(False)
plt.savefig("/data/HeLab/bio/IBD_analysis/output/Step07/Step07b_Characeterization/Proportion_barplot(relate_2_hc,tissue_type, cluster).png")
