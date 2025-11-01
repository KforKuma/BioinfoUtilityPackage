"""
Step07d_Stats(abundance).py
Author: John Hsiung
Update date: 2025-08-29
Description:
    - 进行丰度差异分析，思路如下：
        1）用 Kruskal-Wallis 检验采样是否造成差异；
        2）对采样差异的情况，采用线性混合模型剔除采样影响，保留残差进行分析；
        3）用 Kruskal-Wallis 检验疾病是否造成（残差的）差异（残差几乎一定不是正态分布）；
        4）对结果进行 Tukey HSD 多重比较；
        5）参考对细胞亚群比例直接进行 ANOVA + Tukey HSD多重比较。
        这一整套简称为 KW 工作流
    - 作为对照，我们采取标准的 cna 工作流进行参考（https://github.com/immunogenomics/cna）

Notes:
    - 依赖环境: conda activate scvpy10
"""
##——————————————————————————————————————————————————————————————————————————
import numpy as np
import leidenalg
import sklearn
import scanpy as sc
import scanpy.external as sce
import anndata
import pandas as pd
import os, gc, sys
##——————————————————————————————————————————————————————————————————————————
# 设置scanpy基本属性
import yaml

with open('/data/HeLab/bio/IBD_analysis/assets/io_config.yaml', 'r') as f:
    config = yaml.safe_load(f) # 读取的格式为一系列字典的列表
##——————————————————————————————————————————————————————————————————————————
os.chdir("/data/HeLab/bio/IBD_analysis/")
sys.path.append('/data/HeLab/bio/IBD_analysis/')
from src.ScanpyTools.Scanpy_statistics import (analyze_celltype_residuals,
                                               perform_pca_clustering_on_residuals, plot_residual_heatmap)

##——————————————————————————————————————————————————————————————————————————
sc.set_figure_params(dpi_save=450, color_map = 'viridis_r',fontsize=6)
sc.settings.verbosity = 1
sc.logging.print_header()
##——————————————————————————————————————————————————————————————————————————
# 第一步部分：KW 分析
##——————————————————————————————————————————————————————————————————————————
# 1) 数据处理，获取表达矩阵，检查格式
adata = anndata.read_h5ad("/data/HeLab/bio/IBD_analysis/output/Step07/Step07_finalversion_5.h5ad")
remap_dict = {'A3':'GSM3214201','A2':'GSM3214202','A1':'GSM3214203','B3':'GSM3214204',
              'B2':'GSM3214205','B1':'GSM3214206','C3':'GSM3214207','C2':'GSM3214208','C1':'GSM3214209',
              'SRR23446053':'GSM7041323','SRR23446054':'GSM7041323','SRR23446063':'GSM7041324','SRR23446064':'GSM7041324','SRR23446065':'GSM7041324','SRR23446066':'GSM7041324','SRR23446059':'GSM7041325','SRR23446060':'GSM7041325','SRR23446061':'GSM7041325','SRR23446062':'GSM7041325','SRR23446055':'GSM7041326','SRR23446056':'GSM7041326','SRR23446057':'GSM7041326','SRR23446058':'GSM7041326','SRR23446051':'GSM7041327','SRR23446052':'GSM7041327','SRR23446049':'GSM7041328','SRR23446050':'GSM7041328','SRR23446047':'GSM7041329','SRR23446048':'GSM7041329','SRR23446045':'GSM7041330','SRR23446046':'GSM7041330','SRR23446043':'GSM7041331','SRR23446044':'GSM7041331','SRR23446041':'GSM7041332','SRR23446042':'GSM7041332','SRR23446039':'GSM7041333','SRR23446040':'GSM7041333','SRR23446037':'GSM7041334','SRR23446038':'GSM7041334','SRR23446035':'GSM7041335','SRR23446036':'GSM7041335'}
adata.obs["orig.ident"] = adata.obs["orig.ident"].replace(remap_dict)
adata.obs["orig.ident"] = adata.obs["orig.ident"].str.split("_").str[0]

count_dataframe = (
    adata.obs[["orig.ident", "Subset_Identity"]]
    .groupby(["orig.ident", "Subset_Identity"])
    .size()
    .reset_index(name='count')
)

cell_identity = pd.ExcelFile("/data/HeLab/bio/IBD_analysis/assets/Sample_grouping_new.xlsx")
cell_identity = cell_identity.parse(cell_identity.sheet_names[0])

print([id for id in count_dataframe["orig.ident"].unique() if id not in cell_identity["orig.ident"].unique()])
print([id for id in cell_identity["orig.ident"].unique() if id not in count_dataframe["orig.ident"].unique()])

merge_df = pd.merge(count_dataframe, cell_identity, how='inner', on='orig.ident')
merge_df["sampling_group"] = (merge_df["tissue-origin"].astype(str) + "_" + merge_df["presorted"].astype(str))
merge_df["disease_group"] = (merge_df["disease"].astype(str) + "_" + merge_df["tissue-type"].astype(str))

count_group_df = merge_df[["orig.ident","sampling_group","disease_group","Subset_Identity","count"]]
count_group_df["log_count"] = np.log1p(count_group_df["count"])
count_group_df["percent"] = count_group_df["count"] / count_group_df.groupby("orig.ident")["count"].transform("sum")
count_group_df["logit_percent"] = np.log(count_group_df["percent"] + 1e-5 / (1 - count_group_df["percent"] + 1e-5))
count_group_df["total_count"] = count_group_df.groupby("orig.ident")["count"].transform("sum")

count_group_df = count_group_df[count_group_df["disease_group"] != "CD_mixed"]
##——————————————————————————————————————————————————————————————————————————
# 2) 进行 KW 分析和可视化
subset_list = count_group_df["Subset_Identity"].unique().tolist()

# subset_list.index('T Cell_gdT.g9d2')
for subset in subset_list:
    subset_df = count_group_df[count_group_df["Subset_Identity"] == subset]
    all_zeros = (subset_df['count'] == 0).all()
    if all_zeros:
        print(f"{subset} contains all zero.")
        continue
    analyze_celltype_residuals(subset_df, subset=subset,
                           output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/Step07d_Statistics")
##——————————————————————————————————————————————————————————————————————————
output_path = "/data/HeLab/bio/IBD_analysis/output/Step07/Step07d_Statistics"

pca_df, resid_scaled_row, subset_to_cluster, explained_var = perform_pca_clustering_on_residuals(
    count_group_df,
    output_path,
    auto_choose_k_func=auto_choose_k
)

plot_residual_heatmap(resid_scaled_row, subset_to_cluster, output_path)
##——————————————————————————————————————————————————————————————————————————
# 第二步部分：CNA分析
##——————————————————————————————————————————————————————————————————————————
import cna
np.random.seed(0)
# 按照 count_group_df 重新修订一下 adata.obs
adata.obs["sampling_group"] = (adata.obs["tissue-origin"].astype(str) + "_" + adata.obs["presorted"].astype(str))
adata.obs["disease_group"] = (adata.obs["disease"].astype(str) + "_" + adata.obs["tissue-type"].astype(str))

adata.obs.drop(
    columns=["percent.mt", "percent.ribo", "percent.hb", "Patient",
             "orig.project", "phase", "disease", "tissue-origin",
             "presorted", "tissue-type"],
    inplace=True
)

# 确保 count 矩阵为整数形式（CNA 要求原始 count）
adata_raw = adata.raw.to_adata() if adata.raw is not None else adata.copy()

# Step 1: 标准预处理（也可用你的方式替代）
sc.pp.normalize_total(adata_raw, target_sum=1e4)
sc.pp.log1p(adata_raw)
sc.pp.highly_variable_genes(adata_raw, n_top_genes=2000, subset=True)
sc.pp.scale(adata_raw)
sc.tl.pca(adata_raw, svd_solver='arpack')
sc.pp.neighbors(adata_raw, n_neighbors=30, n_pcs=30)
adata_raw.write_h5ad("/data/HeLab/bio/IBD_analysis/output/Step07/Step07d_Statistics/cna.h5ad")

adata_raw = anndata.read_h5ad("/data/HeLab/bio/IBD_analysis/output/Step07/Step07d_Statistics/cna.h5ad")

# 1. 从细胞层面 metadata 生成样本级 metadata（每个orig.ident一行）
samplem = adata_raw.obs[["orig.ident", "sampling_group", "disease_group"]]

# 2. 对混杂因素做 one-hot 编码（sampling_group是混杂因素）
samplem_dummies = pd.get_dummies(samplem, columns=["sampling_group"], drop_first=True)

# 3. 去重，保留每个orig.ident一行
samplem_unique = samplem_dummies.groupby("orig.ident").first()

# 4. 运行NAM差异邻域分析
# 这里groupby是你的感兴趣分组变量，sample_col是样本ID
res = cna.tl.nam(
    adata_raw,
    groupby="disease_group",
    sample_col="orig.ident",
    sid_name="orig.ident",              # 这里一定要加上
    covariates=samplem_unique.drop(columns=["disease_group"]),
    n_perms=1000,
    random_state=0,
    verbose=True
)
# 5. 结果res是一个字典，通常包含NAM分析的主要结果
NAM, kept = res  # 解包结果

print(NAM.shape)  # (sample数, cell数)
print(kept.sum()) # 多少个细胞被保留

# 6. 可将结果保存到文件
NAM.to_csv("/data/HeLab/bio/IBD_analysis/output/Step07/Step07d_Statistics/NAM_results.csv")


U, svs, V = cna.tl.svd_nam(NAM)

output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/Step07d_Statistics"

#plot neighborhood loadings of NAM PC 1
adata_raw.obs.loc[kept, 'NAMPC1'] = V.PC1
sc.pl.umap(adata_raw, color='NAMPC1', cmap='seismic', title='NAM PC1 neighborhood loadings')
plt.savefig(f"{output_dir}/fig1.png")


# #plot neighborhood loadings of NAM PC 2
adata_raw.obs.loc[kept, 'NAMPC2'] = V.PC2
sc.pl.umap(adata_raw, color='NAMPC2', cmap='seismic', title='NAM PC2 neighborhood loadings')
plt.savefig(f"{output_dir}/fig2.png")

# 只取前 50 个主成分的特征值（解释的方差）
top_n = 50
svs_subset = svs[:top_n]
# 绘图
plt.clf()
plt.plot(svs_subset / svs.sum(), marker='o', linestyle='--')  # 注意仍然除以总和，确保比例一致
plt.xlabel('NAM PC')
plt.ylabel('Fraction of Variance Explained')
plt.xticks(range(0, top_n, 10))  # 每隔10个PC加一个标签（0, 10, 20, 30, 40）
plt.tight_layout()
plt.savefig(f"{output_dir}/fig3.png")


import matplotlib.pyplot as plt


def sample_loading_plot_by_multi_pheno(pheno):
    unique_classes = pheno.unique()
    colors = plt.colormaps['tab10']  # Matplotlib 3.7 以后推荐这样用
    
    for i, cls in enumerate(unique_classes):
        # 对齐索引，生成bool序列
        mask = pheno.eq(cls).reindex(U.index, fill_value=False)
        plt.scatter(U.loc[mask, 'PC1'], U.loc[mask, 'PC2'],
                    s=40, alpha=0.8, label=str(cls), color=colors(i))
    
    plt.legend(title='Phenotype')
    plt.xlabel('NAM PC1')
    plt.ylabel('NAM PC2')


sample_loading_plot_by_multi_pheno(samplem['disease_group'])
plt.title('NAM PC1 vs NAM PC2, colored by disease group')
plt.savefig(f"{output_dir}/fig3.png")


phenotype = samplem[["orig.ident","disease_group"]]  #
phenotype = phenotype.set_index("orig.ident")
results = []

# 建议先处理好X，只做一次
X = pd.get_dummies(phenotype, drop_first=True)
X = sm.add_constant(X)

# 对齐U和X的index（这是关键）
U = U.loc[X.index]

for i in range(U.shape[1]):
    y = U.iloc[:, i]
    model = sm.OLS(y, X).fit()
    pval = model.f_pvalue
    results.append((f"NAM_PC{i+1}", pval))


# 输出前10个主成分及其 p 值
pd.DataFrame(results, columns=["NAM_PC", "pval"]).sort_values("pval").head(10)

import seaborn as sns
import matplotlib.pyplot as plt

# nam_pc_idx 表示要画的 PC 编号
nam_pc_idx = 0
pc_name = U.columns[nam_pc_idx]  # 获取列名，比如 "PC1"

# 添加 NAM PC 到 samplem（确保索引一致）
# 假设 U.index 是每个细胞对应的 orig.ident（样本ID）
U_sample_mean = U.groupby(U.index).mean()  # 每个样本的平均 NAM_PC 值（可改为median）

# 确保 samplem 的 index 是样本 ID
# 然后再赋值，比如取第1个 NAM_PC
samplem.loc[:, "NAM_PC1"] = samplem["orig.ident"].map(U_sample_mean["PC1"])
# 强制转为浮点数，遇到无法转换的会变成NaN
samplem["NAM_PC1"] = pd.to_numeric(samplem["NAM_PC1"], errors='coerce')

# 绘图
plt.figure(figsize=(6, 4))
sns.boxplot(data=samplem, x="disease_group", y="NAM_PC1")
plt.title(f"{pc_name} vs Disease Group")
plt.tight_layout()
plt.savefig(f"{output_dir}/{pc_name}_boxplot.png")


# 确保 samplem 的 index 是样本 ID
# 然后再赋值，比如取第1个 NAM_PC
samplem.loc[:, "NAM_PC2"] = samplem["orig.ident"].map(U_sample_mean["PC2"])
# 强制转为浮点数，遇到无法转换的会变成NaN
samplem["NAM_PC2"] = pd.to_numeric(samplem["NAM_PC2"], errors='coerce')

# 绘图
plt.figure(figsize=(6, 4))
sns.boxplot(data=samplem, x="disease_group", y="NAM_PC2")
plt.title(f"{pc_name} vs Disease Group")
plt.tight_layout()
plt.savefig(f"{output_dir}/{pc_name}_boxplot2.png")

#__________________________________
import statsmodels.api as sm
import pandas as pd


# 准备设计矩阵X，自动为分类变量做哑变量
X = pd.get_dummies(phenotype, drop_first=False)
X = X.drop(columns=['disease_group_HC_normal'])
X = sm.add_constant(X)

y = samplem[["orig.ident","NAM_PC1"]]  #
y = y.set_index("orig.ident")

model = sm.OLS(y, X).fit()

print(model.summary())
