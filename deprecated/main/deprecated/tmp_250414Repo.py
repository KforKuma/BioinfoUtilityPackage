import scanpy as sc
import anndata
import pandas as pd
import os

os.chdir("/data/HeLab/bio/IBD_analysis/")
from src.ScanpyTools.ScanpyTools import ScanpyPlotWrapper

rank_genes_groups_dotplot = ScanpyPlotWrapper(func = sc.pl.rank_genes_groups_dotplot)
dotplot = ScanpyPlotWrapper(func = sc.pl.dotplot)
umap_plot = ScanpyPlotWrapper(func = sc.pl.umap)
tsne_plot = ScanpyPlotWrapper(func = sc.pl.tsne)
adata = anndata.read_h5ad("/data/HeLab/bio/IBD_analysis/tmp/Step12_cleansed(updated0101_clustered).h5ad")




output="/data/HeLab/bio/IBD_analysis/output/Step14/"



my_marke_dict={"TFH":["PDCD1","CXCR5","CXCL13","IL21"],
               "GZMK":["GZMK","EOMES"],
               "STEMLIKE":["TCF7","CD27","TOX","TOX2"]}

from src.ScanpyTools.ScanpyTools import sanitize_filename
from statannotations.Annotator import Annotator

def geneset_dotplot(adata_subset, my_markers, marker_name, output_dir, file_name, groupby_key):
    # 绘制标记基因点图
    dtplt_filename = sanitize_filename(f"{file_name}_{groupby_key}_{marker_name}")
    dotplot(
        save_addr=output_dir,
        filename=dtplt_filename,
        adata=adata_subset,
        groupby=groupby_key,
        layer="log1p_norm",
        standard_scale="var",
        var_names=my_markers
    )
    print(f"--> Dot plot for facet markers saved as '{file_name}_{groupby_key}_{marker_name}.png'.")
    
        

adata_T = adata[adata.obs['Cell_type']=="T_cell"]
adata_T.obs['Subset_Identity'].unique().tolist()
# ['CD8 Tem', 'CD4 Tfh', 'CD8aa', 'CD8 Trm', 'CD8 Tcm',
# 'CD4 Tem', 'CD4 Tcm', 'CD4 Th17',
# 'CD8 Trm IFNG+', 'CD4 Tfr', 'CD4 Treg']

adata_CD4 = adata_T[adata_T.obs['Subset_Identity'].isin(['CD4 Tfh','CD4 Tem', 'CD4 Tcm', 'CD4 Th17','CD4 Tfr', 'CD4 Treg'])]
geneset_dotplot(adata_CD4, my_markers=my_marke_dict,marker_name="stem_like",
                output_dir=output,
                file_name="CD4",
                groupby_key="Subset_Identity")


adata_CD8 = adata[adata.obs['Subset_Identity'].isin(['CD8 Tem','CD8aa', 'CD8 Trm', 'CD8 Tcm','CD8 Trm IFNG+','NK CD56+','NK CD16+','ILC3'])]
geneset_dotplot(adata_CD8, my_markers=my_marke_dict,marker_name="stem_like",
                output_dir=output,
                file_name="CD8",
                groupby_key="Subset_Identity")


TF_dict = {"BCL11B":["BCL11B","ZBTB16","CDKN1A"],
           "IBD":["BACH1","BACH2","XBP1","HIF1A","KLF12"],
           "BS":["ZBTB7A","KLF13","EOMES","PRDM1","RORC"]
           }
geneset_dotplot(adata_CD8, my_markers=TF_dict,marker_name="BCL11B",
                output_dir=output,
                file_name="CD8_dis",
                groupby_key="disease")
adata_CD8_nocm
adata_CD8_nocm = adata_CD8_nocm[adata_CD8_nocm.obs['Subset_Identity'] != "CD8aa"]

geneset_dotplot(adata_CD8_nocm, my_markers=TF_dict,marker_name="BCL11B",
                output_dir=output,
                file_name="CD8_ncm_dis",
                groupby_key="disease")
import matplotlib.pyplot as plt
# --------------------------------------
# 1. 对基因集打分
# --------------------------------------
# 计算 gene_list 的表达打分，并将结果存储在 adata_CD8.obs 中，score_name 可自定义名称
gene_list=['TCF7', 'CD27', 'TOX', 'TOX2', 'PDCD1', 'CXCR5', 'CXCL13', 'IL21']



[g for g in gene_list if g in adata_CD8.var_names]
sc.tl.score_genes(adata_CD8, gene_list=gene_list, score_name='gene_set_score',use_raw=False)
sc.tl.score_genes(adata_CD4, gene_list=gene_list, score_name='gene_set_score',use_raw=False)
# --------------------------------------
# 2. 构建分组变量（同时考虑 "disease" 和 "Subset_Identity"）
# --------------------------------------
# 为了后续按组合分组展示，可以新建一列，比如 "group"，将两个变量的字符串拼接
adata_CD8.obs['group'] = (
    adata_CD8.obs['disease'].astype(str) + "_" +
    adata_CD8.obs['Subset_Identity'].astype(str)
)
adata_CD4.obs['group'] = (
    adata_CD4.obs['disease'].astype(str) + "_" +
    adata_CD4.obs['Subset_Identity'].astype(str)
)
# --------------------------------------
# 3. 可视化打分结果
# --------------------------------------
# (1) 小提琴图：展示每个分组中 gene_set_score 的分布情况
adata_CD4 = adata_CD4[adata_CD4.obs['disease'].notna()]
adata_CD4_Tfh = adata_CD4[adata_CD4.obs['Subset_Identity']=="CD4 Tfh"]
sc.pl.violin(adata_CD4_Tfh, keys='gene_set_score', groupby='group',
             stripplot=True, jitter=0.4, rotation=90,
             title='Gene Set Scoring by Group',save="disease_subset_score_CD4_Tfh.png")

adata_CD4_Tfr = adata_CD4[adata_CD4.obs['Subset_Identity']=="CD4 Tfr"]
sc.pl.violin(adata_CD4_Tfh, keys='gene_set_score', groupby='group',
             stripplot=True, jitter=0.4, rotation=90,
             title='Gene Set Scoring by Group',save="disease_subset_score_CD4_Tfr.png")

adata_CD8 = adata_CD8[adata_CD8.obs['disease'].notna()]
adata_CD8_nocm = adata_CD8[adata_CD8.obs['Subset_Identity'] != "CD8 Tcm"]
adata_CD8_nocm = adata_CD8_nocm[adata_CD8_nocm.obs['Subset_Identity'] != "CD8 Tem"]
adata_CD8_nocm.obs["Subset_Identity"]
adata_CD8.obs['group'].unique().tolist()
sc.pl.violin(adata_CD8, keys='gene_set_score', groupby='disease',stripplot=False,
             rotation=90,
             title='Gene Set Scoring by Group',save="disease_subset_score_CD8.png")

sc.pl.violin(adata_CD8, keys='gene_set_score', groupby='Subset_Identity',stripplot=False,
             rotation=90,
             title='Gene Set Scoring by Group',save="disease_subset_score_CD8_si.png")

sc.pl.violin(adata_CD8_nocm, keys='gene_set_score', groupby='group',stripplot=False,
             rotation=90,
             title='Gene Set Scoring by Group',save="disease_subset_score_CD8ncm_group.png")







# --------------------------------------
# --------------------------------------
# --------------------------------------
# --------------------------------------


disease_list = []
proportion_list = []

for i in adata_CD4.obs['disease'].unique():
    adata_subset = adata_CD4[adata_CD4.obs['disease']==i]
    for j in adata_subset.obs["orig.ident"].unique():
        adata_subsubset = adata_subset[adata_subset.obs["orig.ident"]==j]
        count = adata_subsubset.obs['Subset_Identity'].tolist().count("CD4 Tfh")
        percent = count/adata_subsubset[adata_subsubset.obs["Cell_type"]=="T_cell"].shape[0]
        if percent>=1.0:
            continue
        else:
            # print(i)
            disease_list.append(i)
            # print(percent)
            proportion_list.append(percent)
    
# 构造结果字典
data = {
    "disease": disease_list,
    "proportion": proportion_list
}

# 输出结果
print(data)


df = pd.DataFrame(data)


# --------------------------------------
# 3. 可视化打分结果
# --------------------------------------
# (1) 小提琴图：展示每个分组中 gene_set_score 的分布情况
sc.pl.violin(adata_CD8, keys='gene_set_score', groupby='group',
             stripplot=True, jitter=0.4, rotation=90,
             title='Gene Set Scoring by Group',save="CD8_score_violin.png")

# 创建图形，绘制不显示内部散点（inner=None）的小提琴图
plt.figure(figsize=(8, 6))
ax = sns.violinplot(data=df, x="disease", y="proportion", inner=None, palette="Set2")
# ax.set_title("Violin Plot of Proportion by Disease Group")
ax.set_ylabel("Proportion")
ax.set_xlabel("Disease")

# 定义要比较的组对，示例这里对比 "Healthy" 与 "Disease"
pairs = [("Control", "UC"),("Control","CD"),("Control","BS"),("Control","Colitis")]

# 使用 statannotations 进行显著性检测并在图中标注
annotator = Annotator(ax, pairs, data=df, x="disease", y="proportion")
# 此处设置 test 为独立样本 t 检验 (t-test_ind)，文本格式为星号显示
annotator.configure(test='t-test_ind', text_format='star', loc='outside')
annotator.apply_and_annotate()

plt.tight_layout()
plt.savefig("figures/violin_cd4_Tfh_by_T.png", dpi=300)


adata_CD4_Tfh.obs['disease_new'] = adata_CD4_Tfh.obs['disease'].apply(lambda x: "Control" if x == "Control" else "Inflammatory")

analysis_DEG(adata_CD4_Tfh, "Tfh_iscontrol", "disease_new", "/data/HeLab/bio/IBD_analysis/output/Step14",obs_subset=False)