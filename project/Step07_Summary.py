"""
Step07a_iterative_annot.py
Author: John Hsiung
Update date: 2025-08-29
Description:
    - 进行亚群的细胞身份鉴定，本部分代码有点流水账
Notes:
    - 依赖环境: conda activate scvpy10
"""
####################################
import os, sys

os.environ["ANN_DATA_NO_TORCH"] = "1"

import igraph  # avoid ImportError: dlopen
import anndata
import numpy as np
import pandas as pd
import scanpy as sc

####################################
sc.settings.verbosity = 0
sc.settings.set_figure_params(
    dpi=80,
    facecolor="white",
    frameon=False,
)
####################################
# sys.path.append('/data/HeLab/bio/BioinfoUtilityPackage')
sys.path.append('/public/home/xiongyuehan/data/BioinfoUtilityPackage')

from src.core.handlers.plot_wrapper import ScanpyPlotWrapper
from src.core.handlers.obs_editor import ObsEditor
# from src.core.handlers.geneset_editor import Geneset
from src.core.handlers.identify_focus import *
from src.core.adata.ops import annotate_by_mapping

####################################
# 重新加载
# import importlib
# importlib.reload(sys.modules['src.core.utils.geneset_editor'])

# 删除模块缓存
for module_name in list(sys.modules.keys()):
    if module_name.startswith('src.core'):
        del sys.modules[module_name]

# 重新读入
from src.core.plot.umap import plot_hierarchical_umap
####################################
# 路径初始化
os.chdir("/public/home/xiongyuehan/data/IBD_analysis/output/Step07")
save_addr = "/public/home/xiongyuehan/data/IBD_analysis/output/Step07_Summary"
save_fig_addr = f"{save_addr}/fig"
os.makedirs(save_fig_addr, exist_ok=True)
####################################
# 进行 SCVI 去批次前后的对比
# 我们取 raw 重新进行简单降维聚类
adata = anndata.read_h5ad(f"{save_addr}/Step07_DR_clustered_clean_20260108.h5ad")


# 提取 raw 的表达矩阵（通常是 sparse）
X_raw = adata.raw.X

# 提取 var / obs
var_raw = adata.raw.var.copy()
obs = adata.obs.copy()
obsm = adata.obsm.copy()

# 新建 AnnData
adata_full = anndata.AnnData(
    X=X_raw.copy(),
    var=var_raw.copy(),
    obs=obs,
    obsm=obsm,
)
del adata; gc.collect()

# 标准化
sc.pp.normalize_total(adata_full, target_sum=1e4)
sc.pp.log1p(adata_full)

# 保存备份，但是之后不太会用到
adata_full.write_h5ad(f"{save_addr}/Step07_DR_clustered_clean_20260108(raw_normalised).h5ad")
####################################
# 最后一次改名
ident_remap = {'CAAP epithelium HLA-DR+':"Antigen-presenting colonocyte MHC-II+",
               'Quiescent stem cell LEFTY1+':'Regenerative colonocyte LEFTY1+',
               'Early absorptive progenitor':'Absorptive colonocyte PPARs+'

}


adata.obs["Subset_Identity"] = (
    adata.obs["Subset_Identity"]
    .map(ident_remap)
    .fillna(adata.obs["Subset_Identity"])
)
print(adata.obs['Subset_Identity'].value_counts())

####################################
from src.core.adata.count import print_ref_tab
# 调整细胞顺序
proofread = {'Epithelium': ['Intestinal stem cell OLFM4+LGR5+',
                            'pre-TA cell',
                            'Transit amplifying cell',
                            'Regenerative colonocyte LEFTY1+',
                            'Antigen-presenting colonocyte MHC-II+',
                            'Goblet', 'Paneth cell', 'Tuft cell',  'Enteroendocrine',
                            'Ion-sensing colonocyte BEST4+',
                            'Microfold cell',
                            'Absorptive colonocyte PPARs+','Absorptive colonocyte', 'Absorptive colonocyte Guanylins+'],
             'Fibroblast': ['Fibroblast ADAMDEC1+', 'Fibroblast'],
             'Endothelium': ['Endothelium'],
             'Myeloid Cell': ['Classical monocyte CD14+', 'Nonclassical monocyte CD16A+',
                             'Macrophage', 'Macrophage M1',  'Macrophage M2',
                              'Neutrophil CD16B+',
                              'Mast cell',
                              'cDC1 CLEC9A+', 'cDC2 CD1C+', 'pDC GZMB+'],
             'T Cell': ['CD4 Tnaive', 'CD4 Tmem','CD4 Tmem GZMK+', 'CD4 Tfh', 'CD4 Treg', 'CD4 Th17',
                        'CD8 Tnaive', 'CD8 Tmem', 'CD8 Tmem GZMK+', 'CD8 Trm', 'CD8 Trm GZMA+',
                        'CD8 NKT FCGR3A+','CD8aa IEL',
                        'gdTnaive',  'g9d2T cytotoxic', 'gdTrm',
                        'ILC1', 'ILC3', 'MAIT TRAV1-2+',
                        'Natural killer cell FCGR3A+', 'Natural killer cell NCAM1+',
                        'Mitotic T cell'],
             'B Cell': ['Germinal center B cell',
                        'B cell lambda','B cell kappa', 'B cell IL6+'],
             'Plasma Cell': ['Plasma IgA+', 'Plasma IgG+', 'Mitotic plasma cell']}

adata = annotate_by_mapping(
    adata,
    mapping=proofread,
    col2="Subset_Identity",
    col1="Celltype",
)


cross_tab = print_ref_tab(adata, "Subset_Identity", "Ref_Identity")
# cross_tab.to_csv("/data/HeLab/bio/IBD_analysis/assets/1123_cross_subset.csv")
cross_tab.to_csv(f"{save_addr}/0210_Celltype_Cross_Table.csv")

# 检查
adata.obs["Celltype"] = adata.obs["Celltype"].cat.remove_unused_categories()
adata.obs["Subset_Identity"] = adata.obs["Subset_Identity"].cat.remove_unused_categories()

print(adata.obs['Subset_Identity'].value_counts())
adata.write_h5ad(f"{save_addr}/Step07_DR_clustered_clean_20260210.h5ad")
####################################
adata = anndata.read_h5ad(f"{save_addr}/Step07_DR_clustered_clean_20260210.h5ad")

print(adata.obs['Subset_Identity'].value_counts())
# 简单绘制 UMAP 图
umap_plot = ScanpyPlotWrapper(func=sc.pl.umap)

print(adata.obs.columns)
for i in ["disease", "orig.project","Celltype", "Subset_Identity"]:
    umap_plot(save_addr=save_fig_addr,
              filename=f"UMAP(X_scvi)_{i}",
              adata=adata, color=i)

####################################
# 为了绘制包含所有细胞类群的详细 UMAP 图，我们将对细胞进行改名

ForDraw = {
    'Undifferentiated Epithelium': ['Intestinal stem cell OLFM4+LGR5+',
                                    'pre-TA cell', 'Transit amplifying cell',
                                    'Regenerative colonocyte LEFTY1+',
                                    'Antigen-presenting colonocyte MHC-II+'],
    'Absorptive Epithelium': ['Ion-sensing colonocyte BEST4+',
                              'Absorptive colonocyte PPARs+',
                              'Absorptive colonocyte',
                              'Absorptive colonocyte Guanylins+'],
    'Secretory Epithelium': ['Goblet', 'Paneth cell', 'Tuft cell', 'Enteroendocrine', 'Microfold cell'],
    'Stromal Cell': ['Fibroblast ADAMDEC1+', 'Fibroblast', 'Endothelium'],
    'Myeloid Cell': ['Classical monocyte CD14+', 'Nonclassical monocyte CD16A+',
                     'Macrophage', 'Macrophage M1', 'Macrophage M2',
                     'Neutrophil CD16B+',
                     'Mast cell',
                     'cDC1 CLEC9A+', 'cDC2 CD1C+', 'pDC GZMB+'],
    'CD4 Thymocyte': ['CD4 Tnaive', 'CD4 Tmem', 'CD4 Tmem GZMK+', 'CD4 Tfh', 'CD4 Treg', 'CD4 Th17'],
    'CD8 Thymocyte': ['CD8 Tnaive', 'CD8 Tmem', 'CD8 Tmem GZMK+', 'CD8 Trm', 'CD8 Trm GZMA+',
                      'CD8 NKT FCGR3A+', 'CD8aa IEL'],
    'Thymocyte-Innate Counterpart': ['gdTnaive', 'g9d2T cytotoxic', 'gdTrm',
                                     'ILC1', 'ILC3', 'MAIT TRAV1-2+',
                                     'Natural killer cell FCGR3A+', 'Natural killer cell NCAM1+'],
    'B & Plasma Cell': ['Germinal center B cell', 'B cell lambda', 'B cell kappa', 'B cell IL6+',
                        'Plasma IgA+', 'Plasma IgG+'],
    'Proliferative Cell': ['Mitotic T cell', 'Mitotic plasma cell']
}

adata = annotate_by_mapping(
    adata,
    mapping=ForDraw,
    col2="Subset_Identity",
    col1="Celltype_tmp",
)
# 绘图
from src.core.plot.umap import plot_hierarchical_umap

plot_hierarchical_umap(
    adata=adata,
    save_addr=save_fig_addr,
    filename="UMAP_Subset_Identity",
    hierarchy_dict=ForDraw,
    color_key='Subset_Identity',
    legend_cols=2,
    random_seed=710
)

################################################################
# 细胞大群 - 全部基因热图
celltype_rename_map = {
    'Undifferentiated Epithelium': 'Undiff Epi.',
    'Absorptive Epithelium': 'Absorp Epi.',
    'Secretory Epithelium': 'Secret Epi',
    'Thymocyte-Innate Counterpart': 'Innate Cell'
}
from src.core.plot.matrixplot import plot_deg_matrixplot, plot_matrixplot
print(adata.obs["Celltype_tmp"].value_counts())

order = adata.obs["Celltype_tmp"].cat.categories.tolist()

markers = plot_deg_matrixplot(
    adata,
    save_addr=save_fig_addr,
    filename="Celltype_DEG_Heatmap",
    obs_key='Celltype',
    exclude_groups=[],
    col_order=order,  # 传入先前定义过的 celltype 的排序
    rename_map=celltype_rename_map,
    label_step=5,
    top_n_genes=25,
    save=True,
    use_raw=False
)
plot_matrixplot(
    adata,markers,
    save_addr=save_fig_addr,
    filename="Celltype_DEG_Heatmap(viridis)",
    obs_key='Celltype',
    cmap='viridis',
    label_step=5,
    save=True,
    use_raw=False
)

################################################################
# 补充进行：如果没有按照 scvi 去批次，umap 结果，分别展示 样本来源、疾病在 UMAP 上的嵌入
adata_ctrl = anndata.AnnData(
    X=adata.raw.X.copy(),
    obs=adata.obs.copy(),
    var=adata.raw.var.copy()
)

adata_ctrl.uns = adata.uns.copy()
del adata;gc.collect()

sc.tl.pca(adata_ctrl)
sc.pp.neighbors(adata_ctrl)
sc.tl.umap(adata_ctrl)

umap_plot(save_addr='/public/home/xiongyuehan/data/IBD_analysis/output/Step07/Step07a_Summary',
          filename="UMAP_without_SCVI",
          adata=adata_ctrl, color=["disease", "orig.project"])
umap_plot(save_addr='/public/home/xiongyuehan/data/IBD_analysis/output/Step07/Step07a_Summary',
          filename="UMAP_without_SCVI(celltype)",
          adata=adata_ctrl, color=["Celltype", "Subset_Identity"])

adata_ctrl.write_h5ad("/public/home/xiongyuehan/data/IBD_analysis/output/Step07/Step07_DR_clustered(wo_SCVI).h5ad")

####################################
adata = anndata.read_h5ad(f"{save_addr}/Step07_DR_clustered_clean_20260210.h5ad")

adata_obs = ObsEditor(adata)

adata = adata[adata.obs["Celltype"].isin(['Epithelium', 'Fibroblast', 'Endothelium',
                                          'Myeloid Cell', 'T Cell', 'B Cell', 'Plasma Cell'])]

cells_to_keep = adata.obs["Celltype"].isin([
    'Epithelium', 'Fibroblast', 'Endothelium',
    'Myeloid Cell', 'T Cell', 'B Cell', 'Plasma Cell'
])
adata = adata[cells_to_keep, :].copy()  # 加 copy() 明确触发复制，避免隐式多次复制

adata = adata[adata.obs["Celltype"].isin(
    ['Epithelium', 'Fibroblast', 'Endothelium', 'Myeloid Cell', 'T Cell', 'B Cell', 'Plasma Cell', 'Mitotic Mix'])]

# -----------------------------
# 1. 计算邻接图
# -----------------------------
print("Computing neighbors based on X_scVI...")
sc.pp.neighbors(adata, use_rep="X_scVI", key_added="scvi")
print("Neighbors computed. .obsp['scvi_connectivities'] shape:", adata.obsp["scvi_connectivities"].shape)

# -----------------------------
# 2. 计算 UMAP
# -----------------------------
print("Computing UMAP...")
sc.tl.umap(adata, neighbors_key="scvi")
print("UMAP computed. .obsm['X_umap'] shape:", adata.obsm["X_umap"].shape)

# -----------------------------
# 3. 计算 t-SNE
# -----------------------------
print("Computing t-SNE...")
sc.tl.tsne(adata, use_rep="X_scVI")
print("t-SNE computed. .obsm['X_tsne'] shape:", adata.obsm["X_tsne"].shape)

# -----------------------------
# 4. 计算 Leiden 聚类（低分辨率示例）
# -----------------------------
print("Computing Leiden clustering at resolution 0.25...")
sc.tl.leiden(
    adata,
    neighbors_key="scvi",  # 使用 scVI 邻接图
    key_added="leiden_res0_25",
    resolution=0.25
)
print("Leiden clustering done. Unique clusters:", adata.obs["leiden_res0_25"].unique())

# -----------------------------
# 5. 保存结果
# -----------------------------
out_file = f"{save_addr}/Step07_DR_clustered_clean_20260210.h5ad"
print(f"Writing adata to '{out_file}'...")
adata.write_h5ad(out_file)
print("Done.")

####################################
# 最后一次 focus：拆分亚群并且绘制亚群的高表达基因信息
############################################################################################################################
adata = anndata.read_h5ad(f"{save_addr}/Step07_DR_clustered_clean_20260210.h5ad")
adata_obs = ObsEditor(adata)

from src.core.handlers.geneset_editor import *
my_markers = Geneset("/public/home/xiongyuehan/data/IBD_analysis/assets/Markers-updated_0109.xlsx")

my_markers.get(sheet_name="Myeloid Cell", facet_split=True)
my_markers.get(sheet_name="Epithelium", facet_split=True)

make_a_focus(adata_obs,
             filename="/public/home/xiongyuehan/data/IBD_analysis/assets/0211_final_focus.csv",
             cat_key="Celltype", type_key="Subset_Identity")

Focus = IdentifyFocus(focus_file="/public/home/xiongyuehan/data/IBD_analysis/assets/0211_final_focus.csv",
                      adata=adata_obs._adata)

focus_save_addr = f"{save_fig_addr}/celltype_focus"
os.makedirs(focus_save_addr, exist_ok=True)

Focus.filter_and_save_subsets(h5ad_prefix="02112218",  # 建议使用时间控制版本
                              save_addr=focus_save_addr,  # 取消预设值以避免储存在意外的地方
                              obs_key="Subset_Identity")
Focus.process_filtered_files(Geneset_class=my_markers,
                             save_addr=focus_save_addr,
                             h5ad_prefix="02112218",
                             do_subcluster=True,
                             resolutions_list=[0.5],
                             use_raw=False)

with pd.option_context('display.max_rows', 100):
    adata_obs._adata.obs["Subset_Identity"].value_counts()

################################################
# 简单检查
sc.tl.rank_genes_groups(
    adata_obs,
    groupby="Subset_Identity",
    groups=["CAAP epithelium HLA-DR+"],  # 只算 A
    reference="Early absorptive progenitor",  # 相对 B
    method="wilcoxon",  # 单细胞常用
    pts=True  # 记录表达比例
)
# deg = sc.get.rank_genes_groups_df(
#     adata_obs,
#     group="CAAP epithelium HLA-DR+"
# )
#
# deg.to_csv(f"{save_fig_addr}/CAAP.csv")
################################################
# 样本性别标注
Y_genes = ["RPS4Y1", "DDX3Y", "KDM5D", "UTY", "ZFY", "EIF1AY"]
# 只保留在 adata.var 中存在的基因
Y_genes_present = [g for g in Y_genes if g in adata.raw.var_names]

# 2️⃣ 取出 Y 基因表达矩阵
# 如果 adata.X 是 sparse，需要转成 dense
X_Y = adata.raw[:, Y_genes_present].X
if hasattr(X_Y, "toarray"):  # sparse -> dense
    X_Y = X_Y.toarray()

# 3️⃣ 每个细胞 Y 基因总表达
Y_sum_per_cell = X_Y.sum(axis=1)

# 4️⃣ 添加到 adata.obs
adata.obs["Y_sum"] = Y_sum_per_cell

# 5️⃣ 按 Patient 聚合
patient_Y = adata.obs.groupby("Patient")["Y_sum"].median()  # 用中位数更稳健

# 6️⃣ 根据阈值判定性别（阈值可调）
# 0.1 是 log1p 或 counts 的经验值，可根据数据调
patient_sex = patient_Y.apply(lambda x: "Male" if x > 0.1 else "Female")

# 7️⃣ 查看结果
patient_sex

# 写入 adata 对象
adata.obs["Gender"] = adata.obs["Patient"].map(patient_sex.to_dict())

adata.obs.groupby("disease")["Patient"].nunique()

adata.obs[adata.obs['tissue-origin'] == 'blood'].groupby("disease")["Patient"].nunique()

print(adata.obs["Gender"].value_counts())
adata.write_h5ad(f"{save_addr}/Step07_DR_clustered_clean_20260210.h5ad")