# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 备注：本项目主要将Step12a拆分的小群进行研究，并存在反复研究之可能。
# 运行环境: scvpy10
# 主要输入：拆分的小群，格式如：/data/HeLab/bio/IBD_analysis/tmp/12_adata_En_final.h5ad, .../12_adata_T_final
# 主要输出：
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
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
from src.ScanpyTools.ScanpyTools import ScanpyPlotWrapper, AnnotationMaker, Geneset, easy_DEG, subcluster, \
    filter_and_save_adata, sanitize_filename
from src.EasyInterface.QualityControl import Basic_QC_Plot


rank_genes_groups_dotplot = ScanpyPlotWrapper(func = sc.pl.rank_genes_groups_dotplot)
dotplot = ScanpyPlotWrapper(func = sc.pl.dotplot)
umap_plot = ScanpyPlotWrapper(func = sc.pl.umap)
tsne_plot = ScanpyPlotWrapper(func = sc.pl.tsne)
heatmap_plot = ScanpyPlotWrapper(func = sc.pl.heatmap)


from src.EasyInterface.Scanpy_statistics import plot_cluster_counts, plot_cluster_proportions, plot_piechart

cluster_plot = ScanpyPlotWrapper(func = plot_cluster_counts)
proportion_plot = ScanpyPlotWrapper(func = plot_cluster_proportions)
pie_plot = ScanpyPlotWrapper(func = plot_piechart)
##——————————————————————————————————————————————————————————————————————————
sc.set_figure_params(dpi_save=450, color_map = 'viridis_r',fontsize=6)
sc.settings.verbosity = 1
sc.logging.print_header()

##——————————————————————————————————————————————————————————————————————————
# 1) 数据处理

# 1.1 标准化降维，不需要聚类
adata = anndata.read_h5ad("/data/HeLab/bio/IBD_analysis/tmp/Step12_cleansed(updated0101_clustered).h5ad")
obs_data = pd.read_pickle("/data/HeLab/bio/IBD_analysis/tmp/Step12_cleansed(updated0101_clustered)_obs.pkl")

obs_data
del adata.obs
adata.obs = obs_data

# 1.2 读取注释
adata.obs["Subset_Identity"].value_counts().to_csv("/data/HeLab/bio/IBD_analysis/tmp/0101_value_counts.csv",
                                                   index=True,header=True)
# 加载 Excel 文件
cell_identity = pd.ExcelFile("/data/HeLab/bio/IBD_analysis/assets/cell_identity_assignment_0101.xlsx")
identity_remap = cell_identity.parse(cell_identity.sheet_names[0])

# 获取重排序的列表
reorder_list = identity_remap['Subset_Identity'].astype(str).tolist()

# 遍历除 'Subset_Identity' 外的所有列
for col in identity_remap.columns[1:]:  # 从第二列开始遍历
    # 创建从 'Subset_Identity' 到当前列的映射字典
    identity_type_dict = identity_remap.set_index('Subset_Identity')[col].to_dict()
    
    # 使用 map 函数将 Subset_Identity 映射到当前列
    adata.obs[col] = adata.obs.Subset_Identity.map(identity_type_dict)
    
    # 确保该列为 'category' 类型才能使用 .cat 访问器
    adata.obs[col] = adata.obs[col].astype('category')
    
    # 如果要重新排序分类数据，可以传入新的类别顺序，而不是清空类别
    adata.obs[col] = adata.obs[col].cat.set_categories(sorted(adata.obs[col].cat.categories))

# 2）绘图
output_dir = "/data/HeLab/bio/IBD_analysis/output/Step12c/"
umap_plot(save_addr=output_dir,filename="Celltype_Identity(0101)",
          adata=adata,color=["Cell_type"],legend_loc="on data")
umap_plot(save_addr=output_dir,filename="Subset_Identity(0101)",
          adata=adata,color=["Subset_Identity"],legend_loc="on data")
umap_plot(save_addr=output_dir,filename="Celltype_Identity(0101)_rightlegend",
          adata=adata,color=["Cell_type"],legend_loc="right margin")
umap_plot(save_addr=output_dir,filename="Subset_Identity(0101)_rightlegend",
          adata=adata,color=["Subset_Identity"],legend_loc="right margin")
#

# 3) DEG分析
from src.ScanpyTools.ScanpyTools import analysis_DEG
analysis_DEG(adata, file_name="AllCell(0101)", groupby_key="Cell_type", output_dir=output_dir, obs_subset=4000)
my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")

marker_genes_Celltype = my_markers.data["marker_genes_Celltype"]
heatmap_plot(save_addr=output_dir,filename="Cell_type_marker",
             adata = adata,groupby ="Cell_type",var_names=marker_genes_Celltype,
             swap_axes=True,vmax=2.0,
             show_gene_labels=True,layer="log1p_norm",show=False)

# bdata = adata[adata.obs["Cell_type"].isin(["Epithelial stem cell","Epithelium"])]
# analysis_DEG(bdata, file_name="Epi_vs_Esc(0101)", groupby_key="Cell_type", output_dir=output_dir, obs_subset=4000)
bdata = adata[adata.obs["Cell_type"].isin(["Fibroblast","Smooth Muscle"])]
analysis_DEG(bdata, file_name="SM_vs_FB(0101)", groupby_key="Cell_type", output_dir=output_dir, obs_subset=4000)

#
# 4) 数据统计
from src.EasyInterface.Scanpy_statistics import get_cluster_counts, get_cluster_proportions

adata.obs['Cell_type'].unique().tolist()

output_dir="/data/HeLab/bio/IBD_analysis/output/Step13_CPDB/output/"
adata_Immune = adata[(adata.obs['Cell_type']=="T_cell")|
                     (adata.obs['Cell_type']=="B_cell")|
                     (adata.obs['Cell_type']=="Innate_immunocyte")|
                     (adata.obs['Cell_type']=="Dendritic_cell")|
                     (adata.obs['Cell_type']=="Myeloid cell")|
                     (adata.obs['Cell_type']=="Plasma cell")]

cell_prop = get_cluster_proportions(adata=adata_Immune,cluster_key="Cell_type",sample_key="disease")
proportion_plot(save_addr=output_dir,filename="Cell_type_prop",
                cluster_props=cell_prop)

cell_count = get_cluster_counts(adata_Immune,cluster_key="Cell_type",sample_key="disease")
cluster_plot(save_addr=output_dir,filename="Cell_type_counts",
             cluster_counts=cell_count)

for i in adata.obs['Cell_type'].unique().tolist():
    print(i)
    adata_subset = adata[(adata.obs['Cell_type'] == i)]
 
    save_dir = f"{output_dir}{i}_"
 
    cell_count = get_cluster_counts(adata=adata_subset,cluster_key="Subset_Identity",sample_key="disease")
    cluster_plot(save_addr=save_dir,filename="Cell_type_counts",
                 cluster_counts=cell_count)
    
    cell_prop = get_cluster_proportions(adata=adata_subset, cluster_key="Subset_Identity", sample_key="disease")
    proportion_plot(save_addr=save_dir, filename="Cell_type_prop",
                    cluster_props=cell_prop)

# piechart
for subset in adata_Immune.obs["Subset_Identity"].unique():
    save_dir = f"{output_dir}{subset}_"
    adata_tmp = adata_Immune[adata_Immune.obs["Subset_Identity"] == subset]
    general_count = adata_Immune.obs['disease'].value_counts().sort_index()
    subset_count = adata_tmp.obs['disease'].value_counts().sort_index()
    pie_plot(save_addr=save_dir, filename="prop_pierchart",subset_count=subset_count,general_count=general_count)


