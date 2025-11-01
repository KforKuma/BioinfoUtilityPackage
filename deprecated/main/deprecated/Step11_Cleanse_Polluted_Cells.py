# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 备注：本项目主要将之前按细胞subset分成小群的adata亚群、分析并初步定义出污染的亚群后重新整合，
# 并进行笼统地重新分群。
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##——————————————————————————————————————————————————————————————————————————
import gc
import os
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
##——————————————————————————————————————————————————————————————————————————
# 设置scanpy基本属性
import yaml
with open('/data/HeLab/bio/IBD_analysis/assets/io_config.yaml', 'r') as f:
    config = yaml.safe_load(f) # 读取的格式为一系列字典的列表
##——————————————————————————————————————————————————————————————————————————
os.chdir("/data/HeLab/bio/IBD_analysis/")
from src.ScanpyTools.ScanpyTools import ScanpyPlotWrapper
rank_genes_groups_dotplot = ScanpyPlotWrapper(func = sc.pl.rank_genes_groups_dotplot)
dotplot = ScanpyPlotWrapper(func = sc.pl.dotplot)
umap_plot = ScanpyPlotWrapper(func = sc.pl.umap)
##——————————————————————————————————————————————————————————————————————————
sc.set_figure_params(dpi_save=450, color_map='viridis_r', fontsize=6)
sc.settings.verbosity = 1
sc.logging.print_header()
##——————————————————————————————————————————————————————————————————————————
# 输入文件
adata = anndata.read(config['Step9']['general_output_anndata'])
adata.obs["filtered_celltype_annotation"] = pd.Series(adata.obs["manual_celltype_annotation"].tolist(), )  # 创建一个新列
##——————————————————————————————————————————————————————————————————————————
# 循环读取每一个细胞亚群鉴定结果，并在原anndata文件中标记Doublet/Abberant类型细胞
Subset_list = ["T", "BP", "Epi", "Fb", "M"]
for subset in Subset_list:
    print("Loop working on subset " + subset)
    adata_subset = anndata.read(config['Step10']['original_workdir'] + '10_adata_' + subset + '_Checked.h5ad')
    Polluted_Index = adata_subset.obs_names[(adata_subset.obs["manual_cellsubtype_annotation"] == "Doublet") |
                                            (adata_subset.obs["manual_cellsubtype_annotation"] == "Abberant")]
    adata.obs.loc[Polluted_Index, "filtered_celltype_annotation"] = "Polluted"
    print("Polluted cell counts: " + str(np.sum(adata.obs["filtered_celltype_annotation"] == "Polluted")))
    del Polluted, adata_subset;
    gc.collect()
# 备份，清除并保存
adata.obs["filtered_celltype_annotation"] = adata.obs["filtered_celltype_annotation"].astype('category')
adata.write_h5ad('11_adata_marked.h5ad')  # 683633
adata = adata[(adata.obs['filtered_celltype_annotation'] != "Doublet")]
adata.write_h5ad('11_adata_pure.h5ad')  # 656898
##——————————————————————————————————————————————————————————————————————————
# 重新降维，并分大群
adata = anndata.read('11_adata_pure.h5ad')
adata = subcluster(adata, n_neighbors=20, n_pcs=30, resolutions=[1.0], use_rep="X_pca_harmony")
adata.write_h5ad('11_adata_scran_leidens_npcs.h5ad')
gc.collect()
##——————————————————————————————————————————————————————————————————————————
# 进行细胞类群鉴定和质量管理
adata = anndata.read('11_adata_scran_leidens_npcs.h5ad')
from src.EasyInterface.QualityControl import Basic_QC_Plot
Basic_QC_Plot(adata, prefixx = "Step11_Combined_data", out_dir = config['Step11']['current_workdir']+"fig/")

from src.ScanpyTools.ScanpyTools import Geneset
from src.ScanpyTools.ScanpyTools import AnnotationMaker
# 绘制主要marker点图
my_markers = Geneset(config['Env']['assets']+"Markers.xlsx")
for sheet_name in my_markers.data.keys():
    dotplot(save_addr=config['Step11']['current_workdir']+"fig/", filename = sheet_name,
        adata = adata, groupby="leiden_res1.0", layer="log1p_norm", standard_scale="var",
        var_names = my_markers.get_gene_dict(marker_sheet=sheet_name, celltype_list=""))

# 检查DEG
sc.tl.rank_genes_groups(adata, groupby="leiden_res1.0", use_raw=False,
                        method="wilcoxon", key_added="dea_leiden")
rank_genes_groups_dotplot(save_addr = config['Step11']['current_workdir']+"fig/", filename = "leiden_res1.0",
                          adata = adata, groupby = "manual_cellsubtype_annotation",
                          # standard_scale = "var", # 可选
                          cmap='bwr', # 可选
                          n_genes=5, key="dea_leiden",use_raw=False)
df = sc.get.rank_genes_groups_df(adata = adata, key = "dea_leiden")
df.to_csv(config['Step11']['current_workdir']+"files/"+"leiden_res1.0_Rank_DEG.csv")

# 绘制主要类群UMAP
umap_plot(save_addr="/data/HeLab/bio/IBD_plus/11_Huiguo", filename = "example_gene_expression",
          adata = adata,color=['IGHA1','IGKC','IGHM','CD19','MS4A1','CD28','CD69'],
          layer="log1p_norm")
##——————————————————————————————————————————————————————————————————————————
# 综合上述产出之后，进行操作
ann_list = [] # 待填写
make_anno = AnnotationMaker(adata,"leiden_res1.0","manual_cellsubtype_annotation")
make_anno.annotate_by_list(ann_list)
make_anno.make_annotate()

umap_plot(save_addr = config['Step11']['current_workdir']+"fig/", filename = "manual_annotated",
          adata = make_anno.data, color=["manual_cellsubtype_annotation"],legend_loc="on data")
# 检查DEG
sc.tl.rank_genes_groups(make_anno.data, groupby = "manual_cellsubtype_annotation", use_raw=False,
                        method="wilcoxon", key_added="dea_mca"
)
rank_genes_groups_dotplot(save_addr = config['Step11']['current_workdir']+"fig/", filename = "manual_annotated",
                          adata = make_anno.data, groupby = "manual_cellsubtype_annotation",
                          # standard_scale = "var", # 可选
                          cmap='bwr', # 可选
                          n_genes=5, key="dea_mca",use_raw=False)
df = sc.get.rank_genes_groups_df(make_anno.data,key="dea_mca")
df.to_csv(config['Step11']['current_workdir']+"files/"+ "manual_annotated_Rank_DEG.csv")
##——————————————————————————————————————————————————————————————————————————
make_anno.data.write_h5ad('11_adata_1st_npcs_30_leiden_res2.h5ad')






