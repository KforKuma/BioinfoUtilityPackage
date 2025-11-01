"""
Step07b_Characterization.py
Author: John Hsiung
Update date: 2025-08-29
Description:
    - 在亚群的身份鉴定结束后，检查亚群的基本特征基因，
        绘制足以说明亚群身份和大致功能的图
Notes:
    - 依赖环境: conda activate scvpy10
"""

####################################
import igraph  # avoid ImportError: dlopen
import gc, os
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
####################################
import matplotlib
matplotlib.use('Agg')

sc.settings.verbosity = 0
sc.settings.set_figure_params(
    dpi=80,
    facecolor="white",
    frameon=False,
)
#################
import sys
sys.path.append("/data/HeLab/bio/IBD_analysis/")
from src.ScanpyTools.Scanpy_Plot import ScanpyPlotWrapper
from src.ScanpyTools.ScanpyTools import easy_DEG
umap_plot = ScanpyPlotWrapper(func = sc.pl.umap)
dot_plot = ScanpyPlotWrapper(func = sc.pl.dotplot)

####################################
# 数据读取
####################################
os.chdir("/data/HeLab/bio/IBD_analysis/output/Step07")
adata = anndata.read_h5ad("Step07_finalversion_4.h5ad")

####################################
# 对每个细胞亚群取子集，并进行子集内部的 subset_identity 为精度的 DEG
# 确保这个 DEG 能读出生物学意义
####################################
output_dir = "/data/HeLab/bio/IBD_analysis/output/Step07/Step07b_Characeterization"
cell_main_type_list = adata.obs["Celltype"].unique().tolist()
# cell_main_type_list = cell_main_type_list[1:]
for cell_main_type in cell_main_type_list[4:]:
    print(f"\n Processing: {cell_main_type}")
    
    # 子集提取
    adata_subset = adata[adata.obs["Celltype"] == cell_main_type]
    print(f"Subset contains {adata_subset.n_obs} cells")
    
    # 创建保存目录
    save_dir = f"{output_dir}/{cell_main_type}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saved to: {save_dir}")
    
    # 差异表达分析
    print("Processing easy_DEG() ...")
    easy_DEG(
        adata_subset,
        save_addr=save_dir,
        filename="Subset_DEG",
        obs_key="Subset_Identity",
        save_plot=True,
        plot_gene_num=5,
        downsample=2000,
        use_raw=True
    )
    print("easy_DEG completed.")
    
    # UMAP分析
    print("Starting computing NN and UMAP...")
    sc.pp.neighbors(adata_subset, n_neighbors=20, n_pcs=min(50, adata_subset.obsm["X_scVI"].shape[1]), use_rep="X_scVI")
    sc.tl.umap(adata_subset)
    print("UMAP completed.")
    
    print("Starting UMAP_of_Celltype")
    umap_plot(
        save_addr=save_dir,
        filename="UMAP_of_Celltype",
        adata=adata_subset,
        color="Subset_Identity",
        legend_loc="right margin"
    )
    print("UMAP saved.")
    
    # 保存结果
    adata_subset.write_h5ad(f"{save_dir}/Subset_DEG.h5ad")
    print(f"Data is saved in: {save_dir}/Subset_DEG.h5ad")
    
####################################
# 对每个感兴趣的亚群在做关键基因的点图：B 细胞
####################################
cell_main_type="B Cell"
# adata_subset = adata[adata.obs["Subset_Identity"].isin(["B Cell_B.mem.naive","B Cell_B.mem.activated"])]
adata_subset = adata[adata.obs["Celltype"] == cell_main_type]
save_dir = f"{output_dir}/{cell_main_type}"
easy_DEG(
    adata_subset,
    save_addr=save_dir,
    filename="Subset_DEG_Bmem",
    obs_key="Subset_Identity",
    save_plot=True,
    plot_gene_num=15,
    downsample=3000,
    use_raw=True
)
Bmem_marker = {"B_activated":["TNFRSF13B","CLECL1","CD27","AIM2","PVT1","SAMSN1",
                              "SNED1","KLK1","SSPN","IGHA1","LGALS1","TMEM163","ATXN1"],
               "B_naive":["TCL1A","CD79A","CD52","CD72","CD37","CXCR4"]}
dot_plot(save_addr=f"{output_dir}/{cell_main_type}",
             filename=f"Bmem_marker",
             adata=adata_subset,
             groupby="Subset_Identity",
             # standard_scale="var",
             var_names=Bmem_marker,  # 注意这里传的是 dict
             use_raw=True)

del adata_subset;gc.collect()

####################################
# 对每个感兴趣的亚群在做关键基因的点图：T 细胞
####################################
cell_main_type = "T Cell"
adata_subset = anndata.read_h5ad(f"{output_dir}/{cell_main_type}/Subset_DEG.h5ad")
adata_subset.obs["Subset_Identity"].unique().tolist()
subtype_order = ['T Cell_DN',
                 'T Cell_CD4.naive','T Cell_CD4.early', 'T Cell_CD4.mem',
                 'T Cell_CD4.Treg','T Cell_CD4.Tfh', 'T Cell_CD4.Th1',
                 'T Cell_CD4.Th17','T Cell_MAIT', 'T Cell_CD4.Tr1',
                 'T Cell_CD8.naive', 'T Cell_CD8.mem', 'T Cell_CD8.mem.GZMK','T Cell_CD8.Trm', 'T Cell_CD8.Trm.KLRC2+',
                 'T Cell_CD8.NKT','T Cell_NK.CD16+', 'T Cell_NK.CD56+',
                 'T Cell_gdT.naive','T Cell_gdT.Trm.XCL1+', 'T Cell_gdT.Trm.GZMA+', 'T Cell_gdT.g9d2',
                 'T Cell_ILC.XCL1+', 'T Cell_ILC1', 'T Cell_ILC3',]
adata_subset.obs["Subset_Identity"] = pd.Categorical(
    adata_subset.obs["Subset_Identity"],
    categories=subtype_order,
    ordered=True
)

# T 细胞主要问题是内部差异还是非常大的，还是得分门别类进行区分和绘制差异基因
# 才比较有生物学意义，因此手动做个 dict；
# 至于绘图用的关键基因，用的还是 markers_core-updated.xlsx 里的标准基因，没有做特别细
excel_data = pd.ExcelFile("/data/HeLab/bio/IBD_analysis/assets/Markers_core-updated.xlsx")
assignment_sheet = excel_data.parse(excel_data.sheet_names[2])
assignment_dict = {
    row["cell_type"]: [gene.strip() for gene in row["gene_set"].split(",")]
    for _, row in assignment_sheet.iterrows()
}

draw_dict = {"NK-like":['T Cell_CD8.NKT','T Cell_NK.CD16+', 'T Cell_NK.CD56+'],
             "Innate":['T Cell_gdT.naive','T Cell_gdT.Trm.XCL1+', 'T Cell_gdT.Trm.GZMA+', 'T Cell_gdT.g9d2',
                       'T Cell_ILC.XCL1+', 'T Cell_ILC1', 'T Cell_ILC3'],
             "pan-CD4":['T Cell_CD4.naive','T Cell_CD4.early', 'T Cell_CD4.mem',
                 'T Cell_CD4.Treg','T Cell_CD4.Tfh', 'T Cell_CD4.Th1',
                 'T Cell_CD4.Th17','T Cell_MAIT', 'T Cell_CD4.Tr1'],
             "CD8":['T Cell_CD8.naive', 'T Cell_CD8.mem', 'T Cell_CD8.mem.GZMK','T Cell_CD8.Trm', 'T Cell_CD8.Trm.KLRC2+']}
             
for k,v in draw_dict.items():
    tmp = adata_subset[adata_subset.obs["Subset_Identity"].isin(v)]
    dot_plot(save_addr=f"{output_dir}/{cell_main_type}",
             filename=f"{k}_marker",
             adata=tmp,
             groupby="Subset_Identity",
             standard_scale="var",
             var_names=assignment_dict,  # 注意这里传的是 dict
             use_raw=True)
    

####################################
# 对每个感兴趣的亚群在做关键基因的点图：Fb 细胞
####################################
assignment_sheet = excel_data.parse(excel_data.sheet_names[3])
assignment_dict = {
    row["cell_type"]: [gene.strip() for gene in row["gene_set"].split(",")]
    for _, row in assignment_sheet.iterrows()
}

adata_subset = adata[adata.obs["Celltype"]=="Fibroblast"]
cell_main_type="Fibroblast"
dot_plot(save_addr=f"{output_dir}/{cell_main_type}",
             filename=f"Fb_marker",
             adata=adata_subset,
             groupby="Subset_Identity",
             standard_scale="var",
             var_names=assignment_dict,  # 注意这里传的是 dict
             use_raw=True)

####################################
# 对每个感兴趣的亚群在做关键基因的点图：表达前列环素的细胞
####################################
assignment_sheet = excel_data.parse(excel_data.sheet_names[4])
assignment_dict = {
    row["cell_type"]: [gene.strip() for gene in row["gene_set"].split(",")]
    for _, row in assignment_sheet.iterrows()
}
valid_genes = set(adata.raw.var_names)  # 如果你用的是原始表达矩阵
filtered_dict = {
    k: [gene for gene in v if gene in valid_genes]
    for k, v in assignment_dict.items()
}
# 还可以移除空列表的项（可选）
filtered_dict = {k: v for k, v in filtered_dict.items() if v}

dot_plot(save_addr=f"{output_dir}",
             filename=f"Prost_marker",
             adata=adata,
             groupby="Subset_Identity",
             standard_scale="var",
             var_names=filtered_dict,  # 注意这里传的是 dict
             use_raw=True)

####################################
# 对每个感兴趣的亚群在做关键基因的点图：髓系细胞
####################################
assignment_sheet = excel_data.parse(excel_data.sheet_names[1])
assignment_dict = {
    row["cell_type"]: [gene.strip() for gene in row["gene_set"].split(",")]
    for _, row in assignment_sheet.iterrows()
}
valid_genes = set(adata.raw.var_names)  # 如果你用的是原始表达矩阵
filtered_dict = {
    k: [gene for gene in v if gene in valid_genes]
    for k, v in assignment_dict.items()
}
# 还可以移除空列表的项（可选）
filtered_dict = {k: v for k, v in filtered_dict.items() if v}
adata_subset = adata[adata.obs["Celltype"]=="Myeloid"]

dot_plot(save_addr=f"{output_dir}",
             filename=f"Myeloid",
             adata=adata_subset,
             groupby="Subset_Identity",
             standard_scale="var",
             var_names=filtered_dict,  # 注意这里传的是 dict
             use_raw=True)

