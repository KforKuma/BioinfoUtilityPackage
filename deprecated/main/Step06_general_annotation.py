"""
Step06_general_annotation.py
Author: John Hsiung
Update date: 2025-08-29
Description:
    - 进行初次细胞身份鉴定，和迭代后二次细胞身份鉴定
Notes:
    - 依赖环境: conda activate scvpy10
"""

####################################
import igraph  # avoid ImportError: dlopen
import gc
import os
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
import sys
sys.path.append("/data/HeLab/bio/IBD_analysis/")
from src.ScanpyTools.AnnotationMaker import AnnotationMaker
from src.ScanpyTools.ScanpyTools import subcluster, easy_DEG
from src.EasyInterface.Anndata_Annotator import generate_subclusters_by_identity, run_deg_on_subsets, apply_assignment_annotations
####################################
# 读取数据，DEG 在标准流程中完成了一次
# 因此直接手写 assignment.xlsx，进行第一次细胞定义工作流；输入过程在屏幕上进行
####################################
os.chdir("/data/HeLab/bio/IBD_analysis/output/Step06")
adata = anndata.read_h5ad("/data/HeLab/bio/IBD_analysis/output/Step05/Step05_DR_clustered_DEG.h5ad")


excel_data = pd.ExcelFile("/data/HeLab/bio/IBD_analysis/assets/0527_assignment.xlsx")
assignment_sheet = excel_data.parse(excel_data.sheet_names[0]) # 这个读出来就是pandas.dataframe，我们简单拆一列就行
assignment_sheet["Identity"] = assignment_sheet["Identity"].str.split('_', n=1, expand=True)[0]

make_anno = AnnotationMaker(adata,
                            obs_key = "leiden_res0.5",
                            anno_key = "Primary_Cluster")
make_anno.annotate_by_list(assignment_sheet["Identity"])
make_anno.make_annotate()
##################################################
# 第二次第一步：将文件拆分、重新聚类、进行 DEG
##################################################
cell_list = adata.obs["Primary_Cluster"].unique().tolist()
generate_subclusters_by_identity(
    adata=adata,
    identity_key="Subset_Identity",
    identities=cell_list,  # 或 None 表示处理全部
    resolutions=[0.5, 1.0],
    output_dir="/data/HeLab/bio/IBD_analysis/1.0",
    subcluster_func=subcluster,  # ← 你已有的函数
    use_rep="X_scVI"
)
# 这个时候我们突然发现在 Step04a 第 68 行采用的 ScviTools.processdata 取了 top1000 HVG
# 仔细检查发现adata.raw中还是有21920个基因的

run_deg_on_subsets(
    adata=adata,
    cell_idents_list=adata.obs["Primary_Cluster"].unique(),
    resolutions=[0.5, 1.0],
    base_input_path="/data/HeLab/bio/IBD_analysis/output/Step06/1.0",
    base_output_path="/data/HeLab/bio/IBD_analysis/output/Step06/1.0/DEG",
    easy_deg_func=easy_DEG
)
##################################################
# 第二次第二步：查看后手写 assignment.xlsx，并应用到文件中
##################################################
apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0528_subset_assignment.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step06/1.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)

Subset_statics = adata.obs["Subset_Identity"].value_counts()
large_groups = Subset_statics[Subset_statics > 13000]
large_groups = large_groups.index.tolist()

##################################################
# 第三次第一步：将文件拆分、重新聚类、进行 DEG
##################################################
generate_subclusters_by_identity(
    adata=adata,
    identity_key="Subset_Identity",
    identities=large_groups,  # 或 None 表示处理全部
    resolutions=[0.5, 1.0],
    output_dir="/data/HeLab/bio/IBD_analysis/2.0",
    subcluster_func=subcluster,  # ← 你已有的函数
    use_rep="X_scVI"
)

cell_list = ['B Cell', 'T Cell_CD4', 'T Cell_CD8', 'T Cell_IL17A', 'T Cell_gdT',
             'Plasma_IgA', 'Myeloid_Monocyte', 'B Cell_GC', 'Epithelium_Tuft']

run_deg_on_subsets(
    cell_idents_list=cell_list,
    resolutions=[0.5, 1.0],
    base_input_path="/data/HeLab/bio/IBD_analysis/output/Step06/2.0",
    base_output_path="/data/HeLab/bio/IBD_analysis/output/Step06/2.0/DEG",
    obs_subset = 10000,
    easy_deg_func=easy_DEG
)

##################################################
# 第三次第二步：查看后手写 assignment.xlsx，并应用到文件中
##################################################
apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0529_subset_assignment.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step06/2.0",
    output_key="Subset_Identity",
    # fillna_from_col="Primary_Cluster"
)
Subset_statics = adata.obs["Subset_Identity"].value_counts()
large_groups = Subset_statics[Subset_statics > 14000]
large_groups = large_groups.index.tolist()

##################################################
# 第四次第一步：将文件拆分、重新聚类、进行 DEG
##################################################
generate_subclusters_by_identity(
    adata=adata,
    identity_key="Subset_Identity",
    cell_idents_list=large_groups,  # 或 None 表示处理全部
    resolutions=[1.0],
    output_dir="/data/HeLab/bio/IBD_analysis/output/Step06/3.0",
    subcluster_func=subcluster,  # ← 你已有的函数
    use_rep="X_scVI"
)
run_deg_on_subsets(
    cell_idents_list=large_groups,
    resolutions=[1.0],
    base_input_path="/data/HeLab/bio/IBD_analysis/output/Step06/3.0",
    base_output_path="/data/HeLab/bio/IBD_analysis/output/Step06/3.0/DEG",
    obs_subset = 10000,
    easy_deg_func=easy_DEG
)
# 阶段性保存总文件
adata.write("/data/HeLab/bio/IBD_analysis/output/Step06/Step06_2nd.h5ad")
adata = anndata.read_h5ad("/data/HeLab/bio/IBD_analysis/output/Step06/Step06_2nd.h5ad")

##################################################
# 第四次第二步：查看后手写 assignment.xlsx，并应用到文件中
##################################################
apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0530_subset_assignment.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step06/3.0",
    output_key="Subset_Identity",
    # fillna_from_col="Primary_Cluster"
)
adata.write("/data/HeLab/bio/IBD_analysis/output/Step06/Step06_3rd.h5ad")

# 查看目前的鉴定结果
Subset_statics = adata.obs["Subset_Identity"].value_counts()
print(Subset_statics)




