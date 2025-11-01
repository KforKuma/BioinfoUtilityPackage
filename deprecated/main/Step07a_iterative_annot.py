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
import igraph  # avoid ImportError: dlopen
import gc
import os
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
####################################
import sys
sys.path.append("/data/HeLab/bio/IBD_analysis/")
from src.ScanpyTools.Scanpy_Plot import ScanpyPlotWrapper
umap_plot = ScanpyPlotWrapper(func = sc.pl.umap)
####################################
import yaml
with open('/data/HeLab/bio/IBD_analysis/assets/io_config.yaml', 'r') as f:
    config = yaml.safe_load(f) # 读取的格式为一系列字典的列表
####################################
os.chdir("/data/HeLab/bio/IBD_analysis/output/Step07")

from src.ScanpyTools.ScanpyTools import process_adata
from src.ScanpyTools.Focus import Focus
from src.ScanpyTools.Geneset import Geneset
from src.EasyInterface.Anndata_Annotator import change_one_ident_fast, apply_assignment_annotations
###################################################################################################################
# 第一次修改
###################################################################################################################
adata = anndata.read_h5ad("/data/HeLab/bio/IBD_analysis/output/Step06/Step06_3rd.h5ad")
excel_data = pd.ExcelFile("/data/HeLab/bio/IBD_analysis/assets/0530_subset_focus.xlsx")

focus_sheet = focus_prepare(excel_data)
focus_sheet = focus_sheet.sort_values(by="Name",ascending=False)

my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")

filter_and_save_subsets(adata, focus_sheet,prefix="Step07",
                        output_dir="/data/HeLab/bio/IBD_analysis/output/Step07")

process_filtered_files(focus_sheet.loc[5:,], Geneset_class=my_markers,
                       storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07",
                       prefix="Step07",
                       process_adata_func=process_adata,
                       resolutions_list=[1.0, 1.5],
                       use_raw=True)
apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0604_subset_assignment.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step07/1.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)


change_one_ident(adata,"Subset_Identity","T Cell_CD4_TFH","T Cell_CD4_Tfh")
change_one_ident(adata,"Subset_Identity","B_Cell","B Cell")

value_list = adata.obs["Subset_Identity"].value_counts()
value_list = value_list.sort_index()
with pd.option_context('display.max_rows', None):
    print(value_list)

adata.write_h5ad("Step07_version1.h5ad")
adata = anndata.read_h5ad("Step07_version1.h5ad")
###################################################################################################################
# 第二次修改
###################################################################################################################

my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")


AdataFocus = Focus(focus_file="/data/HeLab/bio/IBD_analysis/assets/0604_subset_focus.xlsx",
                   adata=adata)
AdataFocus.filter_and_save_subsets(h5ad_prefix="Step07",
                                output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/2.0",
                                obs_key="Subset_Identity")
AdataFocus.process_filtered_files(Geneset_class=my_markers,
                               storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07/2.0",
                               h5ad_prefix="Step07",
                               process_adata_func=process_adata,
                               resolutions_list=[0.5,1.0],
                               use_rep="X_scVI",use_raw=True)
apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0605_subset_assignment.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step07/2.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)
change_one_ident_fast(adata,"Subset_Identity","Myeloid_ Macrophage","Myeloid_Macrophage")

value_list = adata.obs["Subset_Identity"].value_counts()
value_list = value_list.sort_index()
with pd.option_context('display.max_rows', None):
    print(value_list)

adata.write_h5ad("Step07_version2.h5ad")

###################################################################################################################
# 第三次修改
###################################################################################################################
adata = anndata.read_h5ad("Step07_version2.h5ad")

excel_data = pd.ExcelFile("/data/HeLab/bio/IBD_analysis/assets/0605_subset_focus.xlsx")
focus_sheet = focus_prepare(excel_data)
focus_sheet = focus_sheet.sort_values(by="Name",ascending=False)

my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")

filter_and_save_subsets(adata, focus_sheet,prefix="Step07",
                        output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/3.0")

process_filtered_files(focus_sheet, Geneset_class=my_markers,
                       storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07/3.0",
                       prefix="Step07",
                       process_adata_func=process_adata,
                       resolutions_list=[0.5,1.0],
                       use_raw=True)

apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0605_subset_assignment2.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step07/3.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)

# 我们突然注意到没有做细胞周期预测
G1S_genes_Tirosh = ['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6', 'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP', 'HELLS', 'RFC2', 'RPA2', 'NASP', 'RAD51AP1', 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', 'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8']
G2M_genes_Tirosh = ['HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2', 'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'FAM64A', 'SMC4', 'CCNB2', 'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP', 'CDCA3', 'HN1', 'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', 'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA']
s_genes = [x for x in G1S_genes_Tirosh if x in adata.raw.var_names]
g2m_genes = [x for x in G2M_genes_Tirosh if x in adata.raw.var_names]

sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
adata.write_h5ad("Step07_version3.h5ad")

###################################################################################################################
# 第四次修改
###################################################################################################################
adata = anndata.read_h5ad("Step07_version3.h5ad")

value_list = adata.obs["Subset_Identity"].value_counts()
value_list = value_list.sort_index()
with pd.option_context('display.max_rows', None):
    print(value_list)

my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")

version4 = Focus(focus_file="/data/HeLab/bio/IBD_analysis/assets/0606_subset_focus.xlsx",
                 adata=adata)
version4.filter_and_save_subsets(h5ad_prefix="Step07",
                                output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/4.0",
                                obs_key="Subset_Identity")
version4.process_filtered_files(Geneset_class=my_markers,
                               storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07/4.0",
                               h5ad_prefix="Step07",
                               process_adata_func=process_adata,
                               resolutions_list=[1.0, 1.5],
                               use_rep="X_scVI",use_raw=True)

change_one_ident_fast(adata,"Subset_Identity","Epithelium_Goblet_GPHN","Epithelium_Goblet")
change_one_ident_fast(adata,"Subset_Identity","Epithelium_Col_CA1","Epithelium")
change_one_ident_fast(adata,"Subset_Identity","B Cell_GZMB+","Myeloid_pDC")
change_one_ident_fast(adata,"Subset_Identity","B Cell_VPREB3","B Cell_GC")
change_one_ident_fast(adata,"Subset_Identity","B Cell_TCL1A_IGLC","B Cell_follicular")
change_one_ident_fast(adata,"Subset_Identity","B Cell_TCL1A_IGKC","B Cell_follicular")

apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0607_subset_assignment.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step07/4.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)
adata.write_h5ad("Step07_version4.h5ad")

###################################################################################################################
# 第五次修改
###################################################################################################################

del version4; gc.collect()
adata = anndata.read_h5ad("Step07_version4.h5ad")
my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")

version5 = Focus(focus_file="/data/HeLab/bio/IBD_analysis/assets/0608_subset_focus.xlsx",
                 adata=adata)
version5.filter_and_save_subsets(h5ad_prefix="Step07",
                                output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/5.0",
                                obs_key="Subset_Identity")
version5.process_filtered_files(Geneset_class=my_markers,
                               storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07/5.0",
                               h5ad_prefix="Step07",
                               process_adata_func=process_adata,
                               resolutions_list=[0.5, 1.5],
                               use_rep="X_scVI",use_raw=True)

apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0608_subset_assignment.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step07/5.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)
adata.write_h5ad("Step07_version5.h5ad")

###################################################################################################################
# 第六次修改
###################################################################################################################
change_one_ident(adata,"Subset_Identity","Epithelium_col_Absorp","Epithelium_Col_Absorp")
change_one_ident(adata,"Subset_Identity","Epithelium_Goblet","Goblet")

del version5; gc.collect()
adata = anndata.read_h5ad("Step07_version5.h5ad")
my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")

version6 = Focus(focus_file="/data/HeLab/bio/IBD_analysis/assets/0609_subset_focus.xlsx",
                 adata=adata)
version6.filter_and_save_subsets(h5ad_prefix="Step07",
                                output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/6.0",
                                obs_key="Subset_Identity")
version6.process_filtered_files(Geneset_class=my_markers,
                               storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07/6.0",
                               h5ad_prefix="Step07",
                               process_adata_func=process_adata,
                               resolutions_list=[1.0, 2.0],
                               use_rep="X_scVI",use_raw=True)
change_one_ident_fast(adata,"Subset_Identity","B Cell_GC_Mitotic","B Cell_GC")
change_one_ident_fast(adata,"Subset_Identity","T Cell_CD4_MT","T Cell_CD4_naive")
change_one_ident_fast(adata,"Subset_Identity","T Cell_CD8_KLRB1","T Cell_CD8_Trm")
change_one_ident_fast(adata,"Subset_Identity","T Cell_CD8_KLRC2","T Cell_CD8_Trm_KLRC2")
change_one_ident(adata,"Subset_Identity","T Cell_CD8_CD16A","T Cell_CD8_NKT")

apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0609_subset_assignment.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step07/6.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)
adata.write_h5ad("Step07_version6.h5ad")

###################################################################################################################
# 第七次修改
###################################################################################################################
adata = anndata.read_h5ad("Step07_version6.h5ad")
my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")

version7 = Focus(focus_file="/data/HeLab/bio/IBD_analysis/assets/0610_subset_focus.xlsx",
                 adata=adata)
version7.filter_and_save_subsets(h5ad_prefix="Step07",
                                output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/7.0",
                                obs_key="Subset_Identity")
version7.process_filtered_files(Geneset_class=my_markers,
                               storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07/7.0",
                               h5ad_prefix="Step07",
                               process_adata_func=process_adata,
                               resolutions_list=[1.0, 2.0],
                               use_rep="X_scVI",use_raw=True)
change_one_ident(adata,"Subset_Identity","Plasma_preplasmablast","Plasma_IgM")
change_one_ident(adata,"Subset_Identity","Epithelium","Epithelium_Rec")
change_one_ident(adata,"Subset_Identity","Epithelium_Col_CA1","Epithelium_Col")
change_one_ident(adata,"Subset_Identity","Epithelium_Col_Secret","Epithelium_Col")
change_one_ident(adata,"Subset_Identity","T Cell_gdT_GZMK","T Cell_gdT_g9d2")

apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0610_subset_assignment.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step07/7.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)

value_list = adata.obs["Subset_Identity"].value_counts()
value_list = value_list.sort_index()
with pd.option_context('display.max_rows', None):
    print(value_list)
    
adata.write_h5ad("Step07_version7.h5ad")

###################################################################################################################
# 第八次修改
###################################################################################################################
adata = anndata.read_h5ad("Step07_version7.h5ad")
my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")

version8 = Focus(focus_file="/data/HeLab/bio/IBD_analysis/assets/0610_subset_focus2.xlsx",
                 adata=adata)
version8.filter_and_save_subsets(h5ad_prefix="Step07",
                                output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/8.0",
                                obs_key="Subset_Identity")
version8.process_filtered_files(Geneset_class=my_markers,
                               storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07/8.0",
                               h5ad_prefix="Step07",
                               process_adata_func=process_adata,
                               resolutions_list=[0.7, 2.0],
                               use_rep="X_scVI",use_raw=True)
gc.collect()

apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0611_subset_assignment.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step07/8.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)

value_list = adata.obs["Subset_Identity"].value_counts()
value_list = value_list.sort_index()
with pd.option_context('display.max_rows', None):
    print(value_list)
    

adata.write_h5ad("Step07_version8.h5ad")

###################################################################################################################
# 第九次修改
###################################################################################################################
del version8; gc.collect()
adata = anndata.read_h5ad("Step07_version8.h5ad")
my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")

version9 = Focus(focus_file="/data/HeLab/bio/IBD_analysis/assets/0611_subset_focus.xlsx",
                 adata=adata)
version9.filter_and_save_subsets(h5ad_prefix="Step07",
                                output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/9.0",
                                obs_key="Subset_Identity")
version9.process_filtered_files(Geneset_class=my_markers,
                               storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07/9.0",
                               h5ad_prefix="Step07",
                               process_adata_func=process_adata,
                               resolutions_list=[0.7, 2.0],
                               use_rep="X_scVI",use_raw=True)
gc.collect()

apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0612_subset_assignment.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step07/9.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)

value_list = adata.obs["Subset_Identity"].value_counts()
value_list = value_list.sort_index()
with pd.option_context('display.max_rows', None):
    print(value_list)

change_one_ident(adata,"Subset_Identity","T Cell_CD4_MT","T Cell_CD4_naive")
change_one_ident(adata,"Subset_Identity","T Cell_CD4_Trm_Inflam","T Cell_CD4_mem")
change_one_ident(adata,"Subset_Identity","T Cell_CD4_Trm_Inhib","T Cell_CD4_mem")
change_one_ident(adata,"Subset_Identity","Epithelium_col","Epithelium_Col")
change_one_ident(adata,"Subset_Identity","Epithelium_Col_lowquality","Epithelium_Col")

change_one_ident(adata,"Subset_Identity","Epithelium_col_Absorp","Epithelium_Col_Absorp")
change_one_ident(adata,"Subset_Identity","Epithelium_col_Apical","Epithelium_Col_Apical")
change_one_ident(adata,"Subset_Identity","Epithelium_col_PTMA","Epithelium_Col_PTMA")
adata.write_h5ad("Step07_version9.h5ad")

###################################################################################################################
# 第十次修改
###################################################################################################################
del version9; gc.collect()
adata = anndata.read_h5ad("Step07_version9.h5ad")
my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")

version10 = Focus(focus_file="/data/HeLab/bio/IBD_analysis/assets/0612_subset_focus.xlsx",
                 adata=adata)
version10.filter_and_save_subsets(h5ad_prefix="Step07",
                                output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/10.0",
                                obs_key="Subset_Identity")
version10.process_filtered_files(Geneset_class=my_markers,
                               storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07/10.0",
                               h5ad_prefix="Step07",
                               process_adata_func=process_adata,
                               resolutions_list=[1.0, 2.0],
                               use_rep="X_scVI",use_raw=True)
gc.collect()

change_one_ident(adata,"Subset_Identity","Epithelium_RHOB","Epithelium_stem_OLFM4")
apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0612_subset_assignment2.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step07/10.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)
value_list = adata.obs["Subset_Identity"].value_counts()
value_list = value_list.sort_index()
with pd.option_context('display.max_rows', None):
    print(value_list)

change_one_ident(adata,"Subset_Identity","T Cell_CD8","T Cell_CD8_mem")
change_one_ident(adata,"Subset_Identity","T Cell_CD4_MT","T Cell_CD4_mem")

adata.write_h5ad("Step07_version10.h5ad")

###################################################################################################################
# 第十一次修改
###################################################################################################################
del version10; gc.collect()
adata = anndata.read_h5ad("Step07_version10.h5ad")
my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")

version11 = Focus(focus_file="/data/HeLab/bio/IBD_analysis/assets/0614_subset_focus.xlsx",
                 adata=adata)
version11.filter_and_save_subsets(h5ad_prefix="Step07",
                                output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/11.0",
                                obs_key="Subset_Identity")
version11.process_filtered_files(Geneset_class=my_markers,
                               storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07/11.0",
                               h5ad_prefix="Step07",
                               process_adata_func=process_adata,
                               resolutions_list=[0.5, 3.0],
                               use_rep="X_scVI",use_raw=True)
gc.collect()
change_one_ident(adata,"Subset_Identity","T Cell_nonLin","T Cell_DN")
change_one_ident(adata,"Subset_Identity","Epithelium_Col_PTMA","Epithelium_Col_CD24")
change_one_ident(adata,"Subset_Identity","Epithelium_Col_MLEC","Epithelium_Col_CD24")
change_one_ident(adata,"Subset_Identity","Epithelium_RBFOX1","Epithelium_Col_CD24")

# Epithelium_Col_AQP8: 本身是 Col_Absorp 的亚群，不过 AQP8 表达不排他，希望能够内部清洗；同时可能与 Apical 亚群有交集
    # 特征包括： AQP8, GUCA2A, GUCA2B
# Epithelium_Col_BEST4：定义良好√
# Epithelium_Col_PTMA：PTMA表达量低于每一群mitotic亚群，
# Epithelium_Col_MLEC：MLEC表达量低于总mitotic亚群，和PTMA相近
# Epithelium_RBFOX1：确实具有最高水平的RBFOX1，但也不甚显著
adata.write_h5ad("Step07_version11.h5ad")

###################################################################################################################
# 第十二次修改
###################################################################################################################
del version11; gc.collect()
# adata = anndata.read_h5ad("Step07_version11.h5ad")
my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")

version12 = Focus(focus_file="/data/HeLab/bio/IBD_analysis/assets/0613_subset_focus.xlsx",
                 adata=adata)
version12.filter_and_save_subsets(h5ad_prefix="Step07",
                                output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/12.0",
                                obs_key="Subset_Identity")
version12.process_filtered_files(Geneset_class=my_markers,
                               storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07/12.0",
                               h5ad_prefix="Step07",
                               process_adata_func=process_adata,
                               resolutions_list=[1.0, 2.0],
                               use_rep="X_scVI",use_raw=True)

apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0614_subset_assignment.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step07/12.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)
value_list = adata.obs["Subset_Identity"].value_counts()
value_list = value_list.sort_index()
with pd.option_context('display.max_rows', None):
    print(value_list)

change_one_ident(adata,"Subset_Identity","Epithelium_Col_TSPAN8","Epithelium_stem_OLFM4")
adata.write_h5ad("Step07_version12.h5ad")

###################################################################################################################
# 第十三次修改
###################################################################################################################
del version12; gc.collect()
# adata = anndata.read_h5ad("Step07_version11.h5ad")
my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")

version13 = Focus(focus_file="/data/HeLab/bio/IBD_analysis/assets/0614_subset_focus2.xlsx",
                 adata=adata)
version13.filter_and_save_subsets(h5ad_prefix="Step07",
                                output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/13.0",
                                obs_key="Subset_Identity")
version13.process_filtered_files(Geneset_class=my_markers,
                               storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07/13.0",
                               h5ad_prefix="Step07",
                               process_adata_func=process_adata,
                               resolutions_list=[1.0, 1.5],
                               use_rep="X_scVI",use_raw=True)
apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0614_subset_assignment2.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step07/13.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)
value_list = adata.obs["Subset_Identity"].value_counts()
value_list = value_list.sort_index()
with pd.option_context('display.max_rows', None):
    print(value_list)

adata.write_h5ad("Step07_version13.h5ad")

###################################################################################################################
# 第十四次修改
###################################################################################################################
del version13; gc.collect()
# adata = anndata.read_h5ad("Step07_version11.h5ad")
my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")

version14 = Focus(focus_file="/data/HeLab/bio/IBD_analysis/assets/0615_subset_focus.xlsx",
                 adata=adata)
version14.filter_and_save_subsets(h5ad_prefix="Step07",
                                output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/14.0",
                                obs_key="Subset_Identity")
version14.process_filtered_files(Geneset_class=my_markers,
                               storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07/14.0",
                               h5ad_prefix="Step07",
                               process_adata_func=process_adata,
                               resolutions_list=[1.0, 2.0],
                               use_rep="X_scVI",use_raw=True)
gc.collect()

change_one_ident(adata,"Subset_Identity","B Cell_kappa","B Cell_mem")
change_one_ident(adata,"Subset_Identity","B Cell_lamda","B Cell_mem")


change_one_ident(adata,"Subset_Identity","Myeloid_SLC8A1","Myeloid_cDC3")

change_one_ident(adata,"Subset_Identity","B Cell_follicular","B Cell_mem")
change_one_ident(adata,"Subset_Identity","T Cell_gdT_g4d1_chemo","T Cell_gdT_Trm_chemo")
change_one_ident(adata,"Subset_Identity","T Cell_gdT_g4d1_cyto","T Cell_gdT_Trm")
apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0615_subset_assignment.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step07/14.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)
value_list = adata.obs["Subset_Identity"].value_counts()
value_list = value_list.sort_index()
with pd.option_context('display.max_rows', None):
    print(value_list)

adata.write_h5ad("Step07_version14.h5ad")

###################################################################################################################
# 第十五次修改
###################################################################################################################
del version14; gc.collect()
# adata = anndata.read_h5ad("Step07_version11.h5ad")
my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")

version15 = Focus(focus_file="/data/HeLab/bio/IBD_analysis/assets/0616_subset_focus.xlsx",
                 adata=adata)
version15.filter_and_save_subsets(h5ad_prefix="Step07",
                                output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/15.0",
                                obs_key="Subset_Identity")
version15.process_filtered_files(Geneset_class=my_markers,
                               storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07/15.0",
                               h5ad_prefix="Step07",
                               process_adata_func=process_adata,
                               resolutions_list=[1.0, 2.0],
                               use_rep="X_scVI",use_raw=True)
gc.collect()
apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0616_subset_assignment.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step07/15.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)
value_list = adata.obs["Subset_Identity"].value_counts()
value_list = value_list.sort_index()
with pd.option_context('display.max_rows', None):
    print(value_list)

adata.write_h5ad("Step07_version15.h5ad")

change_one_ident(adata,"Subset_Identity","Plasma_IgA_pre","Plasma_pre")
change_one_ident(adata,"Subset_Identity","Myeloid_cDC3_mitotic","Myeloid_cDC3")
change_one_ident(adata,"Subset_Identity","T Cell_gdT_g4d1","T Cell_gdT")

###################################################################################################################
# 第十六次修改
###################################################################################################################
del version15; gc.collect()
# adata = anndata.read_h5ad("Step07_version11.h5ad")
my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")

version16 = Focus(focus_file="/data/HeLab/bio/IBD_analysis/assets/0616_subset_focus2.xlsx",
                 adata=adata)

version16.filter_and_save_subsets(h5ad_prefix="Step07",
                                output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/16.0",
                                obs_key="Subset_Identity")
version16.process_filtered_files(Geneset_class=my_markers,
                               storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07/16.0",
                               h5ad_prefix="Step07",
                               process_adata_func=process_adata,
                               resolutions_list=[0.5, 2.0],
                               use_rep="X_scVI",use_raw=True)
gc.collect()
value_list = adata.obs["Subset_Identity"].value_counts()
value_list = value_list.sort_index()
with pd.option_context('display.max_rows', None):
    print(value_list)

apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0616_subset_assignment2.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step07/16.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)
change_one_ident(adata,"Subset_Identity","B Cell_CSR","B Cell_mem")
change_one_ident(adata,"Subset_Identity","T Cell_DN_A","T Cell_DN")
change_one_ident(adata,"Subset_Identity","T Cell_DN_B","T Cell_DN")

###################################################################################################################
# 第十七次修改
###################################################################################################################
del version16; gc.collect()
# adata = anndata.read_h5ad("Step07_version11.h5ad")
my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")

version17 = Focus(focus_file="/data/HeLab/bio/IBD_analysis/assets/0616_subset_focus3.xlsx",
                 adata=adata)

version17.filter_and_save_subsets(h5ad_prefix="Step07",
                                output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/17.0",
                                obs_key="Subset_Identity")
version17.process_filtered_files(Geneset_class=my_markers,
                               storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07/17.0",
                               h5ad_prefix="Step07",
                               process_adata_func=process_adata,
                               resolutions_list=[0.5, 2.0],
                               use_rep="X_scVI",use_raw=True)
gc.collect()

apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0616_subset_assignment3.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step07/17.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)
value_list = adata.obs["Subset_Identity"].value_counts()
value_list = value_list.sort_index()
with pd.option_context('display.max_rows', None):
    print(value_list)

adata.write_h5ad("Step07_version17.h5ad")

###################################################################################################################
# 第十八次修改
###################################################################################################################
del version17; gc.collect()
# adata = anndata.read_h5ad("Step07_version11.h5ad")
my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")

version18 = Focus(focus_file="/data/HeLab/bio/IBD_analysis/assets/0616_subset_focus4.xlsx",
                 adata=adata)

version18.filter_and_save_subsets(h5ad_prefix="Step07",
                                output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/18.0",
                                obs_key="Subset_Identity")
version18.process_filtered_files(Geneset_class=my_markers,
                               storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07/18.0",
                               h5ad_prefix="Step07",
                               process_adata_func=process_adata,
                               resolutions_list=[0.5],
                               use_rep="X_scVI",use_raw=True)
gc.collect()
apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0616_subset_assignment4.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step07/18.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)
value_list = adata.obs["Subset_Identity"].value_counts()
value_list = value_list.sort_index()
with pd.option_context('display.max_rows', None):
    print(value_list)

###################################################################################################################
# 第十九次修改
###################################################################################################################
del version18; gc.collect()
# adata = anndata.read_h5ad("Step07_version11.h5ad")
my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")

version19 = Focus(focus_file="/data/HeLab/bio/IBD_analysis/assets/0616_subset_focus5.xlsx",
                 adata=adata)

version19.filter_and_save_subsets(h5ad_prefix="Step07",
                                output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/19.0",
                                obs_key="Subset_Identity")
version19.process_filtered_files(Geneset_class=my_markers,
                               storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07/19.0",
                               h5ad_prefix="Step07",
                               process_adata_func=process_adata,
                               resolutions_list=[0.5],
                               use_rep="X_scVI",use_raw=True)
gc.collect()
change_one_ident(adata,"Subset_Identity","T Cell_gdT_g8d1","T Cell_gdT_Trm_cyto")

apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0616_subset_assignment5.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step07/19.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)
value_list = adata.obs["Subset_Identity"].value_counts()
value_list = value_list.sort_index()
with pd.option_context('display.max_rows', None):
    print(value_list)
    
adata.write_h5ad("Step07_version19.h5ad")

###################################################################################################################
# 最终版本修改.ver1
###################################################################################################################
# 将低质量细胞移除
remove_cells = adata.obs_names[
    (adata.obs["Subset_Identity"] == "Doublet") | (adata.obs["Subset_Identity"] == "Mito high")
]
adata = adata[~adata.obs_names.isin(remove_cells)]

# 重命名
gc.collect()
change_one_ident_fast(adata,"Subset_Identity","Epithelium_mitotic","Epithelium_Mitotic")
change_one_ident_fast(adata,"Subset_Identity","Myeloid_mitotic","Myeloid_Mitotic")
change_one_ident_fast(adata,"Subset_Identity","T Cell_mitotic","T Cell_Mitotic")

# 拆分成两列，最多只按一个下划线分
split_df = adata.obs["Subset_Identity"].str.split("_", n=1, expand=True)
# 填充第二列中的缺失值为 "default"
split_df[1] = split_df[1].fillna("default")
# 存入 adata.obs
adata.obs["Celltype"] = split_df[0]
adata.obs["Cell_Subtype"] = split_df[1]

adata.obs = adata.obs[['orig.ident', 'percent.mt', 'percent.ribo', 'percent.hb', 'orig.project', 'Patient', 'disease', 'tissue-type', 'tissue-origin', 'presorted',
                        'Subset_Identity', 'phase',"Celltype","Cell_Subtype"]]

adata.write_h5ad("Step07_finalversion_1.h5ad")

###################################################################################################################
# 最终版本修改.ver2
###################################################################################################################
adata=anndata.read_h5ad("Step07_finalversion_1.h5ad")

adata.obs.loc[adata.obs["tissue-type"]=="mixed","tissue-type"]="if"

# 查看交叉关系
ct = pd.crosstab(adata.obs["disease"], adata.obs["tissue-type"])
# 显示交叉表
print(ct)

# 整理交叉关系，并重命名
adata.obs["tissue-disease"] = adata.obs["tissue-type"].astype(str) + "-" + adata.obs["disease"].astype(str)
adata.obs["tissue-disease"].value_counts()
change_one_ident_fast(adata,"tissue-disease","normal-HC","Heathy_control")
change_one_ident_fast(adata,"tissue-disease","if-Colitis","non-IBD colitis")
change_one_ident_fast(adata,"tissue-disease","if-CD","inflamed CD")
change_one_ident_fast(adata,"tissue-disease","nif-CD","non-infl CD")
change_one_ident_fast(adata,"tissue-disease","if-UC","inflamed UC")
change_one_ident_fast(adata,"tissue-disease","nif-UC","non-infl UC")
change_one_ident_fast(adata,"tissue-disease","if-BD","inflamed BD")

td_order = ['Heathy_control', 'non-IBD colitis',
            'non-infl CD', 'inflamed CD',
            'non-infl UC', 'inflamed UC',
            'inflamed BD']
adata.obs["tissue-disease"] = pd.Categorical(adata.obs["tissue-disease"], categories=td_order, ordered=True)
adata.write_h5ad("Step07_finalversion_2.h5ad")

###################################################################################################################
# 进行一些比例检查，这个功能先冻结，因为存在识别算法逻辑上的问题
# 参阅 Step07d 的统计学部分，主要依赖 KW 检验等。
# from src.ScanpyTools.Scanpy_statistics import compute_celltype_proportion
# celltype_list = adata.obs["Subset_Identity"].unique().tolist()
# for i in celltype_list[20:]:
#     input(f"Press enter to run subset {i}...")
#     compute_celltype_proportion(adata,i)
#
# ###
# import pandas as pd
# # 交叉表：每个亚群中，不同细胞周期的细胞数
# ct = pd.crosstab(adata.obs["Subset_Identity"], adata.obs["phase"])
# # 按行归一化，得到每个亚群中各周期的百分比
# ct_percent = ct.div(ct.sum(axis=1), axis=0) * 100
# # 显示结果
# with pd.option_context('display.max_rows', None):
#     print(ct_percent)


###################################################################################################################
# 最终版本修改.ver3
###################################################################################################################
adata = anndata.read_h5ad("Step07_finalversion_2.h5ad")
value_list = adata.obs["Subset_Identity"].value_counts()
value_list = value_list.sort_index()
with pd.option_context('display.max_rows', None):
    print(value_list)


my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")

version_final = Focus(focus_file="/data/HeLab/bio/IBD_analysis/assets/0619_subset_focus.xlsx",
                 adata=adata)
version_final.filter_and_save_subsets(h5ad_prefix="Step07",
                                output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/20.0",
                                obs_key="Subset_Identity")
version_final.process_filtered_files(Geneset_class=my_markers,
                               storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07/20.0",
                               h5ad_prefix="Step07",
                               process_adata_func=process_adata,
                               resolutions_list=[0.7,2.0],
                               use_rep="X_scVI",use_raw=True)
gc.collect()
apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0620_subset_assignment.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step07/20.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)


version_final = Focus(focus_file="/data/HeLab/bio/IBD_analysis/assets/0620_subset_focus.xlsx",
                 adata=adata)
version_final.filter_and_save_subsets(h5ad_prefix="Step07",
                                output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/20.0",
                                obs_key="Subset_Identity")
version_final.process_filtered_files(Geneset_class=my_markers,
                               storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07/20.0",
                               h5ad_prefix="Step07",
                               process_adata_func=process_adata,
                               resolutions_list=[0.7,2.0],
                               use_rep="X_scVI",use_raw=True)
apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0620_subset_assignment2.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step07/20.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)


new_ident_dict = {"Myeloid_Monocyte_IL1B":"Myeloid_Int.Monocyte",
                  "Myeloid_Neutrophil":"Myeloid_N.C.Monocyte",
                  "Myeloid_Monocyte":"Myeloid_C.Monocyte",
                  "Myeloid_Macrophage_M2":"Myeloid_M2.Macrophage",
                  "Myeloid_Mitotic":"Myeloid_Myeloid.mitotic",
                  # "T Cell_CD8aa":"T Cell_MAIT",
                  "T Cell_gdT":"T Cell_gdT.naive",
                  "T Cell_CD8_GZMK":"T Cell_CD8.mem.GZMK",
                  "T Cell_CD4_GZMK":"T Cell_CD4.Tr1",
                  "T Cell_CD4_naive":"T Cell_CD4.naive",
                  "T Cell_CD4_Tfh":"T Cell_CD4.Tfh",
                  "T Cell_CD4_Th1":"T Cell_CD4.Th1",
                  "T Cell_CD4_Th17":"T Cell_CD4.Th17",
                  "T Cell_CD4_Treg":"T Cell_CD4.Treg",
                  "T Cell_CD4_early":"T Cell_CD4.early",
                  "T Cell_CD4_mem":"T Cell_CD4.mem",
                  "T Cell_CD8_mem":"T Cell_CD8.mem",
                  "T Cell_CD8_naive":"T Cell_CD8.naive",
                  "T Cell_CD8_NKT":"T Cell_CD8.NKT",
                  "T Cell_CD8_Trm":"T Cell_CD8.Trm",
                  "T Cell_CD8_Trm_KLRC2":"T Cell_CD8.Trm.KLRC2+",
                  "T Cell_ILC_XCL1":"T Cell_ILC.XCL1+",
                  "T Cell_NK_CD16":"T Cell_NK.CD16+",
                  "T Cell_NK_CD56":"T Cell_NK.CD56+",
                  "T Cell_gdT_Trm_chemo":"T Cell_gdT.Trm.XCL1+",
                  "T Cell_gdT_Trm_cyto":"T Cell_gdT.Trm.GZMA+",
                  "T Cell_gdT_g9d2":"T Cell_gdT.g9d2",
                  "T Cell_gdT_mitotic":"T Cell_gdT.mitotic",
                  "B Cell_GC":"B Cell_GC.B",
                  "B Cell_Mitotic":"B Cell_B.mitotic",
                  "Fibroblast_CXCL12":"Fibroblast_Fb.CXCL12",
                  "Fibroblast_CXCL14":"Fibroblast_Fb.CXCL14",
                  "Epithelium_Goblet.mitotic":"Epi_Goblet.mitotic",
                  "Epithelium_Mitotic":"Epi_Epi.mitotic",
                  "Epithelium_Col_AQP8":"Epi_Absorp.Col.AQP8+",
                  "Epithelium_Col_Absorp":"Epi_Absorp.Col",
                  "Epithelium_Col_Apical":"Epi_Absorp.Col.DUOX2+",
                  "Epithelium_Col":"Epi_Absorp.Col", # 合并
                  "Epithelium_Col_BEST4":"Epi_Col.BEST4+",
                  "Epithelium_Goblet":"Epi_Goblet",
                  "Epithelium_Goblet_TFF1":"Epi_Goblet.TFF1+",
                  "Epithelium_Stem.LGR5":"Epi_Stem.LGR5",
                  "Epithelium_Tuft":"Epi_Tuft",
                  "Epithelium_Paneth":"Epi_Paneth",
                  "Epithelium_EEC":"Epi_EEC",
                  "Epithelium_Microfold":"Epi_Microfold",
                  "Mast Cell":"Myeloid_Mast",
                  "Endothelium":"Endo_Endothelium",
                  "Fibroblast":"Fibroblast_Fb.resting",
                  "Plasma_Mitotic":"Plasma_Plasma.mitotic",
                  "Plasma_IgA":"Plasma_IgA",
                  "Plasma_IgG":"Plasma_IgG",
                  "Plasma_IgM":"Plasma_IgM",
                  "T Cell_Mitotic":"T Cell_T.mitotic"
}


filtered_dict = {k: v for k, v in new_ident_dict.items() if k in adata.obs["Subset_Identity"].unique()}
for k, v in filtered_dict.items():
    change_one_ident_fast(adata,"Subset_Identity",k,v)

adata.obs["Subset_Identity"] = adata.obs["Subset_Identity"].astype("category")
adata.obs["Subset_Identity"].cat.remove_unused_categories(inplace=True)

value_list = adata.obs["Subset_Identity"].value_counts()
value_list = value_list.sort_index()
with pd.option_context('display.max_rows', None):
    print(value_list)


adata.write_h5ad("Step07_finalversion_3.h5ad")

###################################################################################################################
# 最终版本修改.ver4
###################################################################################################################
del version_final;gc.collect()
adata = anndata.read_h5ad("Step07_finalversion_3.h5ad")
version_final = Focus(focus_file="/data/HeLab/bio/IBD_analysis/assets/0620_subset_focus3.xlsx",
                 adata=adata)
version_final.filter_and_save_subsets(h5ad_prefix="Step07",
                                output_dir="/data/HeLab/bio/IBD_analysis/output/Step07/20.0",
                                obs_key="Subset_Identity")
version_final.process_filtered_files(Geneset_class=my_markers,
                               storage_dir="/data/HeLab/bio/IBD_analysis/output/Step07/20.0",
                               h5ad_prefix="Step07",
                               process_adata_func=process_adata,
                               resolutions_list=[0.5],
                               use_rep="X_scVI",use_raw=True)
apply_assignment_annotations(
    assignment_file="/data/HeLab/bio/IBD_analysis/assets/0620_subset_assignment3.xlsx",
    adata_main=adata,
    h5ad_dir="/data/HeLab/bio/IBD_analysis/output/Step07/20.0",
    output_key="Subset_Identity",
    fillna_from_col="Primary_Cluster"
)

remove_cells = adata.obs_names[
    (adata.obs["Subset_Identity"] == "Doublet") | (adata.obs["Subset_Identity"] == "Mito high")
]
adata = adata[~adata.obs_names.isin(remove_cells)]

adata.write_h5ad("Step07_finalversion_4.h5ad")

###################################################################################################################
# 最终版本修改.ver5
###################################################################################################################
adata = anndata.read_h5ad("Step07_finalversion_4.h5ad")
change_one_ident_fast(adata,"Subset_Identity","Fibroblast_Fb.CXCL14","Fibroblast_Fb.activated")
change_one_ident_fast(adata,"Subset_Identity","Fibroblast_Fb.CXCL12","Fibroblast_Fb.activated")
##################
# 拆分成两列，最多只按一个下划线分
split_df = adata.obs["Subset_Identity"].str.split("_", n=1, expand=True)
# 填充第二列中的缺失值为 "default"
split_df[1] = split_df[1].fillna("default")

adata.obs = adata.obs[['orig.ident', 'percent.mt', 'percent.ribo', 'percent.hb', 'orig.project', 'Patient', 'disease',
                       'tissue-type', 'tissue-origin', 'presorted',
                        'Subset_Identity', 'phase']]
# 存入 adata.obs
adata.obs["Celltype"] = split_df[0]
adata.obs["Cell_Subtype"] = split_df[1]

# 添加新类别
adata.obs["Celltype"] = adata.obs["Celltype"].astype(str)
adata.obs.loc[adata.obs["Cell_Subtype"].str.contains("mitotic", case=False, na=False), "Celltype"] = "Mitotic"
adata.obs["Celltype"] = adata.obs["Celltype"].astype("category")

adata.obs = adata.obs[['orig.ident', 'percent.mt', 'percent.ribo', 'percent.hb', 'orig.project', 'Patient', 'disease', 'tissue-type', 'tissue-origin', 'presorted',
                        'Subset_Identity', 'phase',"Celltype","Cell_Subtype"]]
##################
adata.write_h5ad("Step07_finalversion_5.h5ad")



