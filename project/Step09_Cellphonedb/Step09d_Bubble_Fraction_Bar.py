import pandas as pd
import anndata
import os
import gc
import scanpy as sc
import sys


sys.stdout.reconfigure(encoding='utf-8')
####################################
sys.path.append('/public/home/xiongyuehan/data/BioinfoUtilityPackage')

from src.external_adaptor.cellphonedb.cellphone_inspector import *
from src.external_adaptor.cellphonedb.plot import *
from src.external_adaptor.cellphonedb.toolkit import *

####################################
parent_dir = "/public/home/xiongyuehan/data/IBD_analysis/output/Step09_Cellphonedb/260125"

analysis_dir = f"{parent_dir}/cpdb_outcome/analysis_thr15percent"
analy_fig_dir = f"{parent_dir}/output/fig_thr15percent/bubble_and_bar"
os.makedirs(analysis_dir, exist_ok=True)
os.makedirs(analy_fig_dir, exist_ok=True)

adata = anndata.read_h5ad(
    "/public/home/xiongyuehan/data/IBD_analysis/output/Step07_Summary/Step07_DR_clustered_clean_20260210.h5ad")

gene_express_dict = {
    
    "FASLG": ["FASLG"], "FAS": ['FAS'],
    'lymphotoxin': ['LTA', 'LTB'],
    'HVEM': ['TNFRSF14'], 'HVEM_ligand': ['BTLA', 'TNFSF14'],
    'TGFB': ['TGFB1'], 'TGFB_receptor': ['TGFBR1', 'TGFBR2'],
    "IL6": ["IL6"],
    'IL21': ['IL21'], 'IL21_receptor': ['IL21R'], 'CXCR5': ['CXCR5'],
    
    
    
    "Guanylins": ["GUCA2A", "GUCA2B"],
    'CXCR4': ['CXCR4'],
    "CCR1_ligand": ['CCL3', 'CCL4', 'CCL5'],
    "CCR1_ligand_2": ['CCL15', 'CCL23'],
    "CCR1": ['CCR1'],
     'CXCR2': ['CXCR2'],
    'CXCR3': ['CXCR3'], 'CXCR3_ligand': ['CXCL9', 'CXCL10', 'CXCL11'],
    'ephrin': ['EFNA1', 'EFNA2', 'EFNA3', 'EFNA4'],
    # 'Eph_receptor': ['EPHA1', 'EPHA2', 'EPHA3', 'EPHA4', 'EPHA7']
    'CCL28': ['CCL28'], 'CCR10': ['CCR10'],
    'CX3CL1': ['CX3CL1'], 'CX3CR1': ['CX3CR1'],
    'CXCL16': ['CXCL16'], 'CXCR6': ['CXCR6'],
    'IFNGR': ['IFNGR1','IFNGR2'],
    "GZMK": ["GZMK"],
    "IFNG": ["IFNG"],"TNF_receptor": ['TNFRSF1A', 'TNFRSF1B'],
    
    "PARs": ["F2R", "F2RL1"],
    "TNFSF10": ["TNFSF10"],'TNFRSF10D': ['TNFRSF10D'],
    'TNFRSF11B': ['TNFRSF11B'],'CXCR2_ligand': ["CXCL1", "CXCL2", "CXCL3"],
    "DR4_DR5": ["TNFRSF10A", "TNFRSF10B"],
    "TNFA": ["TNF"],"CXCR4_ligand": ["CXCL12", "CXCL14"],"IL1B": ["IL1B"],"PTGS2": ["PTGS2"],
}

gene_express_dict_test = {
    "Anaphylatoxin_receptor": ["C3AR1", "C5AR1"],
    
}

adata = adata[(adata.obs["disease"] == "HC") |
              ((adata.obs["disease"] != "HC") & (adata.obs["tissue-type"] == "if"))]


# 运行一次即可
plot_universal_bubble_legend(analy_fig_dir)

for k, v in gene_express_dict_test.items():
    summary_filtered = plot_gene_bubble_with_cell_fraction(adata, save_addr=analy_fig_dir,
                                                           filename=f"Gene_Expression_cali_{k}",
                                                           gene=v,
                                                           hk_genes=[
                                                               "GAPDH", "ACTB", "B2M",  # 经典代谢
                                                               "RPL13A", "RPL19", "RPL11", "RPL32", "RPLP0",  # 大亚基
                                                               "RPS18", "RPS27", "RPS12", "RPS3A", "RPS6",  # 小亚基
                                                               "EEF1A1", "PABPC1"  # 翻译起始/延伸因子
                                                           ],
                                                           celltype_col="Subset_Identity",
                                                           # celltype_exclude=["Cyc.T"],
                                                           min_frac=0.25,out_frac=0.2,
                                                           figsize=(10, 4),topN=10,
                                                           cmap='crest')

