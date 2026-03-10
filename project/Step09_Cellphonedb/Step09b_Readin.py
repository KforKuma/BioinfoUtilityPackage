"""
Step.py
Author: John Hsiung
Update date: 2025-12-08
Description:
    - 进行 CPDB 数据的分析
Notes:
    - 使用环境：conda activate sc-min
"""

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
####################################
# 重新载入
# 删除模块缓存
for module_name in list(sys.modules.keys()):
    if module_name.startswith('src.external_adaptor.cellphonedb'):
        del sys.modules[module_name]

####################################
# parent_dir = "/data/HeLab/bio/IBD_analysis/output/Step08_Cellphonedb/cellphonedb_input_0625"
# parent_dir = "/public/home/xiongyuehan/data/IBD_analysis/output/Step08_Cellphonedb/251203"
# parent_dir = "/public/home/xiongyuehan/data/IBD_analysis/output/Step08_Cellphonedb/260106"
parent_dir = "/public/home/xiongyuehan/data/IBD_analysis/output/Step08_Cellphonedb/260125"

analysis_dir = f"{parent_dir}/analysis_thr20percent"
output_dir = f"{parent_dir}/output_thr20percent"

os.makedirs(analysis_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

disease_list = ['CD', 'HC', 'UC', 'BD', 'Colitis']
####################################
# 测试
adata = anndata.read_h5ad(
    "/public/home/xiongyuehan/data/IBD_analysis/output/Step07_Summary/Step07_DR_clustered_clean_20260210.h5ad")
obs = adata.obs
del adata;
gc.collect()


print(obs["Subset_Identity"].value_counts())
# 直接复制自 Step06
proofread = {'Epithelium': ['Intestinal stem cell OLFM4+LGR5+',
                            'pre-TA cell',
                            'Transit amplifying cell',
                            'Regenerative colonocyte LEFTY1+',
                            'Antigen-presenting colonocyte MHC-II+',
                            'Goblet', 'Paneth cell', 'Tuft cell', 'Enteroendocrine',
                            'Ion-sensing colonocyte BEST4+',
                            'Microfold cell',
                            'Absorptive colonocyte PPARs+', 'Absorptive colonocyte', 'Absorptive colonocyte Guanylins+'],
             'Fibroblast': ['Fibroblast ADAMDEC1+', 'Fibroblast'],
             'Endothelium': ['Endothelium'],
             'Myeloid Cell': ['Classical monocyte CD14+', 'Nonclassical monocyte CD16A+',
                              'Macrophage', 'Macrophage M1', 'Macrophage M2',
                              'Neutrophil CD16B+',
                              'Mast cell',
                              'cDC1 CLEC9A+', 'cDC2 CD1C+', 'pDC GZMB+'],
             'T Cell': ['CD4 Tnaive', 'CD4 Tmem', 'CD4 Tmem GZMK+', 'CD4 Tfh', 'CD4 Treg', 'CD4 Th17',
                        'CD8 Tnaive', 'CD8 Tmem', 'CD8 Tmem GZMK+', 'CD8 Trm', 'CD8 Trm GZMA+',
                        'CD8 NKT FCGR3A+', 'CD8aa IEL',
                        'gdTnaive', 'g9d2T cytotoxic', 'gdTrm',
                        'ILC1', 'ILC3', 'MAIT TRAV1-2+',
                        'Natural killer cell FCGR3A+', 'Natural killer cell NCAM1+',
                        'Mitotic T cell'],
             'B Cell': ['Germinal center B cell',
                        'B cell lambda', 'B cell kappa', 'B cell IL6+'],
             'Plasma Cell': ['Plasma IgA+', 'Plasma IgG+', 'Mitotic plasma cell']}

for k, v in proofread.items():
    list_unmatched = [i for i in v if i not in obs["Subset_Identity"].unique()]
    print(list_unmatched)


cell_subset_list = []
for k, v in proofread.items():
    cell_subset_list = cell_subset_list + v

cell_types = sorted(cell_subset_list, key=len, reverse=True)

# 初步处理并输出表格
for type in disease_list:
    print("#" * 60)
    print(type)
    
    file_dir = f"{output_dir}/{type}/"
    
    ci = CellphoneInspector(cpdb_outfile=file_dir, degs_analysis=False, cellsign=False)
    ci.prepare_cpdb_tables(add_meta=True)
    
    ci.prepare_cell_metadata(metadata=obs, celltype_key="Subset_Identity", groupby_key="disease")  # 通常直接输入就好
    
    # 准备查询
    ## 输出是可读的，所以这一步可以手动修改
    gene_query = prepare_gene_query(ci)
    celltype_pairs = ci.prepare_celltype_pairs(cell_type1=".", cell_type2=".")
    
    # 过滤
    ci.filter_and_cluster(gene_query=gene_query, celltype_pairs=celltype_pairs, keep_significant_only=False,
                          cluster_rows=False)
    
    # 输出结果表格
    ci.format_outcome(cell_name_list=cell_types)
    df = ci.outcome["final"]
    print(len(df.cell_left.unique()))
    print(len(df.cell_right.unique()))
    print(len(df.celltype_group.unique()))
    # print("CD4 Tmem GZMK+" in df.cell_left.values)
    # print("Ion-transport colonocyte CFTR+" in df.cell_left.values)
    
    df.to_csv(f"{output_dir}/{type}/Arranged_output_0124.csv", index=False)

# 检查
# for type in disease_list:
#     print("#"*60)
#     print(type)
#     df = pd.read_csv(f"{output_dir}/{type}/test.csv",encoding="utf-8", sep=",")
#     print(len(df.cell_left.unique()))
#     print(len(df.cell_right.unique()))
#     print(len(df.celltype_group.unique()))
#     df.cell_left.unique()
#     "CD4 Tmem GZMK+" in df.cell_left.values
#     "CD4 Tmem GZMK+" in df.cell_right.values
####################################
# 整合并分类保存
dfs = []
for type in disease_list:
    df = pd.read_csv(f"{output_dir}/{type}/Arranged_output_0124.csv",
                     encoding="utf-8", sep=",")
    df["group"] = type
    dfs.append(df)

df_merge = combine_outcome(dfs)
split_and_save(df_merge, save_addr=analysis_dir)

df_merge["interaction_group"].unique()
df_merge["cell_right"].value_counts()

mask = df_merge["interaction_group"].str.contains(
    r"GUCA2B|GUCY2C",
    regex=True,
    na=False
)

df_hit = df_merge[mask]

df_hit[df_hit["pvals"] < 0.5]

