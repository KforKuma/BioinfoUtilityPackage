"""
Step.py
Author: John Hsiung
Update date: 2025-08-29
Description:
    - 进行 CPDB 数据的分析
Notes:
    - 使用环境：conda activate scvpy10
"""

import pandas as pd
import anndata
import os, gc
# import scanpy as sc
import sys


sys.path.append('/data/HeLab/bio/IBD_analysis/')
import src.EasyInterface.CPDBTools
importlib.reload(src.EasyInterface.CPDBTools)

from src.EasyInterface.CPDBInterface import extract_cpdb_result,extract_cpdb_table, prepare_metadata


# disease_list = ["HC"] # 虽然这一步是固定的，但为了代码的兼容性仍然建议放在函数之外单独运行
# test_list = ["BD"]

parent_dir = "/data/HeLab/bio/IBD_analysis/output/Step08_Cellphonedb/cellphonedb_input_0625"
analysis_dir = f"{parent_dir}/Analysis"
output_dir = f"{parent_dir}/output_DEG_0625"
# disease_list = adata.obs["disease"].unique()
disease_list = ['CD', 'HC', 'UC', 'BD', 'Colitis']

for type in disease_list:#
    file_dir = f"{output_dir}/{type}/"
    cpdb_results = extract_cpdb_result(file_dir)
    
    h5ad_file = f"{parent_dir}/{type}/counts.h5ad"
    adata_subset = anndata.read_h5ad(h5ad_file)
    
    metadata = prepare_metadata(adata_subset.obs,
                                celltype_key="Subset_Identity",
                                splitby_key=None)
    del adata_subset;gc.collect()
    all = extract_cpdb_table(metadata=metadata,
                             cpdb_outcome_dict=cpdb_results,
                             additional_grouping=False,
                             splitby_key=None,
                             lock_celltype_direction=True,  # 核心功能：加速 && 锁定
                             cell_type1=".", cell_type2=".",  # 方向
                             genes=[],  # 基因（列表行）
                             cluster_rows=False,
                             min_interaction_score=0.1,
                             keep_significant_only=True,
                             debug=False
                             )
    all.to_csv(f"{analysis_dir}/{type}_Allsig.csv")
    del all
    gc.collect()
