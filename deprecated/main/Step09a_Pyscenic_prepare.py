"""
Step09a_Pyscenic_preprare.py
Author: John Hsiung
Update date: 2025-08-29
Description:
    - 准备进行 pyscenic 的运行
Notes:
    - 使用环境：conda activate scvpy10；
    - pyscenic 的实际运行在 pyscenic 环境中完成
"""
import anndata
import pandas as pd
import os
import sys
##——————————————————————————————————————————————————————————————————————————
# 设置scanpy基本属性
import yaml

with open('/data/HeLab/bio/IBD_analysis/assets/io_config.yaml', 'r') as f:
    config = yaml.safe_load(f) # 读取的格式为一系列字典的列表
##——————————————————————————————————————————————————————————————————————————
os.chdir("/data/HeLab/bio/IBD_analysis/")
sys.path.append('/data/HeLab/bio/IBD_analysis/')
from src.ScanpyTools.ScanpyTools import obs_key_wise_subsampling, write_scenic_input


adata = anndata.read("/data/HeLab/bio/IBD_analysis/output/Step07/Step07_finalversion_4.h5ad")

value_list = adata.obs["Subset_Identity"].value_counts()
value_list = value_list.sort_index()
with pd.option_context('display.max_rows', None):
    print(value_list)

# PYSCENIC拆分的依据：pyscenic_assign_dict，包含两列name与groups，用来将细胞按照大类进行拆分，避免文件过大运行pyscenic困难
    # 新增一列mode_by，如果标注disease，则按照disease拆分，新建一个 adata.obs["tmp"] = adata.obs["Subset_Identity"].astype(str) + "-" + adata.obs["disease"].astype(str)
    # 如果标注Subset_Identity，则检查是否包含多个Subset，进行Subset之间的比较
pyscenic_assign_dict = pd.ExcelFile("/data/HeLab/bio/IBD_analysis/assets/PYSCENIC_assignment(250621).xlsx")
pyscenic_assign_dict = pyscenic_assign_dict.parse(pyscenic_assign_dict.sheet_names[0])



adata.obs["tmp"] = adata.obs["Subset_Identity"].astype(str) + "-" + adata.obs["disease"].astype(str)

for index, row in pyscenic_assign_dict.iterrows():
    print(index)
    
    filename = row[0]
    subsets_list = [subset.strip() for subset in row[1].split(',')]
    mode = row[2]
    
    if mode=="disease":
        use_col = "tmp"
    elif mode=="Subset_Identity":
        use_col = "Subset_Identity"
    else:
        raise ValueError(f"Unknown mode value: {mode}")
    
    existing_subsets = set(adata.obs["Subset_Identity"].unique())
    missing_subsets = [s for s in subsets_list if s not in existing_subsets]
    if missing_subsets:
        print(f"Warning: following subsets: {missing_subsets} is/are not found in adata.obs, will skip")
        continue
    
    adata_subset = adata[adata.obs["Subset_Identity"].isin(subsets_list)]
    adata_subset = obs_key_wise_subsampling(bdata = adata_subset,
                                            obs_key = use_col,
                                            downsample = 2000) # 最大保留2000个细胞/每个亚群
    print(adata_subset.shape)
    save_addr = "/data/HeLab/bio/IBD_analysis/output/Step09_PYSCENIC/Pyscenic_0621"
    write_scenic_input(adata_subset = adata_subset,
                       save_addr = save_addr,
                       use_col = use_col,
                       file_name = row[0])

