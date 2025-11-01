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



adata = anndata.read_h5ad("/data/HeLab/bio/IBD_analysis/tmp/Step12_cleansed(updated0101_clustered).h5ad")
obs_data = pd.read_pickle("/data/HeLab/bio/IBD_analysis/tmp/Step12_cleansed(updated0101_clustered)_obs.pkl")

obs_data
del adata.obs
adata.obs = obs_data

# PYSCENIC拆分的依据：pyscenic_assign_dict，包含两列name与groups，用来将细胞按照大类进行拆分，避免文件过大运行pyscenic困难
pyscenic_assign_dict = pd.ExcelFile("/data/HeLab/bio/IBD_analysis/assets/PYSCENIC_assignment(250417).xlsx")
pyscenic_assign_dict = pyscenic_assign_dict.parse(pyscenic_assign_dict.sheet_names[0])



for index, row in pyscenic_assign_dict.iterrows():
    print(index)
    subsets = [subset.strip() for subset in row[1].split(',')]
    print(subsets)
    adata_subset = adata[adata.obs["Subset_Identity"].isin(subsets)]
    adata_subset = obs_key_wise_subsampling(bdata = adata_subset,obs_key = "Subset_Identity", scale = 3000)
    print(adata_subset.shape)
    save_addr = "/data/HeLab/bio/IBD_analysis/output/Step13_PYSCENIC/tmp_250417"
    write_scenic_input(adata_subset,save_addr,row[0])
    

'''
conda activate pyscenic

pyscenic grn \
--num_workers 8 \
--output Adjacencies.tsv \
--method grnboost2 \
matrix.loom \
/data/HeLab/bio/biosoftware/pyscenic/allTFs_hg38.txt

pyscenic ctx \
        Adjacencies.tsv \
        /data/HeLab/bio/biosoftware/pyscenic/hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather \
        /data/HeLab/bio/biosoftware/pyscenic/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather \
        --annotations_fname /data/HeLab/bio/biosoftware/pyscenic/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl \
        --expression_mtx_fname matrix.loom \
        --mode "dask_multiprocessing" \
        --output Regulons.csv \
        --num_workers 8

pyscenic aucell \
        matrix.loom \
        Regulons.csv \
        -o Output.loom \
        --num_workers 8 \
        > pyscenic_aucell.log
'''





