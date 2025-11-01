import scvelo as scv

import anndata
import pandas as pd
import os

##——————————————————————————————————————————————————————————————————————————
# 设置scanpy基本属性
import yaml

with open('/data/HeLab/bio/IBD_analysis/assets/io_config.yaml', 'r') as f:
    config = yaml.safe_load(f) # 读取的格式为一系列字典的列表
##——————————————————————————————————————————————————————————————————————————
os.chdir("/data/HeLab/bio/IBD_analysis/output/Step13_Scvelo")


##################################
# 原则上我们的细胞数量大致削减了2万余，而新增保留应当较少，因此直接使用之前的数据视情况进行对齐和采样
# 先前运算结果的注释和解释
# [1]
# /data/HeLab/bio/IBD_plus/XY
# /data/HeLab/bio/IBD_plus/HRA000072
# /data/HeLab/bio/IBD_plus/GSE225199
# /data/HeLab/bio/IBD_plus/GSE116222/outs
# {以上地址/样本run文件/}下包含“outs”文件夹和"velocyto"文件夹；
# 输入文件：outs文件夹内为cellranger的直接输出；输出文件：velocyto文件夹中包含“.loom”文件
# [2]
# /data/HeLab/bio/IBD_plus/13_velocyto/Merge_w_Splice/13_XY_subset_merge.h5ad
# /data/HeLab/bio/IBD_plus/13_velocyto/Merge_w_Splice/13_HRA00072_subset_merge.h5ad
# /data/HeLab/bio/IBD_plus/13_velocyto/Merge_w_Splice/13_GSE225199_subset_merge.h5ad
# /data/HeLab/bio/IBD_plus/13_velocyto/Merge_w_Splice/13_GSE116222_subset_merge.h5ad
# 包含（主要）经过scv.utils.clean_obs_names和scv.utils.merge，和预处理之后的adata文件合并了的spliced/unspliced结果
# [3]
# /data/HeLab/bio/IBD_plus/13_velocyto/13_combined_velocyto.h5ad
# 直接用anndata.concat将上述subset_merge合并起来的结果
# 我们基本上从这里开始
adata_new = anndata.read_h5ad("/data/HeLab/bio/IBD_analysis/tmp/Step12_cleansed(updated0101_clustered).h5ad")
obs_data = pd.read_pickle("/data/HeLab/bio/IBD_analysis/tmp/Step12_cleansed(updated0101_clustered)_obs.pkl")

obs_data
del adata_new.obs
adata_new.obs = obs_data
adata_new.write("/data/HeLab/bio/IBD_analysis/tmp/Step12_cleansed(updated0101_clustered).h5ad")
adata_new.obs["disease"]


adata_velo = anndata.read_h5ad("/data/HeLab/bio/IBD_plus/13_velocyto/13_combined_velocyto.h5ad")

##
# 对比两个文件的cell_barcode
obs_names_new = adata_new.obs_names
obs_names_velo = adata_velo.obs_names

def analyze_cb(adata_obs_names1,adata_obs_names2):
    # 找出重复的元素（交集）
    common_cells = adata_obs_names1.intersection(adata_obs_names2)
    # 找出 index1 中不在 index2 的元素（差集）
    unique_to_index1 = adata_obs_names1.difference(adata_obs_names2)
    # 找出 index2 中不在 index1 的元素（差集）
    unique_to_index2 = adata_obs_names2.difference(adata_obs_names1)
    
    # 统计结果
    print(f"重复的元素数量: {len(common_cells)}") # 重复的元素数量: 625569
    print(f"仅在 index1 中的元素数量: {len(unique_to_index1)}") # 仅在 index1 中的元素数量: 13597
    print(f"仅在 index2 中的元素数量: {len(unique_to_index2)}") # 9405
    
    return unique_to_index1,unique_to_index2

# 分别进行处理
# 对重复的元素，我们希望进一步确认一下orig.project和是否一致，可以作为一个一致性的估计

if False:
    adata_new.obs[common_cells[0]]
    matched = []
    unmatched = []
    
    for cb in common_cells:
        if adata_velo.obs.loc[cb, "orig.project"] == adata_new.obs.loc[cb, "orig.project"] and adata_velo.obs.loc[
            cb, "phase"] == adata_new.obs.loc[cb, "phase"]:
            matched.append(cb)
        else:
            unmatched.append(cb)
    
    print(len(matched))  # 625569
    print(len(unmatched))  # 0
    
    adata_new_spec = adata_new[unique_to_index1,]
    adata_velo_spec = adata_velo[unique_to_index2,]
    
    # 进行一些对比
    adata_new_spec[adata_new_spec.obs["orig.ident"] == "GSM7041333_Peri_P19_ctrl"]
    adata_velo_spec[adata_velo_spec.obs["orig.ident"] == "GSM7041333"]
    
    adata_new_spec.obs["orig.ident"] = [orii.split("_")[0] for orii in adata_new_spec.obs["orig.ident"]]
    
    list1 = adata_new_spec.obs["orig.ident"].unique().tolist()
    list2 = adata_velo_spec.obs["orig.ident"].unique().tolist()
    
    for i in set(list1).intersection(set(list2)):
        index1, index2 = analyze_cb(adata_new_spec[adata_new_spec.obs["orig.ident"] == i].obs_names,
                                    adata_velo_spec[adata_velo_spec.obs["orig.ident"] == i].obs_names)
# 完成了评估，总而言之不适合进一步合并，那么问题就简单了
adata_new = adata_new[common_cells]
adata_new.obs = adata_new.obs[['orig.ident', 'phase','orig.project', 'filtered_celltype_annotation', 'disease', 'Subset_Identity', 'Cell_type', 'Immune']]
adata_new.var = adata_velo.var

gene_list1 = adata_new.var_names
gene_list2 = adata_velo.var_names
share_genes = gene_list1.intersection(gene_list2)
diff_genes1 = gene_list1.difference(gene_list2)
diff_genes2 = gene_list2.difference(gene_list1)

adata_new = adata_new[:, adata_new.var_names.isin(share_genes)].copy()


adata_new.layers["spliced"] = adata_velo.layers["spliced"]
adata_new.layers["unspliced"] = adata_velo.layers["unspliced"]
adata_new.layers["matrix"] = adata_velo.layers["matrix"]

adata_new.var["Accession"] = adata_velo.var["Accession"]
adata_new.var["Chromosome"] = adata_velo.var["Chromosome"]
adata_new.var["End"] = adata_velo.var["End"]
adata_new.var["Start"] = adata_velo.var["Start"]
adata_new.var["Strand"] = adata_velo.var["Strand"]

adata_new.obs["initial_size_unspliced"] = adata_velo.obs["initial_size_unspliced"]
adata_new.obs["initial_size_spliced"] = adata_velo.obs["initial_size_spliced"]
adata_new.obs["initial_size"] = adata_velo.obs["initial_size"]

del adata_velo
import gc
gc.collect()
adata_new.write("/data/HeLab/bio/IBD_analysis/tmp/Step13_Combine(0113).h5ad")
####################################################################################
####################################################################################
####################################################################################
import sys,gc
sys.path.append('/data/HeLab/bio/IBD_analysis/')
from src.ScanpyTools.ScanpyTools import obs_key_wise_subsampling
####################################################################################
adata = anndata.read_h5ad("/data/HeLab/bio/IBD_analysis/tmp/Step13_Combine(0113).h5ad")

adata.obs["Subset_Identity"].value_counts()
adata = obs_key_wise_subsampling(adata,"Subset_Identity",10000)
adata.write("/data/HeLab/bio/IBD_analysis/tmp/Step13_01_Downsample(0113).h5ad")
gc.collect()
##################
scv.pp.filter_and_normalize(adata, min_shared_counts = 200, n_top_genes = 6000)
scv.pp.moments(adata, n_pcs=30, n_neighbors=50)
gc.collect()
adata.write("/data/HeLab/bio/IBD_analysis/tmp/Step13_02_filter_mmts(0113).h5ad")
##################
scv.pl.proportions(adata,save="_scvelo_proportions.png")
scv.tl.velocity(adata,mode='stochastic')
scv.tl.velocity_graph(adata,n_jobs=12)
adata.write('/data/HeLab/bio/IBD_analysis/tmp/Step13_03_velo(0113).h5ad')


