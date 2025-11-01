# *conda activate cellphonedb*

import pandas as pd
import anndata
import os
import gc
import scanpy as sc
import sys
sys.path.append('/data/HeLab/bio/IBD_analysis/')

# from scripts.Step12b_temp_250101 import data_path

import src.EasyInterface.CPDBTools

os.chdir("/data/HeLab/bio/IBD_analysis/output/Step13_CPDB")

##——————————————————————————————————————————————————————————————————————————
# 1) adata数据预处理及拆分
from cellphonedb.src.core.methods import cpdb_statistical_analysis_method
from src.EasyInterface.CPDBTools import data_split
cpdb_file_path = '/data/HeLab/bio/biosoftware/cpdb/cellphonedb-data-5.0.0/cellphonedb.zip'

adata = anndata.read("/data/HeLab/bio/IBD_analysis/tmp/Step12c_CPDB(0101).h5ad")
exclude = adata[adata.obs["Cell_type"].isin(["Mitotic cell"])].obs_names
adata = adata[~adata.obs_names.isin(exclude), :]
adata.obs["Subset_Identity"].value_counts()
adata.write("/data/HeLab/bio/IBD_analysis/tmp/Step12c_CPDB(0101).h5ad")

diseases = ["Colitis", "CD", "UC", "BS", "Control"]
for disease in diseases:
    data_split(adata, disease, data_path="/data/HeLab/bio/IBD_analysis/output/Step13_CPDB")
    
# # 按照split_key的方法打包保存
# from src.EasyInterface.CPDBTools import filter_by_frequency
# adata = filter_by_frequency(data=adata, column="Subset_Identity", min_count=30)
# print(adata.shape)
# data_path="/data/HeLab/bio/IBD_analysis/output/Step13_CPDB"
# # Define file paths
# count_file_name = f"{data_path}/Combine/counts.h5ad"
# meta_file_name = f"{data_path}/Combine/metadata.tsv"
# # Create and save metadata
# adata.obs['combined'] = adata.obs['Subset_Identity'].str.cat(adata.obs['disease'], sep='_')
#
#
# meta_file = pd.DataFrame({
#     'Cell': adata.obs.index,
#     'cell_type': adata.obs["combined"]
# })
# meta_file.to_csv(meta_file_name, index=False, sep="\t")
#
# # Create a new AnnData object to avoid potential issues with .write() method
# adata = sc.AnnData(adata.X,
#                           obs=pd.DataFrame(index=adata.obs.index),
#                           var=pd.DataFrame(index=adata.var.index))
#
# # Check if the directory exists, if not, create it
# print(count_file_name)
#
# # Save the subset data
# adata.write(count_file_name)
##——————————————————————————————————————————————————————————————————————————
# 2）进行运算
from src.ScanpyTools.ScanpyTools import easy_DEG
from cellphonedb.src.core.methods import cpdb_degs_analysis_method
from itertools import compress
import src.EasyInterface.CPDBTools

diseases = ["Combine"] #"Colitis", "CD", "UC", "BS", "Control"
for disease in diseases:
    print(disease)
    data_path = "/data/HeLab/bio/IBD_analysis/output/Step13_CPDB"
    counts_file_path = f"{data_path}/{disease}/counts.h5ad"
    meta_file_path = f"{data_path}/{disease}/metadata.tsv"
    output_path = f"{data_path}/output/{disease}/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Directory {output_path} created.")
    cpdb_results = cpdb_statistical_analysis_method.call(
        cpdb_file_path=cpdb_file_path,  # mandatory: CellphoneDB database zip file.
        meta_file_path=meta_file_path,  # mandatory: tsv file defining barcodes to cell label.
        counts_file_path=counts_file_path,
        # mandatory: normalized count matrix - a path to the counts file, or an in-memory AnnData object
        counts_data='hgnc_symbol',  # defines the gene annotation in counts matrix.
        # active_tfs_file_path = active_tf_path,           # optional: defines cell types and their active TFs.
        # microenvs_file_path = microenvs_file_path,       # optional (default: None): defines cells per microenvironment.
        score_interactions=True,  # optional: whether to score interactions or not.
        iterations=1000,  # denotes the number of shufflings performed in the analysis.
        threshold=0.1,  # defines the min % of cells expressing a gene for this to be employed in the analysis.
        threads=4,  # number of threads to use in the analysis.
        debug_seed=42,  # debug randome seed. To disable >=0.
        result_precision=3,  # Sets the rounding for the mean values in significan_means.
        pvalue=0.05,  # P-value threshold to employ for significance.
        subsampling=True,  # To enable subsampling the data (geometri sketching).
        subsampling_log=False,  # (mandatory) enable subsampling log1p for non log-transformed data inputs.
        subsampling_num_pc=100,  # Number of componets to subsample via geometric skectching (dafault: 100).
        subsampling_num_cells=20000,  # Number of cells to subsample (integer) (default: 1/3 of the dataset).
        separator='|',  # Sets the string to employ to separate cells in the results dataframes "cellA|CellB".
        debug=False,  # Saves all intermediate tables employed during the analysis in pkl format.
        output_path=output_path,  # Path to save results.
        output_suffix=None
        # Replaces the timestamp in the output files by a user defined string in the  (default: None).
    )


for disease in ["Colitis", "CD", "UC", "BS", "Control"]:  #
    print(disease)
    data_path = "/data/HeLab/bio/IBD_analysis/output/Step13_CPDB"
    counts_file_path = f"{data_path}/{disease}/counts_DEG.h5ad"
    meta_file_path = f"{data_path}/{disease}/metadata_DEG.tsv"
    output_path = f"{data_path}/output_DEG/{disease}/"
    DEG_path = f"{data_path}/{disease}/DEG.txt"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Directory {output_path} created.")
    adata_subset = adata[adata.obs["disease"] == disease]
    if "Mitotic cell" in adata_subset.obs["Cell_type"].unique():
        exclude = adata_subset[adata_subset.obs["Cell_type"].isin(["Mitotic cell"])].obs_names
        adata_subset = adata_subset[~adata_subset.obs_names.isin(exclude), :]
    
    keep = list(compress(adata_subset.obs["Subset_Identity"].value_counts().index.tolist(),
                         list(adata_subset.obs["Subset_Identity"].value_counts() > 30)))
    adata_subset = adata_subset[adata_subset.obs["Subset_Identity"].isin(keep)]
    
    adata_subset = easy_DEG(adata_subset, save_addr=f"{data_path}/{disease}/", filename="DEG_for_CPDB",
                            obs_key="Subset_Identity", save_plot=True, plot_gene_num=5, downsample=0.5)
    df = sc.get.rank_genes_groups_df(adata_subset, group=adata_subset.obs["Subset_Identity"].unique().tolist(),
                                     key="hvg_Subset_Identity")
    # df = pd.read_excel(f"{data_path}/{disease}/HVG_wilcoxon(EN)_file_name_Subset_Identity.xlsx",
    #                    sheet_name='Original', usecols=['group', 'names', 'scores', 'logfoldchanges', 'pvals','pvals_adj'],
    #                    skiprows=1)
    df = df[df['scores'] > 0];
    df = df[df['pvals_adj'] < 0.005];
    df = df.loc[:, ['group',
                    'names']]  # It is a .txt with two columns: the first column should be the cell type name and the second column the associated significant gene id.
    df.to_csv(DEG_path, sep='\t', index=False)
    print(df)
    print("HVG written.")
    print(counts_file_path)
    adata_subset.write(counts_file_path)
    print("Counts file written.")
    meta_file = pd.DataFrame({'Cell': adata_subset.obs.index, 'cell_type': adata_subset.obs["Subset_Identity"]})
    meta_file.to_csv(meta_file_path, index=False, sep="\t")
    print("Meta file written.")
    cpdb_results = cpdb_degs_analysis_method.call(
        cpdb_file_path=cpdb_file_path,  # mandatory: CellphoneDB database zip file.
        meta_file_path=meta_file_path,  # mandatory: tsv file defining barcodes to cell label.
        counts_file_path=counts_file_path,
        degs_file_path=DEG_path,
        counts_data='hgnc_symbol',  # defines the gene annotation in counts matrix.
        score_interactions=True,  # optional: whether to score interactions or not.
        threshold=0.1,  # defines the min % of cells expressing a gene for this to be employed in the analysis.
        threads=5,  # number of threads to use in the analysis.
        result_precision=3,  # Sets the rounding for the mean values in significan_means.
        separator='|',  # Sets the string to employ to separate cells in the results dataframes "cellA|CellB".
        debug=False,  # Saves all intermediate tables employed during the analysis in pkl format.
        output_path=output_path,  # Path to save results.
        output_suffix=None
    )
    gc.collect()


# adata = anndata.read_h5ad("/data/HeLab/bio/IBD_analysis/tmp/Step12c_CPDB(0101).h5ad")
# data_path = "/data/HeLab/bio/IBD_analysis/output/Step13_CPDB"
# disease="Combine"
# counts_file_path = f"{data_path}/{disease}/counts_DEG.h5ad"
# meta_file_path = f"{data_path}/{disease}/metadata_DEG.tsv"
# output_path = f"{data_path}/output_DEG/{disease}/"
# DEG_path = f"{data_path}/{disease}/DEG.txt"
# keep = list(compress(adata.obs["Subset_Identity"].value_counts().index.tolist(),
#                      list(adata.obs["Subset_Identity"].value_counts() > 30)))
# adata = adata[adata.obs["Subset_Identity"].isin(keep)]
#
# adata = easy_DEG(adata, save_addr=f"{data_path}/{disease}/", filename="DEG_for_CPDB",
#                  obs_key="Subset_Identity", save_plot=True, plot_gene_num=5, obs_subset=0.5)
# df = sc.get.rank_genes_groups_df(adata, group=adata.obs["Subset_Identity"].unique().tolist(),
#                                      key="hvg_Subset_Identity")
# df = df[df['scores'] > 0];
# df = df[df['pvals_adj'] < 0.005];
# df = df.loc[:, ['group','names']]  # It is a .txt with two columns: the first column should be the cell type name and the second column the associated significant gene id.
# df.to_csv(DEG_path, sep='\t', index=False)
# print(df)
# print("HVG written.")
# print(counts_file_path)
# adata.write(counts_file_path)
# print("Counts file written.")
# meta_file = pd.DataFrame({'Cell': adata.obs.index, 'cell_type': adata.obs["Subset_Identity"]})
# meta_file.to_csv(meta_file_path, index=False, sep="\t")
# print("Meta file written.")
# cpdb_results = cpdb_degs_analysis_method.call(
#     cpdb_file_path=cpdb_file_path,  # mandatory: CellphoneDB database zip file.
#     meta_file_path=meta_file_path,  # mandatory: tsv file defining barcodes to cell label.
#     counts_file_path=counts_file_path,
#     degs_file_path=DEG_path,
#     # mandatory: normalized count matrix - a path to the counts file, or an in-memory AnnData object
#     counts_data='hgnc_symbol',  # defines the gene annotation in counts matrix.
#     # active_tfs_file_path = active_tf_path,           # optional: defines cell types and their active TFs.
#     # microenvs_file_path = microenvs_file_path,       # optional (default: None): defines cells per microenvironment.
#     score_interactions=True,  # optional: whether to score interactions or not.
#     threshold=0.1,  # defines the min % of cells expressing a gene for this to be employed in the analysis.
#     threads=5,  # number of threads to use in the analysis.
#     result_precision=3,  # Sets the rounding for the mean values in significan_means.
#     separator='|',  # Sets the string to employ to separate cells in the results dataframes "cellA|CellB".
#     debug=False,  # Saves all intermediate tables employed during the analysis in pkl format.
#     output_path=output_path,  # Path to save results.
#     output_suffix=None
#     # Replaces the timestamp in the output files by a user defined string in the  (default: None).
# )
##——————————————————————————————————————————————————————————————————————————
# 3）结果处理
# 3.1 基础绘图
import ktplotspy as kpy
from src.ScanpyTools.ScanpyTools import ScanpyPlotWrapper
kpy_heatmap = ScanpyPlotWrapper(func = kpy.plot_cpdb_heatmap)

save_addr = "/data/HeLab/bio/IBD_analysis/output/Step13_CPDB/output"
for file in ["BS"]: #"Colitis", "CD", "UC", "Control"
    file_dir=f"{save_addr}/{file}/"
    output_filename=f"{file}_kpy_heatmap(symm)"
    pattern="analysis_pvalues|analysis_relevant_interactions"
    file_path = find_file(file_dir, pattern)
    pvals = pd.read_table(file_path, delimiter='\t')
    kpy_heatmap(save_addr=save_addr,filename=output_filename,
                pvals=pvals,
                degs_analysis=False, # DEG时必须打开！
                figsize=(12, 12),
                title="Sum of significant interactions",symmetrical=True)

save_addr = "/data/HeLab/bio/IBD_analysis/output/Step13_CPDB/output_DEG"
for file in ["Colitis", "CD", "UC", "Control"]:
    file_dir=f"{save_addr}/{file}/"
    output_filename=f"{file}_kpy_heatmap(symm)"
    pattern="analysis_pvalues|analysis_relevant_interactions"
    file_path = find_file(file_dir, pattern)
    pvals = pd.read_table(file_path, delimiter='\t')
    kpy_heatmap(save_addr=save_addr,filename=output_filename,
                pvals=pvals,
                degs_analysis=True, # DEG时必须打开！
                figsize=(12, 12),
                title="Sum of significant interactions",symmetrical=True)

##——————————————————————————————————————————————————————————————————————————
# 3.2 表格读取
del extract_cpdb_table
import importlib
importlib.reload(src.EasyInterface.CPDBTools)

from src.EasyInterface.CPDBTools import extract_cpdb_result,extract_cpdb_table
save_addr = "/data/HeLab/bio/IBD_analysis/output/Step13_CPDB/output"


for type in ["Colitis","CD", "UC", "Control","BS"]:#
    adata_test = adata[adata.obs["disease"] == type]
    file_dir = f"{save_addr}/{type}/"
    cpdb_results = extract_cpdb_result(file_dir)
    all = extract_cpdb_table(adata=adata_test,
                             cpdb_outcome_dict=cpdb_results,
                             celltype_key="Subset_Identity",  # 数据&信息行
                             additional_grouping=False, lock_celltype_direction=True,  # 核心功能：加速 && 锁定
                             cell_type1=".", cell_type2=".",  # 方向
                             genes=[],  # 基因（列表行）
                             cluster_rows=False,
                             keep_significant_only=True,
                             debug=False
                             )
    all.to_csv(f"/data/HeLab/bio/IBD_analysis/output/Step13_CPDB/output/{type}_Allsig.csv")
    del all
    gc.collect()



adata_test = adata[adata.obs["disease"] == type]
test1 = f"/data/HeLab/bio/IBD_analysis/output/Step13_CPDB/output/{type}/"
cpdb_results = extract_cpdb_result(test1)
all = extract_cpdb_table(adata=adata_test,
                         cpdb_outcome_dict=cpdb_results,
                         celltype_key="Subset_Identity",  # 数据&信息行
                         additional_grouping=False, lock_celltype_direction=True,  # 核心功能：加速 && 锁定
                         cell_type1=".", cell_type2=".",  # 方向
                         genes=[],  # 基因（列表行）
                         cluster_rows=False,
                         keep_significant_only=True)
all.to_csv(f"/data/HeLab/bio/IBD_analysis/output/Step13_CPDB/output/{type}/Allsig.csv")
del all
gc.collect()


# 测试基于splitby_key的绘图