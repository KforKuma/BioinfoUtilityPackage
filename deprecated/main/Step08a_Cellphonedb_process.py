# *conda activate cellphonedb*
"""
Step08a_Cellphonedb_process.py
Author: John Hsiung
Update date: 2025-08-29
Description:
    - 进行 Cellphonedb 的运行
Notes:
    - 依赖环境: conda activate scvpy10
"""
import pandas as pd
import anndata
import os
import gc
import scanpy as sc
import sys
sys.path.append('/data/HeLab/bio/IBD_analysis/')
# import src.EasyInterface.CPDBTools
import sys
sys.stdout.reconfigure(encoding='utf-8')

os.chdir("/data/HeLab/bio/IBD_analysis/output/Step08_Cellphonedb")

##——————————————————————————————————————————————————————————————————————————
# 1) adata数据预处理及拆分
from cellphonedb.src.core.methods import cpdb_statistical_analysis_method
from src.EasyInterface.CPDBTools import data_split
cpdb_file_path = '/data/HeLab/bio/biosoftware/cpdb/cellphonedb-data-5.0.0/cellphonedb.zip'
##——————————————————————————————————————————————————————————————————————————
adata = anndata.read("/data/HeLab/bio/IBD_analysis/output/Step07/Step07_finalversion_5.h5ad")
exclude = adata[adata.obs["Celltype"].isin(["Mitotic"])].obs_names
adata = adata[~adata.obs_names.isin(exclude), :]
adata.write_h5ad("/data/HeLab/bio/IBD_analysis/output/Step08_Cellphonedb/Step08_0625.h5ad")


diseases = adata.obs["disease"].unique().tolist()
for disease in diseases:
    data_split(adata, disease=disease,
               data_path="/data/HeLab/bio/IBD_analysis/output/Step08_Cellphonedb/cellphonedb_input_0625",
               downsample_by_key="Cell_Subtype", downsample=True,use_raw=True)


##——————————————————————————————————————————————————————————————————————————
# 2）进行运算
from src.ScanpyTools.ScanpyTools import easy_DEG
from cellphonedb.src.core.methods import cpdb_degs_analysis_method
from itertools import compress
import src.EasyInterface.CPDBTools
##——————————————————————————————————————————————————————————————————————————
for disease in diseases:  #
    print(disease)
    data_path = "/data/HeLab/bio/IBD_analysis/output/Step08_Cellphonedb/cellphonedb_input_0625"
    counts_file_path = f"{data_path}/{disease}/counts.h5ad"
    meta_file_path = f"{data_path}/{disease}/metadata.tsv"
    output_path = f"/data/HeLab/bio/IBD_analysis/output/Step08_Cellphonedb/output_DEG_0625/{disease}/"
    DEG_path = f"{data_path}/{disease}/DEG.txt"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Directory {output_path} created.")
    adata_subset = anndata.read_h5ad(counts_file_path)
    
    adata_subset = easy_DEG(adata_subset, save_addr=f"{data_path}/{disease}/",
                            filename="DEG_for_CPDB",
                            use_raw=False,
                            obs_key="Subset_Identity", save_plot=True, plot_gene_num=5, downsample=0.5)
    df = sc.get.rank_genes_groups_df(adata_subset, group=adata_subset.obs["Subset_Identity"].unique().tolist(),
                                     key="hvg_Subset_Identity")
    df = df[df['scores'] > 0];
    df = df[df['pvals_adj'] < 0.005];
    df = df.loc[:, ['group',
                    'names']]  # It is a .txt with two columns: the first column should be the cell type name and the second column the associated significant gene id.
    df.to_csv(DEG_path, sep='\t', index=False)
    print(df)
    print("HVG written.")
    print(counts_file_path)
    cpdb_results = cpdb_degs_analysis_method.call(
        cpdb_file_path=cpdb_file_path,  # mandatory: CellphoneDB database zip file.
        meta_file_path=meta_file_path,  # mandatory: tsv file defining barcodes to cell label.
        counts_file_path=counts_file_path,
        degs_file_path=DEG_path,
        counts_data='hgnc_symbol',  # defines the gene annotation in counts matrix.
        score_interactions=True,  # optional: whether to score interactions or not.
        threshold=0.2,  # defines the min % of cells expressing a gene for this to be employed in the analysis.
        threads=8,  # number of threads to use in the analysis.
        result_precision=3,  # Sets the rounding for the mean values in significan_means.
        separator='|',  # Sets the string to employ to separate cells in the results dataframes "cellA|CellB".
        debug=False,  # Saves all intermediate tables employed during the analysis in pkl format.
        output_path=output_path,  # Path to save results.
        output_suffix=None
    )
    gc.collect()
