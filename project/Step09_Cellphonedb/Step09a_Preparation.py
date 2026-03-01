# *conda activate cellphonedb*
"""
Step08a_Cellphonedb_process.py
Author: John Hsiung
Update date: 2026-02-26
Description:
    - 进行 Cellphonedb 的运行
        - 1203 基于 DEG，1204 基于 statistical
Notes:
    - 依赖环境: conda activate scvpy10
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
os.chdir("/public/home/xiongyuehan/data/IBD_analysis/output/Step08_Cellphonedb")

# save_addr="/public/home/xiongyuehan/data/IBD_analysis/output/Step08_Cellphonedb/251203"
# save_addr="/public/home/xiongyuehan/data/IBD_analysis/output/Step08_Cellphonedb/251204"
# save_addr="/public/home/xiongyuehan/data/IBD_analysis/output/Step08_Cellphonedb/260106"
save_addr = "/public/home/xiongyuehan/data/IBD_analysis/output/Step08_Cellphonedb/260125"

os.makedirs(save_addr, exist_ok=True)
##——————————————————————————————————————————————————————————————————————————
# 确定软件位置
# cpdb_file_path = '/data/HeLab/bio/biosoftware/cpdb/cellphonedb-data-5.0.0/cellphonedb.zip'
# cpdb_file_path = '/public/home/xiongyuehan/data/IBD_analysis/assets/cellphonedb-data-5.0.0/cellphonedb.zip'
##——————————————————————————————————————————————————————————————————————————
# 更新自定义数据库
cpdb_input_dir = '/public/home/xiongyuehan/data/IBD_analysis/assets/cellphonedb-data-5.0.0/data'
os.listdir(cpdb_input_dir)
from cellphonedb.utils import db_utils

# -- Creates new database
db_utils.create_db(cpdb_input_dir)
##——————————————————————————————————————————————————————————————————————————
# 后续：使用新的自定义数据库
cpdb_file_path = '/public/home/xiongyuehan/data/IBD_analysis/assets/cellphonedb-data-5.0.0/data/cellphonedb_01_10_2026_125124.zip'
##——————————————————————————————————————————————————————————————————————————
adata = anndata.read_h5ad(
    "/public/home/xiongyuehan/data/IBD_analysis/output/Step07/Step07a_Summary/Step07_DR_clustered_clean_20260108.h5ad")

print(adata.shape)

gc.collect()
# adata.write_h5ad("/data/HeLab/bio/IBD_analysis/output/Step08_Cellphonedb/Step08_0625.h5ad")
adata.write_h5ad(f"{save_addr}/Step08_for_CPDB_0125.h5ad")
gc.collect()
##——————————————————————————————————————————————————————————————————————————
# 2）进行 DEG 运算
from src.core.adata import subcluster, easy_DEG, obs_keywise_downsample
from cellphonedb.src.core.methods import cpdb_degs_analysis_method

##——————————————————————————————————————————————————————————————————————————
adata = anndata.read_h5ad(f"{save_addr}/Step08_for_CPDB.h5ad")
gc.collect()

for disease in adata.obs.disease.unique():
    print("\n" + "=" * 80)
    print(f"💠 Processing disease: {disease}")
    print("=" * 80)
    
    data_path = f"{save_addr}/input"
    
    subsetfile_path = f"{data_path}/{disease}/Subset.h5ad"
    counts_file_path = f"{data_path}/{disease}/counts.h5ad"
    meta_file_path = f"{data_path}/{disease}/metadata.tsv"
    DEG_path = f"{data_path}/{disease}/DEG.txt"
    
    output_path = f"{save_addr}/output/{disease}/"
    
    print(f"subsetfile_path: {subsetfile_path}")
    print(f"counts_file_path: {counts_file_path}")
    print(f"meta_file_path:   {meta_file_path}")
    print(f"DEG_path:         {DEG_path}")
    print(f"output_path:      {output_path}")
    
    os.makedirs(f"{data_path}/{disease}", exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    # ======================================================================
    # Subset by disease
    # ======================================================================
    print("\nSubsetting adata by disease…")
    adata_subset = adata[adata.obs.disease == disease].copy()
    print(f"   → Subset size: {adata_subset.n_obs} cells, {adata_subset.n_vars} genes")
    
    # Remove small groups
    print("\nRemoving rare Subset_Identity groups (<30 cells)…")
    df = adata_subset.obs.Subset_Identity.value_counts()
    print("   → Original group sizes:")
    print(df)
    
    index_2_remove = df.index[df < 30]
    if len(index_2_remove) > 0:
        print(f"   → Removing groups: {list(index_2_remove)}")
    else:
        print("   → No groups removed.")
    
    adata_subset = adata_subset[~adata_subset.obs.Subset_Identity.isin(index_2_remove)].copy()
    print(f"   → Remaining cells: {adata_subset.n_obs}")
    
    # downsample
    adata_subset = obs_keywise_downsample(adata_subset, obs_key="Subset_Identity", downsample=8000)
    
    # ======================================================================
    # Save meta
    # ======================================================================
    print("\nSaving metadata…")
    meta_file = pd.DataFrame({
        'Cell': adata_subset.obs.index,
        'cell_type': adata_subset.obs["Subset_Identity"]
    })
    meta_file.to_csv(meta_file_path, index=False, sep="\t")
    print("   Meta written.")
    
    # ======================================================================
    # Save raw counts
    # ======================================================================
    print("\nSaving raw X from adata_subset.raw…")
    X = adata_subset.raw.X
    var = adata_subset.raw.var.copy()
    print(f"   → Raw matrix shape: {X.shape}")
    
    adata_out = sc.AnnData(
        X=X,
        obs=adata_subset.obs.copy(),
        var=var
    )
    adata_out.write_h5ad(counts_file_path)
    print(f"   Raw adata written: {counts_file_path}")
    
    # ======================================================================
    # Run easy_DEG
    # ======================================================================
    print("\nRunning easy_DEG()…")
    try:
        adata_subset = easy_DEG(
            adata_subset,
            save_addr=f"{data_path}/{disease}/",
            filename_prefix="DEG_for_CPDB",
            use_raw=False,
            obs_key="Subset_Identity",
            save_plot=True,
            plot_gene_num=5
        )
        adata_subset.write_h5ad(subsetfile_path)
        print(f"   easy_DEG finished, subset saved → {subsetfile_path}")
    
    except Exception as e:
        print(f"Warning:[{disease}] easy_DEG failed: {e}")
        continue
    
    # ======================================================================
    # Collect DEG results
    # ======================================================================
    print("\n📊 Getting DEG results…")
    groups = adata_subset.obs["Subset_Identity"].unique().tolist()
    print(f"   → Groups for DEG: {groups}")
    
    df = sc.get.rank_genes_groups_df(
        adata_subset,
        group=groups,
        key="deg_Subset_Identity"
    )
    print(f"   → DEG total rows: {df.shape}")
    
    df = df[(df['scores'] > 0) & (df['pvals_adj'] < 0.01)]
    print(f"   → Filtered DEG rows: {df.shape}")
    
    df = df.loc[:, ['group', 'names']]
    df.to_csv(DEG_path, sep='\t', index=False)
    print(f"   DEG list saved → {DEG_path}")
    print(df)
    
    # ======================================================================
    # Run CPDB
    # ======================================================================
    print("\n🔬 Running CellPhoneDB (CPDB)…")
    print(f"""
        Input:
        - meta:    {meta_file_path}
        - counts:  {counts_file_path}
        - DEG:     {DEG_path}
        - output:  {output_path}
    """)
    
    cpdb_results = cpdb_degs_analysis_method.call(
        cpdb_file_path=cpdb_file_path,
        meta_file_path=meta_file_path,
        counts_file_path=counts_file_path,
        degs_file_path=DEG_path,
        counts_data='hgnc_symbol',
        score_interactions=True,
        threshold=0.2,
        threads=8,
        result_precision=3,
        separator='|',
        debug=False,
        output_path=output_path,
        output_suffix=None
    )
    print("   CPDB finished.")
    
    gc.collect()
    print("GC collected.\n")

# 检查一下数据
test_dir = "/public/home/xiongyuehan/data/IBD_analysis/output/Step08_Cellphonedb/251204/Step08_for_CPDB.h5ad"
adata_test = anndata.read_h5ad(test_dir)

test_meta = "/public/home/xiongyuehan/data/IBD_analysis/output/Step08_Cellphonedb/251203/input/Colitis/metadata.tsv"
meta_df = pd.read_csv(test_meta, sep="\t")  # sep="\t" 表示制表符分隔

##——————————————————————————————————————————————————————————————————————————
# 3）进行 statistical 运算
from cellphonedb.src.core.methods import cpdb_statistical_analysis_method


for disease in ["HC", "UC", "BD", "CD", "Colitis"]:
    print("\n" + "=" * 80)
    print(f"💠 Processing disease: {disease}")
    print("=" * 80)
    
    data_path = "/public/home/xiongyuehan/data/IBD_analysis/output/Step08_Cellphonedb/260125/input"
    
    subsetfile_path = f"{data_path}/{disease}/Subset.h5ad"
    counts_file_path = f"{data_path}/{disease}/counts.h5ad"
    meta_file_path = f"{data_path}/{disease}/metadata.tsv"
    
    output_path = f"{save_addr}/output_thr20percent/{disease}/"
    
    print(f"subsetfile_path: {subsetfile_path}")
    print(f"counts_file_path: {counts_file_path}")
    print(f"meta_file_path:   {meta_file_path}")
    print(f"output_path:      {output_path}")
    
    os.makedirs(f"{data_path}/{disease}", exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    # ======================================================================
    # Run CPDB
    # ======================================================================
    print("\n🔬 Running CellPhoneDB (CPDB)…")
    print(f"""
        Input:
        - meta:    {meta_file_path}
        - counts:  {counts_file_path}
        - output:  {output_path}
    """)
    
    cpdb_results = cpdb_statistical_analysis_method.call(
        cpdb_file_path=cpdb_file_path,
        meta_file_path=meta_file_path,
        counts_file_path=counts_file_path,
        counts_data='hgnc_symbol',
        score_interactions=True,
        threshold=0.2,
        threads=4,
        result_precision=3,
        separator='|',
        debug=False,
        output_path=output_path,
        output_suffix=None
    )
    print("   CPDB finished.")
    
    gc.collect()
    print("GC collected.\n")

# 检查结果
for disease in ["HC", "UC", "BD", "CD", "Colitis"]:
    print(disease)
    test = pd.read_csv(
        f"/public/home/xiongyuehan/data/IBD_analysis/output/Step08_Cellphonedb/260106/input/{disease}/metadata.tsv",
        sep='\t')
    print(test.cell_type.unique())
    print()
