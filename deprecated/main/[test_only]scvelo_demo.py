"""
[test_only]scvelo_demo.py
Author: John Hsiung
Update date: 2025-09-08
Description:
    - 测试 scvelo 能否正常使用
Notes:
    - 使用环境：conda activate scvelo
    # 如果不能运行，在控制台输入export LD_PRELOAD=$CONDA_PREFIX/lib/libgomp.so.1
"""
###################################################

import scanpy as sc
import scvelo as scv
import pandas as pd
import anndata
scv.set_figure_params()

adata = anndata.read_h5ad("/data/HeLab/bio/IBD_analysis/assets/scvelo-pancreas.h5ad")

scv.pp.filter_genes(adata, min_shared_counts=20)
scv.pp.normalize_per_cell(adata)
scv.pp.filter_genes_dispersion(adata, n_top_genes=2000)
scv.pp.log1p(adata)

scv.tl.velocity(adata)
scv.tl.velocity_graph(adata)
scv.pl.velocity_embedding_stream(adata, basis='umap')
scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi=120)
scv.pl.velocity(adata, ['Cpe',  'Gnao1', 'Ins2', 'Adk'], ncols=2)

scv.tl.rank_velocity_genes(adata, groupby='clusters', min_corr=.3)
df = pd.DataFrame(adata.uns['rank_velocity_genes']['names'])
df.head()


