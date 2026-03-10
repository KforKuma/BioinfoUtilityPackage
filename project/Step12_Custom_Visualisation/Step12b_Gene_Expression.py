import pandas as pd
import anndata
import os
import gc
import scanpy as sc
import sys


sys.stdout.reconfigure(encoding='utf-8')
####################################
sys.path.append('/public/home/xiongyuehan/data/BioinfoUtilityPackage')

from src.core.handlers.plot_wrapper import *

violin_plot = ScanpyPlotWrapper(sc.pl.violin)

####################################
parent_dir = "/public/home/xiongyuehan/data/IBD_analysis/output"

save_fig_dir = f"{parent_dir}/Step12_Custom_Vis/Gene_Expression"

os.makedirs(save_fig_dir, exist_ok=True)

adata = anndata.read_h5ad(
    "/public/home/xiongyuehan/data/IBD_analysis/output/Step07_Summary/Step07_DR_clustered_clean_20260210.h5ad")



adata_ss = adata[adata.obs["Subset_Identity"].isin(["Macrophage M1","Macrophage M2","Macrophage",
                                                    "Neutrophil CD16B+",
                                                    "cDC1 CLEC9A+","cDC2 CD1C+","pDC GZMB+"])]

violin_plot(adata=adata_ss,save_addr=save_fig_dir,groupby="disease",
            keys=["GNAI2","PIK3CG"],
            filename="C3a_C5a_Myeloid_Adaptor")

violin_plot(adata=adata_ss,save_addr=save_fig_dir,groupby="disease",
            keys=["TNFAIP3","NFKBIA"],
            filename="C3a_C5a_Myeloid_Flameout")
