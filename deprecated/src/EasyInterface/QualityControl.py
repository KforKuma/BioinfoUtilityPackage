import scanpy as sc
import pandas as pd
import os
# import matplotlib.pyplot as plt
from src.ScanpyTools.Scanpy_Plot import ScanpyPlotWrapper


umap_plot = ScanpyPlotWrapper(func = sc.pl.umap)
def Basic_QC_Plot(adata, prefixx, out_dir=None):
    if out_dir is None:
        out_dir = os.getcwd()+"/fig/"
    elif not out_dir.endswith("/"):
        out_dir = out_dir + "/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    umap_plot(save_addr=out_dir,filename=prefixx + "_UMAP_origs",
              adata=adata, color=["orig.project","phase"])
    umap_plot(save_addr=out_dir, filename=prefixx + "_UMAP_organelles",
              adata=adata, color=["percent.ribo", "percent.mt"])
    umap_plot(save_addr=out_dir, filename=prefixx + "_UMAP_counts",
              adata=adata, color=["disease", "tissue-type", "tissue-origin"])


def remove_genes(adata, ribo_gene=True, mito_gene=True, sex_chr=True):
    """
    去除可能污染的线粒体基因、核糖体基因，以及性染色体基因
    :param adata: anndata文件
    :param ribo_gene: 核糖体基因是否去除
    :param mito_gene: 线粒体基因是否去除
    :param sex_chr: 性染色体基因是否去除
    :return: 反对清洗后的adata
    """
    chr_annot_path = "/data/HeLab/bio/biosoftware/customised/HSA_chromsome_annot.csv"
    if not os.path.exists(chr_annot_path):
        annot = sc.queries.biomart_annotations(
            "hsapiens",
            ["ensembl_gene_id", "external_gene_name", "start_position", "end_position", "chromosome_name"],
        ).set_index("external_gene_name")
        annot.to_csv("/data/HeLab/bio/biosoftware/customised/HSA_chromsome_annot.csv")
    else:
        annot = pd.read_csv("/data/HeLab/bio/biosoftware/customised/HSA_chromsome_annot.csv")
    # 参考https://nbisweden.github.io/workshop-scRNAseq/labs/compiled/scanpy/scanpy_05_dge.html
    chrY_genes = adata.var_names.intersection(annot.index[annot.chromosome_name == "Y"])
    chrX_genes = adata.var_names.intersection(annot.index[annot.chromosome_name == "X"])
    sex_genes = chrY_genes.union(chrX_genes)
    mt_genes = adata.var_names[adata.var_names.str.startswith('MT-')].tolist()
    rb_genes = adata.var_names[adata.var_names.str.startswith(('RPS', 'RPL', 'RPLP', 'RPSA'))].tolist()
    _remove_genes = mt_genes if mito_gene else []
    _remove_genes = _remove_genes + rb_genes if ribo_gene else []
    _remove_genes = _remove_genes + sex_genes.tolist() if sex_chr else []
    print(len(_remove_genes))
    all_genes = adata.var.index.tolist()
    print(len(all_genes))
    keep_genes = [x for x in all_genes if x not in _remove_genes]
    print(len(keep_genes))
    adata = adata[:, keep_genes]
    return adata
