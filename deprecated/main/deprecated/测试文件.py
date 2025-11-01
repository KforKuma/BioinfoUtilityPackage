
# 制作一个样本
# *conda activate scvpy9*

import anndata

adata = anndata.read("/data/HeLab/bio/IBD_plus/13_DE/13_standard_version_0624.h5ad")
from src.ScanpyTools.ScanpyTools import obs_key_wise_subsampling, subcluster, AnnotationMaker
    # 两种加载module方法
    # 1):
    # import sys
    # sys.path.append("C:\\Users\\16000\\Desktop\\pythonProject\\src\\EasyInterface")
    # import ScanpyTools
    # 2):
    # import src.EasyInterface.ScanpyTools

adata_exp = obs_key_wise_subsampling(adata,obs_key = "Subset_Identity",scale = 100)
adata_exp.write("/data/HeLab/bio/IBD_plus/test_adata.h5ad")
adata_exp = subcluster(adata_exp, n_neighbors=10, n_pcs=20, resolutions=[0.1],
                       use_rep="X_pca_harmony")

make_anno = AnnotationMaker(adata_exp,"leiden_res0.1","anno_maker")
adata_exp.obs["leiden_res0.1"] # 0-8

ann_list = ["a",
            "b","a","b","c","a",
            "a","a","c"]
make_anno.annotate_by_list(ann_list)
make_anno.make_annotate()
# anno_maker
# a    2832
# b    1595
# c     479
# Name: count, dtype: int64

# a: 0,2,5,6,7
# b: 1,3
# c: 4,8
make_anno.plan_annotate()
make_anno.make_annotate()
 # 检查无误
# anno_maker
# a    2832
# b    1595
# Name: count, dtype: int64
