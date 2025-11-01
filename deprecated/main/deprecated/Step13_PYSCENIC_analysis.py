# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 备注：本项目主要将Step12a拆分的小群进行研究，并存在反复研究之可能。
# 运行环境: scvpy10
# 主要输入：/data/HeLab/bio/IBD_analysis/output/Step13_PYSCENIC下的输出文件，每个子文件夹应包含.loom, 一个.tsv和三个.csv
# 主要输出：
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##——————————————————————————————————————————————————————————————————————————
import numpy as np
import leidenalg
import sklearn
import scanpy as sc
import scanpy.external as sce
import anndata
import pandas as pd
import os, gc, sys
import math
##——————————————————————————————————————————————————————————————————————————
# 第1部分：读取输出
##——————————————————————————————————————————————————————————————————————————

output_dir = "/data/HeLab/bio/IBD_analysis/output/Step13_PYSCENIC/tmp_250417/Treg"



loom_file = dir + "Out.loom"



# incidence matrix：行为 regulon，列为基因（包含所有），值为 0 或 1
from src.EasyInterface.PyscenicTools import get_regulons
regulons_incidMat = get_regulons(loom_file, column_attr_name="Regulons")

# 通过查表把regulon恢复位字典，keys为文件中包含的regulon；values为regulon所对应的基因
regulons = regulons_to_gene_lists(regulons_incidMat)

regulonAUC = AUC_mtx # 基本上完全就是这个文件，不需要getAUC


# 将regulon读为列表
regulons_to_gene_lists(AUC_mtx)

##——————————————————————————————————————————————————————————————————————————
# 第2部分：数据处理
##——————————————————————————————————————————————————————————————————————————
tf_oi = getMostVarRegulon(data = auc_output,
                          fitThr = 1.5, minMeanForFit = 0.05,picture = False)
    # 返回的是AUC的子集，不是 regulon/gene 的列表


##——————————————————————————————————————————————————————————————————————————
# 第3部分：画图
##——————————————————————————————————————————————————————————————————————————

# AUC 结果的简单聚类
# sns.clustermap(AUC_mtx, figsize=(12,12))








