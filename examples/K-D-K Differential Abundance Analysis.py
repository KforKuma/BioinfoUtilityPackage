import pandas as pd
import numpy as np

import anndata

import os,gc,re

from src.core.kdk_ops import *
from src.core.kdk_vis import *

# 加载文件
adata = anndata.read_h5ad("/data/HeLab/bio/IBD_analysis/output/Step07/Step07_finalversion_5.h5ad")

# 编辑需要控制的变量
adata.obs["sampling_group"] = (adata.obs["tissue-origin"].astype(str) + "_" + adata.obs["presorted"].astype(str))
adata.obs["disease_group"] = (adata.obs["disease"].astype(str) + "_" + adata.obs["tissue-type"].astype(str))

# 这一步也完全可以在模板里进行，更轻松地把 meta 信息填写进去
meta_file_path="/data/HeLab/bio/IBD_analysis/output/Step07/Step07d_test/meta_template.csv"
make_a_meta(adata,meta_file=meta_file_path)

# 准备数据，应当确保 batch 是存在的
# 不填写 meta_file 的话也会默认按上述方法生成，但建议检查一下
count_df = kdk_prepare(adata.obs,meta_file=meta_file_path,
                       batch_key="orig.ident", type_key="Subset_Identity")

# 运行计算
# 对单独一个细胞亚群进行测试
subset_list = count_df["Subset_Identity"].unique().tolist()
subset=subset_list[0]
subset_df = count_df[count_df["Subset_Identity"] == subset]

# 测试时使用返回值能够清晰看到计算结果
df, posthoc_df = kdk_analyze(subset_df,
                             subset=subset,group_key="disease_group",
                             batch_key="orig.ident", sample_key="sampling_group",
                             save_addr="/data/HeLab/bio/IBD_analysis/output/Step07/Step07d_test",
                             method="Combined", do_return=True)
# 也可以直接用返回值绘图
plot_confidence_interval(posthoc_df, subset,
                         "/data/HeLab/bio/IBD_analysis/output/Step07/Step07d_test","Combined")


# 批量进行
summary = run_kdk(count_df,
                  type_key="Subset_Identity",group_key="disease_group",
                  batch_key="orig.ident",sample_key="sampling_group",
                  save_addr="/data/HeLab/bio/IBD_analysis/output/Step07/Step07d_test",
                  method="Combined")




