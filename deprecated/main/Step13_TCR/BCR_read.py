import numpy as np
import leidenalg
import sklearn
import scanpy as sc
import scanpy.external as sce
import anndata
import pandas as pd
import os
import gc
import sys

# 定义包含所有输出文件夹的父目录
parent_dir = "/data/HeLab/bio/IBD_plus/HRA000072/TCRBCR/"

# 用于存放每个读取后的 AnnData 对象
adata_dict = {}

# 遍历父目录下所有子文件夹
for subfolder in os.listdir(parent_dir):
    if subfolder.endswith("output_filtered_feature_bc_matrix"):
        full_path = os.path.join(parent_dir, subfolder)
        # 读取10x Genomics格式数据
        # 注意：根据需要设置var_names参数，比如'gene_symbols'或'gene_ids'
        print(f"读取数据：{full_path}")
        adata = sc.read_10x_mtx(full_path, var_names='gene_symbols', cache=True)
        adata_dict[subfolder] = adata

print(f"共加载 {len(adata_dict)} 个数据集")


# 合并所有样本的 AnnData 对象，保留样本标签信息
adata_combined = sc.concat(
    list(adata_dict.values()),
    join="outer",
    label="sample",  # 该列会自动存储每个样本的标签
    keys=list(adata_dict.keys())
)

print("合并后的数据集 shape:", adata_combined.shape)

import re
# 根据实际情况构造正则表达式，这里假设 TCR 基因以 TR 开头，免疫球蛋白基因以 IG 开头
pattern = re.compile(r'^(TR[A|B|D|G][V|D|J|C]+|IG[H|K|L][V|D|J|C])+')
# 筛选出符合条件的基因
selected_genes = [gene for gene in adata.var_names if pattern.match(gene)]
print("筛选出的基因数量：", len(selected_genes))
print("示例基因：", selected_genes[:10])

# 执行标准的 scanpy 处理流程

adata_filtered = adata_combined[:, selected_genes].copy()

# 1. 过滤细胞和基因（根据实际情况调整阈值）
sc.pp.filter_cells(adata_filtered, min_genes=5)
sc.pp.filter_genes(adata_filtered, min_cells=3)

# 2. 归一化：将每个细胞的总 UMI 数归一到 1e4
sc.pp.normalize_total(adata_filtered, target_sum=1e4)

# 3. 对数据取对数
sc.pp.log1p(adata_filtered)

# 4. 识别高变基因（保留 top 2000 个高变基因）
sc.pp.highly_variable_genes(adata_filtered, n_top_genes=2000, subset=True)

# 5. 数据标准化（可选：将数据缩放到均值0、方差1）
sc.pp.scale(adata_filtered, max_value=10)

print("数据预处理完成。")
adata.write("/data/HeLab/bio/IBD_analysis/output/Step13_Immuno/Step13_TCR_BCR.h5ad")
