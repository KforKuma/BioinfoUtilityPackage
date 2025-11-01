# 输出文件
# [1] airr_align.tsv
# [2] airr.tsv
    ## airr格式的输出，兼容airr分析软件
# [3] annot.fa
    ## fasta 格式的 consensus  组装注释
    ## consensus_id consensus_length average_coverage annotations
        ## 其中 annotations 也包含几个字段，分别对应 V,D,J,C, CDR1, CDR2 and CDR3
            ## 对于基因的注释，遵循以下格式：
            ## gene_name(reference_gene_length):(consensus_start-consensus_end):(reference_start-reference_length):similarity
            ## 对于 CDR 的注释，遵循以下格式：
            ## CDRx(consensus_start-consensus_end):score=sequence
            ## 对于 CDR1,2，得分为相似性。
            ## 对于 CDR3，得分 0.00 表示部分 CDR3，得分 1.00 表示带有推断核苷酸的 CDR3，其他数字表示基序信号强度，100.00 为最强。坐标以 0 为基准。
# [4] assembled_reads.fa
# [5] barcode_airr.tsv
# [6] barcode_report.tsv
# [7] cdr3.out
    ## consensus_id	index_within_consensus	V_gene	D_gene	J_gene	C_gene	CDR1	CDR2	CDR3	CDR3_score	read_fragment_count CDR3_germline_similarity complete_vdj_assembly
    ## 请注意，trust_cdr3.out 中的 CDR3_score 已除以 100，因此 1.00 是最高分数，0.01 表示估算的 CDR3。
# [8] final.out
# [9] raw.out
# [10] report.tsv
    ## 与 VDJTools 等兼容（仅专注于 CDR3）
    ## read_count	frequency(proportion of read_count)	CDR3_dna	CDR3_amino_acids	V	D	J	C	consensus_id consensus_id_complete_vdj
    ## 频率分别对BCR(IG)和TCR(TR)链进行了标准化。氨基酸序列中，“_”代表终止密码子，“?”代表密码子中存在歧义的核苷酸“N”。
# [11] toassemble_bc.fa, toassemble.fq, toassemble_bc.fa, toassemble.fq


import pandas as pd
import numpy as np
import sklearn
import anndata
import os
import gc
import scanpy as sc
import matplotlib.pyplot as plt
import sys

adata = anndata.read("/data/HeLab/bio/IBD_analysis/tmp/Step12c_CPDB(0101).h5ad")

# 计算交叉表
cross_table = pd.crosstab(index=[adata.obs['orig.project'], adata.obs['orig.ident']],
                          columns=[adata.obs['disease']])

# 保存为 Excel 文件
cross_table.to_excel("/data/HeLab/bio/IBD_analysis/assets/cross_analysis_01.xlsx")

# 计算交叉表
cross_table = pd.crosstab(index=[adata.obs['Subset_Identity'], adata.obs['orig.ident']],
                          columns=[adata.obs['disease']])

# 保存为 Excel 文件
cross_table.to_excel("/data/HeLab/bio/IBD_analysis/assets/cross_analysis_02.xlsx")


# 首先我们需要研究一下如何对结果进行拼接
import pandas as pd
import glob, os, re

from src.EasyInterface.TRUST4Tools import AIRR_combine

# 获取所有符合要求的文件路径
file_pattern = os.path.join(input_dir, "*[0-9]_airr.tsv")
file_list = glob.glob(file_pattern)

# 组织样本文件，按照A1,A2,A3,B1,...,C3进行分类合并
grouped_files = defaultdict(list)
pattern = re.compile(r"TRUST_([ABC]\d)_")

for file_path in input_file_list:
    match = pattern.search(file_path)
    if match:
        sample_id = match.group(1)
        grouped_files[sample_id].append(file_path)
    else:
        print(f"警告：文件 {file_path} 未匹配到样本标识！")


AIRR_combine(input_file_list=file_list,
             output_dir="/data/HeLab/bio/IBD_plus/GSE116222/bam/AIRR")



    






