import os
import re
import pandas as pd
from collections import defaultdict


def AIRR_combine(input_file_list, output_dir):
    """
    处理 AIRR 文件，将相同样本的文件合并。

    参数：
    input_file_list (str): 存放需要合并的 AIRR 文件的列表，建议用glob.glob生成。
    output_dir (str): 合并后文件的存放目录。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 合并每个样本的文件
    for sample, files in grouped_files.items():
        print(f"处理样本 {sample}...")
        output_file = os.path.join(output_dir, f"{sample}_combined_AIRR.tsv")
        
        dfs = []
        for file in files:
            print(f"读取文件: {file}")
            df = pd.read_csv(file, sep="\t")
            dfs.append(df)
        
        merged_df = pd.concat(dfs, ignore_index=True)
        print("所有文件合并完成！")
        
        merged_df.to_csv(output_file, sep="\t", index=False)
        print(f"合并结果已保存到 {output_file}")

# 示例调用
# AIRR_combine(input_dir="/data/HeLab/bio/IBD_plus/GSE116222/bam/TRUST4_output",
#              output_dir="/data/HeLab/bio/IBD_plus/GSE116222/bam/AIRR")
