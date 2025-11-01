import anndata
import pandas as pd
import numpy as np
import scanpy as sc
import gc, os
from itertools import chain
import sys
sys.path.append("/data/HeLab/bio/IBD_analysis/")
sys.stdout.reconfigure(encoding='utf-8')


class AnnotationMaker():
    """
    """
    
    def __init__(self, adata, obs_key, anno_key):
        self.data = adata
        self.obs_key = obs_key
        self.anno_key = anno_key
        self.anno_dict = None  # 初始化为None
    
    def annotate_by_list(self, annot_list):
        # 只需要输入列表，
        unique_obs = self.data.obs[self.obs_key].unique()
        # 检查annot_list长度是否匹配
        if len(annot_list) != len(unique_obs):
            raise ValueError(
                f"Length of 'annot_list' ({len(annot_list)}) does not match the number of unique values in obs_key ({len(unique_obs)}).")
        cl_annotation = dict()
        for i in range(len(unique_obs)):
            cl_annotation[str(i)] = annot_list[i]
        self.anno_dict = cl_annotation
        print(f"Annotations by list have been applied: {self.anno_dict}")
    
    def interactive_annotator(self):
        entries_num = len(self.data.obs[self.obs_key].unique())
        print(f"Number of unique entries to annotate: {entries_num}")
        
        # 用户选择模式
        mode_select = input(
            "Annotate by cell type or by subset series? Enter 'c' for cell type and 's' for series: ").lower()
        # 按照细胞类型 cell type 进行注释时：
        # 1） 每次先输入一个细胞类型，如 T Cell;
        # 2） 输入一串数字，如 1,2,5,7,11
        # 按照数字序列（通常是如 leiden_res1.5) 进行注释时：
        # 1） 自动依次打印数字，询问一个细胞类型
        if mode_select == 'c':
            df = pd.DataFrame(columns=['name', 'value_list'])
            print("Processing by cell type...\n")
            
            while True:
                name = input("Please enter new cell type name (enter 'q' to exit): ")
                if name.lower() == 'q':
                    print("Exiting annotation by cell type mode.")
                    break
                
                value_list_input = input("Please enter subset series, multiple integers separated by comma ',': ")
                value_list = [val.strip() for val in value_list_input.split(',')]
                
                all_values = list(chain(*df['value_list']))  # from itertools
                duplicates = [item for item in value_list if item in all_values]
                
                if duplicates:
                    print(f"Warning: These subsets have been given multiple identities: {duplicates}")
                else:
                    try:
                        combined_values = all_values + value_list
                        converted_list = [int(item) for item in combined_values]
                    except ValueError:
                        print("Error: Value list must contain only integers. Please try again.")
                        continue
                
                # 添加新行到DataFrame中
                df = pd.concat([df, pd.DataFrame({'name': [name], 'value_list': [value_list]})], ignore_index=True)
                
                # 检查所有subset是否被分配
                all_present = all(num in converted_list for num in range(entries_num))
                if all_present:
                    print("All subsets have been assigned a cell type.")
                    break
            
            # 展开 value_list 列并排序
            df = df.explode('value_list').rename(columns={'value_list': 'value'})
            df['value'] = df['value'].astype(int)
            df = df.sort_values(by='value').reset_index(drop=True)
            print("Current record:\n", df)
        
        elif mode_select == 's':
            df = pd.DataFrame(columns=['name', 'value'])
            print("Processing by subset series...\n")
            
            for i in range(entries_num):
                name_input = input(f"Enter cell type for entry {i}: ")
                df = pd.concat([df, pd.DataFrame({'name': name_input, 'value': [i]})], ignore_index=True)
            
            print("All subsets have been assigned. Current record:\n", df)
        
        df['value'] = df['value'].astype(str)
        self.anno_dict = df.set_index('value')['name'].to_dict()
        print("Annotation dictionary created:\n", self.anno_dict)
    
    def make_annotate(self):
        if self.anno_dict is not None:
            print(f"Starting annotation using key '{self.anno_key}'...")
            self.data.obs[self.anno_key] = self.data.obs[self.obs_key].map(self.anno_dict)
            print("Successfully annotated. The statistics of cell identity are as follows:\n")
            print(self.data.obs[self.anno_key].value_counts())
        else:
            print("No annotations available. Please run plan_annotate() or annotate_by_list() first.")
