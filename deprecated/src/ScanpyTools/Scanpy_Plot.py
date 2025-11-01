import anndata
import pandas as pd
import numpy as np
import scanpy as sc
import gc, os
from itertools import chain
import re
import time
from functools import wraps
import sys
sys.path.append("/data/HeLab/bio/IBD_analysis/")

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用无GUI的后端

def split_filename_with_ext(filename, allowed_exts=('.jpg', '.png', '.pdf')):
    """
    智能分离文件名与扩展名，只有当扩展名在允许列表中时才分离。
    返回： (root, ext)，若没有合法扩展名，则 ext = ''
    """
    filename = filename.strip()
    for ext in allowed_exts:
        if filename.lower().endswith(ext):
            return filename[:-len(ext)], ext
    return filename, ''  # 无合法扩展名


class ScanpyPlotWrapper(object):
    """
    Decorator-like wrapper: 调用 scanpy.pl 系列绘图函数后，
    根据 filename 后缀自动保存 PDF/PNG。
    """
    
    def __init__(self, func):
        wraps(func)(self)
        self.func = func
    
    def __call__(self, save_addr, filename, *args, **kwargs):
        print("[Wrapper]: calling {function}()".format(function=self.func.__name__))
        # 确保路径以目录形式存在
        os.makedirs(save_addr, exist_ok=True)
        
        # 调用原绘图函数
        with plt.rc_context():
            self.func(*args, **kwargs)
            
            # 拆分 filename 与 ext
            root, ext = split_filename_with_ext(filename)
            
            ext = ext.lower()
            
            # 根据用户是否指定后缀，决定保存哪些格式
            if not ext or ext == '':
                exts = ['.pdf', '.png']
            
            # 循环保存
            for e in exts:
                out_path = os.path.join(save_addr, root + e)
                plt.savefig(out_path, bbox_inches='tight')
                print(f"Saved: {out_path}")
        
        # 清理当前 figure，防止后续绘图叠加
        plt.close()





##################################################
##################################################

def geneset_dotplot(adata, markers, marker_sheet, output_dir, filename_prefix, groupby_key, use_raw=True, **kwargs):
    """
    绘制指定分组（groupby_key）下的标记基因 dotplot。
    marker_sheet：仅传入一个基因列表（pd.Series格式），而不是把Geneset对象传入
    """
    print("**************** Geneset version control 0.1 ****************")
    dotplot = ScanpyPlotWrapper(func=sc.pl.dotplot)
    from src.ScanpyTools.ScanpyTools import sanitize_filename
    
    if isinstance(marker_sheet, pd.Series):
        raise ValueError("marker_sheet is pd.Series, please recheck input.")
    
    gene_dicts = markers.get_gene_dict(marker_sheet=marker_sheet, facet_split=True)
    
    for facet_name, gene_list_dict in gene_dicts.items():
        # 构造文件名
        filename = sanitize_filename(f"{filename_prefix}_{groupby_key}_{marker_sheet}_{facet_name}")
        
        # 获取有效基因名
        if use_raw and adata.raw is not None:
            valid_genes = adata.raw.var_names
        else:
            valid_genes = adata.var_names
        
        # 检查并过滤子基因集
        cleaned_gene_list_dict = {}
        for subcat, genes in gene_list_dict.items():
            missing_genes = [gene for gene in genes if gene not in valid_genes]
            if missing_genes:
                print(f"[Warning] Genes missing in '{subcat}' ({facet_name}): {missing_genes}")
            
            # 保留有效基因
            valid_sublist = [gene for gene in genes if gene in valid_genes]
            if valid_sublist:
                cleaned_gene_list_dict[subcat] = valid_sublist
        
        if not cleaned_gene_list_dict:
            print(f"[Info] All gene groups for facet '{facet_name}' are empty after filtering. Skipping this plot.")
            continue
        
        # 构造 kwargs（传入 dotplot）
        dotplot_kwargs = dict(
            save_addr=output_dir,
            filename=filename,
            adata=adata,
            groupby=groupby_key,
            standard_scale="var",
            var_names=cleaned_gene_list_dict,  # 注意这里传的是 dict
            use_raw=use_raw,
        )
        
        if use_raw:
            print("Now using raw data of anndata object.")
        if not use_raw:
            if "scvi_normalized" in adata.layers.keys():
                print("Using layer 'scvi_normalized'.")
                dotplot_kwargs["layer"] = "scvi_normalized"
                
        # 删除外部可能传入的 layer
        if "layer" in kwargs and use_raw:
            print("Warning: Ignoring 'layer' argument because use_raw=True.")
            kwargs.pop("layer")
        
        dotplot_kwargs.update(kwargs)
        
        dotplot(**dotplot_kwargs)
        print(f"--> Dotplot saved: {filename}")


