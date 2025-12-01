import anndata
import pandas as pd
import numpy as np
import scanpy as sc
import os
from functools import wraps
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用无GUI的后端

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)

@logged
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

    Example
    -------


    """

    def __init__(self, func):
        wraps(func)(self)
        self.func = func
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def __call__(self, save_addr, filename, *args, **kwargs):
        self.logger.info("Calling {function}()".format(function=self.func.__name__))
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
            else:
                exts = [ext]
            
            # 循环保存
            for e in exts:
                out_path = os.path.join(save_addr, root + e)
                plt.savefig(out_path, bbox_inches='tight')
                self.logger.info(f"Saved: {out_path}")

        # 清理当前 figure，防止后续绘图叠加
        plt.close()
