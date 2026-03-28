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
from src.utils.env_utils import call_with_compatible_args
from src.core.plot.utils import matplotlib_savefig
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
        self.logger.info(f"Calling {self.func.__name__}()")
        os.makedirs(save_addr, exist_ok=True)
        
        # 1. 强制设置返回对象且不立即显示
        kwargs.setdefault('return_fig', True)
        kwargs.setdefault('show', False)
        
        with plt.rc_context():
            result = call_with_compatible_args(self.func, *args, **kwargs)
            
            fig = None
            
            if result is None:
                self.logger.error("Function returned None. Grabbing current figure as fallback.")
                fig = plt.gcf()
            
            # 情况 A: Scanpy 特有的绘图对象 (MatrixPlot, DotPlot, StackedViolationPlot 等)
            elif hasattr(result, 'make_figure'):
                # 检查 fig 是否已经生成
                if not hasattr(result, 'fig') or result.fig is None:
                    self.logger.info("Figure not initialized (likely due to dendrogram). Calling make_figure().")
                    result.make_figure()
                fig = result.fig
            
            # 情况 B: 返回的是普通的 Figure 对象
            elif isinstance(result, plt.Figure):
                fig = result
            
            # 情况 C: 返回的是 Axes 对象
            elif hasattr(result, 'get_figure'):
                fig = result.get_figure()
            
            # 情况 D: 返回的是 Axes 字典 (如 rank_genes_groups 或多颜色 embedding)
            elif isinstance(result, dict):
                # 尝试从字典中提取第一个有效的 ax
                for key, val in result.items():
                    # 某些情况下 val 可能是 list of axes
                    target = val[0] if isinstance(val, (list, tuple)) else val
                    if hasattr(target, 'get_figure'):
                        fig = target.get_figure()
                        break
            
            # 最终兜底
            if fig is None:
                self.logger.warning("Could not extract figure from result. Using plt.gcf().")
                fig = plt.gcf()
            
            # 2. 保存图片
            abs_file_path = os.path.join(save_addr, filename)
            
            # 使用你的 matplotlib_savefig 工具
            # 注意：确保该函数内部能处理 fig 对象
            matplotlib_savefig(fig, abs_file_path, close_after=True)
            
            self.logger.info(f"Successfully saved plot to: {abs_file_path}")
        
        # 3. 彻底清理内存
        plt.close('all')
        return result