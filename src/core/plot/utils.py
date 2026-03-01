import anndata
import pandas as pd
import numpy as np
import scanpy as sc

import os

import seaborn as sns
import matplotlib

matplotlib.use('Agg')  # 使用无GUI的后端

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)

# 频繁调用，没必要
def jitter_color(base_rgb, scale=0.1):
    """
    对 base_rgb 进行轻微 RGB 摆动
    scale: 摆动幅度 (0-0.1 合适)
    """
    r, g, b = base_rgb
    r = min(max(r + np.random.uniform(-scale, scale), 0), 1)
    g = min(max(g + np.random.uniform(-scale, scale), 0), 1)
    b = min(max(b + np.random.uniform(-scale, scale), 0), 1)
    return (r, g, b)

@logged
def matplotlib_savefig(fig, abs_file_path, close_after=True):
    """
    Save a Matplotlib figure safely:
    - Automatically creates directory
    - Supports common formats (png, pdf, svg, eps, jpg, tif)
    - Cleans NaN/Inf in scatter/path collections to avoid PDF errors
    - Optionally closes the figure after saving
    """
    
    # 1️⃣ 创建目录
    os.makedirs(os.path.dirname(abs_file_path), exist_ok=True)
    
    # 2️⃣ 检查文件扩展名
    valid_exts = {".png", ".pdf", ".svg", ".eps", ".jpg", ".jpeg", ".tif", ".tiff"}
    filename = os.path.basename(abs_file_path)
    dirname = os.path.dirname(abs_file_path)
    name_parts = filename.rsplit('.', 1)  # 只拆一次
    if len(name_parts) == 2 and f".{name_parts[1].lower()}" in valid_exts:
        base = os.path.join(dirname, name_parts[0])
        ext = f".{name_parts[1].lower()}"
    else:
        base = os.path.join(dirname, filename)
        ext = ""
    
    # 3️⃣ 清理 NaN / Inf（保证 PDF 保存不报错）
    for coll in fig.findobj(matplotlib.collections.Collection):
        offsets = coll.get_offsets()
        if offsets.size > 0:
            offsets = np.nan_to_num(offsets, nan=0.0, posinf=0.0, neginf=0.0)
            coll.set_offsets(offsets)
        # 颜色数组也可能有 NaN/Inf
        if hasattr(coll, 'get_facecolors'):
            fc = coll.get_facecolors()
            if fc.size > 0:
                fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
                coll.set_facecolors(fc)
        if hasattr(coll, 'get_edgecolors'):
            ec = coll.get_edgecolors()
            if ec.size > 0:
                ec = np.nan_to_num(ec, nan=0.0, posinf=0.0, neginf=0.0)
                coll.set_edgecolors(ec)
    
    # 4️⃣ 保存文件
    if ext == "":
        # 未指定扩展名 → 默认保存 PNG + PDF
        fig.savefig(base + ".png", bbox_inches="tight", dpi=300)
        fig.savefig(base + ".pdf", bbox_inches="tight", dpi=300)
    else:
        fig.savefig(base + ext, bbox_inches="tight", dpi=300)
    
    # 5️⃣ 可选关闭 figure
    if close_after:
        import matplotlib.pyplot as plt
        plt.close(fig)
