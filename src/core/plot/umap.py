import re
import anndata
import pandas as pd
import numpy as np
import scanpy as sc

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
matplotlib.use('Agg')  # 使用无GUI的后端

from src.core.plot.utils import *
from src.core.handlers.plot_wrapper import *

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)


@logged
def process_resolution_umaps(adata, output_dir, resolutions ,use_raw=True ,**kwargs):
    """
    生成 UMAP 图像，用于不同 Leiden 分辨率对比。
    """
    umap_plot = ScanpyPlotWrapper(sc.pl.umap)
    color_keys = [f"leiden_res{res}" for res in resolutions]
    umap_plot(
        save_addr=output_dir,
        filename="Res_Comparison",
        adata=adata,
        color=color_keys,
        legend_loc="on data",
        use_raw=use_raw,
        **kwargs
    )

@logged
def plot_QC_umap(adata, save_addr, filename_prefix):
    umap_plot = ScanpyPlotWrapper(sc.pl.umap)
    
    # 先构造 key_dict
    key_dict = {
        "organelles": [i for i in adata.obs.columns if re.search(r'mt|mito|rb|ribo', i)],
        "phase": [i for i in adata.obs.columns if re.search(r'phase', i)],
        "counts": [i for i in adata.obs.columns if re.search(r'disease|tissue', i)]
    }
    
    # 新字典才安全
    cleaned = {}
    
    for k, cols in key_dict.items():
        if not cols:
            continue
        
        # 过滤掉 bool 和 object 列
        new_cols = []
        for c in cols:
            dtype = adata.obs[c].dtype
            if pd.api.types.is_bool_dtype(dtype):
                continue
            if pd.api.types.is_object_dtype(dtype):
                continue
            new_cols.append(c)
        
        if new_cols:
            cleaned[k] = new_cols
    
    # 展平成 key_list
    key_list = [c for cols in cleaned.values() for c in cols]
    if len(key_list) == 0:
        raise ValueError("[plot_QC_umap] No QC obs_key found; cannot draw UMAP.")
    
    logger.info(f"Find satisfied obs key: {key_list}.")
    
    # 一个 category 画一次
    for name, cols in cleaned.items():
        umap_plot(
            save_addr=save_addr,
            filename=f"{filename_prefix}_UMAP_{name}",
            adata=adata,
            color=cols
        )

@logged
def plot_hierarchical_umap(
        adata,
        save_addr,
        filename,
        hierarchy_dict,
        color_key='Subset_Identity',
        special_celltype_colors={'Proliferative Cell': (0, 0, 0)},
        figsize=(20, 10),
        umap_size=50,
        legend_cols=3,
        jitter_scale=0.15,
        random_seed=42,
        save=True,
        plot=False
):
    """
    绘制带有分级图例的 UMAP。

    参数:
    - adata: AnnData 对象。
    - hierarchy_dict: 字典类型，例如 {'MajorType': ['Subset1', 'Subset2']}。
    - color_key: 储存在 adata.obs 中的列名。
    - special_celltype_colors: 需要指定特殊颜色的类。
    - figsize: 画布大小。
    - umap_size: 点的大小。
    - legend_cols: 图例显示的列数。
    """
    rng = np.random.default_rng(random_seed)
    subset_colors = {}
    abs_fig_path = os.path.join(save_addr, filename)
    
    # 1. 生成调色板
    # 排除特殊指定的类，剩下的用 tab10/20 分配基础色
    normal_celltypes = [ct for ct in hierarchy_dict if ct not in special_celltype_colors]
    base_palette = sns.color_palette("tab10", n_colors=len(normal_celltypes))
    
    base_idx = 0
    for celltype, subsets in hierarchy_dict.items():
        if celltype in special_celltype_colors:
            color = special_celltype_colors[celltype]
            for subset in subsets:
                subset_colors[subset] = color
        else:
            base_color = base_palette[base_idx % len(base_palette)]
            for subset in subsets:
                subset_colors[subset] = jitter_color(base_color, scale=jitter_scale)
            base_idx += 1
    
    # 2. 初始化画布
    fig = plt.figure(figsize=figsize)
    # 左侧 UMAP (占 70% 宽度)
    ax_umap = fig.add_axes([0.0, 0.0, 0.70, 1.0])
    # 右侧图例 (占 24% 宽度)
    ax_leg = fig.add_axes([0.75, 0.0, 0.24, 1.0])
    ax_leg.axis('off')
    
    # 3. 绘制 UMAP
    sc.pl.umap(
        adata,
        color=color_key,
        palette=subset_colors,
        ax=ax_umap,
        size=umap_size,
        alpha=0.8,
        legend_loc='none',
        show=False
    )
    
    # 4. 生成分级图例
    legend_elements = []
    for celltype, subsets in hierarchy_dict.items():
        # 添加大类小标题 (白色背景占位)
        legend_elements.append(Patch(facecolor='white', edgecolor='none', label=celltype))
        for subset in subsets:
            legend_elements.append(Patch(facecolor=subset_colors[subset], label=f"  {subset}"))
    
    leg = ax_leg.legend(
        handles=legend_elements,
        loc='center',
        frameon=False,
        fontsize=9,
        ncol=legend_cols,
        title='Cell Type Hierarchy',
        title_fontsize=10,
        handletextpad=0.5,
        columnspacing=1.0
    )
    
    # 5. 修饰图例：加粗大类标题
    for text in leg.get_texts():
        if not text.get_text().startswith("  "):
            text.set_fontweight("bold")
    
    # 6. 保存
    if save:
        matplotlib_savefig(fig, abs_fig_path)
    
    if plot:
        plt.show()
    else:
        plt.close(fig)