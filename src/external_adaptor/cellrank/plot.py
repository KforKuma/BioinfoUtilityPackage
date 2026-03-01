import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_phase_diff_heatmap(save_addr, filename, df_plot, figsize_ratio=0.4, cmap="RdBu_r", center=0):
    """
    绘制 Source vs Target Velocity 热图并保存为 PNG 和 PDF。

    Parameters
    ----------
    df_plot : pd.DataFrame
        要绘制的矩阵数据，行名为基因。
    sub_addr : str
        保存图片的文件夹路径。
    figsize_ratio : float, optional
        行数与高度的比例，默认为0.4。
    cmap : str, optional
        热图配色方案，默认为 "RdBu_r"。
    center : float, optional
        热图中心值，默认为0。
    """
    if not os.path.exists(save_addr):
        os.makedirs(save_addr)
    
    plt.figure(figsize=(6, figsize_ratio * df_plot.shape[0]+2))
    sns.heatmap(
        df_plot,
        cmap=cmap,
        center=center,
        linewidths=0.5,
        cbar_kws={"label": "Velocity"}
    )
    plt.title("Source vs Target Velocity (Phase Drivers)")
    plt.ylabel("Gene")
    plt.xlabel("")
    plt.tight_layout()
    
    png_path = os.path.join(save_addr, f"{filename}.png")
    pdf_path = os.path.join(save_addr, f"{filename}.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_driver_gene_corr_heatmap(save_addr, filename,
                                  merged_df,
                                  genes_of_interest=None,
                                  corr_suffix="_corr",
                                  min_top_genes=200,
                                  figsize=(10, 12),
                                  cmap="RdBu_r"
                                  ):
    """
    从 merged_df 中提取 *_corr 列，筛选基因并绘制相关性热图。

    Parameters
    ----------
    merged_df : pd.DataFrame
        行为基因，列包含 *_corr 的相关性矩阵
    sub_addr : str
        图像保存路径
    genes_of_interest : list or None
        关注的候选基因列表
    corr_suffix : str
        相关性列的后缀名，默认 "_corr"
    min_top_genes : int
        按 max corr 排序选取的 top 基因数
    figsize : tuple
        图像尺寸
    cmap : str
        颜色映射
    """
    # 1. 取相关性列
    corr_cols = [c for c in merged_df.columns if c.endswith(corr_suffix)]
    df_plot = merged_df[corr_cols].copy()
    df_plot.columns = [c.replace(corr_suffix, "") for c in df_plot.columns]
    
    # 2. 基因筛选
    if genes_of_interest is None:
        genes_of_interest = []
    
    genes_of_interest = list(set(genes_of_interest).intersection(df_plot.index))
    top_genes = (
        df_plot.max(axis=1)
        .sort_values(ascending=False)
        .head(min_top_genes)
        .index
    )
    
    final_genes = genes_of_interest + list(top_genes)
    df_final = df_plot.loc[final_genes].fillna(0)
    
    # 3. 绘图
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        df_final,
        cmap=cmap,
        center=0,
        annot=False,
        cbar_kws={"label": "Correlation with Fate"}
    )
    plt.title("Driver Genes Correlation by Lineage Origin")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    
    plt.savefig(f"{save_addr}/{filename}.png", bbox_inches="tight")
    plt.savefig(f"{save_addr}/{filename}.pdf", bbox_inches="tight")
    plt.close()
