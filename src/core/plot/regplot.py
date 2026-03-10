import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from src.core.plot.utils import *


def plot_significant_regression_by_disease(
        adata_sub,
        subset_cells,
        save_addr,
        filename,
        score_col="C3_C5_Signaling_score",
        pathway_cols=None,
        disease_col="disease",
        alpha=0.01,
        figsize=(4, 4),
        s=10):
    sns.set_style("white")  # 去掉 seaborn 默认 grid
    
    if pathway_cols is None:
        pathway_cols = ['NFkB_score', 'PI3K_AKT_score', 'MAPK_ERK_score', 'Rho_GTPase_score']
    
    # 筛选子集
    responsive_cells = adata_sub.obs[
        adata_sub.obs[disease_col].notna() &
        adata_sub.obs['Subset_Identity'].isin(subset_cells)
        ].copy()
    
    # Melt 数据
    plot_df = responsive_cells.melt(
        id_vars=[score_col, disease_col],
        value_vars=pathway_cols,
        var_name='Pathway',
        value_name='Score'
    )
    # 1. 获取颜色映射表 (这能保证回归线颜色和散点颜色完全一致)
    unique_diseases = responsive_cells[disease_col].cat.categories
    palette = sns.color_palette("Set1", n_colors=len(unique_diseases))  # 或者你喜欢的色盘
    color_dict = dict(zip(unique_diseases, palette))
    # FacetGrid
    g = sns.FacetGrid(
        plot_df,
        col="Pathway",
        hue=disease_col,
        hue_order=unique_diseases,  # 显式指定顺序
        palette=color_dict,  # 使用我们定义的字典
        sharex=True,
        sharey=False,
        height=figsize[0],
        aspect=1
    )
    
    g.map_dataframe(
        sns.scatterplot,
        x=score_col,
        y="Score",
        alpha=0.4,
        s=s
    )
    
    # 每个 facet 单独画显著回归
    for ax, pw in zip(g.axes.flat, pathway_cols):
        
        ax.grid(False)  # 去掉 grid
        
        pw_df = plot_df[plot_df['Pathway'] == pw]
        disease_list = list(pw_df[disease_col].unique())
        
        for disease_name, sub_df in pw_df.groupby(disease_col, observed=False):
            
            rho, p = spearmanr(sub_df[score_col], sub_df['Score'], nan_policy='omit')
            
            if p < alpha:
                sns.regplot(
                    x=score_col,
                    y="Score",
                    data=sub_df,
                    scatter=False,
                    ci=None,
                    ax=ax,
                    line_kws={'color': color_dict[disease_name], 'lw': 2}  # 强制线条颜色同步
                )
                
                idx = disease_list.index(disease_name)
                
                ax.text(
                    0.05,
                    0.9 - 0.1 * idx,
                    f"{disease_name}: ρ={rho:.2f}",
                    transform=ax.transAxes,
                    color=color_dict[disease_name],
                    fontsize=12
                )
    
    g.add_legend(title=disease_col)
    g.set_axis_labels("C3a/C5a Receptor Score", "Downstream Pathway Score")
    g.set_titles("{col_name}")
    
    plt.tight_layout()
    
    fig = g.fig  # 获取 figure
    abs_fig_path = os.path.join(save_addr, filename)
    matplotlib_savefig(fig, abs_fig_path)
