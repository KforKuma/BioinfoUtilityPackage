import statsmodels.api as sm
from scipy.stats import mannwhitneyu
import os

def verify_necessity_via_residuals(obs_df, receptor_col, pathway_col, disease_col, target_group="BD"):
    # 1. 准备数据
    data = obs_df[[receptor_col, pathway_col, disease_col]].dropna()
    X = sm.add_constant(data[receptor_col])
    y = data[pathway_col]
    
    # 2. 线性回归提取残差 (Residuals)
    model = sm.OLS(y, X).fit()
    data['residual'] = model.resid
    
    # 3. 统计检验：BD vs 其他组
    group_target = data[data[disease_col] == target_group]
    group_others = data[data[disease_col] != target_group]
    
    # 原始评分差异
    u_raw, p_raw = mannwhitneyu(group_target[pathway_col], group_others[pathway_col])
    # 剔除受体后的残差差异
    u_res, p_res = mannwhitneyu(group_target['residual'], group_others['residual'])
    
    print(f"--- 验证通路: {pathway_col} ---")
    print(f"原始差异 P-value: {p_raw:.2e}")
    print(f"剔除受体后残差差异 P-value: {p_res:.2e}")
    
    # 计算效应值改善比例 (简单用均值差衡量)
    raw_diff = group_target[pathway_col].mean() - group_others[pathway_col].mean()
    res_diff = group_target['residual'].mean() - group_others['residual'].mean()
    reduction = (1 - res_diff / raw_diff) * 100
    print(f"受体对组间差异的解释贡献度: {reduction:.2f}%")


import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.core.plot.utils import matplotlib_savefig

def virtual_knockout_test(adata, save_addr, filename,
                          receptor_col, pathway_col, disease_col, target_group="BD", control_group="HC"):
    # 1. 提取目标组和对照组数据
    sub = adata.obs[adata.obs[disease_col].isin([target_group, control_group])].copy()
    
    # 2. 定义“虚拟敲除”状态
    low_thresh = sub[receptor_col].quantile(0.25)
    high_thresh = sub[receptor_col].quantile(0.75)
    
    sub['receptor_status'] = 'Middle'
    sub.loc[sub[receptor_col] <= low_thresh, 'receptor_status'] = 'Low'
    sub.loc[sub[receptor_col] >= high_thresh, 'receptor_status'] = 'High'
    
    # 3. 显式调用 fig, ax 进行绘图
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # 过滤数据
    plot_data = sub[sub['receptor_status'].isin(['Low', 'High'])]
    
    sns.boxplot(
        data=plot_data,
        x='receptor_status',
        y=pathway_col,
        hue=disease_col,
        order=['Low', 'High'],
        palette='Set1',
        ax=ax  # 关键：指定在哪个 ax 上绘图
    )
    
    # 使用 ax 对象设置属性
    ax.set_title("Necessity Check: NFkB levels in Receptor-Low vs High cells")
    ax.set_xlabel("Receptor Status")
    ax.set_ylabel(pathway_col)
    
    # 4. 保存图片
    if not os.path.exists(save_addr):
        os.makedirs(save_addr)
    
    abs_file_path = os.path.join(save_addr, filename)
    matplotlib_savefig(fig,abs_file_path)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import statsmodels.formula.api as smf
from statannotations.Annotator import Annotator


def necessity_statistical_test(adata, save_addr, filename,
                               receptor_col, pathway_col, disease_col, target_group="BD", control_group="HC"):
    # 1. 数据准备
    plot_df = adata.obs[adata.obs[disease_col].isin([target_group, control_group])].copy()
    
    # 2. 分类：采用更严格的“虚拟敲除”定义（例如最低 20% 为 Low，最高 20% 为 High）
    low_cutoff = plot_df[receptor_col].quantile(0.20)
    high_cutoff = plot_df[receptor_col].quantile(0.80)
    
    plot_df['receptor_status'] = 'Middle'
    plot_df.loc[plot_df[receptor_col] <= low_cutoff, 'receptor_status'] = 'Low'
    plot_df.loc[plot_df[receptor_col] >= high_cutoff, 'receptor_status'] = 'High'
    
    # 只保留极端的两组进行“必要性”对比
    test_df = plot_df[plot_df['receptor_status'].isin(['Low', 'High'])].copy()
    test_df['receptor_status'] = pd.Categorical(test_df['receptor_status'], categories=['Low', 'High'])
    
    # 3. 统计检验 A：交互作用 (Interaction Effect)
    # 这一步证明 BD 是否对受体更“敏感”
    model = smf.ols(f"{pathway_col} ~ receptor_status * {disease_col}", data=test_df).fit()
    interaction_p = model.pvalues[-1]  # 获取交互项的 p-value
    
    # 4. 统计检验 B：组间两两对比 (Mann-Whitney U)
    pairs = [
        (("Low", target_group), ("Low", control_group)),
        (("High", target_group), ("High", control_group))
    ]
    
    # 5. 可视化与标注
    fig, ax = plt.subplots(figsize=(5, 6))
    sns.boxplot(data=test_df, x='receptor_status', y=pathway_col, hue=disease_col,
                     palette='Set1', width=0.6, showfliers=False, ax=ax)
    
    annotator = Annotator(ax, pairs, data=test_df, x='receptor_status', y=pathway_col, hue=disease_col)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()
    
    # 在 apply_and_annotate() 之后添加
    y_min, y_max = ax.get_ylim()
    # 将上限增加 20%，为星号留出空间
    ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.2)
    
    ax.set_title(f"Necessity Check: {pathway_col}\nInteraction P-value: {interaction_p:.2e}")
    ax.set_ylabel(f"{pathway_col} Score")
    abs_file_path = os.path.join(save_addr, filename)
    matplotlib_savefig(fig,abs_file_path)
    
    # 6. 打印效应值分析
    for status in ['Low', 'High']:
        sub = test_df[test_df['receptor_status'] == status]
        diff = sub[sub[disease_col] == target_group][pathway_col].mean() - \
               sub[sub[disease_col] == control_group][pathway_col].mean()
        print(f"[{status}] 组间均值差 (BD - HC): {diff:.4f}")


import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import scanpy as sc


def analyze_trail_decoy_effect(adata, save_addr, filename,
                               target_cells_mask,
                               gene_dict,
                               color_by='RB_Cluster'):  # 新增颜色列参数
    """
    实现 DcR2 优势比与 NFkB/凋亡信号的关联分析
    """
    subset = adata[target_cells_mask].copy()
    
    # --- 1. 计算 RB Score (逻辑同前) ---
    dcr2_val = subset[:, 'TNFRSF10D'].X.toarray().flatten() if 'TNFRSF10D' in subset.var_names else 0
    # 提取 DR4/DR5 并求和
    dr_genes = [g for g in ['TNFRSF10A', 'TNFRSF10B'] if g in subset.var_names]
    dr_sum = subset[:, dr_genes].X.toarray().sum(axis=1).flatten() if dr_genes else 0
    subset.obs['RB_Score'] = np.log1p(dcr2_val / (dr_sum + 1))
    
    # --- 2. 下游通路打分 ---
    for k, v in gene_dict.items():
        # 过滤掉不在 adata 中的基因
        valid_genes = [g for g in v if g in subset.var_names]
        sc.tl.score_genes(subset, gene_list=valid_genes, score_name=f'{k}_Score')
    
    # --- 3. 准备颜色映射 ---
    # 如果指定列存在，则生成颜色列表
    plot_df = subset.obs.copy()
    scatter_colors = None
    if color_by in plot_df.columns:
        labels = plot_df[color_by].astype(str)
        unique_cats = np.unique(labels)
        palette = sns.color_palette("husl", len(unique_cats))
        color_map = dict(zip(unique_cats, palette))
        scatter_colors = [color_map[l] for l in labels]
    
    num_plots = len(gene_dict)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    if num_plots == 1: axes = [axes]
    for i, (k, v) in enumerate(gene_dict.items()):
        # 修正 ValueError：不要在 scatter_kws 里同时混合 c 和 color
        # 直接在 regplot 级别控制 scatter 的基本属性
        sns.regplot(
            data=plot_df, x='RB_Score', y=f'{k}_Score',
            ax=axes[i],
            scatter_kws={
                'alpha': 0.4,
                's': 15,
                'color': None  # 显式置空，防止干扰
            },
            line_kws={'color': 'red', 'lw': 2}
        )
        
        # 关键修复：手动更新 scatter 的颜色
        # regplot 的 collections[0] 是散点对象
        if scatter_colors is not None:
            axes[i].collections[0].set_facecolor(scatter_colors)
            axes[i].collections[0].set_edgecolor(scatter_colors)
        
        axes[i].set_title(f'DcR2 Dominance vs {k}')
        
        # 计算相关系数 (可选，增加分析深度)
        r = plot_df['RB_Score'].corr(plot_df[f'{k}_Score'])
        axes[i].set_title(f'DcR2 Dominance vs {k}\n(Pearson r = {r:.2f})')
        axes[i].set_xlabel('Log1p(DcR2 / (DRs+1))')
        axes[i].set_ylabel(f'{k} Pathway Score')
    
    plt.tight_layout()
    abs_file_path = os.path.join(save_addr, filename)
    plt.savefig(f"{abs_file_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f'{abs_file_path}.pdf', bbox_inches='tight')
    plt.close(fig)
    
    return subset


def plot_kde_comparison(df, save_addr, filename,
                        x='RB_Score', y='NFkB_Score', hue='RB_Cluster', title_prefix=''):
    """
    绘制二维 KDE 分布图，展示受体优势比与信号通路得分的关系。
    """
    # 1. 正确显式调用 fig, ax
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 定义调色盘逻辑
    current_palette = 'viridis' if hue == 'RB_Cluster' else 'Set2'
    
    # 2. 绘制填充风格的 KDE 图
    sns.kdeplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        fill=True,
        alpha=0.5,
        levels=5,
        palette=current_palette,
        ax=ax  # 显式指定 ax
    )
    
    # 3. 添加散点作为底色
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        s=5,
        alpha=0.2,
        legend=False,
        palette=current_palette,
        ax=ax  # 显式指定 ax
    )
    
    # 4. 使用 ax 对象设置属性（注意：ax 方法通常带有 set_ 前缀）
    ax.set_title(f'{title_prefix} Relation between {x} and {y} (Hue: {hue})')
    ax.set_xlabel('DcR2 Dominance (RB_Score)')
    ax.set_ylabel('NFkB Signaling Activity')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 5. 保存与清理
    fig.tight_layout()
    
    if not os.path.exists(save_addr):
        os.makedirs(save_addr)
    
    abs_file_path = os.path.join(save_addr, filename)
    matplotlib_savefig(fig, abs_file_path)
