import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import os
import re
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from scipy.stats import f_oneway



from src.core.base_anndata_vis import _matplotlib_savefig

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)

@logged
def plot_residual_boxplot(df, subset,group_key, sample_key,save_addr):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 绘制 boxplot
    sns.boxplot(data=df, x=group_key, y="residual", hue=sample_key, ax=ax)
    ax.set_title(f"Residuals of {subset} after correcting for {sample_key}")
    
    # 调整右边距留给 legend
    plt.subplots_adjust(right=0.75)  # 0.75 表示 axes 占 figure 宽度 0~0.75
    # legend 放在 figure 外面
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    abs_fig_path = os.path.join(save_addr, f"{subset}_Residual_Boxplot(by_{sample_key})")
    _matplotlib_savefig(fig, abs_fig_path)

@logged
def plot_confidence_interval(posthoc_df, subset, save_addr, method):
    # 按 meandiff 由大到小排序
    tukey_df = posthoc_df.sort_values("meandiff", ascending=False).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(8, len(tukey_df) * 0.5))
    
    
    # Tukey 显著
    color_tukey = 'firebrick'
    # Dunn 显著空心圈
    edgecolor_dunn = 'lightcoral'
    
    # 绘制每条置信区间线
    for i, row in tukey_df.iterrows():
        # 横线 + 短竖线 caps
        ax.errorbar(
            x=row["meandiff"],
            y=i,
            xerr=[[row["meandiff"] - row["lower"]], [row["upper"] - row["meandiff"]]],
            fmt='o',
            color=color_tukey if str(row["reject"]) == "True" else 'gray',
            ecolor='black',
            capsize=5  # 横线两端短竖线长度
        )
        
        # 如果 Dunn 显著，在同一点画一个圆圈
        if str(row.get("dunn_reject", False)) == "True":
            ax.scatter(
                row["meandiff"], i,
                s=80,  # 圆大小，可调
                facecolors='none',  # 空心
                edgecolors=edgecolor_dunn,  # 边框颜色
                linewidths=3,
                zorder=5  # 确保在误差条之上
            )
    
    # 中心参考线
    ax.axvline(0, color="gray", linestyle="--")
    
    # 设置 y 轴标签
    ax.set_yticks(range(len(tukey_df)))
    ax.set_yticklabels([f"{a} vs {b}" for a, b in zip(tukey_df['group1'], tukey_df['group2'])])
    
    if method == "Tukey":
        ax.set_xlabel("Mean Difference (95% CI)")
        ax.set_title("Tukey HSD Pairwise Comparisons")
    elif method == "Dunn":
        ax.set_xlabel("Mean Rank Difference")
        ax.set_title("Dunn Posthoc Test with Holm Correction")
    elif method == "Combined":
        ax.set_xlabel("Mean Difference (95% CI)")
        ax.set_title("Tukey HSD - Dunn`s Test Cross-Validation")
    
    # 去掉背景网格
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 保存
    abs_fig_path = os.path.join(save_addr, f"{subset}_{method}_ConfidenceInterval")
    _matplotlib_savefig(fig, abs_fig_path)

@logged
def plot_better_residual(df, tukey_df, group_key, subset, save_addr):
    # 计算每组的平均残差
    grouped = df.groupby(group_key)["residual"].mean().reset_index()
    grouped = grouped.sort_values("residual")  # 按 residual 从小到大排序
    
    # 创建索引映射，便于 Tukey 连线定位
    group_order = grouped[group_key].tolist()
    group_to_x = {group: i for i, group in enumerate(group_order)}
    
    # 创建图和坐标轴
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制柱状图
    sns.barplot(
        x=group_key,
        y="residual",
        data=grouped,
        order=group_order,
        palette="viridis",
        ax=ax
    )
    
    ax.set_ylabel("Residual")
    ax.set_xlabel("Disease Group")
    
    # 过滤出显著的比较
    significant = tukey_df[tukey_df["reject"] == "True"].reset_index()
    
    # 连线高度
    y_max = grouped["residual"].max()
    height_step = (grouped["residual"].max() - grouped["residual"].min()) * 0.2
    base_height = y_max + 0.05  # 可调
    
    # 为每个显著组添加线和星号
    for i, row in significant.iterrows():
        g1, g2 = row["group1"], row["group2"]
        x1, x2 = group_to_x[g1], group_to_x[g2]
        x_middle = (x1 + x2) / 2
        h = base_height + i * height_step
        # 连线
        ax.plot([x1, x1, x2, x2], [h - 0.01, h, h, h - 0.01], lw=1.5, c='black')
        # 星号
        pval = row["p-adj"]
        if pval <= 0.001:
            star = "***"
        elif pval <= 0.01:
            star = "**"
        elif pval <= 0.05:
            star = "*"
        else:
            star = "ns"
        ax.text(x_middle, h, star, ha='center', va='bottom', fontsize=16)
    
    # 美化 x 轴标签
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 去掉背景网格
    ax.grid(False)
    
    # 保存并关闭图
    abs_fig_path = os.path.join(save_addr, f"{subset}_Residual_Barplot")
    _matplotlib_savefig(fig, abs_fig_path)

@logged
def plot_de_novo_ANOVA(df, group_key, subset, save_addr):
    # 计算 mean 和 sem
    summary = df.groupby(group_key)["percent"].agg(['mean', 'sem']).reset_index()
    sorted_summary = summary.sort_values("mean").reset_index(drop=True)
    group_order = sorted_summary[group_key].tolist()
    
    # ANOVA 检验
    groups = [df[df[group_key] == group]["percent"] for group in group_order]
    f_stat, p_value = f_oneway(*groups)
    anova_text = f"(ANOVA p = {p_value:.3e})"
    
    # Tukey 检验
    tukey = pairwise_tukeyhsd(endog=df["percent"], groups=df[group_key], alpha=0.05)
    tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
    
    # 创建图和坐标轴
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制柱状图和 error bar
    for i, row in sorted_summary.iterrows():
        ax.bar(i, row["mean"], color=sns.color_palette("viridis")[i % 256], edgecolor='black', zorder=2)
        ax.errorbar(
            x=i,
            y=row["mean"],
            yerr=row["sem"],
            fmt='none',
            ecolor='black',
            capsize=5,
            lw=1.5,
            zorder=3
        )
        # 添加具体数值
        ax.text(i, row["mean"] + 0.001, f"{row['mean']:.3f}", ha='center', va='bottom', fontsize=9, zorder=4)
    
    # 添加 Tukey 显著性标记
    current_height = sorted_summary["mean"].max() + sorted_summary["sem"].max() + 0.01
    height_step = 0.01
    for i, row in tukey_df.iterrows():
        if str(row["reject"]) == "True":
            g1, g2 = row["group1"], row["group2"]
            if g1 in group_order and g2 in group_order:
                x1 = group_order.index(g1)
                x2 = group_order.index(g2)
                x1, x2 = sorted([x1, x2])
                y = current_height
                ax.plot([x1, x1, x2, x2], [y, y + height_step, y + height_step, y], lw=1.2, c='black')
                ax.text((x1 + x2) / 2, y + height_step + 0.001, "*", ha='center', va='bottom', color='black',
                        fontsize=14)
                current_height += height_step * 2
    
    # 坐标轴和标题
    ax.set_ylabel("Fraction of cells in sample")
    ax.set_xlabel("Disease group")
    ax.set_title(f"{subset} relative abundance across disease groups\n{anova_text}")
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(group_order, rotation=45, ha='right')
    
    # 保存并关闭图
    abs_fig_path = os.path.join(save_addr, f"{subset}_Average_Percentage_Barplot")
    _matplotlib_savefig(fig, abs_fig_path)

@logged
def perform_pca_clustering_on_residuals(df, group_key, save_addr, auto_choose_k_func):
    """
    对 residual 进行标准化、PCA、聚类、绘图（fig, ax 风格）。
    参数:
        df: 包含 'Subset_Identity', 'disease_group', 'residual' 列的 DataFrame
        save_addr: 保存图像的基础路径
        auto_choose_k_func: 接收 DataFrame 并返回最佳 k 的函数
    返回:
        pca_df: 包含 PC1, PC2, cluster 的 DataFrame，索引为 Subset_Identity
        resid_scaled_row: 归一化后的 pivot 表（含 residual 值）
        subset_to_cluster: dict，subset → cluster label
        explained_var: PCA 每个主成分的解释度
    """
    
    # pivot + 缺失填补
    resid_pivot = df.groupby(["Subset_Identity", group_key])["residual"].mean().unstack().fillna(0)
    
    # 标准化（按 subset 行）
    scaler = StandardScaler()
    resid_scaled_row = pd.DataFrame(
        scaler.fit_transform(resid_pivot.T).T,
        index=resid_pivot.index,
        columns=resid_pivot.columns
    )
    
    # PCA
    pca = PCA(n_components=2)
    resid_pca = pca.fit_transform(resid_scaled_row)
    explained_var = pca.explained_variance_ratio_ * 100
    
    # PCA dataframe
    pca_df = pd.DataFrame(
        resid_pca, columns=["PC1", "PC2"], index=resid_scaled_row.index
    )
    
    # 自动聚类
    auto_k = auto_choose_k_func(pca_df)
    kmeans = KMeans(n_clusters=auto_k, random_state=0, n_init=10).fit(pca_df)
    pca_df["cluster"] = kmeans.labels_
    
    # 建立映射
    subset_to_cluster = pca_df["cluster"].to_dict()
    
    # 绘图 (fig, ax)
    fig, ax = plt.subplots(figsize=(6, 5))
    for cluster in pca_df["cluster"].unique():
        cluster_data = pca_df[pca_df["cluster"] == cluster]
        ax.scatter(cluster_data["PC1"], cluster_data["PC2"], label=f"Cluster {cluster}", s=40)
        
        if len(cluster_data) >= 3:
            points = cluster_data[["PC1", "PC2"]].values
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                ax.plot(points[simplex, 0], points[simplex, 1], 'k--', linewidth=1)
        
        for subset in cluster_data.index:
            ax.text(pca_df.loc[subset, "PC1"], pca_df.loc[subset, "PC2"], subset, fontsize=8)
    
    ax.set_xlabel(f"PC1 ({explained_var[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_var[1]:.1f}%)")
    ax.set_title("PCA + Clustering on Residuals")
    ax.legend()
    ax.grid(False)
    
    # 保存并关闭图
    fig.tight_layout()
    abs_fig_path = os.path.join(save_addr, "All_Subset_Residual_Heatmap")
    _matplotlib_savefig(fig, abs_fig_path)
    
    return pca_df, resid_scaled_row, subset_to_cluster, explained_var


@logged
def plot_residual_heatmap(resid_scaled_row, subset_to_cluster, save_addr):
    """
    根据 residual 矩阵和 cluster 信息绘制热图。
    参数:
        resid_scaled_row: 标准化后的 residual pivot 表
        subset_to_cluster: subset → cluster 的映射 dict
        save_addr: 输出图像目录
    """
    df = resid_scaled_row.copy()
    df["cluster"] = df.index.map(subset_to_cluster)
    
    df = df.sort_values(by=["cluster", df.index.name or "index"])
    heatmap_data = df.drop(columns=["cluster"])
    
    cluster_labels = df["cluster"].values
    cluster_change_locs = np.where(np.diff(cluster_labels) != 0)[0] + 1
    
    # 创建 fig, ax
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热图
    sns.heatmap(
        heatmap_data,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        yticklabels=True,
        linewidths=0.5,
        linecolor='grey',
        cbar_kws={"label": "Residual"},
        ax=ax
    )
    
    # 添加 cluster 分隔线
    for y in cluster_change_locs:
        ax.axhline(y=y, color="black", linewidth=2)
    
    # 标题和坐标轴标签
    ax.set_title("Mean Residuals by Subset and Disease Group (Cluster-separated)")
    ax.set_xlabel("Disease Group")
    ax.set_ylabel("Subset (Cluster Sorted)")
    
    # 去掉网格
    ax.grid(False)
    
    # 保存并关闭图
    fig.tight_layout()
    abs_fig_path = os.path.join(save_addr, "All_Subset_Residual_Heatmap")
    _matplotlib_savefig(fig, abs_fig_path)

def plot_simulation_benchmarks(df, save_addr, filename):
    # 设置风格
    sns.set_theme(style="whitegrid")
    factors = df['contrast_factor'].unique()
    fig, axes = plt.subplots(len(factors), 1, figsize=(10, 5 * len(factors)), sharex=False)
    
    abs_file_path = os.path.join(save_addr, filename)
    
    if len(factors) == 1: axes = [axes]
    
    for ax, factor in zip(axes, factors):
        sub_df = df[df['contrast_factor'] == factor]
        
        # 创建右侧轴
        ax2 = ax.twinx()
        
        # --- 绘制 Power (左轴) ---
        p1 = sns.lineplot(data=sub_df, x='scale_factor', y='Power', hue='Method',
                          marker='o', ax=ax, legend=True, linewidth=2.5)
        ax.set_ylabel('Power (Sensitivity)', fontsize=12, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        
        # --- 绘制 FDR (右轴) ---
        p2 = sns.lineplot(data=sub_df, x='scale_factor', y='FDR', hue='Method',
                          marker='X', ax=ax2, legend=False, linestyle='--', alpha=0.6)
        ax2.set_ylabel('FDR (1 - Specificity)', fontsize=12, color='red')
        ax2.set_ylim(-0.05, 1.05)
        
        # 绘制 FDR = 0.05 的参考线
        ax2.axhline(0.05, color='red', linestyle=':', alpha=0.5, label='Target FDR (0.05)')
        
        # 设置标题
        ax.set_title(f"Performance Comparison: {factor.capitalize()} Effect", fontsize=14)
        ax.set_xscale('log')  # 使用对数坐标
    
    fig.tight_layout()
    _matplotlib_savefig(fig, abs_file_path)


def plot_simulation_with_inflam_prop(df_plot_combined, save_addr, filename):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.set_theme(style="whitegrid")
    
    abs_file_path = os.path.join(save_addr, filename)
    
    # 2. 准备数据：只筛选你关注的 scale_factor 范围
    
    plot_df = df_plot_combined[df_plot_combined['scale_factor'] <= 2.0].copy()
    
    plot_df['inflam'] = plot_df['inflam'].astype(str)  # 转为分类变量便于上色
    
    # 3. 创建画布：一列看 Power，一列看 FDR
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    
    # 绘制 Power 随 scale_factor 的变化
    
    sns.lineplot(
        
        data=plot_df,
        
        x='scale_factor', y='Power',
        
        hue='inflam', style='method',
        
        marker='o', ax=axes[0]
    
    )
    
    axes[0].set_title('Power Comparison by Inflam Fraction', fontsize=14)
    
    axes[0].set_ylim(0, 1.05)
    
    # 绘制 FDR 随 scale_factor 的变化
    
    sns.lineplot(
        
        data=plot_df,
        
        x='scale_factor', y='FDR',
        
        hue='inflam', style='method',
        
        marker='s', ax=axes[1]
    
    )
    
    axes[1].set_title('FDR Comparison by Inflam Fraction', fontsize=14)
    
    axes[1].axhline(0.05, ls='--', color='red', alpha=0.5)  # 设定显著性阈值线
    
    axes[1].set_ylim(0, 1.05)
    
    fig.tight_layout()
    
    _matplotlib_savefig(fig, abs_file_path)


def plot_simulation_with_inflam_marginalized(
        df_plot_combined, save_addr, filename, max_scale_factor=2.0, ci_type="se"
):
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.set_theme(style="whitegrid")
    abs_file_path = os.path.join(save_addr, filename)
    
    # 1. 数据筛选
    plot_df = df_plot_combined[df_plot_combined["scale_factor"] <= max_scale_factor].copy()
    
    # 2. 预处理：确保数值类型
    plot_df["Power"] = pd.to_numeric(plot_df["Power"])
    plot_df["FDR"] = pd.to_numeric(plot_df["FDR"])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    
    # 定义错误条参数
    # 'se' 对应标准误，('ci', 95) 对应 95% 置信区间
    eb = ('se', 1) if ci_type == "se" else ('ci', 95)
    
    # ---------- Power ----------
    # 直接使用 seaborn 绘制，它会自动按 method 分组并计算均值和 CI
    sns.lineplot(
        data=plot_df,
        x="scale_factor", y="Power",
        hue="method",
        marker="o",
        errorbar=eb,
        ax=axes[0]
    )
    axes[0].set_title(f"Power (Marginalized over Inflam, {ci_type})", fontsize=14)
    axes[0].set_ylim(0, 1.05)
    
    # ---------- FDR ----------
    sns.lineplot(
        data=plot_df,
        x="scale_factor", y="FDR",
        hue="method",
        marker="s",
        errorbar=eb,
        ax=axes[1],
        legend=False  # 左右图共用一个图例即可
    )
    axes[1].axhline(0.05, ls="--", color="red", alpha=0.5)
    axes[1].set_title(f"FDR (Marginalized over Inflam, {ci_type})", fontsize=14)
    axes[1].set_ylim(0, 1.05)
    
    # 3. 布局优化
    fig.suptitle(f"Method Comparison: Robustness across Inflation Proportions", fontsize=16, y=1.02)
    fig.tight_layout()
    
    # 保存（假设 _matplotlib_savefig 已定义）
    fig.savefig(abs_file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_volcano_stratified_label(df, save_path="volcano_stratified.png", p_threshold=0.05, coef_threshold=1.0, bins=6):
    """
    修复 ValueError: The truth value of a Series is ambiguous.
    确保即便有重复索引，也能准确提取单个标量进行绘图。
    """
    df = df.copy()
    
    # 1. 数值化与异常值处理
    df['Coef.'] = pd.to_numeric(df['Coef.'], errors='coerce')
    df['P>|z|'] = pd.to_numeric(df['P>|z|'], errors='coerce')
    df = df.dropna(subset=['Coef.', 'P>|z|'])
    
    # 2. 计算 -log10(p_val)
    min_nonzero_p = df.loc[df['P>|z|'] > 0, 'P>|z|'].min() if (df['P>|z|'] > 0).any() else 1e-4
    df['neg_log10_p'] = -np.log10(df['P>|z|'].replace(0, min_nonzero_p))
    
    # 3. 标记状态
    df['Status'] = 'Non-sig'
    df.loc[(df['P>|z|'] < p_threshold) & (df['Coef.'] >= coef_threshold), 'Status'] = 'Up'
    df.loc[(df['P>|z|'] < p_threshold) & (df['Coef.'] <= -coef_threshold), 'Status'] = 'Down'
    
    # 4. 设置分面
    terms = df['ref'].unique()
    sns.set_theme(style="ticks")
    g = sns.FacetGrid(df, col="ref", hue="Status",
                      palette={'Up': '#e74c3c', 'Down': '#3498db', 'Non-sig': '#bdc3c7'},
                      col_wrap=min(len(terms), 3),
                      height=5, aspect=1.0)
    
    g.map(plt.scatter, "Coef.", "neg_log10_p", alpha=0.5, s=40, edgecolor='none')
    
    # 5. 辅助线
    g.map(lambda **kwargs: plt.axhline(-np.log10(p_threshold), color='black', lw=0.8, ls='--', alpha=0.3))
    
    # 6. 分层标注逻辑 (修复重点：.iloc[0])
    for ax, term_name in zip(g.axes.flat, terms):
        sig_df = df[(df['ref'] == term_name) & (df['Status'] != 'Non-sig')].copy()
        
        if not sig_df.empty:
            p_min, p_max = sig_df['neg_log10_p'].min(), sig_df['neg_log10_p'].max()
            
            # 处理 p 值单一的情况
            if p_max <= p_min:
                target = sig_df.sort_values(by='Coef.', key=abs, ascending=False).iloc[0]
                ax.text(target['Coef.'], target['neg_log10_p'], target['cell_type'], fontsize=8, ha='center')
                continue
            
            # 创建分层
            bin_edges = np.linspace(p_min, p_max, bins + 1)
            sig_df['p_bin'] = pd.cut(sig_df['neg_log10_p'], bins=bin_edges, labels=False, include_lowest=True)
            
            for b in range(bins):
                bin_data = sig_df[sig_df['p_bin'] == b]
                if not bin_data.empty:
                    # 关键修复点：使用 iloc[0] 确保 target 是一个 Series (单行) 而不是含有重复索引的 Dataframe
                    target = bin_data.sort_values(by='Coef.', key=abs, ascending=False).iloc[0]
                    
                    # 此时 target['Coef.'] 必定是一个标量 float
                    ha_val = 'left' if target['Coef.'] > 0 else 'right'
                    
                    ax.text(target['Coef.'], target['neg_log10_p'] + 0.05,
                            target['cell_type'],
                            fontsize=8,
                            ha=ha_val,
                            va='bottom',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.1))
    
    # 7. 整体格式调整
    g.set_axis_labels("Log2 Fold Change", "-log10(p-value)")
    g.add_legend(title="Status")
    plt.subplots_adjust(right=0.85, top=0.9)
    g.fig.suptitle("Stratified Volcano Plot (Fixed Ambiguity)", fontsize=14)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Stratified volcano plot fixed and saved to {save_path}")


def plot_ppv_with_counts(ppv_table, save_path="ppv_trend_analysis.png"):
    # 设置风格
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # 准备 X 轴标签（提取 bin 的中点或直接用字符串）
    x_labels = [str(interval) for interval in ppv_table['coef_bin']]
    x_indices = np.arange(len(x_labels))
    
    # --- 1. 绘制柱形图 (样本量 / Total Predicted Positives) ---
    # 使用淡蓝色表示该区间内我们一共预测了多少个显著点
    color_bar = '#bdc3c7'  # 浅灰色，作为背景
    ax1.bar(x_indices, ppv_table['total_pred_pos'], alpha=0.4, color=color_bar, label='Total Predicted Positives')
    
    # 叠加 TP 计数（深一点的颜色，可选，能看出 FP 的比例）
    ax1.bar(x_indices, ppv_table['tp_count'], alpha=0.6, color='#95a5a6', label='True Positives (TP)')
    
    ax1.set_xlabel('Estimated Coefficient Bin (abs)', fontsize=12)
    ax1.set_ylabel('Count (Data Volume)', fontsize=12, color='#7f8c8d')
    ax1.tick_params(axis='y', labelcolor='#7f8c8d')
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # --- 2. 绘制折线图 (PPV) ---
    # 创建双坐标轴
    ax2 = ax1.twinx()
    color_ppv = '#e74c3c'  # 鲜红色表示准确率
    
    # 绘制带点折线
    ax2.plot(x_indices, ppv_table['PPV'], color=color_ppv, marker='o', linewidth=2.5,
             markersize=8, label='PPV (Precision)')
    
    # 添加一条平滑趋势线 (Optional: Polynomial fit)
    z = np.polyfit(x_indices, ppv_table['PPV'], 3)
    p = np.poly1d(z)
    ax2.plot(x_indices, p(x_indices), "--", color=color_ppv, alpha=0.5, label='PPV Trend')
    
    ax2.set_ylabel('PPV (True Positive / Predicted Positive)', fontsize=12, color=color_ppv)
    ax2.tick_params(axis='y', labelcolor=color_ppv)
    ax2.set_ylim(0, 1.05)  # 概率在 0-1 之间
    
    # --- 3. 细节装饰 ---
    plt.title('PPV Reliability Analysis vs. Estimated Effect Size', fontsize=15, pad=20)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper left', frameon=True,
               facecolor='white',     # 强制背景为白色
            framealpha=0.9,        # 设置微透明，可以看到背后的网格线
           edgecolor='#dcdde1',   # 浅色边框，不突兀
           fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"PPV plot saved to {save_path}")


def plot_multi_layer_ppv(df_combined, save_path, max_x=2.0):
    
    
    df = df_combined.copy()
    
    # 1. 提取 Bin 中点用于数值对齐
    def get_bin_midpoint(bin_str):
        if pd.isna(bin_str): return np.nan
        # 提取括号内的数字，如 "(0.3, 0.4]" -> [0.3, 0.4]
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(bin_str))
        if len(nums) == 2:
            return (float(nums[0]) + float(nums[1])) / 2
        return np.nan
    
    df['bin_midpoint'] = df['coef_bin'].apply(get_bin_midpoint)
    
    # 2. 筛选 X 轴范围
    plot_df = df[df['bin_midpoint'] <= max_x].copy()
    
    plt.figure(figsize=(13, 7))
    sns.set_theme(style="whitegrid")
    
    layers = plot_df['layer'].unique()
    palette = sns.color_palette("Set1", n_colors=len(layers))
    
    for i, layer in enumerate(layers):
        # --- 修复点：使用 plot_df['layer'] 进行筛选 ---
        layer_df = plot_df[plot_df['layer'] == layer].sort_values('bin_midpoint')
        
        # 排除 total_pred_pos 为 0 的无效点，避免拟合时 y 轴全是 0 的干扰
        valid_data = layer_df[layer_df['total_pred_pos'] > 0]
        if valid_data.empty:
            continue
        
        x = valid_data['bin_midpoint']
        y = valid_data['PPV']
        
        # --- A. 绘制原始数据点（半透明，突出趋势） ---
        plt.plot(x, y, marker='o', label=f"{layer} (Raw)",
                 color=palette[i], linewidth=1, alpha=0.4, markersize=5)
        
        # --- B. 绘制 3 次多项式拟合曲线 ---
        if len(x) > 3:
            try:
                z = np.polyfit(x, y, 3)
                p = np.poly1d(z)
                
                # 生成平滑的拟合线
                x_smooth = np.linspace(x.min(), x.max(), 100)
                y_smooth = p(x_smooth)
                
                # 约束拟合线在 [0, 1] 概率范围内，避免绘图溢出
                y_smooth = np.clip(y_smooth, 0, 1)
                
                plt.plot(x_smooth, y_smooth, linestyle='--',
                         color=palette[i], linewidth=2.5, alpha=1.0,
                         label=f"{layer} Trend")
            except np.RankWarning:
                # 如果点太少无法拟合，则跳过拟合部分
                pass
    
    # 3. 细节处理
    plt.xlim(0, max_x)
    plt.ylim(-0.05, 1.05)
    
    # 标注 1.5 之后为数据稀疏区
    plt.axvline(x=1.5, color='gray', linestyle=':', alpha=0.4)
    plt.fill_between([1.5, max_x], -0.05, 1.05, color='gray', alpha=0.03)
    plt.text(1.52, 0.95, "Sparse Data Region", color='gray', fontsize=10, fontstyle='italic')
    
    # 4. 图例美化 (白色背景，无深色阴影)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
               frameon=True, facecolor='white', edgecolor='lightgray',
               fontsize=10, title="Methods & Fitting")
    
    plt.title(f"PPV Reliability Analysis (Layer Comparison up to {max_x})", fontsize=15, pad=20)
    plt.xlabel("Estimated Coefficient Size (|Est_Coef|)", fontsize=12)
    plt.ylabel("Positive Predictive Value (PPV)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"PPV plot saved to {save_path}")


def plot_significance_heatmap(df, term_order=None, save_path="significance_heatmap.png", p_threshold=0.05):
    """
    图2：显著性热图。
    - 移除了顶部的空白。
    - 手动指定了 Term 的展示顺序。
    - cbar 放在正右侧。
    """
    df = df.copy()
    if "term" not in df.columns:
        df["term"] = df.index
    
    terms = df['term'].unique().tolist()
    term_set = set(terms)
    
    if term_order is not None:
        existing_terms = [t for t in term_order if t in term_set]
        remaining_terms = [t for t in terms if t not in existing_terms]
        final_order = existing_terms + remaining_terms
    else:
        final_order = terms
        
    df['term'] = pd.Categorical(df['term'], categories=final_order, ordered=True)
    
    # 1. 转换数据格式 (pivot 会遵循 categorical 的顺序)
    pivot_coef = df.pivot(index='cell_type', columns='term', values='Coef.')
    pivot_pval = df.pivot(index='cell_type', columns='term', values='P>|z|')
    
    # 2. 预处理
    pivot_coef_filled = pivot_coef.fillna(0)
    pivot_pval_filled = pivot_pval.fillna(1.0)
    annot_matrix = pivot_pval_filled.applymap(lambda x: "*" if x < p_threshold else "")
    
    # 3. 绘图
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    fig_height = len(pivot_coef) * 0.4 + 2
    
    # cbar 位置 [左, 下, 宽, 高]
    custom_cbar_pos = (1.02, 0.2, 0.03, 0.4)
    
    g = sns.clustermap(pivot_coef_filled,
                       annot=annot_matrix.values,
                       fmt="",
                       cmap=cmap,
                       center=0,
                       row_cluster=True,
                       col_cluster=False,  # 依然不聚类列
                       # --- 改进 1：挤掉顶部空白 ---
                       # 第一个数是左侧行树状图占比，第二个数是上方列树状图占比
                       dendrogram_ratio=(0.15, 0.02),
                       linewidths=.5,
                       figsize=(10, fig_height),
                       cbar_pos=custom_cbar_pos,
                       cbar_kws={'label': 'Coefficient (LFC)'},
                       annot_kws={"size": 14, "va": 'center'})
    
    # 4. 细节调整
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    
    # 因为顶部变窄了，标题位置 y 需要稍微调低或手动控制
    g.fig.suptitle(f"Cell Type Response (Sorted & Clustered)",
                   fontsize=14, y=0.98, x=0.55)
    
    # 5. 保存
    # 记得 bbox_inches='tight' 否则右侧 cbar 会看不见
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Significance heatmap saved to {save_path}")


