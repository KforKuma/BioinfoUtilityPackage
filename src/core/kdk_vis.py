import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from scipy.stats import f_oneway

from src.core.base_anndata_vis import _matplotlib_savefig


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
        x="disease_group",
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


def plot_de_novo_ANOVA(df, subset, save_addr):
    # 计算 mean 和 sem
    summary = df.groupby("disease_group")["percent"].agg(['mean', 'sem']).reset_index()
    sorted_summary = summary.sort_values("mean").reset_index(drop=True)
    group_order = sorted_summary["disease_group"].tolist()
    
    # ANOVA 检验
    groups = [df[df["disease_group"] == group]["percent"] for group in group_order]
    f_stat, p_value = f_oneway(*groups)
    anova_text = f"(ANOVA p = {p_value:.3e})"
    
    # Tukey 检验
    tukey = pairwise_tukeyhsd(endog=df["percent"], groups=df["disease_group"], alpha=0.05)
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


def perform_pca_clustering_on_residuals(df, output_path, auto_choose_k_func):
    """
    对 residual 进行标准化、PCA、聚类、绘图。
    参数:
        df: 包含 'Subset_Identity', 'disease_group', 'residual' 列的 DataFrame
        output_path: 保存图像的基础路径
        auto_choose_k_func: 接收 DataFrame 并返回最佳 k 的函数
    返回:
        pca_df: 包含 PC1, PC2, cluster 的 DataFrame，索引为 Subset_Identity
        resid_scaled_row: 归一化后的 pivot 表（含 residual 值）
        subset_to_cluster: dict，subset → cluster label
        explained_var: PCA 每个主成分的解释度
    """
    
    # pivot + 缺失填补
    resid_pivot = df.groupby(["Subset_Identity", "disease_group"])["residual"].mean().unstack().fillna(0)
    
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
    
    # 绘图
    plt.figure(figsize=(6, 5))
    for cluster in pca_df["cluster"].unique():
        cluster_data = pca_df[pca_df["cluster"] == cluster]
        plt.scatter(cluster_data["PC1"], cluster_data["PC2"], label=f"Cluster {cluster}", s=40)
        
        if len(cluster_data) >= 3:
            points = cluster_data[["PC1", "PC2"]].values
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k--', linewidth=1)
        
        for subset in cluster_data.index:
            plt.text(pca_df.loc[subset, "PC1"], pca_df.loc[subset, "PC2"], subset, fontsize=8)
    
    plt.xlabel(f"PC1 ({explained_var[0]:.1f}%)")
    plt.ylabel(f"PC2 ({explained_var[1]:.1f}%)")
    plt.title("PCA + Clustering on Residuals")
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{output_path}/All_subset_pca_clusters.png")
    
    return pca_df, resid_scaled_row, subset_to_cluster, explained_var


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
    
    # 保存图像
    fig.tight_layout()
    fig.savefig(f"{save_addr}/All_subset_residual_heatmap.png", bbox_inches='tight')
    plt.close(fig)
    
    # 保存并关闭图
    abs_fig_path = os.path.join(save_addr, "All_Subset_Residual_Heatmap")
    _matplotlib_savefig(fig, abs_fig_path)
