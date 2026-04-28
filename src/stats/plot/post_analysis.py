import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.covariance import GraphicalLassoCV

from src.core.plot.basics import matplotlib_savefig

def plot_glasso_partial_corr(
        beta_matrix,
        save_addr, filename,
        fig=None,
        ax=None,
        cv=5,
        cmap="coolwarm"
):
    """估计稀疏精度矩阵并绘制偏相关热图。

    Args:
        beta_matrix: 行为样本/条件、列为变量的系数矩阵。
        save_addr: 图像输出目录。
        filename: 输出文件名。
        fig: 可选已有 Figure。
        ax: 可选已有 Axes。
        cv: ``GraphicalLassoCV`` 的交叉验证折数。
        cmap: 热图色板。

    Returns:
        ``(partial_corr, gl_model)``，分别为偏相关矩阵和拟合后的模型。

    Example:
        >>> partial_corr, model = plot_glasso_partial_corr(beta_df, "figures", "partial_corr.png")
        >>> partial_corr.shape
        # 用于观察条件/变量之间的稀疏偏相关结构。
    """
    
    # -------- Graphical Lasso --------
    gl_model = GraphicalLassoCV(cv=cv)
    gl_model.fit(beta_matrix.values)
    
    precision_matrix = gl_model.precision_
    cov_matrix = gl_model.covariance_
    
    # -------- Partial correlation --------
    D = np.sqrt(np.diag(precision_matrix))
    partial_corr = -precision_matrix / np.outer(D, D)
    np.fill_diagonal(partial_corr, 1)
    
    # -------- Figure / Axis handling --------
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # -------- Plot heatmap --------
    sns.heatmap(
        partial_corr,
        xticklabels=beta_matrix.columns,
        yticklabels=beta_matrix.columns,
        cmap=cmap,
        center=0,
        annot=False,
        ax=ax
    )
    
    ax.set_title("Cell-cell Partial Correlation (Graphical Lasso)")
    fig.tight_layout()
    
    # -------- Save --------
    
    abs_file_path = os.path.join(save_addr, filename)
    matplotlib_savefig(fig, abs_file_path)
    
    return partial_corr, gl_model


def plot_glasso_partial_corr_celltype(
        beta_matrix,
        save_addr, filename,
        fig=None,
        ax=None,
        cv=5,
        cmap="coolwarm"
):
    """在 cell subtype/subpopulation 维度估计 Graphical Lasso 偏相关。

    Args:
        beta_matrix: cell subtype/subpopulation x 条件 的系数矩阵。
        save_addr: 图像输出目录。
        filename: 输出文件名。
        fig: 可选 Figure。
        ax: 可选 Axes。
        cv: 交叉验证折数。
        cmap: 热图色板。

    Returns:
        ``(partial_corr, gl_model)``。

    Example:
        >>> corr, model = plot_glasso_partial_corr_celltype(beta_df, "figures", "cell_corr.png")
        # 展示亚群之间的偏相关网络。
    """
    
    # -------- transpose beta matrix (cell-cell relation) --------
    beta_matrix_T = beta_matrix.T
    
    # -------- Graphical Lasso --------
    gl_model = GraphicalLassoCV(cv=cv)
    gl_model.fit(beta_matrix_T.values)
    
    precision_matrix = gl_model.precision_
    cov_matrix = gl_model.covariance_
    
    # -------- Partial correlation --------
    D = np.sqrt(np.diag(precision_matrix))
    partial_corr = -precision_matrix / np.outer(D, D)
    np.fill_diagonal(partial_corr, 1)
    
    # -------- Figure / Axis handling --------
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # -------- Plot heatmap --------
    sns.heatmap(
        partial_corr,
        xticklabels=beta_matrix_T.columns,
        yticklabels=beta_matrix_T.columns,
        cmap=cmap,
        center=0,
        annot=False,
        ax=ax
    )
    
    ax.set_title("Cell-cell Partial Correlation (Graphical Lasso)")
    
    fig.tight_layout()
    
    # -------- Save --------
    abs_file_path = os.path.join(save_addr, filename)
    matplotlib_savefig(fig, abs_file_path)
    
    return partial_corr, gl_model


def plot_glasso_partial_corr_celltype_filtered(
        partial_corr,
        beta_matrix,
        save_addr, filename,
        fig=None,
        ax=None
):
    """过滤零连接节点后绘制聚类偏相关热图。

    Args:
        partial_corr: 偏相关矩阵。
        beta_matrix: 原始系数矩阵，用于取 cell subtype/subpopulation 名称。
        save_addr: 图像输出目录。
        filename: 输出文件名。
        fig: 可选 Figure。
        ax: 可选 Axes。

    Returns:
        ``(partial_corr_clustered, filtered_celltypes, Z)``。

    Example:
        >>> clustered, celltypes, linkage_z = plot_glasso_partial_corr_celltype_filtered(
        ...     partial_corr, beta_df, "figures", "clustered_corr.png"
        ... )
    """
    
    # -------- remove diagonal and find non-zero nodes --------
    partial_corr_no_diag = partial_corr.copy()
    np.fill_diagonal(partial_corr_no_diag, 0)
    
    non_zero_mask = np.any(partial_corr_no_diag != 0, axis=0)
    
    partial_corr_filtered = partial_corr[non_zero_mask][:, non_zero_mask]
    filtered_celltypes = beta_matrix.T.columns[non_zero_mask]
    
    # -------- hierarchical clustering --------
    from scipy.cluster.hierarchy import linkage, leaves_list
    
    distance = 1 - partial_corr_filtered
    Z = linkage(distance, method="ward")
    idx = leaves_list(Z)
    
    partial_corr_clustered = partial_corr_filtered[np.ix_(idx, idx)]
    filtered_celltypes = filtered_celltypes[idx]
    
    # -------- Figure / Axis handling --------
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # -------- colormap --------
    from matplotlib.colors import LinearSegmentedColormap
    
    colors = [(0, "blue"), (0.5, "white"), (1, "red")]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    
    # -------- heatmap --------
    sns.heatmap(
        partial_corr_clustered,
        xticklabels=filtered_celltypes,
        yticklabels=filtered_celltypes,
        cmap=cmap,
        center=0,
        annot=False,
        ax=ax
    )
    
    ax.set_title("Cell-cell Partial Correlation (Graphical Lasso)")
    
    fig.tight_layout()
    
    # -------- Save --------
    abs_file_path = os.path.join(save_addr, filename)
    matplotlib_savefig(fig, abs_file_path)
    
    return partial_corr_clustered, filtered_celltypes, Z


def plot_pca_celltype_and_loading(
        beta_matrix,
        save_addr,
        filename_pca,
        filename_loading,
        fig1=None,
        ax1=None,
        fig2=None,
        ax2=None
):
    """对 cell subtype/subpopulation beta 向量做 PCA 并绘制 loading。

    Args:
        beta_matrix: cell subtype/subpopulation x 条件 的系数矩阵。
        save_addr: 图像输出目录。
        filename_pca: PCA scatter 输出文件名。
        filename_loading: loading scatter 输出文件名。
        fig1: 可选 PCA Figure。
        ax1: 可选 PCA Axes。
        fig2: 可选 loading Figure。
        ax2: 可选 loading Axes。

    Returns:
        ``(pcs_df, loading_plot_df, pca)``。

    Example:
        >>> pcs, loadings, pca = plot_pca_celltype_and_loading(
        ...     beta_df, "figures", "pca.png", "loading.png"
        ... )
        >>> pcs.head()
    """
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # -------- Standardize --------
    X = StandardScaler().fit_transform(beta_matrix.values)
    
    # -------- PCA --------
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    
    pcs_df = pd.DataFrame(
        pcs,
        columns=["PC1", "PC2"],
        index=beta_matrix.index
    )
    
    # -------- Figure 1: PCA scatter (cell types) --------
    if fig1 is None or ax1 is None:
        fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    sns.scatterplot(
        x="PC1",
        y="PC2",
        data=pcs_df,
        s=100,
        ax=ax1
    )
    
    for i, txt in enumerate(pcs_df.index):
        ax1.text(
            pcs_df.iloc[i, 0] + 0.05,
            pcs_df.iloc[i, 1] + 0.05,
            txt,
            fontsize=8
        )
    
    ax1.set_title("PCA of Cell Types (β vector)")
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    ax1.grid(True)
    
    fig1.tight_layout()
    
    abs_file_path = os.path.join(save_addr, filename_pca)
    matplotlib_savefig(fig1, abs_file_path)
    
    # -------- PCA loadings --------
    pc1 = pca.components_[0]
    pc2 = pca.components_[1]
    
    loading_plot_df = pd.DataFrame({
        "PC1": pc1,
        "PC2": pc2
    }, index=beta_matrix.columns)
    
    # -------- Figure 2: loading scatter --------
    if fig2 is None or ax2 is None:
        fig2, ax2 = plt.subplots(figsize=(4, 4))
    
    ax2.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax2.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    
    ax2.scatter(
        loading_plot_df["PC1"],
        loading_plot_df["PC2"],
        s=80
    )
    
    for cond in loading_plot_df.index:
        ax2.text(
            loading_plot_df.loc[cond, "PC1"] + 0.02,
            loading_plot_df.loc[cond, "PC2"] + 0.02,
            cond,
            fontsize=9
        )
    
    ax2.set_xlabel("PC1 loading (IBD axis)")
    ax2.set_ylabel("PC2 loading (BD-Colitis axis)")
    ax2.set_title("PCA loading of conditions")
    
    fig2.tight_layout()
    
    abs_file_path = os.path.join(save_addr, filename_loading)
    matplotlib_savefig(fig2, abs_file_path)
    
    return pcs_df, loading_plot_df, pca


def plot_celltype_decomposition(
        beta_matrix,
        save_addr,
        filename_fa,
        filename_nmf,
        filename_ica,
        fig_fa=None,
        ax_fa=None,
        fig_nmf=None,
        ax_nmf=None,
        fig_ica=None,
        ax_ica=None,
        n_components=2
):
    """对 cell subtype/subpopulation beta 向量执行 FA/NMF/ICA 分解。

    Args:
        beta_matrix: cell subtype/subpopulation x 条件 的系数矩阵。
        save_addr: 图像输出目录。
        filename_fa: FA 输出文件名。
        filename_nmf: NMF 输出文件名。
        filename_ica: ICA 输出文件名。
        fig_fa: 可选 FA Figure。
        ax_fa: 可选 FA Axes。
        fig_nmf: 可选 NMF Figure。
        ax_nmf: 可选 NMF Axes。
        fig_ica: 可选 ICA Figure。
        ax_ica: 可选 ICA Axes。
        n_components: 分解维度。

    Returns:
        ``(fa_df, nmf_df, ica_df)``。

    Example:
        >>> fa_df, nmf_df, ica_df = plot_celltype_decomposition(
        ...     beta_df, "figures", "fa.png", "nmf.png", "ica.png"
        ... )
    """
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import FactorAnalysis, NMF, FastICA
    
    # -------- Standardize --------
    X = StandardScaler().fit_transform(beta_matrix.values)
    
    # ============================
    # FA
    # ============================
    
    fa = FactorAnalysis(n_components=n_components, random_state=42)
    fa_components = fa.fit_transform(X)
    
    fa_df = pd.DataFrame(
        fa_components,
        index=beta_matrix.index,
        columns=[f"Factor{i + 1}" for i in range(n_components)]
    )
    
    if fig_fa is None or ax_fa is None:
        fig_fa, ax_fa = plt.subplots(figsize=(10, 8))
    
    sns.scatterplot(
        x="Factor1",
        y="Factor2",
        data=fa_df,
        s=100,
        ax=ax_fa
    )
    
    for i, txt in enumerate(fa_df.index):
        ax_fa.text(
            fa_df.iloc[i, 0] + 0.05,
            fa_df.iloc[i, 1] + 0.05,
            txt,
            fontsize=8
        )
    
    ax_fa.set_title("Factor Analysis of Cell Types")
    ax_fa.set_xlabel("Factor1")
    ax_fa.set_ylabel("Factor2")
    ax_fa.grid(True)
    
    fig_fa.tight_layout()
    
    abs_file_path = os.path.join(save_addr, filename_fa)
    matplotlib_savefig(fig_fa, abs_file_path)
    
    # ============================
    # NMF
    # ============================
    
    X_nonneg = beta_matrix.values - beta_matrix.values.min()
    
    nmf = NMF(n_components=n_components, init="nndsvda", random_state=42)
    
    W = nmf.fit_transform(X_nonneg)
    
    nmf_df = pd.DataFrame(
        W,
        index=beta_matrix.index,
        columns=[f"NMF{i + 1}" for i in range(n_components)]
    )
    
    if fig_nmf is None or ax_nmf is None:
        fig_nmf, ax_nmf = plt.subplots(figsize=(10, 8))
    
    sns.scatterplot(
        x="NMF1",
        y="NMF2",
        data=nmf_df,
        s=100,
        ax=ax_nmf
    )
    
    for i, txt in enumerate(nmf_df.index):
        ax_nmf.text(
            nmf_df.iloc[i, 0] + 0.05,
            nmf_df.iloc[i, 1] + 0.05,
            txt,
            fontsize=8
        )
    
    x_min, x_max = nmf_df.iloc[:, 0].min(), nmf_df.iloc[:, 0].max()
    y_min, y_max = nmf_df.iloc[:, 1].min(), nmf_df.iloc[:, 1].max()
    
    pad_x = 0.05 * (x_max - x_min)
    pad_y = 0.05 * (y_max - y_min)
    
    ax_nmf.set_xlim(x_min - pad_x, x_max + pad_x)
    ax_nmf.set_ylim(y_min - pad_y, y_max + pad_y)
    
    ax_nmf.set_title("NMF of Cell Types")
    ax_nmf.set_xlabel("NMF1")
    ax_nmf.set_ylabel("NMF2")
    ax_nmf.grid(True)
    
    fig_nmf.tight_layout()
    
    abs_file_path = os.path.join(save_addr, filename_nmf)
    matplotlib_savefig(fig_nmf, abs_file_path)
    
    # ============================
    # ICA
    # ============================
    
    ica = FastICA(n_components=n_components, random_state=42)
    
    ica_components = ica.fit_transform(X)
    
    ica_df = pd.DataFrame(
        ica_components,
        index=beta_matrix.index,
        columns=[f"IC{i + 1}" for i in range(n_components)]
    )
    
    if fig_ica is None or ax_ica is None:
        fig_ica, ax_ica = plt.subplots(figsize=(10, 8))
    
    sns.scatterplot(
        x="IC1",
        y="IC2",
        data=ica_df,
        s=100,
        ax=ax_ica
    )
    
    for i, txt in enumerate(ica_df.index):
        ax_ica.text(
            ica_df.iloc[i, 0] + 0.05,
            ica_df.iloc[i, 1] + 0.05,
            txt,
            fontsize=8
        )
    
    ax_ica.set_title("ICA of Cell Types")
    ax_ica.set_xlabel("IC1")
    ax_ica.set_ylabel("IC2")
    ax_ica.grid(True)
    
    fig_ica.tight_layout()
    
    abs_file_path = os.path.join(save_addr, filename_ica)
    matplotlib_savefig(fig_ica, abs_file_path)
    
    return fa_df, nmf_df, ica_df


def plot_ratio_scatter(
        plot_df, save_addr, filename,
        cell_pair,
        disease_col="disease",
        y_scale="log2",
        clr_lmm_result=None,
        alpha=0.85,
        jitter=0.12,
        figsize=(4, 4)
):
    """绘制两个 cell subtype/subpopulation 比例的散点图。

    Args:
        plot_df: ratio 表，需包含 disease 分组列和 ratio/log2_ratio。
        save_addr: 图像输出目录。
        filename: 输出文件名。
        cell_pair: ``(A, B)``，表示 ``A/B``。
        disease_col: 分组列。
        y_scale: ``"log2"`` 或 ``"ratio"``。
        clr_lmm_result: 可选 CLR-LMM 结果，用于显著性标注。
        alpha: 点透明度。
        jitter: x 轴抖动。
        figsize: 图大小。

    Example:
        >>> plot_ratio_scatter(ratio_df, "figures", "ratio.png", ("CD4_Tcm", "CD4_Tem"))
        # 用于查看特定亚群比例在 disease 之间的差异。
    """
    A, B = cell_pair
    
    # ------------------------
    # decide y
    # ------------------------
    if y_scale == "log2":
        y_col = "log2_ratio"
        y_label = f"log2({A}/{B})"
        baseline = 0
    else:
        y_col = "ratio"
        y_label = f"{A}/{B}"
        baseline = 1
    
    fig, ax = plt.subplots(figsize=figsize)
    
    categories = list(plot_df[disease_col].unique())
    xpos = {k: i for i, k in enumerate(categories)}
    
    palette = sns.color_palette("Set2", len(categories))
    
    # ------------------------
    # scatter
    # ------------------------
    for i, cat in enumerate(categories):
        sub = plot_df[plot_df[disease_col] == cat]
        
        xs = xpos[cat] + np.random.normal(0, jitter, size=len(sub))
        
        ax.scatter(
            xs,
            sub[y_col],
            s=45,
            color=palette[i],
            edgecolor="black",
            linewidth=0.4,
            alpha=alpha
        )
    
    ax.axhline(baseline, ls="--", lw=1, color="gray")
    
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels(categories, rotation=20)
    
    ax.set_ylabel(y_label)
    
    # cleaner axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # ------------------------
    # CLR_LMM significance
    # ------------------------
    if clr_lmm_result is not None:
        
        ct = clr_lmm_result["contrast_table"]
        
        y_max = plot_df[y_col].max()
        y_min = plot_df[y_col].min()
        
        h = (y_max - y_min) * 0.08
        level = y_max + h
        
        for other, row in ct.iterrows():
            
            if row["ref"] not in xpos:
                continue
            
            if str(row["significant"]) == "False":
                continue
            
            x1 = xpos[row["ref"]]
            x2 = xpos[other]
            
            ax.plot(
                [x1, x1, x2, x2],
                [level, level + h, level + h, level],
                lw=1.2,
                color="black"
            )
            
            star = "***" if row["P>|z|"] < 0.001 else \
                "**" if row["P>|z|"] < 0.01 else \
                    "*" if row["P>|z|"] < 0.05 else ""
            
            ax.text(
                (x1 + x2) / 2,
                level + h * 1.1,
                star,
                ha="center",
                va="bottom",
                fontsize=10
            )
            
            level += h * 1.8
        
        ax.set_ylim(y_min, level + h)
    
    ax.grid(False)
    fig.tight_layout()
    
    # ------------------------
    # Save
    # ------------------------
    abs_file_path = os.path.join(save_addr, filename)
    matplotlib_savefig(fig, abs_file_path)
