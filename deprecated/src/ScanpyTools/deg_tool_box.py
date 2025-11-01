import seaborn as sns
import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap

import sys

sys.path.append("/data/HeLab/bio/IBD_analysis/")  # 添加 src 的上一级目录

from src.ScanpyTools.ScanpyTools import easy_DEG
##############################
# 服务于 Step07c_DEG_analysis.py
##############################
def load_and_filter_hvg(file_path, load_only = False, desired_order=None):
    import pandas as pd
    excel_data = pd.ExcelFile(file_path)
    df = excel_data.parse(excel_data.sheet_names[0])
    df = df.loc[:, ["names", "scores", "logfoldchanges", "pvals_adj", "cluster"]]
    if not load_only:
        pivot_df = filter_hvg_for_clustermap(
            df,
            filter_by_var=True, var_thr=None,
            filter_by_p_val=True, p_thr=None,
            desired_order=desired_order
        )
    else:
        pivot_df = df
    return pivot_df

def load_merge_and_filter(celllist, output_dir, suffix = "disease", var_quantile=0.9):
    import os
    import pandas as pd
    HVG_list, df_pivot_list = [], []
    for Subset in celllist:
        file = f"{output_dir}/_{Subset}/{Subset}_HVG_wilcoxon_{suffix}.xlsx"
        if not os.path.exists(file):
            print(f"{file} does not exists, skipped")
            continue
        df = load_and_filter_hvg(file, load_only=True)
        df["cluster"] = df["cluster"].astype(str) + "_" + Subset
        df_pivot = df.pivot(index='names', columns='cluster', values='logfoldchanges')
        HVG_list.append(df)
        df_pivot_list.append(df_pivot)
    merged_df = pd.concat(df_pivot_list, axis=1)
    merged_df["variance"] = merged_df.var(axis=1)
    var_thr = merged_df["variance"].quantile(var_quantile)
    filtered = merged_df[merged_df["variance"] > var_thr].drop("variance", axis=1)
    return filtered, HVG_list




def winsorize_df(df, lower_q=0.1, upper_q=0.9):
    values = df.values.ravel()
    lower, upper = np.quantile(values, [lower_q, upper_q])
    return df.clip(lower=lower, upper=upper)


def filter_hvg_for_clustermap(HVG_df, values="logfoldchanges", filter_by_var=True, var_thr=None,
                              filter_by_p_val=True, p_thr=None, desired_order=None):
    import pandas as pd
    
    if HVG_df.empty:
        print("Input HVG_df is empty. Exiting.")
        return pd.DataFrame()
    
    # Step 1: pivot
    HVG_df["cluster"] = HVG_df["cluster"].astype(str)
    pivot_df = HVG_df.pivot_table(
        index="names",
        columns="cluster",
        values=values,
        aggfunc="mean"
    )
    
    if pivot_df.empty:
        print("Pivot table is empty after aggregation. Exiting.")
        return pivot_df
    
    # Step 2: calculate min p-value and variance
    min_pvals = HVG_df.groupby("names")["pvals_adj"].min()
    pivot_df["min_pvals_adj"] = min_pvals.reindex(pivot_df.index)
    pivot_df["variance"] = pivot_df.var(axis=1)
    
    # Step 3: filter by p-value
    if filter_by_p_val:
        p_thr = 1 if p_thr is None else p_thr
        pivot_df = pivot_df[pivot_df["min_pvals_adj"] < p_thr]
        if pivot_df.empty:
            print("No genes passed the p-value threshold.")
            return pivot_df
        pivot_df = pivot_df.fillna(0)
        pivot_df = pivot_df[(pivot_df != 0).any(axis=1)]
        if pivot_df.empty:
            print("All remaining genes are zero across all clusters.")
            return pivot_df
    
    # Step 4: filter by variance
    if filter_by_var:
        var_thr = pivot_df["variance"].quantile(0.75) if var_thr is None else var_thr
        pivot_df = pivot_df[pivot_df["variance"] > var_thr]
        if pivot_df.empty:
            print("No genes passed the variance threshold.")
            return pivot_df
    
    # Step 5: re-order columns & drop helper columns
    if desired_order is None:
        desired_order = ["HC", "Colitis", "UC", "CD", "BD"]
    present_cols = [col for col in desired_order if col in pivot_df.columns]
    if not present_cols:
        print("None of the desired clusters found in columns. Exiting.")
        return pivot_df
    
    pivot_df = pivot_df.drop(columns=["min_pvals_adj", "variance"], errors='ignore')
    pivot_df = pivot_df[present_cols]
    
    return pivot_df


def clustermap_with_custom_cluster(pivot_df, save_dir, figname,
                                   figsize=(12, 24),
                                   distance="correlation", linkage_method="average", fcluster_criterion="distance"
                                   ):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    # 1. 计算距离矩阵和层次聚类
    distance_matrix = pdist(pivot_df.values, metric=distance)
    row_linkage = linkage(distance_matrix, method=linkage_method)
    
    # 2. 根据层次距离阈值裁剪聚类树为 N 个簇
    ts = np.linspace(0.1, 2.0, 50)
    cluster_counts = [len(set(fcluster(row_linkage, t=t, criterion=fcluster_criterion))) for t in ts]
    optimal_t = _elbow_detector(ts, cluster_counts)
    row_clusters = fcluster(row_linkage, t=optimal_t, criterion=fcluster_criterion)
    pivot_df["Cluster_Label"] = row_clusters
    print(pivot_df["Cluster_Label"].value_counts())
    
    # 3. 绘制optimal图
    plt.cla()
    plt.figure(figsize=(8, 6))
    plt.plot(ts, cluster_counts, label="Cluster count")
    plt.axvline(optimal_t, color='red', linestyle='--', label=f"Elbow at t={optimal_t:.2f}")
    plt.xlabel("Distance threshold (t)")
    plt.ylabel("Number of clusters")
    plt.title("Cluster number vs. cut distance threshold")
    plt.grid(True)
    plt.savefig(f"{save_dir}/{figname}_t_detect.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 4. 根据你想要的层级排序行
    import scipy.cluster.hierarchy as sch
    dendro = sch.dendrogram(row_linkage, no_plot=True)
    row_order = dendro['leaves']
    pivot_df_sorted = pivot_df.iloc[row_order]
    # cluster颜色匹配排序后
    row_clusters_sorted = row_clusters[row_order]
    n_clusters = len(set(row_clusters))
    cluster_palette = sns.color_palette("husl", n_clusters)
    cluster_colors_sorted = [cluster_palette[i - 1] for i in row_clusters_sorted]
    
    # Plot clustermap
    from matplotlib.colors import LinearSegmentedColormap
    bl_yel_red = LinearSegmentedColormap.from_list("bl_yel_red", ["navy", "lightyellow", "maroon"])
    print(pivot_df_sorted.describe())
    data = pivot_df_sorted.drop(columns="Cluster_Label")
    vmin = np.percentile(data.values, 1)
    vmax = np.percentile(data.values, 99)
    
    # 先生成 clustermap
    g = sns.clustermap(
        pivot_df_sorted.drop(columns="Cluster_Label"),
        row_cluster=False,
        col_cluster=False,
        row_colors=cluster_colors_sorted,
        cmap=bl_yel_red,
        figsize=figsize,
        vmin=vmin,
        vmax=vmax,
        # dendrogram_ratio=(.1, .2),
        cbar_pos=(.92, .32, .01, 0.4)
    )
    g.cax.set_title("logFC", fontsize=12)
    
    # 可选：进一步调整子图边距让热图最大化利用空间
    g.fig.subplots_adjust(right=0.88)
    
    # 保存
    g.savefig(f"{save_dir}/{figname}_heatmap.png", dpi=300, bbox_inches="tight")
    
    plt.close()


def _elbow_detector(ts, cluster_counts, method="kneed", default_cluster=2):
    """
    :param ts: x轴，簇数列表
    :param cluster_counts: y轴，对应的聚类指标（如 inertia）
    :param method: "MSD" or "kneed"
    :param min_cluster: 最小簇数下限
    :param default_cluster: 检测失败时默认返回值
    :return: optimal cluster number
    """
    optimal_t = None
    
    if method == "MSD":
        # 简单拐点检测：最大二阶差分
        first_diff = np.diff(cluster_counts)
        second_diff = np.diff(first_diff)
        
        # 拐点 = 最大弯曲点
        elbow_idx = np.argmax(np.abs(second_diff)) + 2  # +2 to align with ts index after 2 diffs
        optimal_t = ts[elbow_idx]
    
    elif method == "kneed":
        from kneed import KneeLocator
        kneedle = KneeLocator(ts, cluster_counts, curve='convex', direction='decreasing')
        optimal_t = kneedle.knee
    
    # 检查有效性
    if optimal_t is None:
        print(f"[INFO] Using default elbow number: {default_cluster}")
        optimal_t = default_cluster
    
    return optimal_t


def pca_process(merged_df_filtered, save_dir, figname="among_disease",  figsize=(12, 10)):
    import pandas as pd
    # 仅用于当前文档
    if merged_df_filtered.columns.duplicated().any():
        print("⚠️ Warning: There are duplicated column names!")
        # 可加前缀防止冲突，例如按df编号
        df_list_renamed = [
            df.add_prefix(f"df{i}_") for i, df in enumerate(df_list)
        ]
        merged_df_filtered = pd.concat(df_list_renamed, axis=1)
    
    result_df, pca = run_pca(merged_df_filtered, n_components=3)
    explained_var = pca.explained_variance_ratio_
    print(f"PC1 explains {explained_var[0]:.2%} of variance")
    print(f"PC2 explains {explained_var[1]:.2%} of variance")
    print(f"PC3 explains {explained_var[2]:.2%} of variance")
    plot_pca(result_df, pca, save_dir=save_dir, figname=figname, figsize=figsize, color_by='cell_type')
    return result_df, pca


def run_pca(logfc_matrix, n_components=2):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler
    import pandas as pd
    
    # Step 1: 转置 → 每行是一个“celltype-disease”样本，每列是基因
    df_T = logfc_matrix.T  # shape: [samples x genes]
    
    # Step 2: 标准化（按列，即基因）
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df_T)
    
    # Step 3: PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)
    
    # Step 4: 构造结果 dataframe
    result_df = pd.DataFrame(
        pca_result,
        columns=[f"PC{i + 1}" for i in range(n_components)],
        index=df_T.index  # 每个index是 like "UC_T Cell_NK.CD16+"
    )
    result_df = result_df.copy()
    result_df['label'] = result_df.index
    result_df = result_df.reset_index(drop=True)
    
    # Step 5: 解析 label 中的疾病 & 细胞类型（可根据你的格式微调）
    result_df['group'] = result_df['label'].apply(lambda x: '_'.join(x.split('_')[:-2]))
    result_df['cell_type'] = result_df['label'].apply(lambda x: '_'.join(x.split('_')[-2:]))
    
    return result_df, pca


def plot_pca(result_df, pca, save_dir, figname, figsize=(12, 10), color_by='cell_type'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.cla()
    plt.figure(figsize=(4, 4))
    explained_var = pca.explained_variance_ratio_
    plt.bar(range(1, len(explained_var) + 1), explained_var * 100)
    plt.xlabel("Principal Component", fontsize=12)
    plt.ylabel("Variance Explained (%)", fontsize=12)
    plt.title("PCA Explained Variance", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{figname}_PCA_explaination.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    plt.cla()
    plt.figure(figsize=figsize)
    PC1 = pca.explained_variance_ratio_[0];
    PC1 = "{:.2%}".format(PC1)
    PC2 = pca.explained_variance_ratio_[1];
    PC2 = "{:.2%}".format(PC2)
    sns.scatterplot(data=result_df,
                    x="PC1", y="PC2",
                    hue=color_by, style='group', s=100)
    plt.title(f"PCA of Cell-Disease DEG Patterns")
    plt.xlabel(f'PC1({PC1})')
    plt.ylabel(f'PC2({PC2})')
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    ncol = (len(result_df[color_by].unique()) + len(result_df["group"].unique())) // 25 + 1
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=ncol)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{figname}_PCA.png", dpi=300, bbox_inches="tight")
    plt.close()


########################
def pca_cluster_process(result_df, save_dir, figname, figsize=(10,6)):
    from sklearn.cluster import KMeans
    import seaborn as sns
    
    # 使用 PC1 和 PC2
    X = result_df[["PC1", "PC2"]]
    max_k = min(10, X.shape[0])
    cluster_seq = [i for i in range(2, max_k + 1)]
    inertia_seq = [KMeans(n_clusters=k, random_state=0).fit(X).inertia_ for k in cluster_seq]
    
    optimal_cluster = _elbow_detector(cluster_seq, inertia_seq)
    
    kmeans = KMeans(n_clusters=optimal_cluster, random_state=0)  # 可改成你认为合适的簇数
    result_df['cluster'] = kmeans.fit_predict(X)
    
    # 整理出一个 cluster: celltype list 的字典
    # Step 1: 去重（保留第一个出现的 label）
    dedup_df = result_df.drop_duplicates(subset='label', keep='first')
    # Step 2: 设置 label 为索引，只保留 cluster 列
    label_cluster_map = dedup_df.set_index('cluster')['label']
    cluster_to_labels = label_cluster_map.groupby(label_cluster_map.index).apply(list).to_dict()
    
    plot_pca_with_cluster_legend(result_df, cluster_to_labels, save_dir=save_dir,
                                 figname=figname, only_show=100,figsize=figsize)
    
    return cluster_to_labels


########################
def plot_pca_with_cluster_legend(result_df, cluster_to_labels, save_dir, figname, only_show=5,figsize=(10, 6)):
    plt.cla()
    plt.figure(figsize=figsize)
    
    # Step 1: 画 PCA 聚类散点图
    sns.scatterplot(
        data=result_df,
        x="PC1", y="PC2",
        hue="cluster",
        palette="tab10",
        s=100,
        legend='full'
    )
    
    # Step 2: 构造右侧图注文字
    legend_text = ""
    cluster_to_labels = _formate_tidy_label(cluster_to_labels)
    for cluster_id, labels in cluster_to_labels.items():
        label_str = _format_labels_in_lines(labels, max_label=only_show)
        legend_text += f"Cluster {cluster_id}: \n{label_str}\n"
    
    # Step 3: 添加注释文字（图右侧）
    plt.gcf().text(0.8, 0.5, legend_text,
                   fontsize=10, linespacing=1.8,
                   va='center', ha='left')
    
    # Step 4: 调整图布局
    plt.title("PCA with KMeans Clustering")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # 让右侧有空间
    plt.savefig(f"{save_dir}/{figname}_PCA_cluster.png", dpi=300, bbox_inches="tight")


def _format_labels_in_lines(labels, max_line_length=60, max_label=None):
    '''
    为图注自动换行：限制每行最大字符数
    :param labels: list of str
    :param max_line_length: 每行最多字符数
    :param max_label: 最多展示多少个 label
    :return: formatted string with \n
    '''
    if max_label:
        labels = labels[:max_label]
        if len(labels) < len(labels):
            labels.append("...")
    
    lines = []
    current_line = ""
    
    for label in labels:
        label_str = label if current_line == "" else ", " + label
        # 如果当前加上这个 label 会超限，先收行
        if len(current_line + label_str) > max_line_length:
            lines.append(current_line)
            current_line = label
        else:
            current_line += label_str
    
    if current_line:
        lines.append(current_line)
    
    return "  " + "\n  ".join(lines) + "\n  "


def _formate_tidy_label(cluster_to_labels):
    '''
    返回一个重整细胞名的字典，格式为 "[disease] subtype"
    '''
    new_dict = {}
    for cluster_id, labels in cluster_to_labels.items():
        new_labels = []
        labels.sort()
        for label in labels:
            try:
                dis = "_".join(label.split("_")[:-2])
                celltype = label.split("_")[-2]
                cellsubtype = label.split("_")[-1]
                new_label = f"[{dis}] {cellsubtype}"
            except ValueError:
                # 如果格式不符，保持原样
                new_label = label
            new_labels.append(new_label)
        new_dict[cluster_id] = new_labels
    return new_dict


def split_and_DEG(subset_list, subset_key, split_by_key, output_dir, count_thr=30, downsample=5000):
    for subset in subset_list:
        print(f"Processing subset: {subset}")
        
        save_dir = f"{output_dir}/_{subset}"
        print(f"Creating output directory: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)  # 避免目录已存在时报错
        
        print(f"Subsetting data for: {subset}")
        adata_subset = adata[adata.obs[subset_key] == subset]
        
        # 筛选掉计数小于 30 的疾病亚群；目的是其存在影响在后续 PCA 聚类中对其意义进行挖掘，而且可能存在较大的偏倚
        value_count_df = adata_subset.obs[split_by_key].value_counts()
        disease_accountable = value_count_df.index[value_count_df >= count_thr]
        print(f"Disease group cell counts in {subset}:\n{value_count_df}")
        
        adata_subset = adata_subset[adata_subset.obs[split_by_key].isin(disease_accountable)]
        
        print(f"Running easy_DEG for: {subset}")
        if adata_subset.n_obs < 50:  # 安全保障，实际上不太会出现这种情况
            print(f"Skipped DEG for {subset}: too few cells after filtering.")
            continue
        else:
            easy_DEG(
                adata_subset,
                save_addr=save_dir,
                filename=f"{subset}",
                obs_key=split_by_key,
                save_plot=True,
                plot_gene_num=10,
                downsample=downsample,
                use_raw=True
            )
        
        print(f"Completed DEG analysis for: {subset}\n")
        write_path = f"{save_dir}/Subset_by_disease.h5ad"
        adata_subset.write(write_path)
        del adata_subset
        gc.collect()


def build_adata_from_cluster_dict(adata, cluster_to_labels, tmp_key="tmp"):
    adata_clusters = []
    for cluster_id, label_list in cluster_to_labels.items():
        mask = adata.obs[tmp_key].isin(label_list)
        adata_subset = adata[mask].copy()
        adata_subset.obs["cluster"] = cluster_id
        adata_clusters.append(adata_subset)
    adata_combined = anndata.concat(adata_clusters, join='outer', merge="same")
    adata_combined.obs["cluster"] = adata_combined.obs["cluster"].astype("category")
    return adata_combined


def run_pca_and_deg_for_celltype(celltype, merged_df_filtered, adata, save_dir,
                                 figsize=(12, 10),
                                 pca_fig_prefix="among_disease", DEG_file_suffix="by_PCA_cluster"):
    from src.ScanpyTools.Scanpy_statistics import count_element_list_occurrence
    from src.ScanpyTools.ScanpyTools import easy_DEG
    if isinstance(celltype, (list, tuple)):
        print(f"Processing multiple celltypes.")
        column_mask = [col for col in merged_df_filtered.columns if col.split("_")[-2] in celltype]
        celltype_use_as_name = "-".join(celltype)
    else:
        print(f"Processing {celltype}")
        column_mask = [col for col in merged_df_filtered.columns if col.split("_")[-2] == celltype]
        celltype_use_as_name = celltype
    
    celltype_use_as_name = celltype_use_as_name.replace(" ", "-")
    
    if not column_mask:
        print(f"No columns found for {celltype}")
        return None
    
    df_split = merged_df_filtered.loc[:, column_mask]
    result_df, pca = pca_process(df_split, save_dir, figname=f"{pca_fig_prefix}({celltype_use_as_name})",
                                 figsize=figsize)
    cluster_to_labels = pca_cluster_process(result_df, save_dir,
                                            figname=f"{pca_fig_prefix}({celltype_use_as_name})", figsize=figsize)
    
    if not cluster_to_labels:
        print(f"!{celltype} cannot be clustered, skipped.")
        return None
    
    print(cluster_to_labels)
    adata_combined = build_adata_from_cluster_dict(adata, cluster_to_labels)
    
    easy_DEG(
        adata_combined,
        save_addr=save_dir,
        filename=f"{pca_fig_prefix}_{celltype_use_as_name}({DEG_file_suffix})",
        obs_key="cluster",
        save_plot=True,
        plot_gene_num=10,
        downsample=5000,
        use_raw=True
    )
