# Standard library
import os

# Third-party
import numpy as np
import scanpy as sc
import pandas as pd
import xgboost as xgb
import seaborn as sns
import shap

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import (StandardScaler, label_binarize)
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 roc_curve, roc_auc_score)
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform


import matplotlib
matplotlib.use("Agg")  # 必须在导入 pyplot 之前设置后端
import matplotlib.pyplot as plt

from src.core.base_anndata_vis import _matplotlib_savefig

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)

@logged
def plot_tree_importance(clf,save_path,filename_prefix):
    """
    Plot feature importance of a XGBoost model.

    TODO: 返回数据，兼容对维度的进一步分析。

    Parameters
    ----------
    clf : XGBClassifier
        Trained XGBClassifier model.
    save_path : str
        Path to save the figure.
    filename_prefix : str
        Prefix of the filename to save the figure.

    Returns
    -------
    None

    Notes
    -----
    This function uses the `xgb.plot_importance` function to plot the feature importance of a XGBClassifier model.
    The figure is saved to the specified path with the given filename prefix.
    """
    ax = xgb.plot_importance(clf, max_num_features=20)
    fig = ax.figure
    
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    
    abs_path = os.path.join(save_path, "output", f"{prefix}FeatureImportance")
    _matplotlib_savefig(fig, abs_path)
    
    logger.info(f"Plot saved at {abs_path}.")

@logged
def plot_confusion_matrix(cm_matrix,label_mapping,save_path,filename_prefix):
    """
    Plot a confusion matrix.

    Parameters
    ----------
    cm_matrix : numpy.ndarray
        Confusion matrix.
    label_mapping : dict
        Mapping from feature index to feature name.
    save_path : str
        Path to save the figure.
    filename_prefix : str
        Prefix of the filename to save the figure.

    Returns
    -------
    None

    """
    ax = sns.heatmap(
        cm_matrix,
        annot=True,
        fmt=".2f",  # 因为是浮点数了
        xticklabels=[label_mapping[i] for i in range(len(label_mapping))],
        yticklabels=[label_mapping[i] for i in range(len(label_mapping))]
    )
    fig = ax.figure
    
    ax.set_xlabel('Predicted')  # 用 ax 设置标签
    ax.set_ylabel('True')
    
    fig.tight_layout()  # 紧凑布局
    fig.subplots_adjust(left=0.25, bottom=0.25)  # 调整边距
    
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    abs_path = os.path.join(save_path, "output", f"{prefix}ConfusionMatrix.png")
    _matplotlib_savefig(fig,abs_path)
    
    logger.info(f"Plot saved at {abs_path}.")

@logged
def plot_shap_summary(clf,X_test,label_mapping,save_path,filename_prefix):
    """
    Plot SHAP summary plot.

    TODO: 返回数据，兼容对 SHAP 的进一步分析。

    Parameters
    ----------
    clf : Xgboost.Classifier
        Classifier object trained on the data.
    X_test : numpy.ndarray
        Test data.
    label_mapping : dict
        Mapping from feature index to feature name.
    save_path : str
        Path to save the figure.
    filename_prefix : str
        Prefix of the filename to save the figure.

    Returns
    -------
    None

    Notes
    -----
    This function uses the SHAP library to plot the SHAP summary plot.
    The SHAP summary plot shows the contribution of each feature to the model
    output for a set of test samples. The plot is saved to the specified path with
    the given filename prefix.

    Examples
    --------
    # Plot SHAP summary plot
    plot_shap_summary(clf, X_test, label_mapping, save_path, "test")
    """
    feature_names = [label_mapping.get(i, f"Feature {i}") for i in range(X_test.shape[1])]
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    ax = shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    fig = ax.figure
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)  # 这里留白
    
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    abs_path = os.path.join(save_path, "output", f"{prefix}ShapSummary.png")
    _matplotlib_savefig(fig, abs_path)
    
    logger.info(f"Plot saved at {abs_path}.")

@logged
def plot_roc_per_class(y_test,y_proba,label_mapping,save_path,filename_prefix):
    """
    Plot multi-class ROC curves.

    Parameters
    ----------
    y_test : array-like
        The ground truth target values (class labels; 1 or 0).
    y_proba : array-like
        Target scores, can either be probability estimates or non-thresholded
        measure of decisions (as returned by decision_function).
    label_mapping : dict
        A mapping from integer class labels to their corresponding string names.
    save_path : str
        The path to save the plot.
    filename_prefix : str
        The prefix of the filename to save.

    Returns
    -------
    None

    """
    y_bin = label_binarize(y_test, classes=range(len(np.unique(y_test))))
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i in range(y_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc_i = roc_auc_score(y_bin[:, i], y_proba[:, i])
        class_name = label_mapping[i]  # 使用映射
        ax.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC={auc_i:.2f})')
    
    # 对角线
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    
    # 标签和标题
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Multi-class ROC curves')
    
    # 图例
    ax.legend(loc='lower right')
    
    # 紧凑布局
    fig.tight_layout()
    
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    abs_path = os.path.join(save_path, "output", f"{prefix}ROC.png")
    _matplotlib_savefig(fig, abs_path)
    
    logger.info(f"Plot saved at {abs_path}.")

@logged
def plot_fpr_per_class(y_test, y_pred, label_mapping, save_path, filename_prefix):
    """
    Plot per-class precision/recall/f1 as three subplots in one figure.

    Saves a single figure to: <save_path>/output/<filename_prefix>_per_class_metrics.png
    """
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = ["precision", "recall", "f1-score"]

    # 固定类别与标签顺序
    classes = np.unique(y_test)
    labels = [label_mapping[cls] for cls in classes]

    # 准备保存目录
    out_dir = os.path.join(save_path, "output")
    os.makedirs(out_dir, exist_ok=True)

    # 创建 1x3 子图，共享 y 轴
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for idx, metric in enumerate(metrics):
        values = [report[str(cls)][metric] for cls in classes]
        ax = axes[idx]
        ax.bar(range(len(values)), values, tick_label=labels)
        ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric.capitalize()) if idx == 0 else None
        ax.set_xlabel("Class")
        ax.set_title(f"Per-class {metric.capitalize()}")
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")

    fig.tight_layout()
    
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    abs_path = os.path.join(save_path, "output", f"{prefix}perClassMetrics.png")
    _matplotlib_savefig(fig, abs_path)
    
    logger.info(f"Combined plot saved at {abs_path}.")

@logged
def plot_fpr_violin(results, filename_prefix,save_path,show_points=True):
    """
    绘制 LODO 结果的三面 Boxplot (F1, Precision, Recall)

    Parameters
    ----------
    results : dict of lists
        每个元素是 {"donor": str, "report": dict}，
        其中 report 是 classification_report(..., output_dict=True) 的结果。
    save_path : str or None
        如果给定路径，则保存图像；否则直接 plt.show()
    """
    # 整理数据

    records = []
    for i, donor in enumerate(results["donor"]):
        report = results["report"][i]
        # 取 weighted avg (你也可以改成 macro avg)
        metrics = report["weighted avg"]
        records.append({"donor": donor, "metric": "F1", "value": metrics["f1-score"]})
        records.append({"donor": donor, "metric": "Precision", "value": metrics["precision"]})
        records.append({"donor": donor, "metric": "Recall", "value": metrics["recall"]})

    df = pd.DataFrame(records)

    # 画图
    fig, ax = plt.subplots(figsize=(8, 5))  # 先创建 fig, ax
    

    # 先画 violin
    sns.violinplot(data=df, x="metric", y="value", inner="box", cut=0, palette="Set2", ax=ax)

    # 再叠加 stripplot（每个 donor 的点）
    if show_points:
        sns.stripplot(data=df, x="metric", y="value",
                      color="black", size=3, jitter=True, alpha=0.5, ax=ax)

    ax.set_ylim(-0.05, 1.05)  # 保证 [0,1] 的范围完整显示
    ax.set_title("LODO Performance (F1 / Precision / Recall)", fontsize=14)
    ax.set_ylabel("Value")
    ax.set_xlabel("")
    

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    abs_path = os.path.join(save_path, "output", f"{prefix}perClassMetrics.png")
    _matplotlib_savefig(fig, abs_path)


@logged
def plot_consensus_dendrogram(Z_consensus, donor_labels, branch_supports,
                              save_path,filename_prefix):
    """
    Plot a consensus dendrogram with branch support.

    Parameters
    ----------
    Z_consensus : array-like
        Linkage matrix encoding the hierarchical clustering.
    donor_labels : list of str
        Labels of the donor samples.
    branch_supports : dict
        A mapping from clusters (frozenset of donor labels) to their branch supports.
    save_path : str
        The path to save the plot.
    filename_prefix : str
        The prefix of the filename to save.

    Returns
    -------
    None

    Notes
    ------
    The dendrogram is computed using scipy.cluster.hierarchy.linkage.
    The branch support is annotated on the dendrogram if it is >= 0.10.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    dendro = dendrogram(Z_consensus, labels=donor_labels, ax=ax, leaf_rotation=90)

    # 每个内部节点的坐标
    icoord = dendro["icoord"]
    dcoord = dendro["dcoord"]
    leaves = dendro["leaves"]  # 叶子 donor 的索引

    # 遍历所有内部节点
    for i, (xs, ys) in enumerate(zip(icoord, dcoord)):
        # 这个节点对应的 donor 集合（内部节点的 cluster）
        cluster = frozenset(leaves[int(x / 10)] for x in xs[1:3])  # 中间两个点对应子树

        support = branch_supports.get(cluster, None)
        if support is not None and support >= 0.10:  # 只标注 >=0.10
            x = sum(xs[1:3]) / 2  # 横坐标：子树中点
            y = ys[1]  # 纵坐标：高度
            ax.text(
                x, y, f"{support:.2f}",
                va='bottom', ha='center',
                color='red', fontsize=10, weight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1)
            )

    ax.set_ylabel("Distance")
    ax.set_title("Consensus Dendrogram with Branch Support")
    fig.tight_layout()

    prefix = f"{filename_prefix}_" if filename_prefix else ""
    abs_path = os.path.join(save_path, "output", f"{prefix}ConsensusDendro.png")
    _matplotlib_savefig(fig, abs_path)


@logged
def plot_stability_dendrogram(co_matrix, labels, save_path, filename_prefix):
    """
    co_matrix: n x n 的稳定性矩阵
    labels: donor 名称列表
    """

    # 转换成距离矩阵 (相似度高 = 距离小)
    dist = 1 - co_matrix

    # linkage 聚类
    Z = linkage(dist, method="average")

    # 画图
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=90)
    ax.set_ylabel("Distance (1 - stability)")
    ax.set_title("Stability Dendrogram")

    prefix = f"{filename_prefix}_" if filename_prefix else ""
    abs_path = os.path.join(save_path, "output", f"{prefix}StabilityDendro.png")
    _matplotlib_savefig(fig, abs_path)


@logged
def plot_stability_clustermap(co_matrix, labels, save_path, filename_prefix):
    df = pd.DataFrame(co_matrix, index=labels, columns=labels)
    ax = sns.clustermap(df, cmap="viridis", row_cluster=True, col_cluster=True)
    fig = ax.figure

    prefix = f"{filename_prefix}_" if filename_prefix else ""
    abs_path = os.path.join(save_path, "output", f"{prefix}StabilityCluster.png")
    _matplotlib_savefig(fig, abs_path)

@logged
def plot_xgb_prediction_umap(adata, save_path, filename_prefix=None, skip_subcluster=False,do_return=True,**kwargs):
    # 初始化 UMAP 绘图工具
    """
    Plot UMAP with predicted labels.
    将 XGBoost 预测结果写回 adata.obs 并用 UMAP 可视化。

    △ 提供 subcluster 选项，允许快速进行 de novo 降维以展示结构，或 skip_DR=True 以继承原有的 UMAP 结果。

    △ 如果希望使用稳定的 UMAP，建议先取子集计算完毕后，使用 skip_subcluster=True，跳过计算。


    Parameters
    ----------
    adata : AnnData
        The input AnnData object containing the predicted labels.
    save_path : str
        The path to save the output UMAP figure.
    method_suffix : str, default="scvi"
        The suffix of the output file name.
    do_return : 是否返回原有
    skip_subcluster : bool, default=False
        Whether to skip subclustering process.
    **kwars : 剩余参数会被传入 subcluster

    Returns
    -------
    adata_sub : AnnData
        The subclustered AnnData object containing the predicted labels.

"""
    from src.core.utils.plot_wrapper import ScanpyPlotWrapper
    umap_plot = ScanpyPlotWrapper(func=sc.pl.umap)

    # 读取预测结果
    prefix = f"{filename_prefix}_" if filename_prefix else ""

    y_pred_file = f"{save_path}/{prefix}y_pred.npy"
    y_pred = np.load(y_pred_file)

    # 读取划分索引和 label 映射
    dataset_file = os.path.join(save_path, f"{prefix}dataset.npz")
    data = np.load(dataset_file, allow_pickle=True)
    test_idx = data["test_obs_index"]  # 这是一个 Index 对象数组
    mapping = data["label_mapping"].item()

    # 取出对应测试集的子集
    adata_sub = adata[adata.obs.index.isin(test_idx)].copy()
    if adata_sub.n_obs < 100:
        logger.info("Notice! Anndata subset has less than 100 cell, skip UMAP to avoid corruption.")
        return adata_sub
    if not skip_subcluster:
        from src.core.base_anndata_ops import subcluster
        logger.info(f"Starting subclustering process, this may take some time...")
        default_pars = {"adata":adata_sub,"n_neighbors":20, "n_pcs":20,
                        "skip_DR":False, "resolutions":[1.0], "use_rep":"X_scVI"}
        default_pars.update(kwargs)
        adata_sub = subcluster(**default_pars)

    # 写回预测结果
    adata_sub.obs.loc[test_idx, "predicted_label"] = y_pred
    adata_sub.obs.loc[test_idx, "predicted_label_name"] = adata_sub.obs.loc[test_idx, "predicted_label"].map(mapping)

    logger.info(f"Starting UMAP plotting.")
    # 绘制 UMAP
    umap_plot(
        save_addr=os.path.join(save_path, "output"),
        filename=f"{prefix}XgbPrediction",
        adata=adata_sub,
        color=["disease_type", "predicted_label_name"],
        wspace=0.4
    )
    logger.info("UMAP plotted.")

    if do_return:
        return adata_sub


@logged
def plot_taxonomy(adata, save_path, filename_prefix=None):
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    os.makedirs(f"{save_path}/output",exist_ok=True)

    logger.info("Loading datasets.")
    dataset_file = os.path.join(save_path, f"{prefix}dataset.npz")
    data = np.load(dataset_file, allow_pickle=True)
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    mapping = data["label_mapping"].item()

    logger.info("Loading model.")
    clf = xgb.XGBClassifier()
    clf.load_model(f"{save_path}/{prefix}model.json")

    y_pred = clf.predict(X_test)

    # -------------------
    # 1. 混淆矩阵
    # -------------------
    present_classes = sorted(np.unique(np.concatenate([y_train, y_test])))
    class_names = [mapping[i] for i in present_classes]
    logger.info(f"Present classes (for CM): {present_classes}, "
          f"class_names: {class_names}")

    cm = confusion_matrix(y_test, y_pred, labels=present_classes).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm /= row_sums

    corr = np.corrcoef(cm, rowvar=False)
    dist = 1 - corr
    dist[np.isnan(dist)] = 0
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)

    logger.info(f"CM distance matrix shape: {dist.shape}")
    linkage_matrix = linkage(squareform(dist), method="average")
    logger.info(f"Linkage matrix shape: {linkage_matrix.shape}")

    fig, ax = plt.subplots(figsize=(6, 4))
    dendrogram(linkage_matrix, labels=class_names, orientation="right",ax=ax)
    ax.set_title("Class taxonomy (from confusion matrix)")
    _matplotlib_savefig(fig, abs_path)
    
    logger.info(f"Linkage map (confusion matrix) plotted.")
    del fig, ax
    # -------------------
    # 2. SHAP
    # -------------------
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)  # list, 每个类别一个 [n_samples, n_features]
    labels = np.array(y_test)

    present_classes = np.unique(labels)  # 取测试集里实际出现的类别

    # 对 SHAP 求平均值
    shap_per_class = []
    for i, cname in enumerate(class_names):
        mask = labels == i
        shap_mean = shap_values[mask].mean(axis=0)  # (60, 7)
        shap_per_class.append(shap_mean.flatten())  # 拉平成向量

    shap_per_class = np.vstack(shap_per_class)
    logger.info(f"SHAP per class matrix shape: {shap_per_class.shape}")

    dist_matrix = pdist(shap_per_class, metric="correlation")
    linkage_matrix = linkage(dist_matrix, method="average")
    logger.info(f"SHAP linkage matrix shape: {linkage_matrix.shape}")

    fig, ax = plt.subplots(figsize=(6, 4))
    dendrogram(linkage_matrix, labels=class_names, orientation="right",ax=ax)
    ax.set_title("Class taxonomy (from feature importances)")
    _matplotlib_savefig(fig, abs_path)
    
    logger.info(f"Linkage map (SHAP) plotted.")
    del fig, ax
    # -------------------
    # 3. latent space
    # -------------------
    latent = adata.obsm["X_scVI"]
    labels = adata.obs["disease_type"].values

    class_means = []
    for cname in class_names:
        mask = labels == cname
        if np.sum(mask) == 0:
            logger.info(f"Warning: no samples for class {cname} in latent space, skipping")
            continue
        class_means.append(latent[mask].mean(axis=0))

    class_means = np.vstack(class_means)
    logger.info(f"Latent space class_means shape: {class_means.shape}")

    dist = pdist(class_means, metric="euclidean")
    linkage_matrix = linkage(dist, method="ward")

    fig, ax = plt.subplots(figsize=(6, 4))
    dendrogram(linkage_matrix, labels=class_names, orientation="right",ax=ax)
    ax.set_title("Class taxonomy (from latent space)")
    _matplotlib_savefig(fig, abs_path)
    
    logger.info(f"Linkage map (latent space) plotted.")
    del fig, ax

@logged
def plot_lodo_stripplots(results, save_path,filename_prefix,show_points=True):
    """
    绘制 LODO 结果的三面 Boxplot (F1, Precision, Recall)

    Parameters
    ----------
    results : dict
        每个元素是 {"donor": str, "report": dict}，
        其中 report 是 classification_report(..., output_dict=True) 的结果。
    save_path : str or None
        如果给定路径，则保存图像；否则直接 plt.show()
    """
    # 整理数据
    records = []
    for i, donor in enumerate(results["donor"]):
        report = results["report"][i]
        # 取 weighted avg (你也可以改成 macro avg)
        metrics = report["weighted avg"]
        records.append({"donor": donor, "metric": "F1", "value": metrics["f1-score"]})
        records.append({"donor": donor, "metric": "Precision", "value": metrics["precision"]})
        records.append({"donor": donor, "metric": "Recall", "value": metrics["recall"]})

    df = pd.DataFrame(records)

    # 画图
    fig, ax = plt.subplots(figsize=(8, 6))  # 先创建 fig, ax
    
    
    # 先画 violin
    sns.violinplot(data=df, x="metric", y="value", inner="box", cut=0, palette="Set2", ax=ax)

    # 再叠加 stripplot（每个 donor 的点）
    if show_points:
        sns.stripplot(data=df, x="metric", y="value",
                      color="black", size=3, jitter=True, alpha=0.5, ax=ax)

    ax.set_ylim(-0.05, 1.05)  # 保证 [0,1] 的范围完整显示
    ax.set_title("LODO Performance (F1 / Precision / Recall)", fontsize=14)
    ax.set_ylabel("Value")
    ax.set_xlabel("")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    abs_path = os.path.join(save_path, "output", f"{prefix}LodoStripplots.png")
    _matplotlib_savefig(fig, abs_path)

@logged
def plot_lodo_confusion_matrix(results, save_path, filename_prefix):
    """
    all_results: list of dict，每个 dict 包含:
        - "y_test"
        - "y_pred"
        - "mapping" (label 编码到名称的字典)
    save_path: str, 保存路径（不传则直接 plt.show）
    """
    # 确定类别顺序
    mapping = results["mapping"][0]
    classes = list(mapping.values())

    # 收集混淆矩阵
    conf_list = []
    for i in range(0, len(results["y_test"])):
        y_test = results["y_test"][i]
        y_pred = results["y_pred"][i]
        cm = confusion_matrix(y_test, y_pred, labels=list(mapping.keys()))
        conf_list.append(cm)

    conf_array = np.stack(conf_list, axis=0)  # (n_donors, n_classes, n_classes)

    mean_conf = np.mean(conf_array, axis=0)
    # var_conf  = np.var(conf_array, axis=0) # 没有做oversample等校正，还是标准差比较有意义
    std_conf = np.std(conf_array, axis=0)

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(mean_conf, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title("Mean Confusion Matrix")
    axes[0].set_ylabel("True label")
    axes[0].set_xlabel("Predicted label")

    sns.heatmap(std_conf, annot=True, fmt=".2f", cmap="Reds",
                xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_title("Sigma of Confusion Matrix")
    axes[1].set_ylabel("True label")
    axes[1].set_xlabel("Predicted label")
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    abs_path = os.path.join(save_path, "output", f"{prefix}LodoConfusionMatrix.png")
    _matplotlib_savefig(fig, abs_path)

