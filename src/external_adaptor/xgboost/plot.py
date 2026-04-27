"""XGBoost 相关绘图函数。"""

import logging
import os
from typing import Dict, Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import shap
import xgboost as xgb
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from src.core.plot.utils import matplotlib_savefig
from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


def _ensure_output_dir(save_path: str) -> str:
    """创建并返回输出目录。"""
    if not isinstance(save_path, str) or save_path.strip() == "":
        raise ValueError("Argument `save_path` must be a non-empty string.")
    output_dir = os.path.join(save_path.strip(), "output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _build_output_path(save_path: str, filename_prefix: Optional[str], suffix: str) -> str:
    """构造输出文件基础路径。"""
    output_dir = _ensure_output_dir(save_path)
    prefix = f"{filename_prefix.strip()}_" if isinstance(filename_prefix, str) and filename_prefix.strip() else ""
    return os.path.join(output_dir, f"{prefix}{suffix}")


def _ordered_class_labels(label_mapping: Dict[int, str], classes: Iterable[int]) -> list[str]:
    """根据给定编码顺序获取类别名称。"""
    return [label_mapping.get(int(label), str(label)) for label in classes]


@logged
def plot_tree_importance(clf, save_path, filename_prefix, label_mapping=None):
    """绘制并导出树模型特征重要性。

    Args:
        clf: 已训练的 `xgb.XGBClassifier`。
        save_path: 输出目录。
        filename_prefix: 输出文件名前缀。
        label_mapping: 可选特征名映射，键通常为整数索引。

    Returns:
        特征重要性 DataFrame。

    Example:
        importance_df = plot_tree_importance(
            clf=clf,
            save_path=save_addr,
            filename_prefix="CD4T",
        )
        importance_df.head()
    """
    if clf is None:
        raise ValueError("Argument `clf` must not be `None`.")

    fig, ax = plt.subplots(figsize=(8, 6))
    xgb.plot_importance(clf, max_num_features=20, ax=ax)
    abs_path = _build_output_path(save_path, filename_prefix, "FeatureImportance")
    matplotlib_savefig(fig, abs_path)
    logger.info(f"[plot_tree_importance] Figure was saved to: '{abs_path}'.")

    booster = clf.get_booster()
    score_dict = booster.get_score(importance_type="weight")
    importance_df = pd.DataFrame({"feature": list(score_dict.keys()), "importance": list(score_dict.values())})
    if label_mapping is not None:
        importance_df["feature"] = importance_df["feature"].map(
            lambda value: label_mapping.get(int(str(value).replace("f", "")), value)
        )
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return importance_df


@logged
def plot_confusion_matrix(cm_matrix, label_mapping, save_path, filename_prefix):
    """绘制归一化 confusion matrix。

    Args:
        cm_matrix: confusion matrix。
        label_mapping: 类别编码到类别名的映射字典。
        save_path: 输出目录。
        filename_prefix: 输出文件名前缀。

    Returns:
        `None`。

    Example:
        plot_confusion_matrix(
            cm_matrix=cm,
            label_mapping={0: "HC", 1: "UC"},
            save_path=save_addr,
            filename_prefix="CD4T",
        )
    """
    cm_matrix = np.asarray(cm_matrix, dtype=float)
    if cm_matrix.ndim != 2 or cm_matrix.shape[0] != cm_matrix.shape[1]:
        raise ValueError("Argument `cm_matrix` must be a square matrix.")
    if not isinstance(label_mapping, dict):
        raise TypeError("Argument `label_mapping` must be a dictionary.")

    classes = sorted(label_mapping.keys())
    labels = _ordered_class_labels(label_mapping, classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_matrix, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()

    abs_path = _build_output_path(save_path, filename_prefix, "ConfusionMatrix")
    matplotlib_savefig(fig, abs_path)
    logger.info(f"[plot_confusion_matrix] Figure was saved to: '{abs_path}'.")


@logged
def plot_shap_summary(clf, X_test, label_mapping, save_path, filename_prefix):
    """绘制 SHAP summary plot。

    Args:
        clf: 已训练的 `xgb.XGBClassifier`。
        X_test: 测试特征矩阵。
        label_mapping: 特征名映射或类别映射；若未匹配到，则使用默认名字。
        save_path: 输出目录。
        filename_prefix: 输出文件名前缀。

    Returns:
        字典，包含 `shap_values`、`feature_names` 和 `X_test`。

    Example:
        shap_result = plot_shap_summary(
            clf=clf,
            X_test=X_test,
            label_mapping={0: "HC", 1: "UC"},
            save_path=save_addr,
            filename_prefix="CD4T",
        )
    """
    X_test = np.asarray(X_test)
    feature_names = [label_mapping.get(index, f"Feature {index}") for index in range(X_test.shape[1])]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    fig = plt.gcf()
    fig.tight_layout()

    abs_path = _build_output_path(save_path, filename_prefix, "ShapSummary")
    matplotlib_savefig(fig, abs_path)
    logger.info(f"[plot_shap_summary] Figure was saved to: '{abs_path}'.")

    return {"shap_values": shap_values, "feature_names": feature_names, "X_test": X_test}


@logged
def plot_roc_per_class(y_test, y_proba, label_mapping, save_path, filename_prefix):
    """绘制多分类 ROC 曲线。

    Args:
        y_test: 真实标签数组。
        y_proba: 预测概率矩阵。
        label_mapping: 类别映射字典。
        save_path: 输出目录。
        filename_prefix: 输出文件名前缀。

    Returns:
        `None`。

    Example:
        plot_roc_per_class(
            y_test=y_test,
            y_proba=y_proba,
            label_mapping=mapping,
            save_path=save_addr,
            filename_prefix="CD4T",
        )
    """
    y_test = np.asarray(y_test)
    y_proba = np.asarray(y_proba)
    classes = np.unique(y_test)
    if len(classes) < 2:
        logger.info("[plot_roc_per_class] Warning! Fewer than 2 classes were detected. Skip ROC plotting.")
        return

    y_bin = label_binarize(y_test, classes=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    for index, cls in enumerate(classes):
        if y_bin[:, index].sum() == 0:
            logger.info(
                f"[plot_roc_per_class] Warning! Class '{label_mapping.get(int(cls), cls)}' had no positive samples. "
                "Its ROC curve will be skipped."
            )
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, index], y_proba[:, index])
        auc_value = roc_auc_score(y_bin[:, index], y_proba[:, index])
        class_name = label_mapping.get(int(cls), str(cls))
        ax.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC={auc_value:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multi-class ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()

    abs_path = _build_output_path(save_path, filename_prefix, "ROC")
    matplotlib_savefig(fig, abs_path)
    logger.info(f"[plot_roc_per_class] Figure was saved to: '{abs_path}'.")


@logged
def plot_fpr_per_class(y_test, y_pred, label_mapping, save_path, filename_prefix):
    """绘制每类 precision、recall 和 F1 条形图。

    Args:
        y_test: 真实标签数组。
        y_pred: 预测标签数组。
        label_mapping: 类别映射字典。
        save_path: 输出目录。
        filename_prefix: 输出文件名前缀。

    Returns:
        `None`。

    Example:
        plot_fpr_per_class(
            y_test=y_test,
            y_pred=y_pred,
            label_mapping=mapping,
            save_path=save_addr,
            filename_prefix="CD4T",
        )
    """
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics = ["precision", "recall", "f1-score"]
    classes = np.unique(y_test)
    labels = _ordered_class_labels(label_mapping, classes)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for idx, metric in enumerate(metrics):
        values = [report[str(cls)][metric] for cls in classes]
        axes[idx].bar(range(len(values)), values, tick_label=labels)
        axes[idx].axhline(y=0.5, color="red", linestyle="--", linewidth=1)
        axes[idx].set_ylim(0, 1)
        axes[idx].set_xlabel("Class")
        axes[idx].set_title(f"Per-class {metric.capitalize()}")
        if idx == 0:
            axes[idx].set_ylabel(metric.capitalize())
        for tick in axes[idx].get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")
    fig.tight_layout()

    abs_path = _build_output_path(save_path, filename_prefix, "perClassMetrics")
    matplotlib_savefig(fig, abs_path)
    logger.info(f"[plot_fpr_per_class] Figure was saved to: '{abs_path}'.")


@logged
def plot_fpr_violin(results, filename_prefix, save_path, show_points=True):
    """绘制 LODO 结果的 F1 / Precision / Recall violin 图。

    Args:
        results: `_read_lodo_outcome()` 返回的结果字典。
        filename_prefix: 输出文件名前缀。
        save_path: 输出目录。
        show_points: 是否叠加 donor 级散点。

    Returns:
        `None`。

    Example:
        plot_fpr_violin(
            results=results,
            filename_prefix="CD4T",
            save_path=save_addr,
            show_points=True,
        )
    """
    records = []
    for idx, donor in enumerate(results["donor"]):
        report = results["report"][idx]["weighted avg"]
        records.extend(
            [
                {"donor": donor, "metric": "F1", "value": report["f1-score"]},
                {"donor": donor, "metric": "Precision", "value": report["precision"]},
                {"donor": donor, "metric": "Recall", "value": report["recall"]},
            ]
        )
    df = pd.DataFrame(records)
    if df.empty:
        logger.info("[plot_fpr_violin] Warning! No LODO metrics were available. Skip plotting.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=df, x="metric", y="value", inner="box", cut=0, palette="Set2", ax=ax)
    if show_points:
        sns.stripplot(data=df, x="metric", y="value", color="black", size=3, jitter=True, alpha=0.5, ax=ax)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("LODO Performance (F1 / Precision / Recall)")
    ax.set_ylabel("Value")
    ax.set_xlabel("")

    abs_path = _build_output_path(save_path, filename_prefix, "perClassMetrics")
    matplotlib_savefig(fig, abs_path)
    logger.info(f"[plot_fpr_violin] Figure was saved to: '{abs_path}'.")


@logged
def plot_consensus_dendrogram(Z_consensus, donor_labels, branch_supports, save_path, filename_prefix):
    """绘制 consensus dendrogram，并标注分支支持度。

    Args:
        Z_consensus: linkage matrix。
        donor_labels: donor 名称列表。
        branch_supports: branch support 字典。
        save_path: 输出目录。
        filename_prefix: 输出文件名前缀。

    Returns:
        `None`。

    Example:
        plot_consensus_dendrogram(
            Z_consensus=Z,
            donor_labels=donor_labels,
            branch_supports=branch_supports,
            save_path=save_addr,
            filename_prefix="CD4T",
        )
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    dendro = dendrogram(Z_consensus, labels=donor_labels, ax=ax, leaf_rotation=90)

    icoord = dendro["icoord"]
    dcoord = dendro["dcoord"]
    leaves = dendro["leaves"]
    for xs, ys in zip(icoord, dcoord):
        cluster = frozenset(leaves[int(x / 10)] for x in xs[1:3] if int(x / 10) < len(leaves))
        support = branch_supports.get(cluster, None)
        if support is not None and support >= 0.10:
            ax.text(
                sum(xs[1:3]) / 2,
                ys[1],
                f"{support:.2f}",
                va="bottom",
                ha="center",
                color="red",
                fontsize=10,
                weight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1},
            )

    ax.set_ylabel("Distance")
    ax.set_title("Consensus Dendrogram with Branch Support")
    fig.tight_layout()
    abs_path = _build_output_path(save_path, filename_prefix, "ConsensusDendro")
    matplotlib_savefig(fig, abs_path)
    logger.info(f"[plot_consensus_dendrogram] Figure was saved to: '{abs_path}'.")


@logged
def plot_stability_dendrogram(co_matrix, labels, save_path, filename_prefix):
    """绘制 donor 稳定性 dendrogram。

    Args:
        co_matrix: donor 间相似度矩阵。
        labels: donor 标签列表。
        save_path: 输出目录。
        filename_prefix: 输出文件名前缀。

    Returns:
        `None`。
    """
    co_matrix = np.asarray(co_matrix, dtype=float)
    dist = 1 - co_matrix
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="average")

    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=90)
    ax.set_ylabel("Distance (1 - stability)")
    ax.set_title("Stability Dendrogram")

    abs_path = _build_output_path(save_path, filename_prefix, "StabilityDendro")
    matplotlib_savefig(fig, abs_path)
    logger.info(f"[plot_stability_dendrogram] Figure was saved to: '{abs_path}'.")


@logged
def plot_stability_clustermap(co_matrix, labels, save_path, filename_prefix):
    """绘制 donor 稳定性 clustermap。

    Args:
        co_matrix: donor 间相似度矩阵。
        labels: donor 标签列表。
        save_path: 输出目录。
        filename_prefix: 输出文件名前缀。

    Returns:
        `None`。
    """
    df = pd.DataFrame(co_matrix, index=labels, columns=labels)
    cluster_grid = sns.clustermap(df, cmap="viridis", row_cluster=True, col_cluster=True)
    fig = cluster_grid.figure
    abs_path = _build_output_path(save_path, filename_prefix, "StabilityCluster")
    matplotlib_savefig(fig, abs_path)
    logger.info(f"[plot_stability_clustermap] Figure was saved to: '{abs_path}'.")


@logged
def plot_xgb_prediction_umap(adata, save_path, filename_prefix=None, skip_subcluster=False, do_return=True, **kwargs):
    """将 XGBoost 预测标签写回 AnnData 并绘制 UMAP。

    Args:
        adata: 原始 AnnData 对象。
        save_path: 保存预测文件与图像的目录。
        filename_prefix: 输出文件名前缀。
        skip_subcluster: 是否跳过子聚类/重新降维步骤。
        do_return: 是否返回写回标签后的 AnnData 子集。
        **kwargs: 透传给 `subcluster()` 的参数。

    Returns:
        若 `do_return=True`，返回包含预测标签的 AnnData 子集。

    Example:
        adata_pred = plot_xgb_prediction_umap(
            adata=adata,
            save_path=save_addr,
            filename_prefix="CD4T",
            skip_subcluster=True,
        )
    """
    from src.core.handlers.plot_wrapper import ScanpyPlotWrapper

    prefix = f"{filename_prefix.strip()}_" if isinstance(filename_prefix, str) and filename_prefix.strip() else ""
    y_pred_file = os.path.join(save_path, f"{prefix}y_pred.npy")
    dataset_file = os.path.join(save_path, f"{prefix}dataset.npz")
    if not os.path.isfile(y_pred_file):
        raise FileNotFoundError(f"Prediction file was not found: '{y_pred_file}'.")
    if not os.path.isfile(dataset_file):
        raise FileNotFoundError(f"Dataset file was not found: '{dataset_file}'.")

    y_pred = np.load(y_pred_file, allow_pickle=True)
    data = np.load(dataset_file, allow_pickle=True)
    test_idx = data["test_obs_index"]
    mapping = data["label_mapping"].item()

    adata_sub = adata[adata.obs.index.isin(test_idx)].copy()
    if adata_sub.n_obs < 100:
        logger.info(
            "[plot_xgb_prediction_umap] Warning! The AnnData subset had fewer than 100 cells. "
            "UMAP plotting will be skipped to avoid unstable layouts."
        )
        return adata_sub if do_return else None

    if not skip_subcluster:
        from src.core.adata.ops import subcluster

        logger.info("[plot_xgb_prediction_umap] Starting subclustering before UMAP plotting.")
        default_params = {
            "adata": adata_sub,
            "n_neighbors": 20,
            "n_pcs": 10,
            "skip_DR": False,
            "resolutions": [1.0],
            "use_rep": "X_scVI",
        }
        default_params.update(kwargs)
        adata_sub = subcluster(**default_params)

    adata_sub.obs.loc[test_idx, "predicted_label"] = y_pred
    adata_sub.obs.loc[test_idx, "predicted_label_name"] = adata_sub.obs.loc[test_idx, "predicted_label"].map(mapping)

    umap_plot = ScanpyPlotWrapper(func=sc.pl.umap)
    umap_plot(
        save_addr=_ensure_output_dir(save_path),
        filename=f"{prefix}XgbPrediction",
        adata=adata_sub,
        color=["disease_type", "predicted_label_name"],
        wspace=0.4,
    )
    logger.info("[plot_xgb_prediction_umap] UMAP plotting finished.")

    if do_return:
        return adata_sub


@logged
def plot_taxonomy(adata, save_path, filename_prefix=None):
    """基于 confusion matrix、SHAP 和 latent space 绘制类别 taxonomy。

    Args:
        adata: 原始 AnnData 对象，需包含 `obsm["X_scVI"]` 与 `obs["disease_type"]`。
        save_path: 数据与模型保存目录。
        filename_prefix: 输出文件名前缀。

    Returns:
        `None`。

    Example:
        plot_taxonomy(
            adata=adata,
            save_path=save_addr,
            filename_prefix="CD4T",
        )
    """
    prefix = f"{filename_prefix.strip()}_" if isinstance(filename_prefix, str) and filename_prefix.strip() else ""
    dataset_file = os.path.join(save_path, f"{prefix}dataset.npz")
    model_file = os.path.join(save_path, f"{prefix}model.json")
    if not os.path.isfile(dataset_file):
        raise FileNotFoundError(f"Dataset file was not found: '{dataset_file}'.")
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Model file was not found: '{model_file}'.")
    if "X_scVI" not in adata.obsm:
        raise KeyError("Key `X_scVI` was not found in `adata.obsm`.")
    if "disease_type" not in adata.obs.columns:
        raise KeyError("Column `disease_type` was not found in `adata.obs`.")

    data = np.load(dataset_file, allow_pickle=True)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    mapping = data["label_mapping"].item()

    clf = xgb.XGBClassifier()
    clf.load_model(model_file)
    y_pred = clf.predict(X_test)

    present_classes = sorted(np.unique(np.concatenate([y_train, y_test])))
    class_names = _ordered_class_labels(mapping, present_classes)
    cm = confusion_matrix(y_test, y_pred, labels=present_classes).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm /= row_sums

    corr = np.corrcoef(cm, rowvar=False)
    dist = 1 - corr
    dist[np.isnan(dist)] = 0
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    linkage_matrix = linkage(squareform(dist, checks=False), method="average")

    fig, ax = plt.subplots(figsize=(6, 4))
    dendrogram(linkage_matrix, labels=class_names, orientation="right", ax=ax)
    ax.set_title("Class Taxonomy (Confusion Matrix)")
    matplotlib_savefig(fig, _build_output_path(save_path, filename_prefix, "Class_taxonomy(confusion matrix)"))

    explainer = shap.TreeExplainer(clf)
    shap_values = np.asarray(explainer.shap_values(X_test))
    if shap_values.ndim == 3:
        shap_per_class = []
        labels = np.asarray(y_test)
        for cls in present_classes:
            mask = labels == cls
            if mask.sum() == 0:
                logger.info(
                    f"[plot_taxonomy] Warning! Class '{mapping.get(int(cls), cls)}' had no samples in `y_test` for SHAP taxonomy."
                )
                continue
            shap_per_class.append(shap_values[mask].mean(axis=0).ravel())
        shap_per_class = np.vstack(shap_per_class)
        shap_linkage = linkage(pdist(shap_per_class, metric="correlation"), method="average")
        fig, ax = plt.subplots(figsize=(6, 4))
        dendrogram(shap_linkage, labels=class_names[: shap_per_class.shape[0]], orientation="right", ax=ax)
        ax.set_title("Class Taxonomy (Feature Importances)")
        matplotlib_savefig(fig, _build_output_path(save_path, filename_prefix, "Class_taxonomy(feature importances)"))
    else:
        logger.info("[plot_taxonomy] Warning! Unexpected SHAP array shape. SHAP taxonomy plotting will be skipped.")

    latent = np.asarray(adata.obsm["X_scVI"])
    disease_labels = adata.obs["disease_type"].to_numpy()
    class_means = []
    latent_class_names = []
    for class_name in class_names:
        mask = disease_labels == class_name
        if mask.sum() == 0:
            logger.info(
                f"[plot_taxonomy] Warning! No cells for class '{class_name}' were found in latent space. This class will be skipped."
            )
            continue
        class_means.append(latent[mask].mean(axis=0))
        latent_class_names.append(class_name)

    if len(class_means) >= 2:
        class_means = np.vstack(class_means)
        latent_linkage = linkage(pdist(class_means, metric="euclidean"), method="ward")
        fig, ax = plt.subplots(figsize=(6, 4))
        dendrogram(latent_linkage, labels=latent_class_names, orientation="right", ax=ax)
        ax.set_title("Class Taxonomy (Latent Space)")
        matplotlib_savefig(fig, _build_output_path(save_path, filename_prefix, "Class_taxonomy(latent space)"))
    else:
        logger.info("[plot_taxonomy] Warning! Fewer than 2 classes were available for latent taxonomy plotting.")

    logger.info("[plot_taxonomy] Taxonomy plotting finished.")


@logged
def plot_lodo_stripplots(results, save_path, filename_prefix, show_points=True):
    """绘制 LODO donor 级性能 violin 图。

    Args:
        results: `_read_lodo_outcome()` 返回的结果字典。
        save_path: 输出目录。
        filename_prefix: 输出文件名前缀。
        show_points: 是否叠加 donor 级散点。

    Returns:
        `None`。

    Example:
        plot_lodo_stripplots(
            results=results,
            save_path=save_addr,
            filename_prefix="CD4T",
            show_points=True,
        )
    """
    records = []
    for idx, donor in enumerate(results["donor"]):
        metrics = results["report"][idx]["weighted avg"]
        records.extend(
            [
                {"donor": donor, "metric": "F1", "value": metrics["f1-score"]},
                {"donor": donor, "metric": "Precision", "value": metrics["precision"]},
                {"donor": donor, "metric": "Recall", "value": metrics["recall"]},
            ]
        )

    df = pd.DataFrame(records)
    if df.empty:
        logger.info("[plot_lodo_stripplots] Warning! No donor-level metrics were available. Skip plotting.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(data=df, x="metric", y="value", inner="box", cut=0, palette="Set2", ax=ax)
    if show_points:
        sns.stripplot(data=df, x="metric", y="value", color="black", size=3, jitter=True, alpha=0.5, ax=ax)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("LODO Performance (F1 / Precision / Recall)")
    ax.set_ylabel("Value")
    ax.set_xlabel("")

    abs_path = _build_output_path(save_path, filename_prefix, "LodoStripplots")
    matplotlib_savefig(fig, abs_path)
    logger.info(f"[plot_lodo_stripplots] Figure was saved to: '{abs_path}'.")


@logged
def plot_lodo_confusion_matrix(results, save_path, filename_prefix):
    """绘制 LODO 的平均 confusion matrix 与标准差矩阵。

    Args:
        results: `_read_lodo_outcome()` 返回的结果字典。
        save_path: 输出目录。
        filename_prefix: 输出文件名前缀。

    Returns:
        `None`。

    Example:
        plot_lodo_confusion_matrix(
            results=results,
            save_path=save_addr,
            filename_prefix="CD4T",
        )
    """
    mapping = results["mapping"][0]
    classes = list(mapping.keys())
    class_names = list(mapping.values())

    conf_list = []
    for y_test, y_pred in zip(results["y_test"], results["y_pred"]):
        conf_list.append(confusion_matrix(y_test, y_pred, labels=classes))

    conf_array = np.stack(conf_list, axis=0)
    mean_conf = np.mean(conf_array, axis=0)
    std_conf = np.std(conf_array, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(mean_conf, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Mean Confusion Matrix")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    sns.heatmap(std_conf, annot=True, fmt=".2f", cmap="Reds", xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Sigma of Confusion Matrix")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    fig.tight_layout()

    abs_path = _build_output_path(save_path, filename_prefix, "LodoConfusionMatrix")
    matplotlib_savefig(fig, abs_path)
    logger.info(f"[plot_lodo_confusion_matrix] Figure was saved to: '{abs_path}'.")
