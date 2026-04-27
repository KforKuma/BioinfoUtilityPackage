"""XGBoost 训练结果分析入口。"""

import logging
import os

import numpy as np
import xgboost as xgb
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import accuracy_score, classification_report

from .plot import (
    plot_confusion_matrix,
    plot_consensus_dendrogram,
    plot_fpr_per_class,
    plot_lodo_confusion_matrix,
    plot_lodo_stripplots,
    plot_roc_per_class,
    plot_shap_summary,
    plot_stability_clustermap,
    plot_stability_dendrogram,
    plot_tree_importance,
)
from .support import (
    _compute_cm,
    _donor_vectors_from_proba,
    _read_lodo_outcome,
    bootstrap_consensus_dendrogram,
    compute_donor_similarity_matrix,
)
from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


def _resolve_dataset_and_model(save_path: str, filename_prefix=None) -> tuple[str, str, str]:
    """解析单次训练所需的数据与模型路径。"""
    if not isinstance(save_path, str) or save_path.strip() == "":
        raise ValueError("Argument `save_path` must be a non-empty string.")
    save_path = save_path.strip()
    prefix = f"{filename_prefix.strip()}_" if isinstance(filename_prefix, str) and filename_prefix.strip() else ""
    dataset_path = os.path.join(save_path, f"{prefix}dataset.npz")
    model_path = os.path.join(save_path, f"{prefix}model.json")
    output_dir = os.path.join(save_path, "output")

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file was not found: '{dataset_path}'.")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file was not found: '{model_path}'.")

    os.makedirs(output_dir, exist_ok=True)
    return dataset_path, model_path, output_dir


@logged
def xgb_outcome_analyze(save_path, filename_prefix=None):
    """分析单次 XGBoost 训练结果并导出常见图表。

    Args:
        save_path: 包含 `dataset.npz`、`model.json` 与输出目录的路径。
        filename_prefix: 数据文件与模型文件的前缀。

    Returns:
        字典，包含 `shap`、`confusion_matrix` 和 `importance_df`。

    Example:
        result = xgb_outcome_analyze(
            save_path=save_addr,
            filename_prefix="Tcell",
        )
        result["importance_df"].head()
    """
    dataset_path, model_path, output_dir = _resolve_dataset_and_model(save_path, filename_prefix)
    prefix = f"{filename_prefix.strip()}_" if isinstance(filename_prefix, str) and filename_prefix.strip() else ""

    data = np.load(dataset_path, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]
    label_mapping = data["label_mapping"].item()

    clf = xgb.XGBClassifier()
    clf.load_model(model_path)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    np.save(os.path.join(save_path, f"{prefix}y_pred.npy"), y_pred)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    result_path = os.path.join(output_dir, f"{prefix}classification_result.txt")
    with open(result_path, "w", encoding="utf-8") as file:
        file.write(f"Accuracy: {acc:.4f}\n\n")
        file.write("Classification Report:\n")
        file.write(report)
    logger.info(f"[xgb_outcome_analyze] Classification report was saved to: '{result_path}'.")

    importance_df = plot_tree_importance(clf, save_path, filename_prefix)
    confusion_matrix = _compute_cm(y_test, y_pred)
    plot_confusion_matrix(
        cm_matrix=confusion_matrix,
        label_mapping=label_mapping,
        save_path=save_path,
        filename_prefix=filename_prefix,
    )
    shap_dict = plot_shap_summary(clf, X_test, label_mapping, save_path, filename_prefix)
    plot_roc_per_class(y_test, y_proba, label_mapping, save_path, filename_prefix)
    plot_fpr_per_class(y_test, y_pred, label_mapping, save_path, filename_prefix)

    logger.info("[xgb_outcome_analyze] Finished single-run outcome analysis.")
    return {
        "shap": shap_dict,
        "confusion_matrix": confusion_matrix,
        "importance_df": importance_df,
    }


@logged
def xgb_outcome_analyze_lodo(save_path, filename_prefix=None):
    """分析 Leave-One-Donor-Out 训练结果并导出稳定性相关图表。

    Args:
        save_path: 包含 `LODO_*_dataset.npz` 和对应模型输出的目录。
        filename_prefix: 模型文件名前缀。

    Returns:
        字典，包含 `results`、`similarity_matrix`、`consensus_tree` 和 `branch_supports`。

    Example:
        lodo_summary = xgb_outcome_analyze_lodo(
            save_path=save_addr,
            filename_prefix="Tcell",
        )
        lodo_summary["similarity_matrix"].shape
    """
    results = _read_lodo_outcome(save_path, filename_prefix)
    os.makedirs(os.path.join(save_path, "output"), exist_ok=True)

    logger.info("[xgb_outcome_analyze_lodo] Starting `plot_lodo_stripplots`.")
    plot_lodo_stripplots(results, save_path=save_path, filename_prefix=filename_prefix)

    logger.info("[xgb_outcome_analyze_lodo] Starting `plot_lodo_confusion_matrix`.")
    plot_lodo_confusion_matrix(results, save_path=save_path, filename_prefix=filename_prefix)

    donor_labels = results["donor"]
    label_mapping = results["mapping"][0]
    if not isinstance(label_mapping, dict):
        raise TypeError("Object `results['mapping'][0]` must be a dictionary.")
    donor_mat = _donor_vectors_from_proba(results["y_proba"])
    sim = compute_donor_similarity_matrix(donor_mat, metric="cosine")

    logger.info("[xgb_outcome_analyze_lodo] Starting `plot_stability_dendrogram`.")
    plot_stability_dendrogram(sim, donor_labels, save_path=save_path, filename_prefix=filename_prefix)

    logger.info("[xgb_outcome_analyze_lodo] Starting `plot_stability_clustermap`.")
    plot_stability_clustermap(sim, donor_labels, save_path=save_path, filename_prefix=filename_prefix)

    logger.info("[xgb_outcome_analyze_lodo] Starting `plot_consensus_dendrogram`.")
    consensus_tree, branch_supports = bootstrap_consensus_dendrogram(
        sim_matrix=sim,
        n_bootstrap=100,
        method="average",
        support_threshold=0.5,
    )

    sim_for_real = sim.copy()
    np.fill_diagonal(sim_for_real, 0)
    distance_condensed = squareform(sim_for_real, checks=False)
    Z_real = linkage(distance_condensed, method="average")
    plot_consensus_dendrogram(
        Z_real,
        donor_labels,
        branch_supports,
        save_path=save_path,
        filename_prefix=filename_prefix,
    )

    logger.info("[xgb_outcome_analyze_lodo] Finished LODO outcome analysis.")
    return {
        "results": results,
        "similarity_matrix": sim,
        "consensus_tree": consensus_tree,
        "branch_supports": branch_supports,
    }
