"""XGBoost 结果分析辅助函数。"""

import logging
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


def _resolve_lodo_files(save_path: str) -> List[str]:
    """收集 LODO 数据文件。"""
    if not isinstance(save_path, str) or save_path.strip() == "":
        raise ValueError("Argument `save_path` must be a non-empty string.")
    if not os.path.isdir(save_path):
        raise FileNotFoundError(f"Directory `save_path` was not found: '{save_path}'.")

    files = sorted(
        file_name
        for file_name in os.listdir(save_path)
        if file_name.startswith("LODO_") and file_name.endswith("_dataset.npz")
    )
    if not files:
        raise FileNotFoundError(f"No LODO dataset files were found in `save_path`: '{save_path}'.")
    return files


@logged
def _read_lodo_outcome(save_path, filename_prefix=None):
    """读取 LODO 数据文件并基于已保存模型回填预测结果。

    Args:
        save_path: 包含 `LODO_*_dataset.npz` 与模型文件的目录。
        filename_prefix: 模型文件名前缀；若为 `None`，默认读取 `model.json`。

    Returns:
        字典，包含 donor、预测结果、概率、分类报告和标签映射。

    Example:
        results = _read_lodo_outcome(
            save_path=save_path,
            filename_prefix="Tcell",
        )
        # 结果可继续送入 `plot_lodo_stripplots()` 或 `plot_lodo_confusion_matrix()`
        len(results["donor"])
    """
    files = _resolve_lodo_files(save_path)
    results = {
        "donor": [],
        "y_test": [],
        "y_pred": [],
        "y_proba": [],
        "report": [],
        "accuracy": [],
        "mapping": [],
    }

    total = len(files)
    for index, file_name in enumerate(files, start=1):
        match = re.match(r"^LODO_(.+)_dataset\.npz$", file_name)
        if not match:
            raise ValueError(f"Unexpected LODO dataset filename: '{file_name}'.")

        donor = match.group(1)
        logger.info(f"[_read_lodo_outcome] ({index}/{total}) Processing donor: '{donor}'.")
        dataset_path = os.path.join(save_path, file_name)
        data = np.load(dataset_path, allow_pickle=True)
        X_test = data["X_test"]
        y_test = data["y_test"]
        mapping = data["label_mapping"].item()

        if isinstance(filename_prefix, str) and filename_prefix.strip():
            donor_model_name = f"{filename_prefix.strip()}_LODO_{donor}_model.json"
            shared_model_name = f"{filename_prefix.strip()}_model.json"
        else:
            donor_model_name = f"LODO_{donor}_model.json"
            shared_model_name = "model.json"

        donor_model_path = os.path.join(save_path, donor_model_name)
        shared_model_path = os.path.join(save_path, shared_model_name)
        if os.path.isfile(donor_model_path):
            model_path = donor_model_path
        elif os.path.isfile(shared_model_path):
            model_path = shared_model_path
            logger.info(
                f"[_read_lodo_outcome] Warning! Donor-specific model was not found for donor '{donor}'. "
                f"Fallback to shared model: '{shared_model_path}'."
            )
        else:
            raise FileNotFoundError(
                f"Neither donor-specific model '{donor_model_path}' nor shared model '{shared_model_path}' was found."
            )

        clf = xgb.XGBClassifier()
        clf.load_model(model_path)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results["donor"].append(donor)
        results["y_test"].append(y_test)
        results["y_pred"].append(y_pred)
        results["y_proba"].append(y_proba)
        results["report"].append(report)
        results["accuracy"].append(acc)
        results["mapping"].append(mapping)

    logger.info(f"[_read_lodo_outcome] Loaded {len(results['donor'])} donor-specific outcomes.")
    return results


def _compute_cm(y_test, y_pred):
    """计算按行归一化的 confusion matrix。

    Args:
        y_test: 真实标签数组。
        y_pred: 预测标签数组。

    Returns:
        归一化后的 confusion matrix。

    Example:
        cm = _compute_cm(
            y_test=np.array([0, 0, 1, 1]),
            y_pred=np.array([0, 1, 1, 1]),
        )
        cm.shape
    """
    cm = confusion_matrix(y_test, y_pred)
    cm_sum = cm.sum(axis=1, keepdims=True)
    return np.divide(
        cm.astype(np.float32),
        cm_sum,
        out=np.zeros_like(cm, dtype=np.float32),
        where=cm_sum != 0,
    )


def _donor_vectors_from_preds(y_pred_list, n_classes, label_shift=1):
    """将 donor 级预测标签转换为类别占比矩阵。

    Args:
        y_pred_list: 每个 donor 的预测标签数组列表。
        n_classes: 类别总数。
        label_shift: 若标签从 1 开始编码则设为 1，否则设为 0。

    Returns:
        形状为 `(n_donors, n_classes)` 的占比矩阵。
    """
    if n_classes <= 0:
        raise ValueError("Argument `n_classes` must be greater than 0.")

    vectors = []
    for y_pred in y_pred_list:
        if label_shift == 1:
            counts = np.bincount(np.asarray(y_pred, dtype=int), minlength=n_classes + 1)[1:]
        else:
            counts = np.bincount(np.asarray(y_pred, dtype=int), minlength=n_classes)
        total = counts.sum()
        vectors.append(np.zeros(n_classes) if total == 0 else counts / total)
    return np.vstack(vectors)


def _donor_vectors_from_proba(y_proba_list):
    """将 donor 级概率矩阵压缩为平均概率矩阵。"""
    if not y_proba_list:
        raise ValueError("Argument `y_proba_list` must not be empty.")
    return np.vstack([np.asarray(prob).mean(axis=0) for prob in y_proba_list])


def compute_donor_similarity_matrix(donor_matrix, metric="cosine"):
    """计算 donor 间相似度矩阵。

    Args:
        donor_matrix: 形状为 `(n_donors, n_features)` 的 donor 表征矩阵。
        metric: 距离度量，例如 `'cosine'` 或 `'jensenshannon'`。

    Returns:
        形状为 `(n_donors, n_donors)` 的相似度矩阵。

    Example:
        sim = compute_donor_similarity_matrix(
            donor_matrix=np.array([[0.2, 0.8], [0.1, 0.9], [0.8, 0.2]]),
            metric="cosine",
        )
        sim.shape
    """
    donor_matrix = np.asarray(donor_matrix, dtype=float)
    if donor_matrix.ndim != 2:
        raise ValueError("Argument `donor_matrix` must be a 2-dimensional array.")
    if donor_matrix.shape[0] < 2:
        raise ValueError("Argument `donor_matrix` must contain at least 2 donors.")

    if metric == "cosine":
        distance = pdist(donor_matrix, metric="cosine")
    elif metric == "jensenshannon":
        distance = pdist(donor_matrix, metric="jensenshannon")
    else:
        distance = pdist(donor_matrix, metric=metric)

    similarity = 1 - squareform(distance)
    np.fill_diagonal(similarity, 1.0)
    return similarity


@logged
def get_internal_clusters(Z, n_leaves):
    """遍历 linkage 矩阵，返回每个内部节点对应的叶子索引。

    Args:
        Z: `scipy` 的 linkage matrix。
        n_leaves: 叶子节点数量。

    Returns:
        字典，键为 cluster index，值为该内部节点包含的叶子索引列表。
    """
    clusters: Dict[int, List[int]] = {}
    for index, row in enumerate(Z):
        left, right = int(row[0]), int(row[1])
        left_leaves = [left] if left < n_leaves else clusters[left]
        right_leaves = [right] if right < n_leaves else clusters[right]
        clusters[index + n_leaves] = left_leaves + right_leaves
    return clusters


@logged
def bootstrap_consensus_dendrogram(sim_matrix, n_bootstrap=100, method="average", support_threshold=0.7):
    """基于 donor 相似度矩阵构建 bootstrap consensus dendrogram。

    Args:
        sim_matrix: 形状为 `(n_donors, n_donors)` 的 donor 相似度矩阵。
        n_bootstrap: bootstrap 次数。
        method: linkage 方法。
        support_threshold: 仅保留支持度不低于该阈值的 branch。

    Returns:
        二元组 `(Z_consensus, branch_supports)`。

    Example:
        Z_consensus, branch_supports = bootstrap_consensus_dendrogram(
            sim_matrix=sim,
            n_bootstrap=100,
            method="average",
            support_threshold=0.5,
        )
    """
    sim_matrix = np.asarray(sim_matrix, dtype=float)
    if sim_matrix.ndim != 2 or sim_matrix.shape[0] != sim_matrix.shape[1]:
        raise ValueError("Argument `sim_matrix` must be a square matrix.")
    if n_bootstrap < 1:
        raise ValueError("Argument `n_bootstrap` must be greater than or equal to 1.")

    n_donors = sim_matrix.shape[0]
    if n_donors < 2:
        raise ValueError("Argument `sim_matrix` must contain at least 2 donors.")

    branch_counter: Counter[frozenset[int]] = Counter()

    for _ in range(n_bootstrap):
        sampled_idx = np.random.choice(n_donors, size=n_donors, replace=True)
        sim_sampled = sim_matrix[np.ix_(sampled_idx, sampled_idx)]
        dist_matrix = 1 - sim_sampled
        np.fill_diagonal(dist_matrix, 0)
        Z = linkage(squareform(dist_matrix, checks=False), method=method)

        clusters = get_internal_clusters(Z, len(sampled_idx))
        for cluster_leaves in clusters.values():
            if len(cluster_leaves) > 1:
                global_leaves = frozenset(sampled_idx[leaf] for leaf in cluster_leaves)
                branch_counter[global_leaves] += 1

    branch_supports = {branch: count / n_bootstrap for branch, count in branch_counter.items()}
    consensus_branches = {
        branch: support for branch, support in branch_supports.items() if support >= support_threshold
    }

    dist_matrix = np.ones((n_donors, n_donors), dtype=float)
    np.fill_diagonal(dist_matrix, 0)
    for branch, support in consensus_branches.items():
        idx = list(branch)
        for i in idx:
            for j in idx:
                if i != j:
                    dist_matrix[i, j] = min(dist_matrix[i, j], 1 - support)

    Z_consensus = linkage(squareform(dist_matrix, checks=False), method=method)
    return Z_consensus, branch_supports
