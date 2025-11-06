# Standard library
import os
import re
from collections import Counter

# Third-party
import numpy as np
import scanpy as sc
import xgboost as xgb
import seaborn as sns
import shap

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import (StandardScaler, label_binarize)
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report, plot_confusion_matrix,
                                 roc_curve, roc_auc_score)
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

import matplotlib
matplotlib.use("Agg")  # 必须在导入 pyplot 之前设置后端
import matplotlib.pyplot as plt


def _read_lodo_outcome(save_path, filename_prefix=None):
    # 默认名为 f"{save_path}/LODO_{donor}_dataset.npz"
    files = [f for f in os.listdir(save_path)
             if f.startswith("LODO_") and f.endswith(f"dataset.npz")]
    total = len(files)

    # 准备字典
    results = {"donor": [], "y_test": [], "y_pred": [], "y_proba": [],
               "report": [], "accuracy": [], "mapping": []}
    for i, file in enumerate(files, start=1):
        # 获取 {donor} 字段；为这个写一个正则是不是有点搞笑了……
        m = re.match(r"^LODO_(.+)_dataset\.npz$", file)
        if not m:
            raise ValueError(f"Unexpected filename: {file}")
        donor = m.group(1)

        # 开始处理数据
        print(f"[xgb_outcome_analyze_lodo] ({i}/{total}) Processing donor: {donor}")
        npz_path = os.path.join(save_path, file)
        data = np.load(npz_path, allow_pickle=True)
        X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"];
        mapping = data["label_mapping"].item()

        # 加载模型
        clf = xgb.XGBClassifier()
        prefix = f"{filename_prefix}_" if filename_prefix else ""
        clf.load_model(f"{save_path}/{prefix}model.json")

        # 预测
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        # 分类报告
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # 结果整合
        results["donor"].append(donor)
        results["y_test"].append(y_test)
        results["y_pred"].append(y_pred)
        results["y_proba"].append(y_proba)
        results["report"].append(report)
        results["accuracy"].append(acc)
        results["mapping"].append(mapping)


    return results

def _compute_cm(y_test, y_pred):
    """
    Compute a normalized confusion matrix.

    Parameters
    ----------
    y_test : array-like, shape (n_samples,)
        Ground truth target values.

    y_pred : array-like, shape (n_samples,)
        Estimated targets as returned by a classifier.

    Returns
    -------
    cm_normalized : ndarray, shape (n_classes, n_classes)
        Normalized confusion matrix. The rows correspond to the true labels and
        the columns correspond to the predicted labels. The entries are normalized
        such that the sum of each row is 1.
    """
    cm = confusion_matrix(y_test, y_pred)
    # 每行归一化
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm.astype(np.float32),
        cm_sum,
        out=np.zeros_like(cm, dtype=np.float32),
        where=cm_sum != 0
    )
    return cm_normalized

def _donor_vectors_from_preds(y_pred_list, n_classes, label_shift=1):
    """
    label_shift: 如果标签是1..K，设为1；如果是0..K-1，设为0
    返回 shape (n_donors, n_classes) 的占比矩阵
    """
    vecs = []
    for y in y_pred_list:
        if label_shift == 1:
            counts = np.bincount(y, minlength=n_classes+1)[1:]
        else:
            counts = np.bincount(y, minlength=n_classes)
        total = counts.sum()
        if total == 0:
            vecs.append(np.zeros(n_classes))
        else:
            vecs.append(counts / total)
    return np.vstack(vecs)

def _donor_vectors_from_proba(y_proba_list):
    """每个 donor 的 y_proba_list[i] 形状 (n_cells_i, n_classes)"""
    return np.vstack([p.mean(axis=0) for p in y_proba_list])

def compute_donor_similarity_matrix(donor_matrix, metric='cosine'):
    """
    donor_matrix: n_donors x n_classes (概率或占比)
    metric: 'cosine' or 'jensenshannon'
    返回相似度矩阵 (n_donors x n_donors)
    """
    if metric == 'cosine':
        dist = pdist(donor_matrix, metric='cosine')
        sim = 1 - squareform(dist)
    elif metric == 'jensenshannon':
        # pdist(..., 'jensenshannon') 返回 sqrt(JS divergence) in newer scipy
        dist = pdist(donor_matrix, metric='jensenshannon')
        sim = 1 - squareform(dist)  # 注意尺度和解释
    else:
        dist = pdist(donor_matrix, metric=metric)
        sim = 1 - squareform(dist)
    return sim


def get_internal_clusters(Z, n_leaves):
    """
    遍历 linkage 矩阵，返回每个内部节点对应的叶子 index 列表
    """
    clusters = {}
    for i, row in enumerate(Z):
        left, right = int(row[0]), int(row[1])
        if left < n_leaves:
            left_leaves = [left]
        else:
            left_leaves = clusters[left]
        if right < n_leaves:
            right_leaves = [right]
        else:
            right_leaves = clusters[right]
        clusters[i + n_leaves] = left_leaves + right_leaves
    return clusters  # key = cluster index, value = list of leaf indices

def bootstrap_consensus_dendrogram(sim_matrix, n_bootstrap=100, method="average", support_threshold=0.7):
    """
    基于 donor × donor 相似度矩阵 (sim_matrix) 计算 bootstrap consensus dendrogram

    Parameters
    ----------
    sim_matrix : ndarray, shape (n_donors, n_donors)
        donor × donor 相似性矩阵，基于 y_proba 计算
    n_bootstrap : int
        bootstrap 重采样次数
    method : str
        linkage 方法
    support_threshold : float
        保留的最小 branch support

    Returns
    -------
    Z_consensus : ndarray
        consensus linkage matrix
    branch_supports : dict
        {frozenset(donor_indices): support_value}
    """
    import numpy as np
    from collections import Counter
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    n_donors = sim_matrix.shape[0]
    branch_counter = Counter()

    for b in range(n_bootstrap):
        # bootstrap donors（这里是对 donor 索引采样，而不是重新计算 sim）
        sampled_idx = np.random.choice(n_donors, size=n_donors, replace=True)
        sim_sampled = sim_matrix[np.ix_(sampled_idx, sampled_idx)]
        dist_matrix = 1 - sim_sampled
        np.fill_diagonal(dist_matrix, 0)

        Z = linkage(squareform(dist_matrix), method=method)

        # 获取 bootstrap 树的所有 cluster
        clusters = get_internal_clusters(Z, len(sampled_idx))
        for cluster_leaves in clusters.values():
            if len(cluster_leaves) > 1:
                # 注意：cluster_leaves 是在 bootstrap 子集中，需要映射回全局 donor index
                global_leaves = frozenset(sampled_idx[i] for i in cluster_leaves)
                branch_counter[global_leaves] += 1

    # 计算分支支持
    branch_supports = {branch: count / n_bootstrap for branch, count in branch_counter.items()}

    # 只保留高支持分支
    consensus_branches = {branch: support for branch, support in branch_supports.items()
                          if support >= support_threshold}

    # 构建 consensus distance matrix
    dist_matrix = np.ones((n_donors, n_donors))
    np.fill_diagonal(dist_matrix, 0)
    for branch, support in consensus_branches.items():
        idx = list(branch)
        for i in idx:
            for j in idx:
                if i != j:
                    dist_matrix[i, j] = min(dist_matrix[i, j], 1 - support)

    Z_consensus = linkage(squareform(dist_matrix), method=method)
    return Z_consensus, branch_supports