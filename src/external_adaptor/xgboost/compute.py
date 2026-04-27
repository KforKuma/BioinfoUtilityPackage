"""XGBoost 数据准备、训练与相关计算函数。"""

import logging
import os
from collections import Counter
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import xgboost as xgb
from scipy.sparse import issparse
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


class SkipThis(Exception):
    """在当前数据子集不满足训练条件时抛出的跳过异常。"""


def _require_obs_column(adata, column_name: str) -> None:
    """检查 `adata.obs` 是否包含指定列。"""
    if column_name not in adata.obs.columns:
        raise KeyError(
            f"Column `{column_name}` was not found in `adata.obs`. "
            f"Available columns are: {list(adata.obs.columns)}."
        )


def _ensure_save_path(save_path: str) -> str:
    """检查并创建输出目录。"""
    if not isinstance(save_path, str) or save_path.strip() == "":
        raise ValueError("Argument `save_path` must be a non-empty string.")
    save_path = save_path.strip()
    os.makedirs(save_path, exist_ok=True)
    return save_path


@logged
def _xgb_oversample_data(
    X,
    y,
    train_obs_index=None,
    mode="random",
    random_state=42,
):
    """对训练集进行过采样。

    Args:
        X: 训练特征矩阵。
        y: 训练标签数组。
        train_obs_index: 训练样本原始索引，可选。
        mode: 过采样方式，支持 `'random'` 和 `'smote'`。
        random_state: 随机种子。

    Returns:
        三元组 `(X_res, y_res, idx_res)`。

    Example:
        X_res, y_res, idx_res = _xgb_oversample_data(
            X=X_train,
            y=y_train,
            train_obs_index=train_obs_index,
            mode="random",
        )
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("Argument `X` must be a 2-dimensional array.")
    if X.shape[0] != len(y):
        raise ValueError("Arguments `X` and `y` must contain the same number of samples.")
    if mode not in {"random", "smote"}:
        raise ValueError("Argument `mode` must be either 'random' or 'smote'.")

    if len(np.unique(y)) <= 1:
        logger.info("[_xgb_oversample_data] Warning! Only one class was detected. Skip oversampling.")
        return X, y, train_obs_index

    if mode == "random":
        counter = Counter(y)
        max_count = max(counter.values())
        X_res, y_res, idx_res = [], [], []

        for label, count in counter.items():
            cls_idx = np.where(y == label)[0]
            repeat = int(np.ceil(max_count / count))
            sampled_idx = np.tile(cls_idx, repeat)[:max_count]
            X_res.append(X[sampled_idx])
            y_res.append(y[sampled_idx])
            if train_obs_index is not None:
                idx_res.append(np.asarray(train_obs_index)[sampled_idx])

        X_res = np.vstack(X_res)
        y_res = np.concatenate(y_res)
        idx_res = np.concatenate(idx_res) if train_obs_index is not None else None
        logger.info("[_xgb_oversample_data] Oversampling finished with random replication.")
        return X_res, y_res, idx_res

    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError(
            "SMOTE requires package `imbalanced-learn`. "
            "Please install it in the target environment before using `mode='smote'`."
        ) from exc

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    if train_obs_index is not None:
        train_obs_index = np.asarray(train_obs_index)
        n_new = X_res.shape[0] - len(train_obs_index)
        idx_res = np.concatenate([train_obs_index, np.array([f"SMOTE_{i}" for i in range(n_new)], dtype=object)])
    else:
        idx_res = None

    logger.info("[_xgb_oversample_data] Oversampling finished with SMOTE.")
    return X_res, y_res, idx_res


@logged
def _prepare_subset(
    adata,
    obs_select,
    obs_key,
    group_key,
    categorical_key="disease_type",
    min_samples_per_class=10,
    verbose=True,
):
    """按 cell subtype/subpopulation 子集并做初步过滤。

    Args:
        adata: 输入 AnnData 对象。
        obs_select: 目标 cell subtype/subpopulation，可为 `None`、字符串或列表。
        obs_key: `adata.obs` 中用于筛选子集的列名。
        group_key: 实验最小单位列名，通常为患者或 donor。
        categorical_key: 待分类标签列名。
        min_samples_per_class: 每个类别至少需要的细胞数。
        verbose: 是否打印过程日志。

    Returns:
        过滤后的 AnnData 子集。

    Example:
        adata_sub = _prepare_subset(
            adata=adata,
            obs_select=["CD4 T", "CD8 T"],
            obs_key="Subset_Identity",
            group_key="Patient",
            categorical_key="disease_type",
        )
    """
    _require_obs_column(adata, obs_key)
    _require_obs_column(adata, group_key)
    _require_obs_column(adata, categorical_key)

    if obs_select is None:
        adata_sub = adata.copy()
    elif isinstance(obs_select, str):
        adata_sub = adata[adata.obs[obs_key] == obs_select].copy()
    elif isinstance(obs_select, (list, tuple, np.ndarray, set)):
        obs_select = list(obs_select)
        if not obs_select:
            raise ValueError("Argument `obs_select` must not be an empty sequence.")
        adata_sub = adata[adata.obs[obs_key].isin(obs_select)].copy()
    else:
        raise TypeError("Argument `obs_select` must be `None`, a string, or a sequence of strings.")

    if adata_sub.n_obs == 0:
        raise SkipThis("No cells were selected for the current subset.")

    if verbose:
        logger.info(f"[_prepare_subset] Selected {adata_sub.n_obs} cells from `{obs_key}`.")

    if isinstance(obs_select, str) or (isinstance(obs_select, (list, tuple, np.ndarray, set)) and len(obs_select) == 1):
        group_counts = adata_sub.obs.groupby(categorical_key)[group_key].nunique()
        total_counts = adata_sub.obs.groupby(categorical_key).size()
        valid_categories = group_counts[(group_counts >= 2) & (total_counts >= min_samples_per_class)].index
        removed = group_counts.index.difference(valid_categories)
        adata_sub = adata_sub[adata_sub.obs[categorical_key].isin(valid_categories)].copy()
        if verbose and len(removed) > 0:
            logger.info(
                f"[_prepare_subset] Warning! Categories {list(removed)} were removed because they did not satisfy "
                f"`min_samples_per_class`: {min_samples_per_class} or donor-count requirements."
            )
    elif isinstance(obs_select, (list, tuple, np.ndarray, set)):
        valid_celltypes = []
        for celltype_name in obs_select:
            adata_ct = adata_sub[adata_sub.obs[obs_key] == celltype_name]
            group_counts = adata_ct.obs.groupby(categorical_key)[group_key].nunique()
            total_counts = adata_ct.shape[0]
            if not group_counts.empty and (group_counts >= 2).all() and total_counts >= min_samples_per_class:
                valid_celltypes.append(celltype_name)
            elif verbose:
                logger.info(
                    f"[_prepare_subset] Warning! Cell subtype '{celltype_name}' was removed because it did not "
                    "satisfy donor-count or total-cell requirements."
                )

        if not valid_celltypes:
            raise SkipThis("No cell subtypes satisfied the donor-count and sample-count requirements.")
        adata_sub = adata_sub[adata_sub.obs[obs_key].isin(valid_celltypes)].copy()

    if adata_sub.n_obs == 0:
        raise SkipThis("No cells remained after subset filtering.")
    return adata_sub


@logged
def _check_and_filter_diseases(
    adata_sub,
    group_key,
    categorical_key="disease_type",
    min_samples_per_class=10,
    verbose=True,
):
    """进一步过滤疾病标签，确保可用于分类。

    Args:
        adata_sub: 已初步筛选的 AnnData 子集。
        group_key: donor / patient 列名。
        categorical_key: 分类标签列名。
        min_samples_per_class: 每类最少细胞数。
        verbose: 是否打印过程日志。

    Returns:
        过滤后的 AnnData 子集。
    """
    _require_obs_column(adata_sub, group_key)
    _require_obs_column(adata_sub, categorical_key)

    group_counts = adata_sub.obs.groupby(categorical_key)[group_key].nunique()
    total_counts = adata_sub.obs.groupby(categorical_key).size()
    valid_categories = group_counts[(group_counts >= 2) & (total_counts >= min_samples_per_class)].index
    removed = group_counts.index.difference(valid_categories)
    adata_sub = adata_sub[adata_sub.obs[categorical_key].isin(valid_categories)].copy()

    if verbose:
        if len(removed) > 0:
            logger.info(f"[_check_and_filter_diseases] Removed categories: {list(removed)}.")
        logger.info(
            f"[_check_and_filter_diseases] Remaining categories: "
            f"{list(adata_sub.obs[categorical_key].astype(str).unique())}."
        )

    if adata_sub.obs[categorical_key].nunique() <= 1:
        raise SkipThis("Not enough disease groups remained after filtering.")
    return adata_sub


@logged
def _encode_labels(adata_sub, categorical_key="disease_type", verbose=True):
    """将分类标签编码为整数。

    Args:
        adata_sub: 输入 AnnData 子集。
        categorical_key: 分类标签列名。
        verbose: 是否打印标签映射。

    Returns:
        三元组 `(y_codes, mapping, y_category)`。
    """
    _require_obs_column(adata_sub, categorical_key)

    y_category = adata_sub.obs[categorical_key].astype("category").cat.remove_unused_categories()
    y_codes = y_category.cat.codes.to_numpy()
    mapping = dict(enumerate(y_category.cat.categories))

    if verbose:
        logger.info(f"[_encode_labels] Label mapping: {mapping}.")
        logger.info(f"[_encode_labels] Label distribution:\n{y_category.value_counts()}")
    return y_codes, mapping, y_category


@logged
def _extract_features(
    adata_sub,
    method,
    train_idx,
    test_idx,
    n_components=50,
    random_state=42,
    verbose=True,
):
    """提取训练与测试特征。

    支持三种模式：
    1. `scvi`: 直接使用 `adata.obsm["X_scVI"]`。
    2. `pca`: 在训练集上拟合 PCA，再转换测试集。
    3. `combined`: 拼接标准化后的 scVI 与 PCA 特征。

    Args:
        adata_sub: 输入 AnnData 子集。
        method: 特征提取方法。
        train_idx: 训练集索引。
        test_idx: 测试集索引。
        n_components: PCA 维度。
        random_state: 随机种子。
        verbose: 是否打印说明信息。

    Returns:
        二元组 `(X_train, X_test)`。
    """
    method = str(method).lower()
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    if method not in {"scvi", "pca", "combined"}:
        raise SkipThis("Argument `method` must be 'scvi', 'pca', or 'combined'.")

    X_scvi_train = X_scvi_test = X_train_pca = X_test_pca = None

    if method in {"scvi", "combined"}:
        if "X_scVI" not in adata_sub.obsm:
            raise KeyError("Key `X_scVI` was not found in `adata_sub.obsm`.")
        X_scvi = np.asarray(adata_sub.obsm["X_scVI"])
        scaler_scvi = StandardScaler()
        X_scvi_train = scaler_scvi.fit_transform(X_scvi[train_idx])
        X_scvi_test = scaler_scvi.transform(X_scvi[test_idx])

    if method in {"pca", "combined"}:
        X_all = adata_sub.X.toarray() if issparse(adata_sub.X) else np.asarray(adata_sub.X)
        X_train_raw = X_all[train_idx]
        X_test_raw = X_all[test_idx]
        if X_train_raw.shape[0] == 0 or X_test_raw.shape[0] == 0:
            raise SkipThis("PCA feature extraction failed because train or test split was empty.")

        max_components = min(X_train_raw.shape[0], X_train_raw.shape[1], n_components)
        if max_components < 1:
            raise SkipThis("PCA feature extraction failed because no valid component could be computed.")

        scaler_pca = StandardScaler()
        X_train_scaled = scaler_pca.fit_transform(X_train_raw)
        X_test_scaled = scaler_pca.transform(X_test_raw)
        pca = PCA(n_components=max_components, random_state=random_state)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        if verbose:
            logger.info(
                f"[_extract_features] PCA explained variance ratio (first 5): "
                f"{pca.explained_variance_ratio_[:5].tolist()}."
            )

    if method == "scvi":
        return X_scvi_train, X_scvi_test
    if method == "pca":
        return X_train_pca, X_test_pca
    return np.hstack([X_scvi_train, X_train_pca]), np.hstack([X_scvi_test, X_test_pca])


@logged
def _stratified_group_split(
    y_codes,
    groups,
    n_splits=5,
    random_state=42,
    verbose=True,
    min_samples_per_class=2,
):
    """执行分层分组切分，并在必要时回退过滤问题类别。

    Args:
        y_codes: 整数标签数组。
        groups: donor / patient 分组数组。
        n_splits: `StratifiedGroupKFold` 的折数。
        random_state: 随机种子。
        verbose: 是否打印过程日志。
        min_samples_per_class: 过滤后每类最少样本数。

    Returns:
        四元组 `(train_idx, test_idx, y_codes_used, class_mapping)`。

    Example:
        train_idx, test_idx, y_used, class_mapping = _stratified_group_split(
            y_codes=y_codes,
            groups=groups,
            n_splits=5,
        )
    """
    y_codes = np.asarray(y_codes)
    groups = np.asarray(groups)
    if len(y_codes) != len(groups):
        raise ValueError("Arguments `y_codes` and `groups` must contain the same number of samples.")
    if len(np.unique(y_codes)) <= 1:
        raise SkipThis("Only one class was detected, so no stratified split could be created.")

    n_splits = max(2, int(n_splits))
    X_dummy = np.zeros(len(y_codes))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for attempt, (train_idx, test_idx) in enumerate(sgkf.split(X_dummy, y_codes, groups), start=1):
        y_train, y_test = y_codes[train_idx], y_codes[test_idx]
        if set(y_train) == set(y_test):
            if verbose:
                logger.info(
                    f"[_stratified_group_split] Successful split on attempt {attempt} with "
                    f"{len(set(y_train))} shared classes."
                )
            return train_idx, test_idx, y_codes, {label: label for label in np.unique(y_codes)}

    if verbose:
        logger.info(
            "[_stratified_group_split] Warning! No split contained all classes in both train and test. "
            "A fallback filtering path will be used."
        )

    train_idx, test_idx = next(sgkf.split(X_dummy, y_codes, groups))
    keep_classes = set(y_codes[train_idx]) & set(y_codes[test_idx])
    dropped = sorted(set(y_codes) - keep_classes)
    if verbose and dropped:
        logger.info(f"[_stratified_group_split] Warning! Dropping classes not shared by train and test: {dropped}.")

    mask_keep = np.isin(y_codes, list(keep_classes))
    y_codes_filtered = y_codes[mask_keep]
    groups_filtered = groups[mask_keep]

    for cls in np.unique(y_codes_filtered):
        if np.sum(y_codes_filtered == cls) < min_samples_per_class:
            raise SkipThis(
                f"Class `{cls}` had fewer than `min_samples_per_class`: {min_samples_per_class} after filtering."
            )

    unique_classes = np.unique(y_codes_filtered)
    class_mapping = {old: new for new, old in enumerate(unique_classes)}
    y_codes_remap = np.array([class_mapping[label] for label in y_codes_filtered])

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for attempt, (train_idx, test_idx) in enumerate(
        sgkf.split(np.zeros(len(y_codes_remap)), y_codes_remap, groups_filtered),
        start=1,
    ):
        y_train, y_test = y_codes_remap[train_idx], y_codes_remap[test_idx]
        if set(y_train) == set(y_test):
            if verbose:
                logger.info(f"[_stratified_group_split] Successful fallback split on attempt {attempt}.")
            return train_idx, test_idx, y_codes_remap, class_mapping

    raise SkipThis("Unable to find a valid train/test split with shared classes after fallback filtering.")


@logged
def _compute_sample_weights(y_train):
    """根据类别频率计算 sample weights。

    Args:
        y_train: 训练标签数组。

    Returns:
        样本权重数组。
    """
    y_train = np.asarray(y_train)
    counter = Counter(y_train)
    total = sum(counter.values())
    return np.asarray([total / counter[label] for label in y_train], dtype=float)


def _save_dataset(
    save_path,
    X_train,
    X_test,
    y_train,
    y_test,
    train_obs_index,
    test_obs_index,
    sample_weights,
    mapping,
    filename_prefix=None,
    verbose=True,
):
    """保存训练/测试数据集为压缩 `npz` 文件。

    Args:
        save_path: 保存目录。
        X_train: 训练特征矩阵。
        X_test: 测试特征矩阵。
        y_train: 训练标签数组。
        y_test: 测试标签数组。
        train_obs_index: 训练集对应的 obs index。
        test_obs_index: 测试集对应的 obs index。
        sample_weights: 样本权重数组，可为 `None`。
        mapping: 标签映射字典。
        filename_prefix: 文件名前缀。
        verbose: 是否打印保存信息。
    """
    save_path = _ensure_save_path(save_path)
    file_name = f"{filename_prefix}_dataset.npz" if filename_prefix else "dataset.npz"
    file_path = os.path.join(save_path, file_name)
    np.savez_compressed(
        file_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_obs_index=np.asarray(train_obs_index, dtype=object),
        test_obs_index=np.asarray(test_obs_index, dtype=object),
        sample_weights=sample_weights if sample_weights is not None else np.array([], dtype=float),
        label_mapping=mapping,
    )
    if verbose:
        logger.info(f"[_save_dataset] Dataset was saved to: '{file_path}'.")


@logged
def xgb_data_prepare(
    adata,
    obs_select=None,
    save_path=None,
    file_name=None,
    method="scvi",
    obs_key="Subset_Identity",
    group_key="Patient",
    categorical_key="disease_type",
    test_size=0.2,
    verbose=True,
    random_state=42,
    oversample=True,
    weightsample=True,
    min_samples_per_class=10,
):
    """准备单次 XGBoost 训练所需数据集。

    Args:
        adata: 输入 AnnData 对象。
        obs_select: 目标 cell subtype/subpopulation，可为 `None`、字符串或列表。
        save_path: 输出目录。
        file_name: 导出数据文件前缀。
        method: 特征提取方法，支持 `'scvi'`、`'pca'` 和 `'combined'`。
        obs_key: `adata.obs` 中表示 cell subtype 的列名。
        group_key: donor / patient 列名。
        categorical_key: 待预测标签列名。
        test_size: 测试集比例。
        verbose: 是否打印过程日志。
        random_state: 随机种子。
        oversample: 是否对训练集做过采样。
        weightsample: 是否计算 sample weights。
        min_samples_per_class: 每类最少细胞数。

    Returns:
        `None`。函数会把数据集导出为 `npz`。

    Example:
        xgb_data_prepare(
            adata=adata,
            obs_select="CD4 T",
            save_path=save_addr,
            file_name="CD4T_scvi",
            method="scvi",
            obs_key="Subset_Identity",
            group_key="Patient",
            categorical_key="disease_type",
            test_size=0.2,
        )
    """
    save_path = _ensure_save_path(save_path)
    if not 0 < test_size < 1:
        raise ValueError("Argument `test_size` must be between 0 and 1.")

    logger.info("[xgb_data_prepare] Starting XGBoost dataset preparation.")
    adata_sub = _prepare_subset(
        adata=adata,
        obs_select=obs_select,
        obs_key=obs_key,
        group_key=group_key,
        categorical_key=categorical_key,
        min_samples_per_class=min_samples_per_class,
        verbose=verbose,
    )
    adata_sub = _check_and_filter_diseases(
        adata_sub=adata_sub,
        group_key=group_key,
        categorical_key=categorical_key,
        min_samples_per_class=min_samples_per_class,
        verbose=verbose,
    )

    y_codes, mapping, _ = _encode_labels(adata_sub, categorical_key=categorical_key, verbose=verbose)
    groups = adata_sub.obs[group_key].to_numpy()
    min_groups_per_class = adata_sub.obs.groupby(categorical_key)[group_key].nunique().min()
    n_splits = max(2, min(int(round(1 / test_size)), int(min_groups_per_class)))

    train_idx, test_idx, y_codes_used, class_mapping = _stratified_group_split(
        y_codes=y_codes,
        groups=groups,
        n_splits=n_splits,
        random_state=random_state,
        verbose=verbose,
    )

    if class_mapping != {label: label for label in np.unique(y_codes)}:
        mask_keep = np.isin(y_codes, list(class_mapping.keys()))
        adata_sub = adata_sub[mask_keep].copy()
        groups = groups[mask_keep]
        y_codes = y_codes_used
        original_mapping = mapping.copy()
        mapping = {new: original_mapping[old] for old, new in class_mapping.items()}
    else:
        y_codes = y_codes_used

    X_train, X_test = _extract_features(
        adata_sub=adata_sub,
        method=method,
        train_idx=train_idx,
        test_idx=test_idx,
        random_state=random_state,
        verbose=verbose,
    )

    y_train = y_codes[train_idx]
    y_test = y_codes[test_idx]
    train_obs_index = adata_sub.obs.index[train_idx]
    test_obs_index = adata_sub.obs.index[test_idx]

    if oversample:
        logger.info("[xgb_data_prepare] Starting oversampling.")
        try:
            X_train, y_train, train_obs_index = _xgb_oversample_data(
                X=X_train,
                y=y_train,
                train_obs_index=train_obs_index,
                mode="smote",
                random_state=random_state,
            )
        except Exception as exc:
            logger.info(
                f"[xgb_data_prepare] Warning! SMOTE oversampling failed and random oversampling will be used instead. "
                f"Details: '{exc}'."
            )
            X_train, y_train, train_obs_index = _xgb_oversample_data(
                X=X_train,
                y=y_train,
                train_obs_index=train_obs_index,
                mode="random",
                random_state=random_state,
            )
    else:
        logger.info("[xgb_data_prepare] Skip oversampling.")

    sample_weights = _compute_sample_weights(y_train) if weightsample else None
    _save_dataset(
        save_path=save_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_obs_index=train_obs_index,
        test_obs_index=test_obs_index,
        sample_weights=sample_weights,
        mapping=mapping,
        filename_prefix=file_name,
        verbose=verbose,
    )
    logger.info("[xgb_data_prepare] Dataset preparation finished successfully.")


@logged
def xgb_data_prepare_lodo(
    adata,
    save_path,
    obs_select=None,
    obs_key="Subset_Identity",
    method="combined",
    group_key="Patient",
    categorical_key="disease_type",
    verbose=False,
    weightsample=True,
    min_samples_per_class=10,
):
    """准备 Leave-One-Donor-Out 所需数据集。

    Args:
        adata: 输入 AnnData 对象。
        save_path: 输出目录。
        obs_select: 目标 cell subtype/subpopulation，可为 `None`、字符串或列表。
        obs_key: `adata.obs` 中表示 cell subtype 的列名。
        method: 特征提取方法，支持 `'scvi'`、`'pca'` 和 `'combined'`。
        group_key: donor / patient 列名。
        categorical_key: 待预测标签列名。
        verbose: 是否打印过程日志。
        weightsample: 是否计算 sample weights。
        min_samples_per_class: 每类最少细胞数。

    Returns:
        `None`。函数会为每个 donor 导出一个 `LODO_*_dataset.npz`。

    Example:
        xgb_data_prepare_lodo(
            adata=adata,
            save_path=save_addr,
            obs_select="CD4 T",
            obs_key="Subset_Identity",
            method="combined",
            group_key="Patient",
            categorical_key="disease_type",
        )
    """
    save_path = _ensure_save_path(save_path)
    logger.info("[xgb_data_prepare_lodo] Starting LODO dataset preparation.")

    adata_sub = _prepare_subset(
        adata=adata,
        obs_select=obs_select,
        obs_key=obs_key,
        group_key=group_key,
        categorical_key=categorical_key,
        min_samples_per_class=min_samples_per_class,
        verbose=verbose,
    )
    adata_sub = _check_and_filter_diseases(
        adata_sub=adata_sub,
        group_key=group_key,
        categorical_key=categorical_key,
        min_samples_per_class=min_samples_per_class,
        verbose=verbose,
    )

    method = str(method).lower()
    if method not in {"scvi", "pca", "combined"}:
        raise ValueError("Argument `method` must be 'scvi', 'pca', or 'combined'.")

    if method in {"pca", "combined"}:
        if "X_pca" not in adata_sub.obsm:
            logger.info(
                "[xgb_data_prepare_lodo] Warning! Key `X_pca` was not found in `adata_sub.obsm`. "
                "A global PCA embedding will be computed for the current subset."
            )
            adata_tmp = adata_sub.copy()
            sc.pp.normalize_total(adata_tmp, target_sum=1e4)
            sc.pp.log1p(adata_tmp)
            sc.pp.scale(adata_tmp, max_value=10)
            sc.tl.pca(adata_tmp, n_comps=min(50, adata_tmp.n_vars, adata_tmp.n_obs))
            adata_sub.obsm["X_pca"] = adata_tmp.obsm["X_pca"]

    if method in {"scvi", "combined"} and "X_scVI" not in adata_sub.obsm:
        raise KeyError("Key `X_scVI` was not found in `adata_sub.obsm`.")

    features_dict = {}
    if method in {"scvi", "combined"}:
        features_dict["scvi"] = np.ascontiguousarray(adata_sub.obsm["X_scVI"])
    if method in {"pca", "combined"}:
        features_dict["pca"] = np.ascontiguousarray(adata_sub.obsm["X_pca"])

    y_codes, mapping, _ = _encode_labels(adata_sub, categorical_key=categorical_key, verbose=verbose)
    donors = adata_sub.obs[group_key].to_numpy()
    unique_donors = np.unique(donors)
    total = len(unique_donors)

    for index, donor in enumerate(unique_donors, start=1):
        logger.info(f"[xgb_data_prepare_lodo] ({index}/{total}) Processing donor: '{donor}'.")
        train_mask = donors != donor
        test_mask = donors == donor
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            logger.info(
                f"[xgb_data_prepare_lodo] Warning! Donor '{donor}' produced an empty train or test split and will be skipped."
            )
            continue

        X_parts_train = []
        X_parts_test = []
        if method in {"scvi", "combined"}:
            X_parts_train.append(features_dict["scvi"][train_mask])
            X_parts_test.append(features_dict["scvi"][test_mask])
        if method in {"pca", "combined"}:
            X_parts_train.append(features_dict["pca"][train_mask])
            X_parts_test.append(features_dict["pca"][test_mask])

        X_train = np.hstack(X_parts_train)
        X_test = np.hstack(X_parts_test)
        y_train = y_codes[train_mask]
        y_test = y_codes[test_mask]
        train_obs_index = adata_sub.obs.index[train_mask]
        test_obs_index = adata_sub.obs.index[test_mask]
        sample_weights = _compute_sample_weights(y_train) if weightsample else None

        _save_dataset(
            save_path=save_path,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            train_obs_index=train_obs_index,
            test_obs_index=test_obs_index,
            sample_weights=sample_weights,
            mapping=mapping,
            filename_prefix=f"LODO_{donor}",
            verbose=True,
        )

    logger.info("[xgb_data_prepare_lodo] LODO dataset preparation finished.")


@logged
def polarised_f1(y_true, y_pred):
    """计算二分类或“极化”双类场景的 macro F1。

    若输入中包含超过 2 个类别，则默认仅使用频率最高的两个类别，并输出 warning。

    Args:
        y_true: 真实标签数组。
        y_pred: 预测标签数组。

    Returns:
        二元组 `("polarised_f1", score)`。

    Example:
        metric_name, metric_value = polarised_f1(
            y_true=np.array([0, 0, 1, 1]),
            y_pred=np.array([0, 1, 1, 1]),
        )
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("Arguments `y_true` and `y_pred` must have the same shape.")

    classes, counts = np.unique(y_true, return_counts=True)
    if len(classes) < 2:
        raise ValueError("At least 2 classes are required to compute `polarised_f1`.")
    if len(classes) > 2:
        selected_classes = classes[np.argsort(counts)[-2:]]
        logger.info(
            f"[polarised_f1] Warning! More than 2 classes were detected. Only the 2 most frequent classes "
            f"will be used: {selected_classes.tolist()}."
        )
    else:
        selected_classes = classes

    mask = np.isin(y_true, selected_classes)
    score = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
    return "polarised_f1", score


@logged
def xgboost_process(
    save_path,
    filename_prefix=None,
    eval_metric="mlogloss",
    npz_file=None,
    do_return=False,
    verbose=True,
    **kwargs,
):
    """训练单个 XGBoost 多分类模型。

    Args:
        save_path: 数据与模型输出目录。
        filename_prefix: 数据文件和模型文件前缀。
        eval_metric: XGBoost 评估指标。
        npz_file: 若指定，则直接读取该数据文件而不是默认 `dataset.npz`。
        do_return: 是否返回训练后的模型对象。
        verbose: 是否打印评估日志。
        **kwargs: 透传给 `xgb.XGBClassifier` 的附加参数。

    Returns:
        若 `do_return=True`，返回训练后的 `xgb.XGBClassifier`；否则返回 `None`。

    Example:
        clf = xgboost_process(
            save_path=save_addr,
            filename_prefix="CD4T_scvi",
            eval_metric="mlogloss",
            do_return=True,
        )
    """
    save_path = _ensure_save_path(save_path)

    if npz_file is None:
        data_path = os.path.join(save_path, f"{filename_prefix}_dataset.npz" if filename_prefix else "dataset.npz")
    else:
        data_path = str(npz_file)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Dataset file was not found: '{data_path}'.")

    data = np.load(data_path, allow_pickle=True)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    sample_weights = data["sample_weights"] if "sample_weights" in data.files else np.array([], dtype=float)
    sample_weights = sample_weights if sample_weights.size > 0 else None

    default_params = {
        "objective": "multi:softprob",
        "max_depth": 6,
        "colsample_bytree": 0.8,
        "min_split_loss": 1,
        "min_child_weight": 1,
        "reg_lambda": 0,
        "n_estimators": 500,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "tree_method": "exact",
        "random_state": 42,
        "use_label_encoder": False,
    }
    default_params.update(kwargs)
    clf = xgb.XGBClassifier(**default_params)

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric=eval_metric,
        sample_weight=sample_weights,
        verbose=False,
    )

    y_pred = clf.predict(X_test)
    if verbose:
        logger.info(f"[xgboost_process] Accuracy: {accuracy_score(y_test, y_pred):.4f}.")
        logger.info(f"[xgboost_process] Classification report:\n{classification_report(y_test, y_pred, zero_division=0)}")

    prefix = f"{filename_prefix}_" if filename_prefix else ""
    model_path = os.path.join(save_path, f"{prefix}model.json")
    clf.save_model(model_path)
    logger.info(f"[xgboost_process] Model was saved to: '{model_path}'.")

    if do_return:
        return clf


@logged
def xgboost_process_lodo(save_path, filename_prefix=None, eval_metric="mlogloss", verbose=False, **kwargs):
    """对所有 LODO 数据文件循环训练 XGBoost 模型。

    Args:
        save_path: 包含 `LODO_*_dataset.npz` 的目录。
        filename_prefix: 模型文件名前缀。
        eval_metric: XGBoost 评估指标。
        verbose: 是否打印每轮分类报告。
        **kwargs: 透传给 `xgb.XGBClassifier` 的附加参数。

    Returns:
        `None`。

    Example:
        xgboost_process_lodo(
            save_path=save_addr,
            filename_prefix="CD4T_scvi",
            eval_metric="mlogloss",
        )
    """
    save_path = _ensure_save_path(save_path)
    files = sorted(file_name for file_name in os.listdir(save_path) if file_name.startswith("LODO_") and file_name.endswith("_dataset.npz"))
    if not files:
        raise FileNotFoundError(f"No LODO dataset files were found in `save_path`: '{save_path}'.")

    total = len(files)
    for index, file_name in enumerate(files, start=1):
        donor = file_name.replace("LODO_", "").replace("_dataset.npz", "")
        logger.info(f"[xgboost_process_lodo] ({index}/{total}) Processing donor: '{donor}'.")
        donor_model_prefix = f"{filename_prefix}_LODO_{donor}" if filename_prefix else f"LODO_{donor}"
        xgboost_process(
            save_path=save_path,
            filename_prefix=donor_model_prefix,
            eval_metric=eval_metric,
            npz_file=os.path.join(save_path, file_name),
            verbose=verbose,
            **kwargs,
        )

    logger.info("[xgboost_process_lodo] All donor-specific datasets were processed.")


@logged
def compute_scvi_gene_corr_fast(
    adata,
    layer=None,
    chunk_size=1000,
    method="spearman",
    top_n=50,
    verbose=True,
):
    """计算 scVI latent 与基因表达的相关性，并按绝对值返回 Top 基因。

    Args:
        adata: 包含 `obsm["X_scVI"]` 的 AnnData 对象。
        layer: 若提供，则从 `adata.layers[layer]` 读取表达矩阵；否则使用 `adata.X`。
        chunk_size: 分块计算时每块基因数。
        method: 相关性方法，支持 `'spearman'` 和 `'pearson'`。
        top_n: 每个 latent 返回的 Top 基因数量。
        verbose: 是否打印过程日志。

    Returns:
        字典，键为 `scVI_i`，值为对应 latent 的 Top 相关基因 DataFrame。

    Example:
        corr_dict = compute_scvi_gene_corr_fast(
            adata=adata,
            layer="counts",
            chunk_size=2000,
            method="spearman",
            top_n=30,
        )
        corr_dict["scVI_0"].head()
    """
    if "X_scVI" not in adata.obsm:
        raise KeyError("Key `X_scVI` was not found in `adata.obsm`.")
    if method not in {"spearman", "pearson"}:
        raise ValueError("Argument `method` must be either 'spearman' or 'pearson'.")
    if chunk_size < 1:
        raise ValueError("Argument `chunk_size` must be greater than or equal to 1.")
    if top_n < 1:
        raise ValueError("Argument `top_n` must be greater than or equal to 1.")
    if layer is not None and layer not in adata.layers:
        raise KeyError(f"Layer `{layer}` was not found in `adata.layers`.")

    X_scvi = np.asarray(adata.obsm["X_scVI"], dtype=float)
    X_gene = adata.X if layer is None else adata.layers[layer]
    gene_names = np.asarray(adata.var_names)
    n_cells, n_genes = X_gene.shape
    n_latent = X_scvi.shape[1]

    if verbose:
        logger.info(
            f"[compute_scvi_gene_corr_fast] Input summary: {n_cells} cells, {n_genes} genes, {n_latent} latent dimensions."
        )

    X_scvi_rank = np.apply_along_axis(rankdata, 0, X_scvi) if method == "spearman" else X_scvi
    result = {}

    for latent_idx in range(n_latent):
        latent_vec = X_scvi_rank[:, latent_idx]
        latent_centered = latent_vec - latent_vec.mean()
        corr_list = []
        gene_list = []

        for start in range(0, n_genes, chunk_size):
            end = min(start + chunk_size, n_genes)
            chunk = X_gene[:, start:end].toarray() if issparse(X_gene) else np.asarray(X_gene[:, start:end])
            if method == "spearman":
                chunk = np.apply_along_axis(rankdata, 0, chunk)

            chunk = chunk - chunk.mean(axis=0)
            numerator = latent_centered @ chunk
            denominator = np.sqrt((latent_centered ** 2).sum() * (chunk ** 2).sum(axis=0))
            corr = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator, dtype=float),
                where=denominator != 0,
            )
            corr_list.append(corr)
            gene_list.append(gene_names[start:end])

        all_corr = np.concatenate(corr_list)
        all_genes = np.concatenate(gene_list)
        df = pd.DataFrame({"gene": all_genes, "corr": all_corr})
        df = df.reindex(df["corr"].abs().sort_values(ascending=False).index)
        result[f"scVI_{latent_idx}"] = df.head(top_n).reset_index(drop=True)

    logger.info("[compute_scvi_gene_corr_fast] Correlation analysis finished.")
    return result
