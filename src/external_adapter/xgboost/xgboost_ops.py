# Standard library
import os
import sys
import gc
from collections import Counter

# Third-party
import numpy as np
import scanpy as sc
import xgboost as xgb
import seaborn as sns
import shap
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")  # 必须在导入 pyplot 之前设置后端
import matplotlib.pyplot as plt


# from src.core.base_anndata_ops import subcluster



class SkipThis(Exception):
    """Raised when the dataset cannot be split properly and should be skipped."""
    pass


def _xgb_oversample_data(
        X,
        y,
        train_obs_index=None,
        mode="random",
        random_state=42
):
    """
    Oversampling utility function.

    Parameters
    ----------
    X : np.ndarray
        Training feature matrix.
    y : np.ndarray
        Training labels (encoded as int or str).
    train_obs_index : np.ndarray or None
        Original index for tracking (optional).
    mode : {"random", "smote"}
        Oversampling strategy:
        - "random": naive resampling (复制 minority 类).
        - "smote": synthetic oversampling.
    random_state : int
        Random seed.

    Returns
    -------
    X_res : np.ndarray
        Oversampled training features.
    y_res : np.ndarray
        Oversampled labels.
    idx_res : np.ndarray or None
        Oversampled obs_index (if given).
    """

    rng = np.random.default_rng(random_state)
    y = np.array(y)

    if mode == "random":
        # --- 简单随机复制 ---
        counter = Counter(y)
        max_count = max(counter.values())
        X_res, y_res, idx_res = [], [], []

        for cls, count in counter.items():
            idx_cls = np.where(y == cls)[0]
            n_repeat = int(np.ceil(max_count / count))
            idx_aug = np.tile(idx_cls, n_repeat)[:max_count]

            X_res.append(X[idx_aug])
            y_res.append(y[idx_aug])
            if train_obs_index is not None:
                idx_res.append(train_obs_index[idx_aug])

        X_res = np.vstack(X_res)
        y_res = np.concatenate(y_res)
        idx_res = np.concatenate(idx_res) if train_obs_index is not None else None

        return X_res, y_res, idx_res

    elif mode == "smote":
        # --- 使用 imbalanced-learn 的 SMOTE ---
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            raise ImportError("SMOTE requires 'imbalanced-learn' package. Install via `pip install imbalanced-learn`.")

        sm = SMOTE(random_state=random_state)
        X_res, y_res = sm.fit_resample(X, y)

        if train_obs_index is not None:
            # 新生成的样本没有对应 index，用占位符补充
            n_new = X_res.shape[0] - len(train_obs_index)
            idx_res = np.concatenate([train_obs_index, np.array([f"SMOTE_{i}" for i in range(n_new)])])
        else:
            idx_res = None

        return X_res, y_res, idx_res

    else:
        raise ValueError("mode must be 'random' or 'smote'")


def _prepare_subset(adata, obs_select, obs_key, group_key, min_samples_per_class=10, verbose=True):
    """
    根据 obs_select 选择子集，并做初步过滤：
    - 如果 obs_select=None，返回全数据
    - 如果 obs_select 是单个亚群，要求该亚群每个 disease 至少有2个 patient，且总样本数>=min_samples_per_class
    - 如果 obs_select 是多个亚群，逐个检查，去掉不满足条件的
    """
    import numpy as np

    if obs_select is None:
        adata_sub = adata.copy()
    elif isinstance(obs_select, (list, np.ndarray)):
        adata_sub = adata[adata.obs[obs_key].isin(obs_select)].copy()
    else:  # 单个 str
        adata_sub = adata[adata.obs[obs_key] == obs_select].copy()

    if verbose:
        print("--> Subset selected:")
        print(adata_sub.obs[obs_key].value_counts())

    # （1）单亚群情况
    if isinstance(obs_select, str) or (isinstance(obs_select, (list, np.ndarray)) and len(obs_select) == 1):
        counts_per_disease = adata_sub.obs.groupby("disease_type")[group_key].nunique()
        valid_diseases = counts_per_disease[(counts_per_disease >= 2) &
                                            (adata_sub.obs.groupby(
                                                "disease_type").size() >= min_samples_per_class)].index
        removed = counts_per_disease.index.difference(valid_diseases)
        adata_sub = adata_sub[adata_sub.obs["disease_type"].isin(valid_diseases)].copy()
        if verbose and len(removed) > 0:
            print(
                f"--> Removed disease groups (not enough patients or samples <{min_samples_per_class}): {list(removed)}")

    # （2）多亚群情况
    elif isinstance(obs_select, (list, np.ndarray)):
        valid_celltypes = []
        for ct in obs_select:
            adata_ct = adata_sub[adata_sub.obs[obs_key] == ct]
            counts_per_disease = adata_ct.obs.groupby("disease_type")[group_key].nunique()
            total_counts = adata_ct.shape[0]
            if (counts_per_disease >= 2).all() and total_counts >= min_samples_per_class:
                valid_celltypes.append(ct)
            elif verbose:
                print(f"--> Removed cell type {ct} (not enough patients per disease or total <{min_samples_per_class})")
        if len(valid_celltypes) == 0:
            raise SkipThis("No cell types satisfy the patient/sample count requirements.")
        adata_sub = adata_sub[adata_sub.obs[obs_key].isin(valid_celltypes)].copy()

    return adata_sub


def _check_and_filter_diseases(adata_sub, group_key, min_samples_per_class=10, verbose=True):
    """
    检查并过滤疾病分组：
    - 每个疾病至少有 2 个不同的 patient
    - 每个疾病总样本数 >= min_samples_per_class
    - 至少保留 2 个疾病分类，否则报错
    """
    counts_per_disease = adata_sub.obs.groupby("disease_type")[group_key].nunique()
    total_counts = adata_sub.obs.groupby("disease_type").size()

    valid_diseases = counts_per_disease[(counts_per_disease >= 2) & (total_counts >= min_samples_per_class)].index
    removed = counts_per_disease.index.difference(valid_diseases)

    adata_sub = adata_sub[adata_sub.obs["disease_type"].isin(valid_diseases)].copy()

    if verbose:
        if len(removed) > 0:
            print(f"--> Removed disease groups: {list(removed)}")
        print("--> Remaining diseases:", list(adata_sub.obs["disease_type"].unique()))

    if adata_sub.obs["disease_type"].nunique() <= 1:
        raise SkipThis("Not enough disease groups left for classification after filtering.")

    return adata_sub


def _encode_labels(adata_sub, label_key="disease_type", verbose=True):
    """
    将标签编码为整数 codes，同时返回映射字典
    """
    y = adata_sub.obs[label_key].astype("category")
    y = y.cat.remove_unused_categories()

    y_codes = y.cat.codes.values
    mapping = dict(enumerate(y.cat.categories))

    if verbose:
        print("--> Label mapping:", mapping)
        print(y.value_counts())

    return y_codes, mapping, y


def _extract_features(adata_sub, method, train_idx, test_idx, n_components=50, random_state=42, verbose=True):
    """
    提取特征矩阵 X
    - scVI: 直接取 obsm["X_scVI"]
    - pca: 只在训练集上拟合，再 transform 测试集，避免信息泄露
    - combined: scVI + PCA 特征拼接，每种特征独立标准化
    """

    # --- scVI 特征 ---
    if method.lower() in ["scvi", "combined"]:
        X_scvi = adata_sub.obsm["X_scVI"]
        X_scvi_train_raw, X_scvi_test_raw = X_scvi[train_idx], X_scvi[test_idx]

        # 独立标准化 scVI
        scaler_scvi = StandardScaler()
        X_scvi_train = scaler_scvi.fit_transform(X_scvi_train_raw)
        X_scvi_test = scaler_scvi.transform(X_scvi_test_raw)

    # --- PCA 特征 ---
    if method.lower() in ["pca", "combined"]:
        # 先基础预处理
        adata_tmp = adata_sub.copy()
        # sc.pp.normalize_total(adata_tmp, target_sum=1e4)
        # sc.pp.log1p(adata_tmp)

        # 转为 numpy
        X_all = adata_tmp.X.toarray() if not isinstance(adata_tmp.X, np.ndarray) else adata_tmp.X
        X_train_raw, X_test_raw = X_all[train_idx], X_all[test_idx]

        # 标准化 + PCA
        scaler_pca = StandardScaler()
        X_train_scaled = scaler_pca.fit_transform(X_train_raw)
        X_test_scaled = scaler_pca.transform(X_test_raw)
        try:
            pca = PCA(n_components=n_components, random_state=random_state)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)
        except ValueError as e:
            # 自动调小 n_components
            n_components_new = min(X_train_raw.shape[0], X_test_raw.shape[0])
            print(f"[Warning] PCA ValueError: {e}. Reset n_components={n_components_new}")
            pca = PCA(n_components=n_components_new, random_state=random_state)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)
        if verbose:
            print(f"--> PCA explained variance (first 5): {pca.explained_variance_ratio_[:5]}")

    # --- 根据方法返回结果 ---
    if method.lower() == "scvi":
        return X_scvi_train, X_scvi_test
    elif method.lower() == "pca":
        return X_train_pca, X_test_pca
    elif method.lower() == "combined":
        # 拼接 scVI + PCA
        X_train = np.hstack([X_scvi_train, X_train_pca])
        X_test = np.hstack([X_scvi_test, X_test_pca])
        return X_train, X_test
    else:
        raise SkipThis("method must be 'scvi', 'pca' or 'combined'")


def _stratified_group_split(y_codes, groups, n_splits=5, random_state=42, verbose=True, min_samples_per_class=2):
    """
    使用 StratifiedGroupKFold 拆分训练/测试集，确保每个类别都在训练和测试中出现。
    如果无法保证，则剔除只出现在 train 或 test 的类别，重新 remap。
    如果仍无法保证或样本数不足，则抛出 SkipThis。
    """

    X_dummy = np.zeros(len(y_codes))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # 第一轮尝试
    for attempt, (train_idx, test_idx) in enumerate(sgkf.split(X_dummy, y_codes, groups), 1):
        y_train, y_test = y_codes[train_idx], y_codes[test_idx]
        if set(y_train) == set(y_test):
            if verbose:
                print(f"--> Successful split on attempt {attempt}: all {len(set(y_train))} classes present.")
            return train_idx, test_idx, y_codes, {c: c for c in np.unique(y_codes)}

    if verbose:
        print("--> Warning! Could not find a split with all classes in both train/test.")

    # 兜底：剔除只在 train 或 test 中的类别
    train_idx, test_idx = next(sgkf.split(X_dummy, y_codes, groups))
    y_train, y_test = y_codes[train_idx], y_codes[test_idx]
    keep_classes = set(y_train) & set(y_test)
    dropped = set(y_codes) - keep_classes
    if verbose and dropped:
        print(f"--> Dropping problematic classes (not in both train/test): {dropped}")

    mask_keep = np.isin(y_codes, list(keep_classes))
    y_codes_filtered = y_codes[mask_keep]
    groups_filtered = np.array(groups)[mask_keep]

    # 检查是否样本足够
    for cls in np.unique(y_codes_filtered):
        if np.sum(y_codes_filtered == cls) < min_samples_per_class:
            raise SkipThis(f"Class {cls} has fewer than {min_samples_per_class} samples after filtering.")

    # remap
    unique_classes = np.unique(y_codes_filtered)
    class_mapping = {old: new for new, old in enumerate(unique_classes)}
    y_codes_remap = np.array([class_mapping[y] for y in y_codes_filtered])

    # 第二轮尝试拆分
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for attempt, (train_idx, test_idx) in enumerate(
            sgkf.split(np.zeros(len(y_codes_remap)), y_codes_remap, groups_filtered), 1):
        y_train, y_test = y_codes_remap[train_idx], y_codes_remap[test_idx]
        if set(y_train) == set(y_test):
            if verbose:
                print(f"--> Successful split after filtering on attempt {attempt}")
            return train_idx, test_idx, y_codes_remap, class_mapping

    # 如果仍失败，则抛出 SkipThis
    raise SkipThis("Unable to find train/test split with all classes present after filtering.")


def _compute_sample_weights(y_train):
    """
    根据类别频率计算 sample weights
    核心就是 weight = total_samples / count_of_label
    少数类别权重高，多数类别权重低
    """

    counter = Counter(y_train)
    total = sum(counter.values())
    sample_weights = np.array([total / counter[label] for label in y_train])

    return sample_weights


def _save_dataset(save_path, method, X_train, X_test, y_train, y_test,
                  train_obs_index, test_obs_index, sample_weights, mapping,
                  file_name=None, verbose=True):
    """
    保存数据集为压缩 npz 文件
    """

    os.makedirs(save_path, exist_ok=True)

    if file_name is None:
        file_path = os.path.join(save_path, f"dataset_{method}.npz")
    else:
        file_path = os.path.join(save_path, f"{file_name}_{method}.npz")
    np.savez_compressed(
        file_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_obs_index=train_obs_index,
        test_obs_index=test_obs_index,
        sample_weights=sample_weights,
        label_mapping=mapping
    )

    if verbose:
        print(f"--> Successfully saved dataset to {file_path}")


def xgb_data_prepare(
        adata,
        obs_select=None,
        save_path=None,
        method="scvi",
        obs_key="Subset_Identity",
        group_key="Patient",
        test_size=0.2,
        verbose=True,
        random_state=42,
        oversample=True,
        weightsample=True,
        min_samples_per_class=10
):
    os.makedirs(save_path, exist_ok=True)

    print("[xgb_data_prepare] Start preparting XGB training data.")
    # 1. 子集 & 初步过滤
    print("[xgb_data_prepare] Start filtering.")
    adata_sub = _prepare_subset(adata, obs_select, obs_key, group_key, min_samples_per_class, verbose)
    adata_sub = _check_and_filter_diseases(adata_sub, group_key, min_samples_per_class, verbose)

    # 2. 标签编码
    print("[xgb_data_prepare] Start labeling tags.")
    y_codes, mapping, y = _encode_labels(adata_sub, "disease_type", verbose)

    # 3. 分层分组划分
    print("[xgb_data_prepare] Start stratifying.")
    groups = adata_sub.obs[group_key].values
    min_patients_per_disease = adata_sub.obs.groupby("disease_type")[group_key].nunique().min()
    n_splits = min(int(1 / test_size), min_patients_per_disease)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx, y_codes, class_mapping = _stratified_group_split(
        y_codes, groups,
        n_splits=n_splits,
        random_state=random_state,
        verbose=verbose
    )

    # 3.x） 修剪 mapping，避免幽灵类别
    print("[xgb_data_prepare] Trimming missing categories.")
    used_classes = sorted(set(y_codes[train_idx]) | set(y_codes[test_idx]))
    mapping = {cls: name for cls, name in mapping.items() if cls in used_classes}

    # 4. 特征提取（PCA 必须在 train 上 fit）
    print("[xgb_data_prepare] Start extracting eigenvalues.")
    X_train, X_test = _extract_features(adata_sub, method, train_idx, test_idx)

    # 5. Oversample
    if oversample:
        print("[xgb_data_prepare] Start oversampling.")
        if len(set(y_codes)) > 1:
            try:
                X_train, y_train, train_obs_index = _xgb_oversample_data(
                    X_train, y_codes[train_idx], adata_sub.obs.index[train_idx], mode="smote"
                )
            except ValueError as e:
                if "Expected n_neighbors" in str(e):
                    print("[xgb_data_prepare] SMOTE failed due to too few samples, falling back to random oversampling")
                    X_train, y_train, train_obs_index = _xgb_oversample_data(
                        X_train, y_codes[train_idx], adata_sub.obs.index[train_idx], mode="random"
                    )
                else:
                    raise
        else:
            raise SkipThis("--> Only one class present in training data, skipping.")
    else:
        print("[xgb_data_prepare] Skip oversampling.")
        y_train, train_obs_index = y_codes[train_idx], adata_sub.obs.index[train_idx]

    # 6. Sample weights
    print("[xgb_data_prepare] Start computing sample weights.")
    sample_weights = _compute_sample_weights(y_train) if weightsample else None

    # 7. 保存
    _save_dataset(save_path, method, X_train, X_test, y_train, y_codes[test_idx],
                  train_obs_index, adata_sub.obs.index[test_idx],
                  sample_weights, mapping)
    print("[xgb_data_prepare] Finished and successfully saved.")

    return


def xgb_data_prepare_lodo(
        adata,
        obs_select=None,
        save_path=None,
        method="scvi",
        obs_key="Subset_Identity",
        group_key="Patient",
        test_size=0.2,
        verbose=False,
        random_state=42,
        oversample=True,
        weightsample=True,
        min_samples_per_class=10
):
    '''
    lodo 模式的数据准备，虽然数据量大但是还是一次性准备了，
    一次性生成符合标准
    有助于函数的整体性，时候可以清理保存的数据
    :return: 不返回，只保存
    '''
    os.makedirs(save_path, exist_ok=True)
    print("[xgb_data_prepare_lodo] Start preparting XGB-lodo data.")

    # 1. 子集 & 初步过滤
    # 这一步规则完全相同：确保每个样本有 >= 2个patient，这样才能lodo；>= 2个疾病分类，这样才能学习
    print("[xgb_data_prepare_lodo] Start filtering.")
    adata_sub = _prepare_subset(adata, obs_select, obs_key, group_key, min_samples_per_class, verbose)
    adata_sub = _check_and_filter_diseases(adata_sub, group_key, min_samples_per_class, verbose)

    # 2. 标签编码
    # 这一步完全相同
    print("[xgb_data_prepare_lodo] Start labeling tags.")
    y_codes, mapping, y = _encode_labels(adata_sub, "disease_type", verbose)

    # 3. 更新：循环分组生成子数据集，并顺带检查是否有幽灵类别不能满足
    # 由于在子集规范步骤（_check_and_filter_diseases）已经做了基本的保证，
    # 在 lodo 中采取宽松的要求，只做基本存在检查，不存在则跳过

    patients = adata_sub.obs["Patient"]  # pandas Series
    uniques = patients.dropna().unique()  # 保留出现顺序；若需排序可用 np.sort(np.unique(...))
    total = len(uniques)

    for i, donor in enumerate(uniques, 1):
        print(f"[xgb_data_prepare_lodo] {i}/{total} • working on {donor}")

        # ...你的处理逻辑...
        cell_num = (adata_sub.obs["Patient"].eq(donor)).sum()
        print(f"[xgb_data_prepare_lodo] Donor: {donor},total cells: {cell_num}")

        # 定义训练/测试集
        train_idx = adata_sub.obs["Patient"] != donor
        test_idx = adata_sub.obs["Patient"] == donor

        if train_idx.sum() == 0 or test_idx.sum() == 0:
            # 不应该发生这种情况
            print(f"--> Triggers length boundary condition, check data preprocessing.")
            continue

        # 4. 特征提取（PCA 必须在 train 上 fit）
        X_train, X_test = _extract_features(adata_sub, method, train_idx, test_idx,verbose)
        y_train = y_codes[train_idx]
        y_test = y_codes[test_idx]
        train_obs_index = adata_sub.obs.index[train_idx.values]
        test_obs_index = adata_sub.obs.index[test_idx.values]

        # 5: 省略了 oversample 这一步

        # 6. Sample weights
        sample_weights = _compute_sample_weights(y_train,verbose) if weightsample else None

        # 7. 保存
        # 按照新格式来命名
        file_name = f"LODO_{donor}"
        _save_dataset(save_path, method, X_train, X_test, y_train, y_test,
                      train_obs_index, test_obs_index,
                      sample_weights, mapping, file_name)
        print(f"[xgb_data_prepare_lodo] Sample successfully finished and saved.)

    return


def xgboost_read(save_path, method_suffix):
    data = np.load(os.path.join(save_path, f"dataset_{method_suffix}.npz"), allow_pickle=True)
    file_list = data.files
    print(f"Containing files: {file_list}.")
    return data
