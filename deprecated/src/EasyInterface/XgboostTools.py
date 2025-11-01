import numpy as np
import matplotlib
matplotlib.use('Agg')  # 将图输出到文件而不是屏幕
import matplotlib.pyplot as plt

import scanpy as sc
import os, gc, sys
from src.ScanpyTools.ScanpyTools import subcluster
import os
import xgboost as xgb
import os
import seaborn as sns
import shap

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
    import numpy as np
    from collections import Counter
    
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
                                            (adata_sub.obs.groupby("disease_type").size() >= min_samples_per_class)].index
        removed = counts_per_disease.index.difference(valid_diseases)
        adata_sub = adata_sub[adata_sub.obs["disease_type"].isin(valid_diseases)].copy()
        if verbose and len(removed) > 0:
            print(f"--> Removed disease groups (not enough patients or samples <{min_samples_per_class}): {list(removed)}")
    
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
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import scanpy as sc
    
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
    from sklearn.model_selection import StratifiedGroupKFold
    import numpy as np
    
    
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
    for attempt, (train_idx, test_idx) in enumerate(sgkf.split(np.zeros(len(y_codes_remap)), y_codes_remap, groups_filtered), 1):
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
    import numpy as np
    from collections import Counter
    
    counter = Counter(y_train)
    total = sum(counter.values())
    sample_weights = np.array([total / counter[label] for label in y_train])
    
    return sample_weights

def _save_dataset(save_path, method, X_train, X_test, y_train, y_test,
                  train_obs_index, test_obs_index, sample_weights, mapping,
                  file_name=None,verbose=True):
    """
    保存数据集为压缩 npz 文件
    """
    import os
    import numpy as np
    
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
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # 1. 子集 & 初步过滤
    adata_sub = _prepare_subset(adata, obs_select, obs_key, group_key, min_samples_per_class, verbose)
    adata_sub = _check_and_filter_diseases(adata_sub, group_key, min_samples_per_class, verbose)
    
    # 2. 标签编码
    y_codes, mapping, y = _encode_labels(adata_sub, "disease_type", verbose)
    
    # 3. 分层分组划分
    from sklearn.model_selection import StratifiedGroupKFold
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
    used_classes = sorted(set(y_codes[train_idx]) | set(y_codes[test_idx]))
    mapping = {cls: name for cls, name in mapping.items() if cls in used_classes}
    
    # 4. 特征提取（PCA 必须在 train 上 fit）
    X_train, X_test = _extract_features(adata_sub, method, train_idx, test_idx)
    
    # 5. Oversample
    if oversample:
        if len(set(y_codes)) > 1:
            try:
                X_train, y_train, train_obs_index = _xgb_oversample_data(
                    X_train, y_codes[train_idx], adata_sub.obs.index[train_idx], mode="smote"
                )
            except ValueError as e:
                if "Expected n_neighbors" in str(e):
                    print("--> SMOTE failed due to too few samples, falling back to random oversampling")
                    X_train, y_train, train_obs_index = _xgb_oversample_data(
                        X_train, y_codes[train_idx], adata_sub.obs.index[train_idx], mode="random"
                    )
                else:
                    raise
        else:
            raise SkipThis("--> Only one class present in training data, skipping.")
    else:
        y_train, train_obs_index = y_codes[train_idx], adata_sub.obs.index[train_idx]
    
    # 6. Sample weights
    sample_weights = _compute_sample_weights(y_train) if weightsample else None
    
    # 7. 保存
    _save_dataset(save_path, method, X_train, X_test, y_train, y_codes[test_idx],
                 train_obs_index, adata_sub.obs.index[test_idx],
                 sample_weights, mapping)
    
    return


def xgb_data_prepare_lodo(
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
    '''
    lodo 模式的数据准备，虽然数据量大但是还是一次性准备了，
    一次性生成符合标准
    有助于函数的整体性，时候可以清理保存的数据
    :return: 不返回，只保存
    '''
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # 1. 子集 & 初步过滤
    # 这一步规则完全相同：确保每个样本有 >= 2个patient，这样才能lodo；>= 2个疾病分类，这样才能学习
    adata_sub = _prepare_subset(adata, obs_select, obs_key, group_key, min_samples_per_class, verbose)
    adata_sub = _check_and_filter_diseases(adata_sub, group_key, min_samples_per_class, verbose)
    
    # 2. 标签编码
    # 这一步完全相同
    y_codes, mapping, y = _encode_labels(adata_sub, "disease_type", verbose)
    
    # 3. 更新：循环分组生成子数据集，并顺带检查是否有幽灵类别不能满足
    # 由于在子集规范步骤（_check_and_filter_diseases）已经做了基本的保证，
    # 在 lodo 中采取宽松的要求，只做基本存在检查，不存在则跳过
    import numpy as np
    
    for donor in np.unique(adata_sub.obs["Patient"]):
        cell_num = sum(adata_sub.obs["Patient"]==donor)
        print(f"--> Donor: {donor},total cells: {cell_num}")
        
        # 定义训练/测试集
        train_idx = adata_sub.obs["Patient"] != donor
        test_idx = adata_sub.obs["Patient"] == donor
        
        if train_idx.sum() == 0 or test_idx.sum() == 0:
            # 不应该发生这种情况
            print(f"--> Triggers length boundary condition, check data preprocessing.")
            continue
            
         # 4. 特征提取（PCA 必须在 train 上 fit）
        X_train, X_test = _extract_features(adata_sub, method, train_idx, test_idx)
        y_train = y_codes[train_idx]
        y_test = y_codes[test_idx]
        train_obs_index = adata_sub.obs.index[train_idx.values]
        test_obs_index = adata_sub.obs.index[test_idx.values]
        
        # 5: 省略了 oversample 这一步
        
        # 6. Sample weights
        sample_weights = _compute_sample_weights(y_train) if weightsample else None
    
        # 7. 保存
        # 按照新格式来命名
        file_name = f"LODO_{donor}"
        _save_dataset(save_path, method, X_train, X_test, y_train, y_test,
                      train_obs_index,test_obs_index,
                      sample_weights, mapping, file_name)
    
    return


def xgboost_read(save_path,method_suffix):
    data = np.load(os.path.join(save_path, f"dataset_{method_suffix}.npz"), allow_pickle=True)
    file_list = data.files
    print(f"Containing files: {file_list}.")
    return data

def xgboost_process(save_path, method_suffix,
                    eval_metric,objective="softmax",max_depth=6,colsample_bytree=0.8,
                    min_split_loss=0,min_child_weight=1,reg_lambda=0,
                    verbose=True,do_return=False):
    """
    训练 XGBoost 模型，多分类，使用 sample_weights
    """
    from sklearn.metrics import accuracy_score, classification_report
    
    os.makedirs(save_path, exist_ok=True)
    
    # 读取数据
    data = np.load(os.path.join(save_path, f"dataset_{method_suffix}.npz"), allow_pickle=True)
    X_train = data["X_train"];
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    sample_weights = data["sample_weights"]
    
    # 初始化模型
    clf = xgb.XGBClassifier(
        objective=objective,
        n_estimators=500,
        max_depth=max_depth,
        learning_rate=0.1,
        min_child_weight=min_child_weight,
        subsample=0.8,
        reg_lambda=reg_lambda,
        colsample_bytree=colsample_bytree,
        tree_method="hist",
        min_split_loss=min_split_loss,
        random_state=42,
        use_label_encoder=False
    )
    
    # 训练
    clf.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric=eval_metric,
            sample_weight=sample_weights,
            verbose=False
            )
    
    # 评估
    y_pred = clf.predict(X_test)
    
    if verbose:
        print("--> Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
    
    # 保存模型
    clf.save_model(f"{save_path}/categorical-model_{method_suffix}.json")
    print("--> XGB Model successfully saved.")
    
    if not do_return:
        return
    else:
        return clf


def xgboost_process_lodo(save_path, method_suffix,
                         eval_metric, objective="softmax", max_depth=6, colsample_bytree=0.8,
                         min_split_loss=0, min_child_weight=1, reg_lambda=0,
                         verbose=True):
    """
    训练 XGBoost 模型，多分类，使用 sample_weights
    """
    from sklearn.metrics import accuracy_score, classification_report
    
    # 读取数据
    fl = os.listdir(save_path)
    files = [item for item in fl if item.startswith('LODO')]
    total = len(files)
    
    for i, file in enumerate(files):
        head = "LODO_"
        tail = f"_{method_suffix}.npz"
        
        donor = file[len(head):-len(tail)]
        print(f"Processing {i+1}/{total}: {donor}")
        
        data = np.load(os.path.join(save_path, file), allow_pickle=True)
        X_train = data["X_train"];
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
        sample_weights = data.get("sample_weights", None)
        
        # 初始化模型
        clf = xgb.XGBClassifier(
            objective=objective,
            n_estimators=500,
            max_depth=max_depth,
            learning_rate=0.1,
            min_child_weight=min_child_weight,
            subsample=0.8,
            reg_lambda=reg_lambda,
            colsample_bytree=colsample_bytree,
            tree_method="hist",
            min_split_loss=min_split_loss,
            random_state=42,
            use_label_encoder=False
        )
        
        # 训练
        clf.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric=eval_metric,
                sample_weight=sample_weights,
                verbose=False
                )
        
        # 评估
        y_pred = clf.predict(X_test)
        
        if verbose:
            print("--> Accuracy:", accuracy_score(y_test, y_pred))
            print(classification_report(y_test, y_pred))
        
        # 保存模型
        save_name = f"LODO_{donor}"
        clf.save_model(f"{save_path}/{save_name}_categorical-model_{method_suffix}.json")
        print("--> XGB Model successfully saved.")
        
    return


def xgb_outcome_analyze(save_path, method_suffix):
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 roc_curve, roc_auc_score)
    from sklearn.preprocessing import label_binarize
    
    os.makedirs(f"{save_path}/output", exist_ok=True)
    
    # 读取数据
    data = np.load(f"{save_path}/dataset_{method_suffix}.npz", allow_pickle=True)
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]; mapping = data["label_mapping"].item()
    
    # 加载模型
    clf = xgb.XGBClassifier()
    clf.load_model(f"{save_path}/categorical-model_{method_suffix}.json")
    
    # 预测
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    
    # 把预测结果单独做一个保存，回头做umap用
    np.save(f"{save_path}/y_pred_{method_suffix}.npy", y_pred)
    
    # 分类报告
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    filename = f"{save_path}/output/classification_result_{method_suffix}.txt"
    
    with open(filename, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"--> Result is saved at {filename}")
    
    # 特征重要性
    xgb.plot_importance(clf, max_num_features=20)
    plt.savefig(f"{save_path}/output/feature_importance_{method_suffix}.png")
    plt.clf()
    print(f"--> XGB importance plotted.")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    # 每行归一化
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm.astype(np.float32),
        cm_sum,
        out=np.zeros_like(cm, dtype=np.float32),
        where=cm_sum != 0
    )
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",  # 因为是浮点数了
        xticklabels=[mapping[i] for i in range(len(mapping))],
        yticklabels=[mapping[i] for i in range(len(mapping))]
    )
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.25)  # 这里调整边距
    plt.savefig(f"{save_path}/output/confusion_mat_{method_suffix}.png")
    plt.clf()
    print(f"--> XGB confusion matrix plotted.")
    
    # SHAP 分析
    feature_names = [mapping.get(i, f"Feature {i}") for i in range(X_test.shape[1])]
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 这里留白
    plt.savefig(f"{save_path}/output/SHAP_summary_{method_suffix}.png")
    plt.clf()
    print(f"--> XGB SHAP plotted.")
    
    # 多分类 ROC
    y_bin = label_binarize(y_test, classes=range(len(np.unique(y_test))))
    plt.figure(figsize=(8, 6))
    for i in range(y_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc_i = roc_auc_score(y_bin[:, i], y_proba[:, i])
        class_name = mapping[i]  # 这里用映射替换
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC={auc_i:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC curves')
    plt.legend(loc="lower right")
    plt.savefig(f"{save_path}/output/ROC_{method_suffix}.png")
    plt.clf()
    print(f"--> XGB AUC plotted.")
    
    # Per-class Precision/Recall/F1
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = ["precision", "recall", "f1-score"]
    
    for metric in metrics:
        values = [report[str(cls)][metric] for cls in np.unique(y_test)]
        
        # 映射 class label
        labels = [mapping[cls] for cls in np.unique(y_test)]
        
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(values)), values, tick_label=labels)
        
        # 添加红色阈值线 y=0.5
        plt.axhline(y=0.5, color="red", linestyle="--", linewidth=1)
        
        plt.ylim(0, 1)
        plt.ylabel(metric.capitalize())
        plt.xlabel("Class")
        plt.title(f"Per-class {metric.capitalize()}")
        plt.savefig(f"{save_path}/output/{metric}_{method_suffix}.png")
        plt.clf()
    
    print(f"--> XGB precision/recall/F1 plotted.")


def xgb_outcome_analyze_lodo(save_path, method_suffix):
    import json
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 roc_curve, roc_auc_score)
    from sklearn.preprocessing import label_binarize
    
    os.makedirs(f"{save_path}/output", exist_ok=True)
    
    # 读取数据
    results = {"donor": [], "y_test": [], "y_pred": [], "y_proba": [], "report": [], "accuracy": [],"mapping":[]}
    
    
    fl = os.listdir(save_path)
    files = [item for item in fl if item.startswith('LODO')]
    files = [item for item in files if item.endswith('npz')]
    total = len(files)
    
    for i, file in enumerate(files):
        head = "LODO_"
        tail = f"_{method_suffix}.npz"
        
        donor = file[len(head):-len(tail)]
        print(f"Processing {i+1}/{total}: {donor}")
        
        # 读取数据
        data = np.load(os.path.join(save_path, file), allow_pickle=True)
        X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"];
        mapping = data["label_mapping"].item()
        
        # 加载模型
        clf = xgb.XGBClassifier()
        save_name = f"LODO_{donor}"
        clf.load_model(f"{save_path}/{save_name}_categorical-model_{method_suffix}.json")
        
        # 预测
        y_pred = clf.predict(X_test); y_proba = clf.predict_proba(X_test)
        
        # 分类报告
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # 保存
        results["donor"].append(donor)
        results["y_test"].append(y_test)
        results["y_pred"].append(y_pred)
        results["y_proba"].append(y_proba)
        results["report"].append(report)
        results["accuracy"].append(acc)
        results["mapping"].append(mapping)
        
    # with open(f"{save_path}/output/output_dict.json", "w") as f:
    #     json.dump(results, f, indent=4)
    
    return results
    
    
def plot_xgb_prediction_umap(adata, save_path, method_suffix="scvi",skip_subcluster=False):
    """
    将 XGBoost 预测结果写回 adata.obs 并用 UMAP 可视化。

    参数：
        adata : AnnData
            原始或子集 AnnData 对象
        save_path : str
            存放预测结果的文件夹
        method_suffix : str
            区分不同 embedding 方法，如 "scvi" 或 "pca"
    """
    from src.ScanpyTools.Scanpy_Plot import ScanpyPlotWrapper
    
    # 初始化 UMAP 绘图工具
    umap_plot = ScanpyPlotWrapper(func=sc.pl.umap)
    
    # 读取预测结果
    y_pred_file = os.path.join(save_path, f"y_pred_{method_suffix}.npy")
    y_pred = np.load(y_pred_file)
    
    # 读取划分索引和 label 映射
    dataset_file = os.path.join(save_path, f"dataset_{method_suffix}.npz")
    data = np.load(dataset_file, allow_pickle=True)
    test_idx = data["test_obs_index"]  # 这是一个 Index 对象数组
    mapping = data["label_mapping"].item()
    
    # 取出对应测试集的子集
    adata_sub = adata[adata.obs.index.isin(test_idx)].copy()
    if adata_sub.n_obs < 100:
        print("--> Notice! Anndata subset has less than 100 cell, skip UMAP to avoid corruption.")
        return adata_sub
    if not skip_subcluster:
        print(f"--> Starting subclustering process, this may take some time...")
        adata_sub = subcluster(adata_sub,n_pcs=10,resolutions=[1.0])
    
    # 写回预测结果
    adata_sub.obs.loc[test_idx, "predicted_label"] = y_pred
    adata_sub.obs.loc[test_idx, "predicted_label_name"] = adata_sub.obs.loc[test_idx, "predicted_label"].map(mapping)
    
    print(f"--> Starting UMAP plotting.")
    # 绘制 UMAP
    umap_plot(
        save_addr=os.path.join(save_path, "output"),
        filename=f"Umap_by_prediction_{method_suffix}",
        adata=adata_sub,
        color=["disease_type", "predicted_label_name"],
        wspace=0.4
    )
    print("--> UMAP plotted.")
    
    return adata_sub

def plot_taxonomy(adata, save_path, method_suffix="scvi"):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import xgboost as xgb
    from scipy.cluster.hierarchy import linkage, dendrogram
    from sklearn.metrics import confusion_matrix
    from scipy.spatial.distance import pdist, squareform
    import shap
    
    print(f"--> Loading dataset {method_suffix}")
    dataset_file = os.path.join(save_path, f"dataset_{method_suffix}.npz")
    data = np.load(dataset_file, allow_pickle=True)
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    mapping = data["label_mapping"].item()
    
    print("--> Loading model")
    clf = xgb.XGBClassifier()
    clf.load_model(f"{save_path}/categorical-model_{method_suffix}.json")
    
    y_pred = clf.predict(X_test)
    
    # -------------------
    # 1. 混淆矩阵
    # -------------------
    present_classes = sorted(np.unique(np.concatenate([y_train, y_test])))
    class_names = [mapping[i] for i in present_classes]
    print(f"Present classes (for CM): {present_classes}, class_names: {class_names}")
    
    cm = confusion_matrix(y_test, y_pred, labels=present_classes).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums==0] = 1
    cm /= row_sums
    
    corr = np.corrcoef(cm, rowvar=False)
    dist = 1 - corr
    dist[np.isnan(dist)] = 0
    dist = (dist + dist.T)/2
    np.fill_diagonal(dist, 0)
    
    print(f"CM distance matrix shape: {dist.shape}")
    linkage_matrix = linkage(squareform(dist), method="average")
    print(f"Linkage matrix shape: {linkage_matrix.shape}")
    
    plt.figure(figsize=(6, 4))
    dendrogram(linkage_matrix, labels=class_names, orientation="right")
    plt.title("Class taxonomy (from confusion matrix)")
    plt.savefig(f"{save_path}/output/Taxonomy_CM_{method_suffix}.png", bbox_inches="tight")
    plt.clf()
    print(f"--> Linkage map (confusion matrix) plotted.")
    
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
    print(f"SHAP per class matrix shape: {shap_per_class.shape}")
    
    dist_matrix = pdist(shap_per_class, metric="correlation")
    linkage_matrix = linkage(dist_matrix, method="average")
    print(f"SHAP linkage matrix shape: {linkage_matrix.shape}")
    
    plt.figure(figsize=(6, 4))
    dendrogram(linkage_matrix, labels=class_names, orientation="right")
    plt.title("Class taxonomy (from feature importances)")
    plt.savefig(f"{save_path}/output/Taxonomy_SHAP_{method_suffix}.png", bbox_inches="tight")
    plt.clf()
    print(f"--> Linkage map (SHAP) plotted.")
    
    # -------------------
    # 3. latent space
    # -------------------
    latent = adata.obsm["X_scVI"]
    labels = adata.obs["disease_type"].values
    
    class_means = []
    for cname in class_names:
        mask = labels == cname
        if np.sum(mask)==0:
            print(f"--> Warning: no samples for class {cname} in latent space, skipping")
            continue
        class_means.append(latent[mask].mean(axis=0))
    
    class_means = np.vstack(class_means)
    print(f"Latent space class_means shape: {class_means.shape}")
    
    dist = pdist(class_means, metric="euclidean")
    linkage_matrix = linkage(dist, method="ward")
    
    plt.figure(figsize=(6, 4))
    dendrogram(linkage_matrix, labels=class_names, orientation="right")
    plt.title("Class taxonomy (from latent space)")
    plt.savefig(f"{save_path}/output/Taxonomy_LS_{method_suffix}.png", bbox_inches="tight")
    plt.clf()
    print(f"--> Linkage map (latent space) plotted.")

  

def plot_lodo_boxplots(results, show_points=True, save_path=None):
    """
    绘制 LODO 结果的三面 Boxplot (F1, Precision, Recall)

    Parameters
    ----------
    results : list of dict
        每个元素是 {"donor": str, "report": dict}，
        其中 report 是 classification_report(..., output_dict=True) 的结果。
    save_path : str or None
        如果给定路径，则保存图像；否则直接 plt.show()
    """
    # 整理数据
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report
    
    records = []
    for i,donor in enumerate(results["donor"]):
        report = results["report"][i]
        # 取 weighted avg (你也可以改成 macro avg)
        metrics = report["weighted avg"]
        records.append({"donor": donor, "metric": "F1", "value": metrics["f1-score"]})
        records.append({"donor": donor, "metric": "Precision", "value": metrics["precision"]})
        records.append({"donor": donor, "metric": "Recall", "value": metrics["recall"]})
    
    df = pd.DataFrame(records)
    
    # 画图
    plt.figure(figsize=(8, 6))
    
    # 先画 violin
    sns.violinplot(data=df, x="metric", y="value", inner="box", cut=0, palette="Set2")
    
    # 再叠加 stripplot（每个 donor 的点）
    if show_points:
        sns.stripplot(data=df, x="metric", y="value",
                      color="black", size=3, jitter=True, alpha=0.5)
    
    plt.ylim(-0.05, 1.05)  # 保证 [0,1] 的范围完整显示
    plt.title("LODO Performance (F1 / Precision / Recall)", fontsize=14)
    plt.ylabel("Value")
    plt.xlabel("")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_confusion_stats(results, save_path=None):
    """
    all_results: list of dict，每个 dict 包含:
        - "y_test"
        - "y_pred"
        - "mapping" (label 编码到名称的字典)
    save_path: str, 保存路径（不传则直接 plt.show）
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    # 确定类别顺序
    mapping = results["mapping"][0]
    classes = list(mapping.values())
    
    # 收集混淆矩阵
    conf_list = []
    for i in range(0, len(results["y_test"])):
        y_test = results["y_test"][i]; y_pred = results["y_pred"][i]
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
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()




# def compute_donor_similarity(y_pred_list, n_classes):
#     """
#     y_pred_list: list of array，每个 array 是一个 donor 的 y_pred
#     n_classes: 总类别数
#     返回: donor x donor 的相似性矩阵
#     """
#     import numpy as np
#     from scipy.spatial.distance import pdist, squareform
#
#     donor_vectors = []
#     for y in y_pred_list:
#         vec = np.bincount(y, minlength=n_classes)  # 统计每个类别数量
#         vec = vec / vec.sum()  # 转为占比
#         donor_vectors.append(vec)
#
#     donor_matrix = np.stack(donor_vectors)  # shape: n_donors x n_classes
#     # 计算 1 - cosine 相似度作为距离
#     dist = pdist(donor_matrix, metric='cosine')
#     co_matrix = 1 - squareform(dist)  # 转为 n_donors x n_donors
#     return co_matrix

def donor_vectors_from_preds(y_pred_list, n_classes, label_shift=1):
    """
    label_shift: 如果标签是1..K，设为1；如果是0..K-1，设为0
    返回 shape (n_donors, n_classes) 的占比矩阵
    """
    import numpy as np
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

def donor_vectors_from_proba(y_proba_list):
    """每个 donor 的 y_proba_list[i] 形状 (n_cells_i, n_classes)"""
    import numpy as np
    return np.vstack([p.mean(axis=0) for p in y_proba_list])

def compute_donor_similarity_matrix(donor_matrix, metric='cosine'):
    """
    donor_matrix: n_donors x n_classes (概率或占比)
    metric: 'cosine' or 'jensenshannon'
    返回相似度矩阵 (n_donors x n_donors)
    """
    from scipy.spatial.distance import pdist, squareform
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



def plot_stability_dendrogram(co_matrix, labels, save_path=None):
    """
    co_matrix: n x n 的稳定性矩阵
    labels: donor 名称列表
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage, dendrogram
    
    # 转换成距离矩阵 (相似度高 = 距离小)
    dist = 1 - co_matrix
    
    # linkage 聚类
    Z = linkage(dist, method="average")
    
    # 画图
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=90)
    ax.set_ylabel("Distance (1 - stability)")
    ax.set_title("Stability Dendrogram")
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()


def plot_stability_clustermap(co_matrix, labels, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    df = pd.DataFrame(co_matrix, index=labels, columns=labels)
    sns.clustermap(df, cmap="viridis", row_cluster=True, col_cluster=True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from collections import Counter

def compute_donor_similarity_lodo(y_list, n_classes):
    """
    简单的 donor x donor 相似性矩阵：
    - 对每个 donor，计算每个类别比例
    - donor 之间相似性 = cosine similarity of class proportions
    """
    from sklearn.preprocessing import normalize
    
    n_donors = len(y_list)
    class_counts = np.zeros((n_donors, n_classes))
    for i, y in enumerate(y_list):
        for c in range(1, n_classes+1):
            class_counts[i, c-1] = np.sum(y == c)
    class_props = normalize(class_counts, norm='l1', axis=1)
    sim_matrix = class_props @ class_props.T  # cosine similarity
    return sim_matrix

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


def plot_consensus_dendrogram(Z_consensus, donor_labels, branch_supports, save_path=None):
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dendro = dendrogram(Z_consensus, labels=donor_labels, ax=ax, leaf_rotation=90)
    
    # 每个内部节点的坐标
    icoord = dendro["icoord"]
    dcoord = dendro["dcoord"]
    leaves = dendro["leaves"]  # 叶子 donor 的索引
    
    # 遍历所有内部节点
    for i, (xs, ys) in enumerate(zip(icoord, dcoord)):
        # 这个节点对应的 donor 集合（内部节点的 cluster）
        cluster = frozenset(leaves[int(x/10)] for x in xs[1:3])  # 中间两个点对应子树
        
        support = branch_supports.get(cluster, None)
        if support is not None and support >= 0.10:  # 只标注 >=0.10
            x = sum(xs[1:3]) / 2  # 横坐标：子树中点
            y = ys[1]             # 纵坐标：高度
            ax.text(
                x, y, f"{support:.2f}",
                va='bottom', ha='center',
                color='red', fontsize=10, weight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1)
            )
    
    ax.set_ylabel("Distance")
    ax.set_title("Consensus Dendrogram with Branch Support")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()




