import os
import sys
from typing import Literal, Tuple

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import logging
from src.utils.hier_logger import logged

sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


def _validate_input_dataframe(df: pd.DataFrame, required_columns: list[str]) -> None:
    """检查输入表是否包含 PCA 所需列。"""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Required columns are missing from input DataFrame: {missing_cols}.")


def _prepare_score_matrix(
        df: pd.DataFrame,
        agg_func: Literal["mean", "median", "sum", "max", "min"] | str = "mean",
) -> pd.DataFrame:
    """将长表整理为可供 PCA 使用的宽表。"""
    _validate_input_dataframe(df, ["folder_name", "names", "scores"])

    prepared_df = df.copy()
    prepared_df["cell_type"] = prepared_df["folder_name"].astype(str).str.lstrip("_")

    data_matrix = prepared_df.pivot_table(
        index="cell_type",
        columns="names",
        values="scores",
        aggfunc=agg_func,
    )

    if data_matrix.empty:
        raise ValueError("The pivoted score matrix is empty. Please check the input DataFrame.")

    return data_matrix


@logged
def merge_xlsx_from_subfolders(parent_dir: str, file_suffix: str = ".xlsx") -> pd.DataFrame:
    """递归合并子目录中的 Excel 文件。

    仅会读取目录名以下划线 `_` 开头的子目录，并为每条记录补充 `folder_name` 列。

    Args:
        parent_dir: 待扫描的父目录。
        file_suffix: 需要合并的文件后缀名。

    Returns:
        合并后的 DataFrame。
    """
    if not os.path.exists(parent_dir):
        raise FileNotFoundError(f"`parent_dir`: '{parent_dir}' does not exist.")

    combined_data = []
    for subdir, _, files in os.walk(parent_dir):
        if not os.path.basename(subdir).startswith("_"):
            continue

        for file_name in files:
            if not file_name.endswith(file_suffix):
                continue

            file_path = os.path.join(subdir, file_name)
            logger.info(f"[merge_xlsx_from_subfolders] Reading file: {file_path}")
            try:
                df = pd.read_excel(file_path)
                df["folder_name"] = os.path.basename(subdir)
                combined_data.append(df)
            except Exception as exc:
                logger.info(
                    f"[merge_xlsx_from_subfolders] Warning! Failed to read file: {file_path}. "
                    f"Details: {exc}"
                )

    if not combined_data:
        raise ValueError(
            f"No file ending with '{file_suffix}' was found in subfolders starting with '_' under: {parent_dir}."
        )

    merged_df = pd.concat(combined_data, ignore_index=True)
    logger.info(
        f"[merge_xlsx_from_subfolders] Merged DataFrame generated with shape: {merged_df.shape}."
    )
    return merged_df


@logged
def run_pca_analysis(
        df: pd.DataFrame,
        n_components: int = 5,
        agg_func: Literal["mean", "median", "sum", "max", "min"] | str = "mean",
) -> Tuple[pd.DataFrame, pd.Series]:
    """执行 PCA 降维分析。

    Args:
        df: 输入长表，至少包含 `folder_name`、`names`、`scores` 三列。
        n_components: 需要保留的主成分数量。
        agg_func: 当同一 cell subtype/subpopulation 与同一基因重复出现时的聚合方式。

    Returns:
        一个二元组：
        1. 含 PCA 坐标的 DataFrame；
        2. 各主成分解释方差比例。
    """
    if n_components < 1:
        raise ValueError("`n_components` must be greater than or equal to 1.")

    data_matrix = _prepare_score_matrix(df, agg_func=agg_func).fillna(0)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_matrix)

    n_components_final = min(n_components, data_matrix.shape[0], data_matrix.shape[1])
    if n_components_final < 1:
        raise ValueError("No valid PCA component can be computed from the current data matrix.")

    pca = PCA(n_components=n_components_final)
    pca_results = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(
        pca_results,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
        index=data_matrix.index,
    ).reset_index()
    pca_df = pca_df.rename(columns={"cell_type": "cell_type"})

    logger.info(
        f"[run_pca_analysis] PCA completed with `n_components`: {pca.n_components_}. "
        f"Input matrix shape: {data_matrix.shape}."
    )
    return pca_df, pca.explained_variance_ratio_


@logged
def run_pca_and_clustering(
        df: pd.DataFrame,
        n_components: int = 5,
        n_clusters: int = 3,
        agg_func: Literal["mean", "median", "sum", "max", "min"] | str = "mean",
) -> Tuple[pd.DataFrame, pd.Series]:
    """执行 PCA 降维并在 PCA 空间中进行 KMeans 聚类。

    Args:
        df: 输入长表，至少包含 `folder_name`、`names`、`scores` 三列。
        n_components: 需要保留的主成分数量。
        n_clusters: KMeans 聚类簇数。
        agg_func: 当同一 cell subtype/subpopulation 与同一基因重复出现时的聚合方式。

    Returns:
        一个二元组：
        1. 含 PCA 坐标和聚类标签的 DataFrame；
        2. 各主成分解释方差比例。
    """
    if n_components < 1:
        raise ValueError("`n_components` must be greater than or equal to 1.")
    if n_clusters < 1:
        raise ValueError("`n_clusters` must be greater than or equal to 1.")

    data_matrix = _prepare_score_matrix(df, agg_func=agg_func)

    # 使用 0 作为缺失分数兜底，避免 PCA/KMeans 被 NaN 阻断。
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    imputed_data = imputer.fit_transform(data_matrix)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)

    n_comp = min(n_components, data_matrix.shape[0], data_matrix.shape[1])
    if n_comp < 1:
        raise ValueError("No valid PCA component can be computed from the current data matrix.")
    if n_clusters > data_matrix.shape[0]:
        logger.info(
            f"[run_pca_and_clustering] Warning! `n_clusters`: {n_clusters} is greater than sample size: "
            f"{data_matrix.shape[0]}. Falling back to sample size."
        )
        n_clusters = data_matrix.shape[0]

    pca = PCA(n_components=n_comp)
    pca_results = pca.fit_transform(scaled_data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(pca_results)

    pca_columns = [f"PC{i + 1}" for i in range(pca_results.shape[1])]
    final_df = pd.DataFrame(
        pca_results,
        columns=pca_columns,
        index=data_matrix.index,
    ).reset_index()
    final_df["cluster"] = cluster_labels.astype(str)

    logger.info(
        f"[run_pca_and_clustering] PCA and clustering completed with `n_components`: {n_comp}, "
        f"`n_clusters`: {n_clusters}."
    )
    return final_df, pca.explained_variance_ratio_
