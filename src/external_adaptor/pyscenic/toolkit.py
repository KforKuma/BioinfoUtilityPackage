"""pySCENIC 结果整理与输入导出工具。"""

import logging
import os
import warnings
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import jensenshannon

from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


def _ensure_save_dir(save_addr: str) -> str:
    """检查并创建输出目录。"""
    if not isinstance(save_addr, str) or save_addr.strip() == "":
        raise ValueError("Argument `save_addr` must be a non-empty string.")
    save_addr = save_addr.strip()
    os.makedirs(save_addr, exist_ok=True)
    return save_addr


@logged
def regulons_to_gene_lists(incid_mat: pd.DataFrame, omit_empty: bool = True) -> Dict[str, List[Any]]:
    """将 regulon incidence matrix 转换为 regulon 对应的基因列表字典。

    这个函数会尽量兼容两种常见方向：
    1. 行是 regulon、列是基因。
    2. 行是基因、列是 regulon。

    若检测到第一种形式，就按行解析；若更像第二种形式，则自动按列解析。

    Args:
        incid_mat: 0/1 incidence matrix。
        omit_empty: 是否删除没有基因成员的 regulon。

    Returns:
        字典，键为 regulon 名称，值为对应基因列表。

    Example:
        regulon_dict = regulons_to_gene_lists(
            incid_mat=incidence_df,
            omit_empty=True,
        )
        regulon_dict.get("IRF1(+)")
    """
    if not isinstance(incid_mat, pd.DataFrame):
        raise TypeError("Argument `incid_mat` must be a pandas DataFrame.")
    if incid_mat.empty:
        raise ValueError("Argument `incid_mat` must not be empty.")

    binary_values = incid_mat.fillna(0).astype(float)
    row_sum = (binary_values.sum(axis=1) > 0).mean()
    col_sum = (binary_values.sum(axis=0) > 0).mean()

    if row_sum <= col_sum:
        regulon_dict = {
            regulon: incid_mat.columns[binary_values.loc[regulon] == 1].tolist()
            for regulon in incid_mat.index
        }
    else:
        regulon_dict = {
            regulon: incid_mat.index[binary_values[regulon] == 1].tolist()
            for regulon in incid_mat.columns
        }

    if omit_empty:
        regulon_dict = {key: value for key, value in regulon_dict.items() if value}

    logger.info(f"[regulons_to_gene_lists] Parsed {len(regulon_dict)} regulons.")
    return regulon_dict


@logged
def get_regulons_incid_matrix(loom_file: str) -> pd.DataFrame:
    """从 pySCENIC 的 loom 文件中读取 regulon incidence matrix。

    Args:
        loom_file: pySCENIC 输出的 `.loom` 文件路径。

    Returns:
        行为基因、列为 regulon 的 incidence matrix。

    Example:
        incid_mat = get_regulons_incid_matrix(
            loom_file="pyscenic_output.loom",
        )
        incid_mat.iloc[:5, :5]
    """
    if not os.path.isfile(loom_file):
        raise FileNotFoundError(f"File `loom_file` was not found: '{loom_file}'.")

    with h5py.File(loom_file, "r") as file:
        regulon_struct = file["row_attrs/Regulons"][()]
        genes = file["row_attrs/Gene"][()].astype(str)
        regulon_names = regulon_struct.dtype.names
        data = np.vstack([regulon_struct[name] for name in regulon_names]).T

    df = pd.DataFrame(data=data, index=genes, columns=regulon_names)
    logger.info(
        f"[get_regulons_incid_matrix] Loaded incidence matrix with {df.shape[0]} genes and {df.shape[1]} regulons."
    )
    return df


@logged
def get_regulons_auc_from_h5(loom_file: str, col_attr_name: str = "RegulonsAUC") -> pd.DataFrame:
    """从 pySCENIC 的 loom 文件中读取 regulon AUC 矩阵。

    Args:
        loom_file: pySCENIC 输出的 `.loom` 文件路径。
        col_attr_name: 用于存储 regulon AUC 的列属性名。

    Returns:
        行为 regulon、列为 cell barcode 的 AUC 矩阵。

    Example:
        auc_df = get_regulons_auc_from_h5(
            loom_file="pyscenic_output.loom",
            col_attr_name="RegulonsAUC",
        )
        auc_df.iloc[:5, :5]
    """
    if not os.path.isfile(loom_file):
        raise FileNotFoundError(f"File `loom_file` was not found: '{loom_file}'.")

    with h5py.File(loom_file, "r") as file:
        auc_struct = file[f"col_attrs/{col_attr_name}"][()]
        cells = file["col_attrs/CellID"][()].astype(str)
        regulons = auc_struct.dtype.names
        data = [auc_struct[regulon] for regulon in regulons]

    df = pd.DataFrame(data=data, index=regulons, columns=cells)
    df.index.name = "regulons"
    df.columns.name = "cells"
    logger.info(f"[get_regulons_auc_from_h5] Loaded AUC matrix with shape: {df.shape}.")
    return df


@logged
def get_most_var_regulon(
    data: pd.DataFrame,
    fit_thr: float = 1.5,
    min_mean_for_fit: float = 0.05,
    return_names: bool = False,
):
    """识别高变 regulon。

    该函数复用了类似 HVG 的思路：先计算均值与 `CV^2`，再用 `CV^2 ~ 1/mean` 的模型做拟合，
    最后把显著高于拟合线的 regulon 视为高变 regulon。

    Args:
        data: 行为 regulon、列为细胞的 AUC 或活性矩阵。
        fit_thr: 高变阈值倍数。
        min_mean_for_fit: 参与拟合的最小均值阈值。
        return_names: 若为 `True`，仅返回 regulon 名称列表。

    Returns:
        若 `return_names=False`，返回四元组 `(hv_data, mean_auc, cv2, fit_model)`；
        否则返回高变 regulon 名称列表。

    Example:
        hv_data, mean_auc, cv2, fit_model = get_most_var_regulon(
            data=regulon_auc_matrix,
            fit_thr=1.5,
        )
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Argument `data` must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("Argument `data` must not be empty.")

    data_no0 = data.loc[data.sum(axis=1) > 0].copy()
    if data_no0.empty:
        raise ValueError("No non-zero regulons were found in `data`.")

    mean_auc = data_no0.mean(axis=1)
    var_auc = data_no0.var(axis=1, ddof=1)
    cv2 = var_auc / (mean_auc**2)

    use_for_fit = mean_auc >= min_mean_for_fit
    if use_for_fit.sum() < 3:
        raise ValueError("Too few regulons satisfied `min_mean_for_fit` for model fitting.")

    x = 1 / mean_auc[use_for_fit].values
    y = cv2[use_for_fit].values
    X = sm.add_constant(x)
    glm_gamma = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.identity()))
    fit_res = glm_gamma.fit()
    fit_model = pd.Series(fit_res.fittedvalues, index=mean_auc[use_for_fit].index)

    hv_mask = cv2[use_for_fit] > fit_model * fit_thr
    hv_regulons = fit_model[hv_mask]
    logger.info(f"[get_most_var_regulon] Selected {len(hv_regulons)} highly variable regulons.")

    if return_names:
        return hv_regulons.index.tolist()
    return data_no0.loc[hv_regulons.index], mean_auc, cv2, fit_model


@logged
def scale_tf_matrix(data):
    """按 regulon 行对矩阵做 z-score 标准化。"""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Argument `data` must be a pandas DataFrame.")
    return data.sub(data.mean(axis=1), axis=0).div(data.std(axis=1), axis=0)


@logged
def calc_rss_one_regulon(p_regulon, p_celltype):
    """计算单个 regulon 与单个 cell type 分布之间的 RSS。"""
    jsd = jensenshannon(p_regulon, p_celltype, base=2) ** 2
    return 1 - np.sqrt(jsd)


@logged
def calc_rss(auc: pd.DataFrame, cell_annotation: pd.Series, cell_types: list = None) -> pd.DataFrame:
    """计算 regulon specificity score, RSS。

    Args:
        auc: 行为 regulon、列为细胞的 AUC 矩阵。
        cell_annotation: 细胞注释，索引需与 `auc.columns` 对齐。
        cell_types: 若提供，则仅计算这些 cell subtype/subpopulation 的 RSS。

    Returns:
        行为 regulon、列为 cell subtype 的 RSS 矩阵。

    Example:
        rss = calc_rss(
            auc=regulon_auc_matrix,
            cell_annotation=meta["Cell_Identity"],
        )
    """
    if not isinstance(auc, pd.DataFrame):
        raise TypeError("Argument `auc` must be a pandas DataFrame.")
    if not isinstance(cell_annotation, pd.Series):
        raise TypeError("Argument `cell_annotation` must be a pandas Series.")

    cell_annotation = cell_annotation.reindex(auc.columns)
    if cell_annotation.isna().any():
        raise ValueError("The index of `cell_annotation` did not match `auc.columns`.")

    auc = auc.loc[auc.sum(axis=1) > 0]
    if auc.empty:
        raise ValueError("No non-zero regulons were found in `auc`.")

    norm_auc = auc.div(auc.sum(axis=1), axis=0)
    if cell_types is None:
        cell_types = cell_annotation.unique().tolist()

    rss_dict = {}
    for cell_type in cell_types:
        mask = cell_annotation.to_numpy() == cell_type
        if mask.sum() == 0:
            warnings.warn(f"No cells were found for cell type '{cell_type}', so it will be skipped.")
            continue

        p_celltype = mask.astype(float)
        p_celltype /= p_celltype.sum()
        rss_scores = []
        for regulon_name, row in norm_auc.iterrows():
            p_regulon = row.to_numpy()
            try:
                jsd = jensenshannon(p_regulon, p_celltype, base=2)
                rss_scores.append(1 - jsd)
            except Exception as exc:
                raise RuntimeError(
                    f"JSD calculation failed for regulon '{regulon_name}' and cell type '{cell_type}'."
                ) from exc
        rss_dict[cell_type] = pd.Series(rss_scores, index=norm_auc.index)

    rss_df = pd.DataFrame(rss_dict)
    logger.info(f"[calc_rss] Calculated RSS matrix with shape: {rss_df.shape}.")
    return rss_df


@logged
def write_scenic_input(adata_subset, save_addr, use_col, file_name):
    """导出 pySCENIC 所需的 loom 输入和元数据文件。

    Args:
        adata_subset: 待导出的 AnnData 子集。
        save_addr: 输出目录。
        use_col: `adata.obs` 中需要随元数据一并导出的列名。
        file_name: 子目录名称。

    Returns:
        `None`。

    Example:
        write_scenic_input(
            adata_subset=adata_subset,
            save_addr=save_addr,
            use_col="Cell_Identity",
            file_name="Subset_A",
        )
    """
    import loompy as lp

    if use_col not in adata_subset.obs.columns:
        raise KeyError(
            f"Column `{use_col}` was not found in `adata_subset.obs`. "
            f"Available columns are: {list(adata_subset.obs.columns)}."
        )
    required_meta_cols = ["orig.ident", use_col, "disease"]
    missing = [column for column in required_meta_cols if column not in adata_subset.obs.columns]
    if missing:
        raise KeyError(f"Required metadata columns were not found in `adata_subset.obs`: {missing}.")

    path = _ensure_save_dir(os.path.join(save_addr, file_name))
    print(f"[write_scenic_input] Output directory: '{path}'")
    print("[write_scenic_input] Writing loom file.")

    row_attrs = {"Gene": np.array(adata_subset.var_names)}
    X = adata_subset.X.toarray() if hasattr(adata_subset.X, "toarray") else np.asarray(adata_subset.X)
    col_attrs = {
        "CellID": np.array(adata_subset.obs_names),
        "nGene": np.array(np.sum(X.transpose() > 0, axis=0)).flatten(),
        "nUMI": np.array(np.sum(X.transpose(), axis=0)).flatten(),
    }
    lp.create(os.path.join(path, "matrix.loom"), X.transpose(), row_attrs, col_attrs)

    print("[write_scenic_input] Writing metadata table.")
    meta_df = adata_subset.obs[required_meta_cols].copy()
    meta_df.to_csv(os.path.join(path, "meta_data.csv"))
    print("[write_scenic_input] Finished.")
