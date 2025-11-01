import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon

import os, gc, sys, re
from typing import Dict, List, Any
# from MulticoreTSNE import MulticoreTSNE as TSNE
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import h5py
import statsmodels.api as sm
from adjustText import adjust_text

######################################

def regulons_to_gene_lists(incid_mat: pd.DataFrame,
                           omit_empty=True) -> Dict[str, List[Any]]:
    """
    将 incidence matrix 转换为 regulon 对应的基因列表字典。

    参数
    ----
    incid_mat : pd.DataFrame
        一个以 regulon 名称为行索引、基因名称为列名的 DataFrame，
        元素为 0 或 1，表示该 regulon 是否包含对应基因。

    返回
    ----
    Dict[str, List[Any]]
        一个字典，键是每个 regulon（行索引），
        值是该行中所有值为 1 的列名列表（基因列表）。
    """
    # 通过布尔索引 + 列名过滤的方式构建字典
    regulon_dict = {regulon: incid_mat.columns[incid_mat.loc[regulon] == 1].tolist()
                    for regulon in incid_mat.index}
    if omit_empty:
        empty_keys = [k for k, v in regulon_dict.items() if v == []]
        for keys in empty_keys:
            regulon_dict.pop(keys)
    return regulon_dict


def get_regulons_incid_matrix(loom_file: str) -> pd.DataFrame:
    with h5py.File(loom_file, "r") as f:
        regulon_struct = f["row_attrs/Regulons"][()]  # structured array
        genes = f["row_attrs/Gene"][()].astype(str)

        regulon_names = regulon_struct.dtype.names
        data = np.vstack([regulon_struct[name] for name in regulon_names]).T

        df = pd.DataFrame(data=data, index=genes, columns=regulon_names)
        return df


def get_regulons_auc_from_h5(loom_file: str,
                             col_attr_name: str = "RegulonsAUC") -> pd.DataFrame:
    '''
    从 pyscenic 生成的 loom 文件中读取结构化列属性（如 RegulonsAUC）并转为 DataFrame。

    :param loom_file:
    :param col_attr_name:
    :return:
    '''
    with h5py.File(loom_file, "r") as f:
        # 取出结构化字段
        auc_struct = f[f"col_attrs/{col_attr_name}"][()]
        cells = f["col_attrs/CellID"][()].astype(str)

        # regulon 名称是 dtype 中的字段名
        regulons = auc_struct.dtype.names
        data = []
        for reg in regulons:
            data.append(auc_struct[reg])

        df = pd.DataFrame(data=data, index=regulons, columns=cells)
        df.index.name = "regulons"
        df.columns.name = "cells"

    return df

def get_most_var_regulon(
    data: pd.DataFrame,
    fit_thr: float = 1.5,
    min_mean_for_fit: float = 0.05,
    return_names: bool = False
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Identify highly variable regulons using a CV² ~ 1/mean model.

    Example
    -------
    # Step 1: 计算高变调控子
    hv_data, mean_auc, cv2, fit_model = get_most_var_regulon(regulon_auc_matrix)

    # Step 2: 绘图
    plot_regulon_variability(mean_auc, cv2, fit_model,
                             fit_thr=1.5,
                             hv_regulons=hv_data.index.tolist(),
                             plt_savedir="./figures",
                             plt_name="AUC_CV2_summary")

    Parameters
    ----------
    data : pd.DataFrame
        Regulon activity matrix (rows: regulons, columns: cells)
    fit_thr : float
        Threshold multiplier above the fitted CV2 line.
    min_mean_for_fit : float
        Minimum mean AUC for inclusion in model fitting.
    return_names : bool
        If True, return only regulon names.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]
        (subset of HV regulons, mean_auc, cv2, fitted_model)
    """
    # Remove regulons not expressed in any cell
    data_no0 = data.loc[data.sum(axis=1) > 0].copy()

    # --- Compute mean and CV² ---
    mean_auc = data_no0.mean(axis=1)
    var_auc = data_no0.var(axis=1, ddof=1)
    cv2 = var_auc / (mean_auc ** 2)

    # --- Fit gamma model: CV² ~ 1/mean ---
    use_for_fit = mean_auc >= min_mean_for_fit
    x = 1 / mean_auc[use_for_fit].values
    y = cv2[use_for_fit].values
    X = sm.add_constant(x)
    glm_gamma = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.identity()))
    fit_res = glm_gamma.fit()
    a0, a1 = fit_res.params
    fit_model = pd.Series(fit_res.fittedvalues, index=mean_auc[use_for_fit].index)

    # --- Select highly variable regulons ---
    hv_mask = cv2[use_for_fit] > fit_model * fit_thr
    hv_regulons = fit_model[hv_mask]

    print(f"[get_most_var_regulon] {len(hv_regulons)} highly variable regulons selected.")

    if return_names:
        return hv_regulons.index.tolist()
    else:
        return data_no0.loc[hv_regulons.index], mean_auc, cv2, fit_model



def scale_tf_matrix(data):
    return data.sub(data.mean(axis=1), axis=0).div(data.std(axis=1), axis=0)


def calc_rss_one_regulon(p_regulon,
                         p_celltype):
    jsd = jensenshannon(p_regulon, p_celltype, base=2) ** 2
    return 1 - np.sqrt(jsd)


def calc_rss(auc: pd.DataFrame,
             cell_annotation: pd.Series,
             cell_types: list = None) -> pd.DataFrame:
    import warnings


    if cell_annotation.isna().any():
        raise ValueError("NAs in annotation")

    auc = auc.copy()

    # Remove regulons with all-zero AUC
    auc = auc.loc[auc.sum(axis=1) > 0]

    # Normalize AUC by row
    norm_auc = auc.div(auc.sum(axis=1), axis=0)

    if cell_types is None:
        cell_types = cell_annotation.unique()

    rss_dict = {}

    for this_type in cell_types:
        # 找出该细胞类型的掩码
        mask = (cell_annotation == this_type).values
        if mask.sum() == 0:
            warnings.warn(f"No cells found for cell type '{this_type}', skipping.")
            continue

        p_celltype = mask.astype(float)
        p_celltype /= p_celltype.sum()

        rss_scores = []

        for reg_name, row in norm_auc.iterrows():
            p_regulon = row.values
            if np.sum(p_regulon) == 0:
                rss_scores.append(np.nan)
                continue
            try:
                jsd = jensenshannon(p_regulon, p_celltype, base=2) ** 2
                rss = 1 - np.sqrt(jsd)
                rss_scores.append(rss)
            except Exception as e:
                warnings.warn(f"JSD failed for regulon {reg_name}: {e}")
                rss_scores.append(np.nan)

        rss_dict[this_type] = pd.Series(rss_scores, index=norm_auc.index)

    rss_df = pd.DataFrame(rss_dict)
    return rss_df
