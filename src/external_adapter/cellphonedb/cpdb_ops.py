from tkinter import BooleanVar

import numpy as np
import pandas as pd
import scanpy as sc
import os, re

from anndata import AnnData
import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)

@logged
def prepare_CPDB_input(adata: AnnData,
                       output_dir: str,
                       cell_by_obs: str = "Subset_Identity",
                       group_by_obs: str = "disease",
                       downsample: bool = True,
                       min_count: int = 30, max_cells: int = 2000,
                       random_state: object = 0, use_raw: bool = False) -> None:
    '''
    替换原 CPDBTools.data_split

    Examples
    --------
    prepare_CPDB_input(adata,
                       output_dir=os.path.join(save_addr,"CPDB_251101"),
                       cell_by_obs="Subset_Identity",
                       group_by_obs="disease",
                       max_cells=5000)



    Parameters
    ----------
    :param adata:
    :param output_dir:
    :param cell_by_obs:
    :param group_by_obs:
    :param downsample:
    :param min_count:
    :param max_cells:
    :param random_state:
    :param use_raw:
    :return:
    '''
    if group_by_obs is None:
        logger.info(f"Skip grouping, take AnndataObject as whole.")
    else:
        if cell_by_obs not in adata.obs.columns or group_by_obs not in adata.obs.columns:
            raise ValueError("Please recheck adata.obs column keys.")
        logger.info(f"Split by {group_by_obs}")

    np.random.seed(random_state)

    def _adata_subset_process(adata_subset, dirname, downsample, min_count, max_cells):
        before_filtered = adata_subset.shape[0]
        subset_counts = adata_subset.obs[cell_by_obs].value_counts()
        valid_subsets = subset_counts[subset_counts >= min_count].index
        adata_subset = adata_subset[adata_subset.obs[cell_by_obs].isin(valid_subsets)].copy()
        after_filtered = adata_subset.shape[0]
        logger.info(f"Cell count before {before_filtered} → after {after_filtered}")

        if downsample:
            selected_indices = []
            for group, indices in adata_subset.obs.groupby(cell_by_obs).indices.items():
                n = min(len(indices), max_cells)
                selected_indices.extend(np.random.choice(indices, n, replace=False))
            adata_subset = adata_subset[selected_indices].copy()

        logger.info("Current subset components:\n")
        logger.info(adata_subset.obs[cell_by_obs].value_counts())

        save_dir = os.path.join(output_dir, dirname)
        os.makedirs(save_dir, exist_ok=True)

        meta_file = pd.DataFrame({
            'Cell': adata_subset.obs.index,
            'cell_type': adata_subset.obs[cell_by_obs]
        })
        meta_file_path = os.path.join(save_dir, "metadata.tsv")
        meta_file.to_csv(meta_file_path, index=False, sep="\t")

        if use_raw:
            if adata.raw is None:
                raise ValueError("adata.raw is None, cannot use raw matrix.")
            X = adata.raw[adata_subset.obs_names, :].X
            var = adata.raw.var.copy()
        else:
            X = adata_subset.X
            var = adata_subset.var.copy()

        var["gene_name"] = var.index

        adata_out = sc.AnnData(X=X, obs=adata_subset.obs.copy(), var=var)
        count_file_path = os.path.join(save_dir, "counts.h5ad")
        adata_out.write(count_file_path)

        logger.info(f"File successfully saved in: {save_dir}")

    if group_by_obs is None:
        _adata_subset_process(adata_subset=adata, dirname="total",
                              downsample=downsample, min_count=min_count, max_cells=max_cells)
    else:
        for key in adata.obs[group_by_obs].unique():
            adata_subset = adata[adata.obs[group_by_obs] == key].copy()
            _adata_subset_process(adata_subset=adata_subset, dirname=key,
                                  downsample=downsample, min_count=min_count, max_cells=max_cells)


# 全局统一的点大小映射
def size_map(x):
    return 40 + x * 200


def search_df(df_all, dict):
    '''
    dict = {"interaction_group":["TNF-TNFRSF1A","TNF-TNFRSF1B"],
           "cell_left":["CD4 Th17","ILC3", "g9d2T cytotoxic"]
           }
    :param df_all:
    :param dict:
    :return:
    '''
    df = df_all.copy()
    
    # 循环筛选
    for k, v in dict.items():
        if isinstance(v, list):
            df = df[df_all[k].isin(v)]
        elif isinstance(v, str):
            df = df[df_all[k] == v]
        else:
            raise ValueError("Please revise the search dictionary.")
    
    # 自动补全缺失组合
    interaction_list = sorted(df["interaction_group"].unique())
    celltype_list = sorted(df["celltype_group"].unique())
    all_group = sorted(df["group"].unique())
    
    full_index = pd.MultiIndex.from_product(
        [interaction_list, celltype_list, all_group],
        names=["interaction_group", "celltype_group", "group"]
    )
    
    df_full = df.set_index(["interaction_group", "celltype_group", "group"])
    df_full = df_full.reindex(full_index).reset_index()
    
    # 填充默认值
    df_full["scaled_means"] = df_full["scaled_means"].fillna(0)
    df_full["pvals"] = df_full["pvals"].fillna(1)
    df_full["scores"] = df_full["scores"].fillna(0)
    df_full["significant"] = df_full["significant"].fillna("no")
    
    df_full["dot_size"] = size_map(df_full["scaled_means"])
    
    return df_full


def vline_generator(df_full, by_ligand=True, ligand_order=None, receptor_order=None):
    
    df = df_full.assign(
        cell_left=df_full["celltype_group"].str.split("-").str[0],
        cell_right=df_full["celltype_group"].str.split("-").str[1]
    )
    
    # 自动生成顺序
    if ligand_order is None:
        ligand_order = sorted(df["cell_left"].unique())
    if receptor_order is None:
        receptor_order = sorted(df["cell_right"].unique())
    
    def sort_df(df, col1, col2, col1_order=None, col2_order=None):
        return (
            df.assign(
                **({
                       col1: pd.Categorical(df[col1], categories=col1_order, ordered=True)
                   } if col1_order else {}),
                **({
                       col2: pd.Categorical(df[col2], categories=col2_order, ordered=True)
                   } if col2_order else {})
            )
            .sort_values([col1, col2])
            .reset_index(drop=True)
        )
    
    # 排序
    if by_ligand:
        df_sorted = sort_df(df, "cell_left", "cell_right",
                            col1_order=ligand_order,
                            col2_order=receptor_order)
    else:
        df_sorted = sort_df(df, "cell_right", "cell_left",
                            col1_order=receptor_order,
                            col2_order=ligand_order)
    
    x_ticks_order = df_sorted["celltype_group"].tolist()
    
    # 生成 vline pairs
    vline_pairs = []
    if by_ligand:
        for ligand in ligand_order:
            last_idx = _find_last_index(x_ticks_order, rf"^{ligand}")
            if last_idx + 1 < len(x_ticks_order):
                vline_pairs.append((x_ticks_order[last_idx], x_ticks_order[last_idx + 1]))
    else:
        for receptor in receptor_order:
            last_idx = _find_last_index(x_ticks_order, rf"{receptor}$")
            if last_idx + 1 < len(x_ticks_order):
                vline_pairs.append((x_ticks_order[last_idx], x_ticks_order[last_idx + 1]))
    
    return df_sorted, vline_pairs

def pad_strings(series, sep="-"):
    # series 是 pandas Series
    parts = series.str.split(sep, expand=True)
    heads = parts[0]
    tails = parts[1]
    
    padded_heads = heads.str.ljust(heads.str.len().max())
    padded_tails = tails.str.rjust(tails.str.len().max())
    
    return padded_heads + "->-" + padded_tails


def _find_last_index(lst, pattern):
    """
    返回 lst 中最后一个匹配正则 pattern 的元素索引。
    若无匹配，则返回 -1。
    """
    prog = re.compile(pattern)
    
    for i in range(len(lst) - 1, -1, -1):
        if prog.search(lst[i]):
            return i
    return -1
