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


def search_df(df_all, search_dict):
    """
    从 df_all 里筛选、聚合，并生成 df_full。
    自动保留原始 cell_left / cell_right 信息，保证最终列可用于排序。

    search_dict 示例：
    {
        "interaction_group": ["TNF-TNFRSF1A", "TNF-TNFRSF1B"],
        "cell_left": ["CD4 Th17", "ILC3", "g9d2T cytotoxic"]
    }
    """
    df = df_all.copy()
    
    # 1️⃣ 连续条件筛选
    for k, v in search_dict.items():
        if isinstance(v, list):
            df = df[df[k].isin(v)]
        elif isinstance(v, str):
            df = df[df[k] == v]
        else:
            raise ValueError(f"Invalid type for key {k}")
    
    # 2️⃣ collapse 重复 index
    df = (
        df
        .groupby(
            ["interaction_group", "celltype_group", "group"],
            as_index=False
        )
        .agg({
            "scaled_means": "mean",
            "pvals": "min",
            "scores": "mean",
            "significant": lambda x: "yes" if (x == "yes").any() else "no",
            "cell_left": "first",  # 保留原始值
            "cell_right": "first"
        })
    )
    
    # 3️⃣ 构造 full index
    full_index = pd.MultiIndex.from_product(
        [
            df["interaction_group"].unique(),
            df["celltype_group"].unique(),
            df["group"].unique()
        ],
        names=["interaction_group", "celltype_group", "group"]
    )
    
    # 4️⃣ reindex
    df_full = (
        df
        .set_index(["interaction_group", "celltype_group", "group"])
        .reindex(full_index)
        .reset_index()
    )
    
    # 5️⃣ 填默认值
    df_full = df_full.assign(
        scaled_means=df_full["scaled_means"].fillna(0),
        pvals=df_full["pvals"].fillna(1),
        scores=df_full["scores"].fillna(0),
        significant=df_full["significant"].fillna("no")
    )
    
    # 6️⃣ 生成 dot_size
    df_full["dot_size"] = size_map(df_full["scaled_means"])
    
    # 7️⃣ 保留原始正确值
    mapping = df_all.set_index("celltype_group")[["cell_left", "cell_right"]].drop_duplicates()
    df_full = df_full.merge(mapping, on="celltype_group", how="left", suffixes=("", "_orig"))
    
    # ✅ 覆盖 cell_left / cell_right 用原始值，避免 NaN
    df_full["cell_left"] = df_full["cell_left_orig"]
    df_full["cell_right"] = df_full["cell_right_orig"]
    
    return df_full


def vline_generator(df_full, by_ligand=True, ligand_order=None, receptor_order=None):
    """
    生成排序后的 df 和 vline 分割对，用于 ligand-receptor heatmap 等图。

    Parameters
    ----------
    df_full : pd.DataFrame
        包含 "celltype_group" 列，例如 "CD4 Th17 - B cell"
    by_ligand : bool, default True
        是否以 ligand (cell_left) 优先排序。False 时按 receptor (cell_right) 优先。
    ligand_order : list, optional
        指定 cell_left 顺序
    receptor_order : list, optional
        指定 cell_right 顺序

    Returns
    -------
    df_sorted : pd.DataFrame
        按指定顺序排序后的 DataFrame
    vline_pairs : list of tuple
        每个 tuple 是 (当前组最后一个 celltype_group, 下一组第一个 celltype_group)
        可用于绘制纵向分隔线
    """
    df = df_full.copy()
    # 拆分 cell_left / cell_right
    # df["cell_left"] = df["celltype_group"].str.split("-").str[0].str.strip()
    # df["cell_right"] = df["celltype_group"].str.split("-").str[1].str.strip()
    
    # 默认顺序按字母排序
    if ligand_order is None:
        ligand_order = sorted(df["cell_left"].unique())
    else:
        df = df[df["cell_left"].isin(ligand_order)]
    if receptor_order is None:
        receptor_order = sorted(df["cell_right"].unique())
    else:
        df = df[df["cell_right"].isin(receptor_order)]
    
    if by_ligand:
        df["cell_left"] = pd.Categorical(df["cell_left"], ligand_order, ordered=True)
        df["cell_right"] = pd.Categorical(df["cell_right"], receptor_order, ordered=True)
        df_sorted = df.sort_values(["cell_left", "cell_right"])
        groups = ligand_order
    else:
        df["cell_right"] = pd.Categorical(df["cell_right"], receptor_order, ordered=True)
        df["cell_left"] = pd.Categorical(df["cell_left"], ligand_order, ordered=True)
        df_sorted = df.sort_values(["cell_right", "cell_left"])
        groups = receptor_order
    
    df_sorted = df_sorted.reset_index(drop=True)
    x_ticks = df_sorted["celltype_group"].tolist()
    
    # 生成 vline 对
    vline_pairs = []
    for g in groups:
        # 找到组内最后一个索引
        indices = [i for i, x in enumerate(x_ticks) if (x.startswith(g) if by_ligand else x.endswith(g))]
        if not indices:
            continue  # 如果该组不存在，跳过
        idx = max(indices)
        if idx + 1 < len(x_ticks):
            vline_pairs.append((x_ticks[idx], x_ticks[idx + 1]))
    
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


def build_interaction_matrix(
        df,
        cells=None,
        interactions=None,
        weight_col="scores",
        cell_left_col="cell_left",
        cell_right_col="cell_right",
        interaction_col="interaction_group"
):
    """
    构建 cell × cell 的 interaction 权重矩阵

    Parameters
    ----------
    df : pd.DataFrame
        interaction 表（如 df_all）
    cells : list or None
        按圆周顺序排列的 celltype 列表；None 表示自动推断
    interactions : list or None
        允许的 interaction_group；None 表示全部
    weight_col : str
        用于加权的列（如 scores）
    """
    
    # -------- 1. 自动推断 cells --------
    if cells is None:
        cells = (
            pd.concat([df[cell_left_col], df[cell_right_col]])
            .dropna()
            .unique()
            .tolist()
        )
    
    # -------- 2. 自动推断 interactions --------
    if interactions is None:
        interactions = df[interaction_col].dropna().unique().tolist()
    
    # -------- 3. 初始化矩阵 --------
    mat = pd.DataFrame(0.0, index=cells, columns=cells)
    
    # -------- 4. 子集筛选 --------
    sub = df[
        df[interaction_col].isin(interactions) &
        df[cell_left_col].isin(cells) &
        df[cell_right_col].isin(cells)
        ]
    
    # -------- 5. 累加权重 --------
    for _, r in sub.iterrows():
        w = r.get(weight_col, 0.0)
        if pd.notna(w):
            mat.loc[r[cell_left_col], r[cell_right_col]] += w
    
    return mat


def CCI_sankey_table(df, center_cell="Neutro",
                     keep_self=False,
                     score_col="scores", min_score=0.0):
    """
    生成 Sankey 图用的三部分节点和链接表格。

    功能：
    - center_cell: str 或 list[str]，中间细胞或细胞集
    - keep_self: 是否保留 middle->middle 的正反馈（仅单 middle cell 时可选）
    - 自动添加左右节点视觉区分空格，避免 middle 同时出现在左右导致环路
    - 返回 nodes_dict {'left':[], 'middle':[], 'right':[]} 和 links_df

    注意：
    - 当 center_cell 是 list 且长度>1 时，内部 self-link 会被强制忽略
    """
    
    df = df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    
    # -----------------------------
    # normalize center_cell
    # -----------------------------
    if isinstance(center_cell, str):
        center_cells = [center_cell.strip()]
    else:
        center_cells = [c.strip() for c in center_cell]
    
    center_set = set(center_cells)
    
    # 多 middle cell 时，强制不保留 self-link
    if len(center_set) > 1:
        keep_self = False
    
    # 统一 strip 避免匹配问题
    df["cell_left"] = df["cell_left"].str.strip()
    df["cell_right"] = df["cell_right"].str.strip()
    
    # -----------------------------
    # 构建左右 df
    # -----------------------------
    left_df = df[
        (df["cell_right"].isin(center_set)) &
        ((~df["cell_left"].isin(center_set)) if not keep_self else True) &
        (df[score_col] >= min_score)
        ].copy()
    
    right_df = df[
        (df["cell_left"].isin(center_set)) &
        ((~df["cell_right"].isin(center_set)) if not keep_self else True) &
        (df[score_col] >= min_score)
        ].copy()
    
    if left_df.empty and right_df.empty:
        raise ValueError("没有找到与 center_cell 相关的有效流，请检查数据或降低 min_score。")
    
    # -----------------------------
    # 给左右节点加视觉区分空格
    # -----------------------------
    left_df["cell_left_L"] = left_df["cell_left"] + " "
    right_df["cell_right_R"] = "  " + right_df["cell_right"]
    
    # -----------------------------
    # 构建节点列表
    # -----------------------------
    left_nodes = list(pd.unique(left_df["cell_left_L"]))
    right_nodes = list(pd.unique(right_df["cell_right_R"]))
    
    nodes_dict = {
        "left": left_nodes,
        "middle": center_cells,
        "right": right_nodes
    }
    
    all_nodes = left_nodes + center_cells + right_nodes
    node_idx = {n: i for i, n in enumerate(all_nodes)}
    
    # -----------------------------
    # 构建 links
    # -----------------------------
    link_rows = []
    
    # 左 -> middle
    for _, r in left_df.iterrows():
        link_rows.append({
            "source": r["cell_left_L"],
            "target": r["cell_right"],  # 指向具体 middle cell
            "value": r[score_col],
            "label": str(r.get("interaction_group", ""))
        })
    
    # middle -> 右
    for _, r in right_df.iterrows():
        link_rows.append({
            "source": r["cell_left"],  # 从具体 middle cell
            "target": r["cell_right_R"],
            "value": r[score_col],
            "label": str(r.get("interaction_group", ""))
        })
    
    links_df = pd.DataFrame(link_rows)
    
    if links_df.empty:
        raise ValueError("没有有效的链接，绘图会空白。")
    
    # -----------------------------
    # 聚合重复 source -> target
    # -----------------------------
    links_df = (
        links_df
        .groupby(["source", "target"], as_index=False)
        .agg({
            "value": "sum",
            "label": lambda x: ";".join(x.unique())
        })
    )
    
    # 索引映射
    links_df["source_idx"] = links_df["source"].map(node_idx)
    links_df["target_idx"] = links_df["target"].map(node_idx)
    
    return nodes_dict, links_df

