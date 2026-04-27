import logging
import os
import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


@logged
def prepare_CPDB_input(
    adata: AnnData,
    output_dir: str,
    cell_by_obs: str = "Subset_Identity",
    group_by_obs: Optional[str] = "disease",
    downsample: bool = True,
    min_count: int = 30,
    max_cells: int = 2000,
    random_state: int = 0,
    use_raw: bool = False,
) -> None:
    """准备 CellPhoneDB 所需的 `metadata.tsv` 和 `counts.h5ad`。

    该函数会按 `group_by_obs` 拆分 AnnData，并在每个分组内部按
    cell subtype/subpopulation 过滤与下采样，最终输出适用于
    CellPhoneDB 的输入文件。

    Args:
        adata: 输入 AnnData 对象。
        output_dir: 输出根目录。
        cell_by_obs: 表示 cell subtype/subpopulation 的 `obs` 列名。
        group_by_obs: 分组列名；若为 None，则将整个对象视为一个整体输出。
        downsample: 是否对每个 cell subtype 进行下采样。
        min_count: 每个 cell subtype 至少保留多少个细胞。
        max_cells: 每个 cell subtype 最多保留多少个细胞。
        random_state: 随机种子。
        use_raw: 是否使用 `adata.raw` 作为输出表达矩阵。

    Returns:
        None

    Example:
        prepare_CPDB_input(
            adata=adata,
            output_dir=os.path.join(save_addr, "CPDB_251101"),
            cell_by_obs="Subset_Identity",
            group_by_obs="disease",
            downsample=True,
            max_cells=5000,
            use_raw=False,
        )
        # 生成后的每个子目录中会包含 `metadata.tsv` 和 `counts.h5ad`
    """
    if not isinstance(adata, AnnData):
        raise TypeError("Argument `adata` must be an AnnData object.")
    if cell_by_obs not in adata.obs.columns:
        raise KeyError(f"Column `{cell_by_obs}` was not found in `adata.obs`.")
    if group_by_obs is not None and group_by_obs not in adata.obs.columns:
        raise KeyError(f"Column `{group_by_obs}` was not found in `adata.obs`.")
    if min_count <= 0 or max_cells <= 0:
        raise ValueError("Arguments `min_count` and `max_cells` must be greater than 0.")
    if use_raw and getattr(adata, "raw", None) is None:
        raise ValueError("Argument `use_raw` is True but `adata.raw` is not available.")

    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(random_state)

    def _adata_subset_process(adata_subset: AnnData, dirname: str) -> None:
        before_filtered = adata_subset.n_obs
        subset_counts = adata_subset.obs[cell_by_obs].value_counts()
        valid_subsets = subset_counts[subset_counts >= min_count].index
        adata_subset = adata_subset[adata_subset.obs[cell_by_obs].isin(valid_subsets)].copy()
        after_filtered = adata_subset.n_obs
        logger.info(
            f"[prepare_CPDB_input] Subset '{dirname}': cell count before filtering: {before_filtered}, "
            f"after filtering: {after_filtered}."
        )

        if adata_subset.n_obs == 0:
            logger.warning(
                f"[prepare_CPDB_input] Warning! No cells remain in subset '{dirname}' after filtering by `min_count`."
            )
            return

        if downsample:
            selected_indices = []
            for group_name, indices in adata_subset.obs.groupby(cell_by_obs).indices.items():
                n_keep = min(len(indices), max_cells)
                selected_indices.extend(np.random.choice(indices, n_keep, replace=False))
            adata_subset = adata_subset[selected_indices].copy()

        logger.info(
            f"[prepare_CPDB_input] Current cell subtype counts in subset '{dirname}':\n"
            f"{adata_subset.obs[cell_by_obs].value_counts()}"
        )

        save_dir = os.path.join(output_dir, str(dirname))
        os.makedirs(save_dir, exist_ok=True)

        meta_file = pd.DataFrame({"Cell": adata_subset.obs.index, "cell_type": adata_subset.obs[cell_by_obs]})
        meta_file_path = os.path.join(save_dir, "metadata.tsv")
        meta_file.to_csv(meta_file_path, index=False, sep="\t")

        if use_raw:
            source = adata.raw
            X = source[adata_subset.obs_names, :].X
            var = source.var.copy()
        else:
            X = adata_subset.X
            var = adata_subset.var.copy()
        var["gene_name"] = var.index

        adata_out = sc.AnnData(X=X, obs=adata_subset.obs.copy(), var=var)
        count_file_path = os.path.join(save_dir, "counts.h5ad")
        adata_out.write(count_file_path)
        logger.info(f"[prepare_CPDB_input] Files were saved successfully to: '{save_dir}'.")

    if group_by_obs is None:
        logger.info("[prepare_CPDB_input] `group_by_obs` is None. The whole AnnData object will be exported as one subset.")
        _adata_subset_process(adata, "total")
    else:
        logger.info(f"[prepare_CPDB_input] Split AnnData by `{group_by_obs}`.")
        for group_value in adata.obs[group_by_obs].dropna().unique():
            adata_subset = adata[adata.obs[group_by_obs] == group_value].copy()
            _adata_subset_process(adata_subset, str(group_value))


def size_map(x):
    """将数值映射为统一的 dotplot 点大小。"""
    return 40 + x * 200


def search_df(df_all: pd.DataFrame, search_dict=None, cell_any=None) -> pd.DataFrame:
    """筛选、聚合并补全 CellPhoneDB 长表。

    Args:
        df_all: 输入长表，通常来自 `CellphoneInspector.format_outcome()`。
        search_dict: 形如 `{column_name: value_or_list}` 的筛选字典。
        cell_any: 若提供，则要求 cell name 同时匹配 `cell_left` 或 `cell_right`。

    Returns:
        补全缺失组合后的 DataFrame。

    Example:
        df_full = search_df(
            df_all=df_all,
            search_dict={"interaction_group": ["TNF-TNFRSF1A", "TNF-TNFRSF1B"]},
            cell_any=["CD4 Th17", "ILC3"],
        )
        # 返回结果会按 interaction / celltype / group 组合补全，并新增 `dot_size`
    """
    if not isinstance(df_all, pd.DataFrame):
        raise TypeError("Argument `df_all` must be a pandas DataFrame.")
    if df_all.empty:
        return pd.DataFrame(columns=df_all.columns)

    required_cols = {
        "interaction_group", "celltype_group", "group", "scaled_means",
        "pvals", "scores", "significant", "cell_left", "cell_right",
    }
    missing_cols = required_cols - set(df_all.columns)
    if missing_cols:
        raise KeyError(f"Required columns are missing in `df_all`: {sorted(missing_cols)}.")

    df = df_all.copy()
    if search_dict is not None:
        for key, value in search_dict.items():
            if key not in df.columns:
                raise KeyError(f"Column `{key}` was not found in `df_all`.")
            if isinstance(value, list):
                df = df[df[key].isin(value)]
            elif isinstance(value, str):
                df = df[df[key] == value]
            else:
                raise TypeError(f"Invalid filter type for key `{key}`. Expected string or list.")

    if cell_any is not None:
        if isinstance(cell_any, str):
            cell_any = [cell_any]
        df = df[df["cell_left"].isin(cell_any) | df["cell_right"].isin(cell_any)]

    if df.empty:
        return pd.DataFrame(columns=df_all.columns)

    df = (
        df.groupby(["interaction_group", "celltype_group", "group"], as_index=False)
        .agg(
            scaled_means=("scaled_means", "mean"),
            pvals=("pvals", "min"),
            scores=("scores", "mean"),
            significant=("significant", lambda x: "yes" if (x == "yes").any() else "no"),
            cell_left=("cell_left", "first"),
            cell_right=("cell_right", "first"),
        )
    )

    full_index = pd.MultiIndex.from_product(
        [df["interaction_group"].unique(), df["celltype_group"].unique(), df["group"].unique()],
        names=["interaction_group", "celltype_group", "group"],
    )

    df_full = df.set_index(["interaction_group", "celltype_group", "group"]).reindex(full_index).reset_index()
    df_full = df_full.assign(
        scaled_means=df_full["scaled_means"].fillna(0),
        pvals=df_full["pvals"].fillna(1),
        scores=df_full["scores"].fillna(0),
        significant=df_full["significant"].fillna("no"),
    )
    df_full["dot_size"] = size_map(df_full["scaled_means"])

    mapping = df_all.set_index("celltype_group")[["cell_left", "cell_right"]].drop_duplicates()
    df_full = df_full.merge(mapping, on="celltype_group", how="left", suffixes=("", "_orig"))
    df_full["cell_left"] = df_full["cell_left_orig"]
    df_full["cell_right"] = df_full["cell_right_orig"]
    return df_full


def vline_generator(df_full: pd.DataFrame, by_ligand: bool = True, ligand_order=None, receptor_order=None):
    """为组合 dotplot / heatmap 生成排序结果和竖分隔线位置。

    Args:
        df_full: 包含 `celltype_group`、`cell_left`、`cell_right` 的长表。
        by_ligand: 是否按 ligand 端优先排序。
        ligand_order: 指定 ligand 侧顺序。
        receptor_order: 指定 receptor 侧顺序。

    Returns:
        `(df_sorted, vline_pairs)`。

    Example:
        df_sorted, vline_pairs = vline_generator(
            df_full=df_full,
            by_ligand=True,
            ligand_order=["CD4 Th17", "ILC3"],
        )
        # `vline_pairs` 可直接用于 draw_combine_dotplot 标注大类分隔
    """
    if not isinstance(df_full, pd.DataFrame):
        raise TypeError("Argument `df_full` must be a pandas DataFrame.")
    for col in ["celltype_group", "cell_left", "cell_right"]:
        if col not in df_full.columns:
            raise KeyError(f"Column `{col}` was not found in `df_full`.")

    df = df_full.copy()
    if ligand_order is None:
        ligand_order = sorted(df["cell_left"].dropna().unique())
    else:
        df = df[df["cell_left"].isin(ligand_order)]
    if receptor_order is None:
        receptor_order = sorted(df["cell_right"].dropna().unique())
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
    vline_pairs = []
    for group_name in groups:
        indices = [i for i, x in enumerate(x_ticks) if (x.startswith(group_name) if by_ligand else x.endswith(group_name))]
        if not indices:
            continue
        idx = max(indices)
        if idx + 1 < len(x_ticks):
            vline_pairs.append((x_ticks[idx], x_ticks[idx + 1]))
    return df_sorted, vline_pairs


def pad_strings(series: pd.Series, sep: str = "-") -> pd.Series:
    """将 `A-B` 形式的字符串左右对齐，便于固定宽度展示。"""
    parts = series.str.split(sep, expand=True)
    heads = parts[0]
    tails = parts[1]
    padded_heads = heads.str.ljust(heads.str.len().max())
    padded_tails = tails.str.rjust(tails.str.len().max())
    return padded_heads + "->-" + padded_tails


def _find_last_index(lst, pattern: str) -> int:
    """返回列表中最后一个匹配正则表达式的位置。"""
    prog = re.compile(pattern)
    for i in range(len(lst) - 1, -1, -1):
        if prog.search(lst[i]):
            return i
    return -1


def build_interaction_matrix(
    df: pd.DataFrame,
    cells=None,
    interactions=None,
    weight_col: str = "scores",
    cell_left_col: str = "cell_left",
    cell_right_col: str = "cell_right",
    interaction_col: str = "interaction_group",
):
    """构建 cell-by-cell 交互权重矩阵。

    Args:
        df: 输入交互长表。
        cells: 需要保留的 cell 列表；为空时自动推断。
        interactions: 需要保留的 interaction 列表；为空时自动推断。
        weight_col: 用于加权的数值列名。
        cell_left_col: 左侧细胞列名。
        cell_right_col: 右侧细胞列名。
        interaction_col: interaction 名称列名。

    Returns:
        方阵形式的交互权重矩阵。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Argument `df` must be a pandas DataFrame.")

    cells = cells or pd.concat([df[cell_left_col], df[cell_right_col]]).dropna().unique().tolist()
    interactions = interactions or df[interaction_col].dropna().unique().tolist()
    mat = pd.DataFrame(0.0, index=cells, columns=cells)

    sub = df[
        df[interaction_col].isin(interactions)
        & df[cell_left_col].isin(cells)
        & df[cell_right_col].isin(cells)
    ]
    for _, row in sub.iterrows():
        weight = row.get(weight_col, 0.0)
        if pd.notna(weight):
            mat.loc[row[cell_left_col], row[cell_right_col]] += weight
    return mat


def CCI_sankey_table(df: pd.DataFrame, center_cell="Neutro", keep_self: bool = False,
                     score_col: str = "scores", min_score: float = 0.0):
    """构建 Sankey 图所需的节点和链接表。

    Args:
        df: 输入交互长表。
        center_cell: 中间层细胞名称或名称列表。
        keep_self: 是否保留 middle-to-middle 自环。
        score_col: 作为流量的分数列名。
        min_score: 保留边的最小分数阈值。

    Returns:
        `(nodes_dict, links_df)`。

    Example:
        nodes, links = CCI_sankey_table(
            df=df_full,
            center_cell=["Neutro", "Mono"],
            score_col="scores",
            min_score=0.1,
        )
        # 返回值可继续传给自定义 Sankey 绘图函数
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Argument `df` must be a pandas DataFrame.")
    for col in ["cell_left", "cell_right", score_col]:
        if col not in df.columns:
            raise KeyError(f"Column `{col}` was not found in `df`.")

    df = df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    center_cells = [center_cell.strip()] if isinstance(center_cell, str) else [c.strip() for c in center_cell]
    center_set = set(center_cells)
    if len(center_set) > 1:
        keep_self = False

    df["cell_left"] = df["cell_left"].astype(str).str.strip()
    df["cell_right"] = df["cell_right"].astype(str).str.strip()

    left_df = df[
        df["cell_right"].isin(center_set)
        & ((~df["cell_left"].isin(center_set)) if not keep_self else True)
        & (df[score_col] >= min_score)
    ].copy()
    right_df = df[
        df["cell_left"].isin(center_set)
        & ((~df["cell_right"].isin(center_set)) if not keep_self else True)
        & (df[score_col] >= min_score)
    ].copy()

    if left_df.empty and right_df.empty:
        raise ValueError(
            "No valid flows were found around `center_cell`. Please recheck the input data or reduce `min_score`."
        )

    left_df["cell_left_L"] = left_df["cell_left"] + " "
    right_df["cell_right_R"] = "  " + right_df["cell_right"]
    left_nodes = list(pd.unique(left_df["cell_left_L"]))
    right_nodes = list(pd.unique(right_df["cell_right_R"]))

    nodes_dict = {"left": left_nodes, "middle": center_cells, "right": right_nodes}
    all_nodes = left_nodes + center_cells + right_nodes
    node_idx = {node: i for i, node in enumerate(all_nodes)}

    link_rows = []
    for _, row in left_df.iterrows():
        link_rows.append({
            "source": row["cell_left_L"],
            "target": row["cell_right"],
            "value": row[score_col],
            "label": str(row.get("interaction_group", "")),
        })
    for _, row in right_df.iterrows():
        link_rows.append({
            "source": row["cell_left"],
            "target": row["cell_right_R"],
            "value": row[score_col],
            "label": str(row.get("interaction_group", "")),
        })

    links_df = pd.DataFrame(link_rows)
    if links_df.empty:
        raise ValueError("No valid links remain for Sankey plotting.")

    links_df = (
        links_df.groupby(["source", "target"], as_index=False)
        .agg({"value": "sum", "label": lambda x: ";".join(x.unique())})
    )
    links_df["source_idx"] = links_df["source"].map(node_idx)
    links_df["target_idx"] = links_df["target"].map(node_idx)
    return nodes_dict, links_df
