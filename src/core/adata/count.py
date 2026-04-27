import sys
from typing import Iterable, Optional

import pandas as pd
from anndata import AnnData

import logging
from src.utils.hier_logger import logged

sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


def _validate_obs_columns(adata: AnnData, columns: Iterable[str]) -> None:
    """检查 `adata.obs` 中是否存在指定列。"""
    missing_cols = [col for col in columns if col not in adata.obs.columns]
    if missing_cols:
        raise KeyError(f"Columns do not exist in `adata.obs`: {missing_cols}.")


def _validate_drop_axis(drop_axis: str) -> int:
    """将 `drop_axis` 转换为 pandas 可接受的轴参数。"""
    if drop_axis == "index":
        return 0
    if drop_axis == "columns":
        return 1
    raise ValueError("`drop_axis` must be either 'index' or 'columns'.")


@logged
def get_cluster_counts(
        adata: AnnData,
        obs_key: str = "Subset_Identity",
        group_by: str = "orig.project",
        drop_values: Optional[Iterable[str]] = None,
        drop_axis: str = "index",
) -> pd.DataFrame:
    """统计每个分组下不同 cell subtype/subpopulation 的细胞数。

    Args:
        adata: 输入的 AnnData 对象。
        obs_key: `adata.obs` 中表示 cell subtype/subpopulation 的列名。
        group_by: `adata.obs` 中用于分组统计的列名。
        drop_values: 需要从结果中移除的行名或列名。
        drop_axis: `drop_values` 的作用方向，可选 `"index"` 或 `"columns"`。

    Returns:
        行为 `group_by`、列为 `obs_key` 的计数矩阵。

    Example:
        ```python
        counts = get_cluster_counts(
            adata,
            obs_key="Subset_Identity",
            group_by="disease",
        )
        ```
    """
    _validate_obs_columns(adata, [obs_key, group_by])
    axis = _validate_drop_axis(drop_axis)

    counts = (
        adata.obs.groupby([group_by, obs_key], observed=False)
        .size()
        .unstack(fill_value=0)
    )

    if drop_values is not None:
        logger.info(f"[get_cluster_counts] Dropping values on `drop_axis`: '{drop_axis}'.")
        counts = counts.drop(drop_values, axis=axis, errors="ignore")

    counts.attrs["obs_key"] = obs_key
    counts.attrs["group_by"] = group_by
    logger.info(
        f"[get_cluster_counts] Count table generated with shape: {counts.shape}. "
        f"`group_by`: '{group_by}', `obs_key`: '{obs_key}'."
    )
    return counts


@logged
def get_cluster_proportions(
        adata: AnnData,
        obs_key: str = "Subset_Identity",
        group_by: str = "orig.project",
        drop_values: Optional[Iterable[str]] = None,
        drop_axis: str = "index",
) -> pd.DataFrame:
    """统计每个分组下不同 cell subtype/subpopulation 的百分比。

    输出结果按行归一化，每一行总和为 100。

    Args:
        adata: 输入的 AnnData 对象。
        obs_key: `adata.obs` 中表示 cell subtype/subpopulation 的列名。
        group_by: `adata.obs` 中用于分组统计的列名。
        drop_values: 需要从结果中移除的行名或列名。
        drop_axis: `drop_values` 的作用方向，可选 `"index"` 或 `"columns"`。

    Returns:
        行为 `group_by`、列为 `obs_key` 的百分比矩阵。

    Example:
        ```python
        props = get_cluster_proportions(
            adata,
            obs_key="Subset_Identity",
            group_by="disease",
        )
        ```
    """
    _validate_obs_columns(adata, [obs_key, group_by])
    axis = _validate_drop_axis(drop_axis)

    props = (
        adata.obs.groupby([group_by, obs_key], observed=False)
        .size()
        .groupby(level=0)
        .apply(lambda x: 100 * x / x.sum())
        .unstack(fill_value=0)
    )

    if drop_values is not None:
        logger.info(f"[get_cluster_proportions] Dropping values on `drop_axis`: '{drop_axis}'.")
        props = props.drop(drop_values, axis=axis, errors="ignore")

    props.attrs["obs_key"] = obs_key
    props.attrs["group_by"] = group_by
    logger.info(
        f"[get_cluster_proportions] Proportion table generated with shape: {props.shape}. "
        f"`group_by`: '{group_by}', `obs_key`: '{obs_key}'."
    )
    return props


@logged
def print_ref_tab(adata: AnnData, obs_key: str, ref_key: str) -> pd.DataFrame:
    """生成每个 cell subtype/subpopulation 在参考标签中的 top2 对应关系表。

    Args:
        adata: 输入的 AnnData 对象。
        obs_key: `adata.obs` 中待查询的 cell subtype/subpopulation 列名。
        ref_key: `adata.obs` 中作为参考的标签列名。

    Returns:
        包含每个 `obs_key` 对应 top1/top2 标签及其比例的宽表。
    """
    _validate_obs_columns(adata, [obs_key, ref_key])

    ct = pd.crosstab(adata.obs[obs_key], adata.obs[ref_key])
    ct_frac = ct.div(ct.sum(axis=1), axis=0).fillna(0)

    top2 = (
        ct_frac.apply(lambda row: row.nlargest(2), axis=1)
        .stack()
        .reset_index()
    )
    top2.columns = [obs_key, "Top_name", "Top_fraction"]

    # 为每个 cell subtype/subpopulation 标记第 1 和第 2 高的参考标签。
    top2["rank"] = top2.groupby(obs_key)["Top_fraction"].rank(
        method="first",
        ascending=False,
    ).astype(int)

    top2_df = top2.pivot(index=obs_key, columns="rank", values=["Top_name", "Top_fraction"])
    top2_df.columns = [f"Top{rank}_{name}" for name, rank in top2_df.columns]

    logger.info(
        f"[print_ref_tab] Reference table generated with shape: {top2_df.shape}. "
        f"`obs_key`: '{obs_key}', `ref_key`: '{ref_key}'."
    )
    return top2_df
