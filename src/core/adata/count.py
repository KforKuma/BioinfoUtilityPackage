import anndata

import pandas as pd
import scanpy as sc
import numpy as np
import scipy as sp

import sys
sys.stdout.reconfigure(encoding='utf-8')

import logging
from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)

@logged
def get_cluster_counts(adata, obs_key="Subset_Identity", group_by="orig.project",
                       drop_values=None, drop_axis="index"):
    """
    统计每个 group_by 下各个 obs_key 的细胞数量。

    Examples
    --------
    counts = get_cluster_counts(adata,obs_key="Subset_Identity", group_by="disease")

    """
    counts = (
        adata.obs.groupby([group_by, obs_key])
        .size()
        .unstack(fill_value=0)
    )

    if drop_values is not None:
        counts.drop(drop_values, axis=0 if drop_axis=="index" else 1, inplace=True)

    counts.attrs["obs_key"] = obs_key
    counts.attrs["group_by"] = group_by
    return counts

@logged
def get_cluster_proportions(adata, obs_key="Subset_Identity", group_by="orig.project",
                            drop_values=None, drop_axis="index"):
    """
    统计每个 group_by 下各个 obs_key 的百分比（行和为100%）。

    Examples
    --------
    props = get_cluster_proportions(adata,obs_key="Subset_Identity", group_by="disease")


    """
    props = (
        adata.obs.groupby([group_by, obs_key])
        .size()
        .groupby(level=0)
        .apply(lambda x: 100 * x / x.sum())
        .unstack(fill_value=0)
    )

    if drop_values is not None:
        props.drop(drop_values, axis=0 if drop_axis=="index" else 1, inplace=True)

    props.attrs["obs_key"] = obs_key
    props.attrs["group_by"] = group_by
    return props
@logged
def print_ref_tab(adata, obs_key, ref_key):
    # crosstable
    ct = pd.crosstab(adata.obs[obs_key], adata.obs[ref_key])
    
    # row-wise fraction
    ct_frac = ct.div(ct.sum(axis=1), axis=0)
    
    # find top2 per row
    top2 = (
        ct_frac.apply(lambda r: r.nlargest(2), axis=1)
        .stack()  # convert to multi-index Series
        .reset_index()  # columns: [row, level_1, 0]
    )
    
    # rename columns
    top2.columns = [obs_key, "Top_name", "Top_fraction"]
    
    # 给每行加一个 rank（1 or 2）
    top2["rank"] = top2.groupby(obs_key)["Top_fraction"].rank(
        method="first", ascending=False
    ).astype(int)
    
    # pivot 到宽格式
    top2_df = top2.pivot(index=obs_key, columns="rank", values=["Top_name", "Top_fraction"])
    
    # 展开 MultiIndex 列名
    top2_df.columns = [f"Top{r}_{name}" for name, r in top2_df.columns]
    
    return top2_df
