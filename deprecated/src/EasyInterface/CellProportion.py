import anndata
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scanpy as sc
import gc
import seaborn as sns

# 迁移完成

def get_cluster_counts(adata,
                       cluster_key="cluster_final",
                       sample_key="replicate",
                       drop_values=None):
    adata_tmp = adata.copy()
    sizes = adata_tmp.obs.groupby([cluster_key, sample_key]).size()
    props = sizes.groupby(level=1, group_keys=True).apply(lambda x: 1 * x)
    props = props.droplevel(0)
    props = props.reset_index()
    props = props.pivot(columns=sample_key, index=cluster_key).T
    props.index = props.index.droplevel(0)
    props.fillna(0, inplace=True)
    if drop_values is not None:
        for drop_value in drop_values:
            props.drop(drop_value, axis=0, inplace=True)
    return props


def plot_cluster_counts(cluster_counts,
                        cluster_palette=None,
                        xlabel_rotation=0):
    fig, ax = plt.subplots(dpi=300)
    fig.patch.set_facecolor("white")
    cmap = None
    if cluster_palette is not None:
        cmap = sns.palettes.blend_palette(
            cluster_palette,
            n_colors=len(cluster_palette),
            as_cmap=True)
    cluster_counts.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        legend=None,
        colormap=cmap
    )
    ax.legend(bbox_to_anchor=(1.01, 1), frameon=False, title="Cluster")
    sns.despine(fig, ax)
    ax.tick_params(axis="x", rotation=xlabel_rotation)
    ax.set_xlabel(cluster_counts.index.name.capitalize())
    ax.set_ylabel("Counts")
    fig.tight_layout()
    return fig


def get_cluster_proportions(adata,
                            cluster_key="cluster_final",
                            sample_key="replicate",
                            drop_values=None):
    adata_tmp = adata.copy()
    sizes = adata_tmp.obs.groupby([cluster_key, sample_key]).size()
    props = sizes.groupby(level=1, group_keys=True).apply(lambda x: 100 * x / x.sum())
    props = props.droplevel(0)
    props = props.reset_index()
    props = props.pivot(columns=sample_key, index=cluster_key).T
    props.index = props.index.droplevel(0)
    props.fillna(0, inplace=True)
    if drop_values is not None:
        for drop_value in drop_values:
            props.drop(drop_value, axis=0, inplace=True)
    return props


def plot_cluster_proportions(cluster_props,
                             cluster_palette=None,
                             xlabel_rotation=0):
    fig, ax = plt.subplots(dpi=300)
    fig.patch.set_facecolor("white")
    cmap = None
    if cluster_palette is not None:
        cmap = sns.palettes.blend_palette(
            cluster_palette,
            n_colors=len(cluster_palette),
            as_cmap=True)
    cluster_props.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        legend=None,
        colormap=cmap
    )
    ax.legend(bbox_to_anchor=(1.01, 1), frameon=False, title="Cluster")
    sns.despine(fig, ax)
    ax.tick_params(axis="x", rotation=xlabel_rotation)
    ax.set_xlabel(cluster_props.index.name.capitalize())
    ax.set_ylabel("Proportion")
    fig.tight_layout()
    return fig
