from tkinter import BooleanVar

import numpy as np
import pandas as pd
import scanpy as sc
import os, re

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







