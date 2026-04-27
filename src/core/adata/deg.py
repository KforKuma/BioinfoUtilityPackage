from collections.abc import Iterable
from typing import Callable

import gc
import os
import sys

import anndata
import pandas as pd
import scanpy as sc

from src.core.adata.ops import obs_keywise_downsample

import logging
from src.utils.hier_logger import logged

sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


def _normalize_filename_prefix(filename_prefix: str | None) -> str:
    """规范化文件名前缀。"""
    if not filename_prefix:
        return ""
    return filename_prefix if filename_prefix.endswith("_") else f"{filename_prefix}_"


def _should_downsample(downsample: bool | int | float) -> bool:
    """判断是否需要执行下采样。"""
    # bool 是 int 的子类，这里需要显式排除，避免 `downsample=True` 被误判为 1。
    return not isinstance(downsample, bool) and isinstance(downsample, (int, float)) and downsample > 0


@logged
def easy_DEG(
        adata: anndata.AnnData,
        save_addr: str,
        filename_prefix: str | None,
        obs_key: str = "Subset_Identity",
        save_plot: bool = True,
        plot_gene_num: int = 5,
        downsample: bool | int | float = False,
        method: str = "wilcoxon",
        use_raw: bool = False,
        min_cells: int = 3,
) -> anndata.AnnData:
    """执行分组差异表达分析并导出结果。

    该函数会先检查分组列、过滤过小分组，再调用 `scanpy.tl.rank_genes_groups`
    计算每个分组的差异表达基因，并按需输出 dotplot 与 Excel 结果表。

    Args:
        adata: 输入的 AnnData 对象。
        save_addr: 结果输出目录。
        filename_prefix: 输出文件名前缀；若为空，则直接使用默认文件名。
        obs_key: `adata.obs` 中用于分组的列名。
        save_plot: 是否输出 DEG dotplot。
        plot_gene_num: dotplot 中每组展示的基因数量。
        downsample: 下采样参数。`False` 表示不下采样；正整数表示每组最多保留
            指定数量细胞；0 到 1 之间的小数表示按比例下采样。
        method: `scanpy.tl.rank_genes_groups` 使用的统计方法。
        use_raw: 是否优先使用 `adata.raw` 中的表达矩阵。
        min_cells: 保留分组所需的最少细胞数。小于该阈值的分组会被剔除。

    Returns:
        处理后的 AnnData 对象。若成功执行，会在 `adata.uns` 中新增对应的 DEG 结果。

    Raises:
        KeyError: 当 `obs_key` 不存在于 `adata.obs` 时抛出。
        ValueError: 当过滤后分组不足以进行差异分析，或 `min_cells` 不合法时抛出。
    """
    if obs_key not in adata.obs.columns:
        raise KeyError(f"`obs_key`: '{obs_key}' does not exist in `adata.obs`.")
    if min_cells < 1:
        raise ValueError("`min_cells` must be greater than or equal to 1.")

    if use_raw and adata.raw is None:
        logger.info("[easy_DEG] Warning! Received `use_raw`: True, but `adata.raw` does not exist. Falling back to `adata.X`.")
        use_raw = False

    if not pd.api.types.is_categorical_dtype(adata.obs[obs_key]):
        adata.obs[obs_key] = adata.obs[obs_key].astype("category")

    group_sizes = adata.obs[obs_key].value_counts()
    logger.info(f"[easy_DEG] Cell counts for each cell subtype/subpopulation in `{obs_key}`: {group_sizes.to_dict()}")

    small_groups = group_sizes[group_sizes < min_cells].index.tolist()
    if small_groups:
        logger.info(
            f"[easy_DEG] Removing cell subtype/subpopulation with cell counts below `min_cells`: "
            f"{min_cells}. Removed groups: {small_groups}"
        )
        adata = adata[~adata.obs[obs_key].isin(small_groups)].copy()
        # 过滤后同步清理未使用的类别，避免下游绘图和统计结果混入空分组。
        adata.obs[obs_key] = adata.obs[obs_key].cat.remove_unused_categories()

    remaining_group_sizes = adata.obs[obs_key].value_counts()
    logger.info(f"[easy_DEG] Cell counts after filtering in `{obs_key}`: {remaining_group_sizes.to_dict()}")

    if remaining_group_sizes.empty:
        logger.info("[easy_DEG] Warning! No cells remain after filtering. Skipping DEG analysis.")
        return adata

    if remaining_group_sizes.size < 2:
        logger.info(
            "[easy_DEG] Warning! Only one cell subtype/subpopulation remains after filtering. "
            f"Skipping DEG analysis. Current groups: {remaining_group_sizes.to_dict()}"
        )
        return adata

    os.makedirs(save_addr, exist_ok=True)
    output_prefix = _normalize_filename_prefix(filename_prefix)
    deg_key = f"deg_{obs_key}"

    if _should_downsample(downsample):
        logger.info(f"[easy_DEG] Downsampling enabled with `downsample`: {downsample}.")
        adata = obs_keywise_downsample(adata, obs_key, downsample)
    else:
        logger.info("[easy_DEG] Downsampling is disabled.")

    logger.info(f"[easy_DEG] Running DEG analysis with grouping key: `{obs_key}`.")
    sc.tl.rank_genes_groups(
        adata,
        groupby=obs_key,
        use_raw=use_raw,
        method=method,
        key_added=deg_key,
    )
    logger.info(f"[easy_DEG] DEG results were added to `adata.uns['{deg_key}']`.")

    if save_plot:
        try:
            from src.core.handlers.plot_wrapper import ScanpyPlotWrapper

            rank_genes_groups_dotplot = ScanpyPlotWrapper(sc.pl.rank_genes_groups_dotplot)
            rank_genes_groups_dotplot(
                save_addr=save_addr,
                filename=f"{output_prefix}{obs_key}_HVG_Dotplot",
                adata=adata,
                groupby=obs_key,
                key=deg_key,
                standard_scale="var",
                n_genes=plot_gene_num,
                dendrogram=False,
                use_raw=use_raw,
                show=False,
            )
            logger.info("[easy_DEG] Dotplot was saved successfully.")
        except Exception as exc:
            logger.info(f"[easy_DEG] Warning! Dotplot generation failed: {exc}")

    groups = adata.uns[deg_key]["names"].dtype.names
    df_all = pd.concat(
        [
            # 为每个 group 添加 `cluster` 列，便于后续统一排序和导出。
            sc.get.rank_genes_groups_df(adata, group=group, key=deg_key).assign(cluster=group)
            for group in groups
        ],
        ignore_index=True,
    )

    df_sorted_logfc = df_all.sort_values(
        by=["names", "logfoldchanges"],
        ascending=[True, False],
    )
    df_sorted_pval = (
        df_all[df_all["scores"] > 0]
        .sort_values(by=["cluster", "pvals_adj"], ascending=[True, True])
    )

    excel_path = os.path.join(save_addr, f"{output_prefix}{obs_key}_HVG.xlsx")
    try:
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            df_sorted_logfc.to_excel(writer, sheet_name="Sorted_by_logFC", index=False)
            df_sorted_pval.to_excel(writer, sheet_name="Sorted_by_pval", index=False)
        logger.info(f"[easy_DEG] Excel results were saved to: {excel_path}")
    except Exception as exc:
        logger.info(f"[easy_DEG] Warning! Excel export failed: {exc}")

    logger.info("[easy_DEG] DEG analysis completed.")
    return adata


@logged
def run_deg_on_subsets(
        cell_idents_list: Iterable[str] | None = None,
        resolutions: Iterable[float] | None = None,
        base_input_path: str = ".",
        base_output_path: str = ".",
        deg_method: str = "wilcoxon",
        save_plot: bool = True,
        plot_gene_num: int = 5,
        use_raw: bool = True,
        downsample: bool | int | float = False,
        save_prefix: str | None = None,
        output_suffix: str | None = None,
        easy_deg_func: Callable | None = None,
) -> None:
    """对多个子集文件执行多分辨率 DEG 分析。

    该函数通常与 `generate_subclusters_by_identity` 搭配使用：前者负责产生
    不同子集的 h5ad 文件，当前函数再对这些子集在不同 Leiden 分辨率下执行 DEG。

    Args:
        cell_idents_list: 需要处理的子集名称列表。
        resolutions: 需要分析的 Leiden 分辨率列表。
        base_input_path: 输入 h5ad 文件所在目录。
        base_output_path: DEG 结果输出目录。
        deg_method: 差异分析方法，默认使用 `wilcoxon`。
        save_plot: 是否保存 dotplot。
        plot_gene_num: dotplot 中每组展示的基因数量。
        use_raw: 是否优先使用 `adata.raw`。
        downsample: 下采样参数，语义与 `easy_DEG` 一致。
        save_prefix: 子集文件名前缀。
        output_suffix: 输出 h5ad 文件名后缀。
        easy_deg_func: 自定义 DEG 执行函数；为空时默认使用 `easy_DEG`。

    Raises:
        ValueError: 当 `cell_idents_list` 为空时抛出。
        FileNotFoundError: 当输入 h5ad 文件不存在时抛出。
    """
    if cell_idents_list is None:
        raise ValueError("`cell_idents_list` cannot be `None`. Please provide cell subtype/subpopulation names to process.")

    cell_idents_list = list(cell_idents_list)
    if not cell_idents_list:
        raise ValueError("`cell_idents_list` cannot be an empty list.")

    easy_deg_func = easy_DEG if easy_deg_func is None else easy_deg_func
    save_prefix = "SubsetSplit" if save_prefix is None else save_prefix.rstrip("_")
    output_suffix = "_DEG" if output_suffix is None else output_suffix
    resolutions = [0.5, 1.0] if resolutions is None else list(resolutions)

    if not output_suffix.endswith(".h5ad"):
        output_suffix = f"{output_suffix}.h5ad"

    os.makedirs(base_output_path, exist_ok=True)

    for cell_ident in cell_idents_list:
        logger.info(f"[run_deg_on_subsets] Starting DEG processing for cell subtype/subpopulation: '{cell_ident}'.")

        input_file = os.path.join(base_input_path, f"{save_prefix}_{cell_ident}.h5ad")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file does not exist: {input_file}")

        logger.info(f"[run_deg_on_subsets] Reading input file: {input_file}")
        adata_subset = anndata.read_h5ad(input_file)

        for resolution in resolutions:
            group_key = f"leiden_res{resolution}"
            logger.info(
                f"[run_deg_on_subsets] Running DEG with grouping key: '{group_key}', "
                f"`resolution`: {resolution}."
            )
            adata_subset = easy_deg_func(
                adata_subset,
                save_addr=base_output_path,
                filename_prefix=f"Secondary_Cluster_{cell_ident}(For clean up)",
                obs_key=group_key,
                save_plot=save_plot,
                plot_gene_num=plot_gene_num,
                downsample=downsample,
                method=deg_method,
                use_raw=use_raw,
            )

        output_file = os.path.join(base_output_path, f"{save_prefix}_{cell_ident}{output_suffix}")
        adata_subset.write_h5ad(output_file)
        logger.info(f"[run_deg_on_subsets] DEG is successfully saved at: {output_file}")


@logged
def split_and_DEG(
        adata: anndata.AnnData,
        subset_list: Iterable[str],
        obs_key: str,
        groupby_key: str,
        output_dir: str,
        count_thr: int = 30,
        downsample: bool | int | float = 5000,
        **kwargs,
) -> None:
    """按指定子集拆分数据，并在每个子集内执行 DEG 分析。

    该函数会先按 `obs_key` 拆分数据，再根据 `groupby_key` 过滤掉细胞数不足的组别，
    最后对每个符合条件的子集执行 `easy_DEG`。

    Args:
        adata: 输入的 AnnData 对象。
        subset_list: 需要逐个处理的子集名称列表。
        obs_key: 用于切分子集的 `adata.obs` 列名。
        groupby_key: 子集内部执行 DEG 时使用的分组列名。
        output_dir: 每个子集的输出根目录。
        count_thr: 子集内部每个分组的最小细胞数阈值。
        downsample: 下采样参数，语义与 `easy_DEG` 一致。
        **kwargs: 透传给 `easy_DEG` 的其余参数。

    Example:
        ```python
        cell_list = adata.obs["Subset_Identity"].unique().tolist()
        split_and_DEG(
            adata=adata,
            subset_list=cell_list,
            obs_key="Subset_Identity",
            groupby_key="disease",
            output_dir=output_dir,
        )
        ```
    """
    if obs_key not in adata.obs.columns:
        raise KeyError(f"`obs_key`: '{obs_key}' does not exist in `adata.obs`.")
    if groupby_key not in adata.obs.columns:
        raise KeyError(f"`groupby_key`: '{groupby_key}' does not exist in `adata.obs`.")
    if count_thr < 1:
        raise ValueError("`count_thr` must be greater than or equal to 1.")

    subset_list = list(subset_list)
    if not subset_list:
        logger.info("[split_and_DEG] Warning! `subset_list` is empty. Returning without any computation.")
        return

    for subset in subset_list:
        logger.info(f"[split_and_DEG] Starting processing for cell subtype/subpopulation: '{subset}'.")

        save_dir = os.path.join(output_dir, f"_{subset}")
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"[split_and_DEG] Output directory: {save_dir}")

        adata_subset = adata[adata.obs[obs_key] == subset].copy()
        logger.info(
            f"[split_and_DEG] Cell subtype/subpopulation '{subset}' contains "
            f"{adata_subset.n_obs} cells."
        )

        group_counts = adata_subset.obs[groupby_key].value_counts()
        valid_groups = group_counts.index[group_counts >= count_thr]
        logger.info(
            f"[split_and_DEG] Cell counts for each cell subtype/subpopulation in `{groupby_key}`:\n"
            f"{group_counts}"
        )

        adata_subset = adata_subset[adata_subset.obs[groupby_key].isin(valid_groups)].copy()
        if adata_subset.n_obs < (2 * count_thr):
            logger.info(
                f"[split_and_DEG] Warning! Cell subtype/subpopulation '{subset}' does not have enough "
                f"cells for DEG analysis after filtering. Skipping."
            )
            del adata_subset
            gc.collect()
            continue

        logger.info(f"[split_and_DEG] Starting DEG analysis for cell subtype/subpopulation: '{subset}'.")
        easy_DEG(
            adata_subset,
            save_addr=save_dir,
            filename_prefix=str(subset),
            obs_key=groupby_key,
            save_plot=True,
            plot_gene_num=10,
            downsample=downsample,
            **kwargs,
        )

        write_path = os.path.join(save_dir, "Subset_by_disease.h5ad")
        adata_subset.write(write_path)
        logger.info(f"[split_and_DEG] Subset result was saved to: {write_path}")

        del adata_subset
        gc.collect()
