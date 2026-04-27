import logging
import os

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

from src.core.plot.utils import matplotlib_savefig
from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


@logged
def plot_significant_regression_by_disease(
    adata_sub,
    subset_cells,
    save_addr,
    filename,
    score_col="C3_C5_Signaling_score",
    pathway_cols=None,
    disease_col="disease",
    alpha=0.01,
    figsize=(4, 4),
    s=10,
):
    """按 disease 绘制显著相关的回归线。

    该函数会先在指定 cell subtype/subpopulation 中，将多个 pathway score
    转成长表，再分别判断每个 disease 内 `score_col` 与 pathway score
    是否达到显著相关，只有显著时才绘制回归线。

    Args:
        adata_sub: 输入 AnnData 对象。
        subset_cells: 需要保留的 cell subtype/subpopulation 名称列表。
        save_addr: 输出目录。
        filename: 输出文件名。
        score_col: 横轴评分列名。
        pathway_cols: 纵轴通路评分列名列表。
        disease_col: disease 分组列名。
        alpha: 相关性显著性阈值。
        figsize: 单个 facet 的尺寸。
        s: 散点大小。

    Returns:
        `FacetGrid` 对象。

    Example:
        grid = plot_significant_regression_by_disease(
            adata_sub=adata,
            subset_cells=["Mono", "Macro"],
            save_addr=save_addr,
            filename="myeloid_regression",
            score_col="C3_C5_Signaling_score",
            pathway_cols=["NFkB_score", "MAPK_ERK_score"],
            disease_col="disease",
            alpha=0.01,
        )
        # 只有在某个 disease 内达到显著相关时，对应 facet 才会叠加回归线
    """
    if disease_col not in adata_sub.obs.columns:
        raise KeyError(f"Column `{disease_col}` was not found in `adata.obs`.")
    if score_col not in adata_sub.obs.columns:
        raise KeyError(f"Column `{score_col}` was not found in `adata.obs`.")

    pathway_cols = pathway_cols or ["NFkB_score", "PI3K_AKT_score", "MAPK_ERK_score", "Rho_GTPase_score"]
    missing_pathways = [col for col in pathway_cols if col not in adata_sub.obs.columns]
    if missing_pathways:
        raise KeyError(f"Columns were not found in `adata.obs`: {missing_pathways}.")
    if "Subset_Identity" not in adata_sub.obs.columns:
        raise KeyError("Column `Subset_Identity` was not found in `adata.obs`.")

    sns.set_style("white")
    responsive_cells = adata_sub.obs[
        adata_sub.obs[disease_col].notna() & adata_sub.obs["Subset_Identity"].isin(subset_cells)
    ].copy()
    if responsive_cells.empty:
        raise ValueError("No cells remain after filtering `subset_cells` and non-null disease labels.")

    plot_df = responsive_cells.melt(
        id_vars=[score_col, disease_col],
        value_vars=pathway_cols,
        var_name="Pathway",
        value_name="Score",
    )

    disease_values = responsive_cells[disease_col]
    if hasattr(disease_values, "cat"):
        unique_diseases = disease_values.cat.categories.tolist()
    else:
        unique_diseases = sorted(disease_values.astype(str).unique().tolist())
        logger.warning(
            f"[plot_significant_regression_by_disease] Warning! Column `{disease_col}` is not categorical. "
            "Fallback to sorted unique values."
        )

    palette = sns.color_palette("Set1", n_colors=len(unique_diseases))
    color_dict = dict(zip(unique_diseases, palette))
    grid = sns.FacetGrid(
        plot_df,
        col="Pathway",
        hue=disease_col,
        hue_order=unique_diseases,
        palette=color_dict,
        sharex=True,
        sharey=False,
        height=figsize[0],
        aspect=1,
    )
    grid.map_dataframe(sns.scatterplot, x=score_col, y="Score", alpha=0.4, s=s)

    for ax, pathway in zip(grid.axes.flat, pathway_cols):
        ax.grid(False)
        pathway_df = plot_df[plot_df["Pathway"] == pathway]
        disease_list = list(pathway_df[disease_col].dropna().unique())

        for disease_name, sub_df in pathway_df.groupby(disease_col, observed=False):
            if len(sub_df) < 3:
                logger.warning(
                    f"[plot_significant_regression_by_disease] Warning! Too few cells for disease '{disease_name}' "
                    f"in pathway '{pathway}'. Skip regression."
                )
                continue

            rho, p_value = spearmanr(sub_df[score_col], sub_df["Score"], nan_policy="omit")
            if p_value < alpha:
                sns.regplot(
                    x=score_col,
                    y="Score",
                    data=sub_df,
                    scatter=False,
                    ci=None,
                    ax=ax,
                    line_kws={"color": color_dict[disease_name], "lw": 2},
                )
                disease_index = disease_list.index(disease_name)
                ax.text(
                    0.05,
                    0.9 - 0.1 * disease_index,
                    f"{disease_name}: rho={rho:.2f}",
                    transform=ax.transAxes,
                    color=color_dict[disease_name],
                    fontsize=12,
                )

    grid.add_legend(title=disease_col)
    grid.set_axis_labels("C3a/C5a Receptor Score", "Downstream Pathway Score")
    grid.set_titles("{col_name}")
    plt.tight_layout()

    abs_fig_path = os.path.join(save_addr, filename)
    matplotlib_savefig(grid.fig, abs_fig_path, close_after=False)
    plt.close(grid.fig)
    return grid
