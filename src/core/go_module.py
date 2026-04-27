import logging
import os
from typing import Mapping, Optional, Sequence

import gseapy as gp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from src.core.plot.utils import matplotlib_savefig
from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)

matplotlib.use("Agg")


@logged
def balance_cell_subsets(
    adata,
    subset_col: str = "Subset_Identity",
    cond_col: str = "cond_group",
    groups: Sequence[str] = ("Inflammatory", "Control"),
    random_state: int = 42,
):
    """按条件组平衡每个细胞亚群的细胞数量。

    该函数会在每个 cell subtype/subpopulation 内，分别统计两个条件组的细胞数，
    并按较小值进行下采样，从而尽量减少组间数量不平衡对后续 DEG 或富集分析的影响。

    Args:
        adata: 输入 AnnData 对象。
        subset_col: 表示 cell subtype/subpopulation 的 `obs` 列名。
        cond_col: 表示条件分组的 `obs` 列名。
        groups: 需要平衡的两个条件组名称。
        random_state: 随机抽样种子。

    Returns:
        平衡后的 AnnData 对象。

    Example:
        # 在每个细胞亚群内部，让 Inflammatory 与 Control 的细胞数保持一致
        adata_balanced = balance_cell_subsets(
            adata=adata,
            subset_col="Subset_Identity",
            cond_col="cond_group",
            groups=["Inflammatory", "Control"],
            random_state=42,
        )
        # 适合在做 group-vs-group DEG 前先降低样本量失衡带来的影响
    """
    if subset_col not in adata.obs.columns:
        raise KeyError(f"Column `{subset_col}` was not found in `adata.obs`.")
    if cond_col not in adata.obs.columns:
        raise KeyError(f"Column `{cond_col}` was not found in `adata.obs`.")
    if groups is None or len(groups) != 2:
        raise ValueError("Argument `groups` must contain exactly 2 group names.")

    obs = adata.obs[[subset_col, cond_col]].copy()
    counts = obs.groupby([subset_col, cond_col]).size().unstack(fill_value=0)

    missing_groups = [group for group in groups if group not in counts.columns]
    if missing_groups:
        raise KeyError(f"Groups were not found in column `{cond_col}`: {missing_groups}.")

    min_counts_per_subset = counts[list(groups)].min(axis=1)
    sampled_indices = []

    for subset, n_target in min_counts_per_subset.items():
        if n_target <= 0:
            logger.warning(
                f"[balance_cell_subsets] Warning! Subset '{subset}' has zero cells in at least one target group. "
                "Skip balancing for this subset."
            )
            continue

        for group in groups:
            current_idx = obs[(obs[subset_col] == subset) & (obs[cond_col] == group)].index
            sampled_idx = pd.Series(current_idx).sample(n=int(n_target), random_state=random_state)
            sampled_indices.extend(sampled_idx.tolist())

    if len(sampled_indices) == 0:
        raise ValueError("No cells were sampled during balancing. Please recheck `subset_col`, `cond_col`, and `groups`.")

    adata_balanced = adata[sampled_indices].copy()
    print(
        f"[balance_cell_subsets] Balanced cell subsets successfully. Original cells: {adata.n_obs}, "
        f"balanced cells: {adata_balanced.n_obs}."
    )
    return adata_balanced


@logged
def analyze_DEG(
    adata,
    save_addr,
    filename,
    subset_col: str = "Subset_Identity",
    group1: str = "Absorptive colonocyte Guanylins+",
    group2: str = "Absorptive colonocyte",
):
    """计算两个细胞亚群之间的差异表达基因。

    该函数会从指定的 `subset_col` 中提取 `group1` 和 `group2` 两个
    cell subtype/subpopulation，执行 Wilcoxon 检验，并将结果保存为 CSV。

    Args:
        adata: 输入 AnnData 对象。
        save_addr: 输出目录。
        filename: 输出文件名前缀。
        subset_col: 用于区分细胞亚群的 `obs` 列名。
        group1: 目标组名称。
        group2: 参考组名称。

    Returns:
        DEG 结果 DataFrame。

    Example:
        deg_result = analyze_DEG(
            adata=adata,
            save_addr=save_addr,
            filename="Guanylins_vs_Colonocyte",
            subset_col="Subset_Identity",
            group1="Absorptive colonocyte Guanylins+",
            group2="Absorptive colonocyte",
        )
        # 结果会保存为 CSV，也会返回 DataFrame，方便继续接到 volcano 或 GO enrichment
    """
    if subset_col not in adata.obs.columns:
        raise KeyError(f"Column `{subset_col}` was not found in `adata.obs`.")

    subset_adata = adata[adata.obs[subset_col].isin([group1, group2])].copy()
    if subset_adata.n_obs == 0:
        raise ValueError(
            f"No cells were matched in `{subset_col}` for `group1`: '{group1}' and `group2`: '{group2}'."
        )

    matched_groups = subset_adata.obs[subset_col].astype(str).unique().tolist()
    if group1 not in matched_groups or group2 not in matched_groups:
        raise ValueError(
            f"Both `group1`: '{group1}' and `group2`: '{group2}' must be present in `{subset_col}`."
        )

    sc.tl.rank_genes_groups(
        subset_adata,
        groupby=subset_col,
        reference=group2,
        groups=[group1],
        method="wilcoxon",
    )

    result = sc.get.rank_genes_groups_df(subset_adata, group=group1)
    os.makedirs(save_addr, exist_ok=True)
    abs_csv_path = os.path.join(save_addr, f"{filename}.csv")
    result.to_csv(abs_csv_path, index=False)
    logger.info(f"[analyze_DEG] DEG result was saved to: '{abs_csv_path}'.")
    return result


@logged
def plot_volcano(
    result: pd.DataFrame,
    save_addr,
    filename,
    cluster_genes_dict: Optional[Mapping[str, Sequence[str]]] = None,
    lfc_limit: float = 10,
    p_thresh: float = 0.05,
    lfc_thresh: float = 1.0,
):
    """绘制 DEG 火山图并可选标注感兴趣基因。

    Args:
        result: DEG 结果表，通常来自 `analyze_DEG` 或 `sc.get.rank_genes_groups_df`。
        save_addr: 输出目录。
        filename: 输出文件名前缀。
        cluster_genes_dict: 需要重点标注的基因字典。
            若提供该参数，建议 `result` 中包含 `cluster` 列；若缺失则会回退为全表匹配。
        lfc_limit: 用于显示的 log fold change 截断上限。
        p_thresh: 判定显著性的校正 p 值阈值。
        lfc_thresh: 判定显著性的 log fold change 阈值。

    Returns:
        用于绘图的处理后 DataFrame。

    Example:
        volcano_df = plot_volcano(
            result=deg_result,
            save_addr=save_addr,
            filename="Guanylins_volcano",
            cluster_genes_dict={
                "Absorptive colonocyte Guanylins+": ["GUCA2A", "GUCA2B", "SLC26A3"],
            },
            lfc_limit=8,
            p_thresh=0.05,
            lfc_thresh=1.0,
        )
        # 如果只想画标准火山图，也可以不传 `cluster_genes_dict`
    """
    if not isinstance(result, pd.DataFrame):
        raise TypeError("Argument `result` must be a pandas DataFrame.")

    required_cols = {"pvals_adj", "logfoldchanges"}
    missing_cols = required_cols - set(result.columns)
    if missing_cols:
        raise KeyError(f"Required columns are missing in `result`: {sorted(missing_cols)}.")

    gene_col = "names" if "names" in result.columns else "gene"
    if gene_col not in result.columns:
        raise KeyError("Column `names` or `gene` must exist in `result`.")

    df = result.copy()
    df["sig"] = "Normal"
    df.loc[(df["pvals_adj"] < p_thresh) & (df["logfoldchanges"] > lfc_thresh), "sig"] = "Up"
    df.loc[(df["pvals_adj"] < p_thresh) & (df["logfoldchanges"] < -lfc_thresh), "sig"] = "Down"
    df["plot_lfc"] = df["logfoldchanges"].clip(-lfc_limit, lfc_limit)
    df["nlog10"] = -np.log10(df["pvals_adj"].fillna(1.0) + 1e-300)

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {"Up": "#e41a1c", "Down": "#377eb8", "Normal": "#d9d9d9"}
    sns.scatterplot(
        data=df,
        x="plot_lfc",
        y="nlog10",
        hue="sig",
        palette=colors,
        s=25,
        alpha=0.7,
        edgecolor=None,
        ax=ax,
        rasterized=True,
    )

    ax.axvline(x=lfc_thresh, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(x=-lfc_thresh, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(y=-np.log10(p_thresh), color="gray", linestyle="--", linewidth=0.8)

    texts = []
    if cluster_genes_dict is not None:
        try:
            from adjustText import adjust_text
        except Exception as exc:
            raise ImportError("Package `adjustText` is required when `cluster_genes_dict` is provided.") from exc

        has_cluster = "cluster" in df.columns
        if cluster_genes_dict and not has_cluster:
            logger.warning(
                "[plot_volcano] Warning! Column `cluster` was not found in `result`. Fallback to global gene matching."
            )

        for cluster, genes in cluster_genes_dict.items():
            if has_cluster:
                genes_to_plot = df[(df["cluster"] == cluster) & (df[gene_col].isin(genes))]
            else:
                genes_to_plot = df[df[gene_col].isin(genes)]

            found_genes = genes_to_plot[gene_col].tolist()
            missing_genes = sorted(set(genes) - set(found_genes))
            if missing_genes:
                print(f"[plot_volcano] Warning! Missing genes for cluster '{cluster}': {missing_genes}")
            if not genes_to_plot.empty:
                print(f"[plot_volcano] Annotating {len(found_genes)} genes for cluster '{cluster}'.")

            for _, row in genes_to_plot.iterrows():
                texts.append(
                    ax.text(
                        row["plot_lfc"],
                        row["nlog10"],
                        row[gene_col],
                        fontsize=9,
                        fontweight="medium",
                    )
                )

        if texts:
            # 仅在确实存在文本时调用自动排版，避免空列表时无意义开销。
            adjust_text(
                texts,
                ax=ax,
                only_move={"points": "y", "text": "xy"},
                arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
                expand_points=(1.5, 1.5),
            )

    if "cluster" in df.columns and df["cluster"].nunique() > 0:
        clusters = df["cluster"].dropna().astype(str).unique().tolist()
        title = f"{clusters[0]} vs Others"
    else:
        title = "Volcano Plot"

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel(r"$\log_{2}(\mathrm{Fold\ Change})$ (Clipped)", fontsize=12)
    ax.set_ylabel(r"$-\log_{10}(\mathrm{Adjusted\ P\ value})$", fontsize=12)
    ax.set_xlim(-lfc_limit * 1.1, lfc_limit * 1.1)
    ax.legend(title="Expression", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

    sns.despine()
    plt.tight_layout()

    abs_path = os.path.join(save_addr, filename)
    matplotlib_savefig(fig, abs_path, close_after=False)
    plt.close(fig)
    print(f"[plot_volcano] Volcano plot was saved successfully. LFC values were clipped at: {lfc_limit}.")
    return df


@logged
def trim_redundant_terms(res_df: pd.DataFrame, overlap_thresh: float = 0.6) -> pd.DataFrame:
    """基于基因重叠度去除冗余 GO terms。

    该函数会按显著性排序后，依次保留与已选 term 的 Jaccard overlap
    不超过阈值的 term，从而减少 GO 富集结果中语义高度重复的条目。

    Args:
        res_df: GO enrichment 结果表，至少需要包含 `Adjusted P-value` 和 `Genes` 两列。
        overlap_thresh: Jaccard overlap 阈值，超过该值的 term 将视为冗余。

    Returns:
        去冗余后的结果表。

    Example:
        trimmed_res = trim_redundant_terms(
            res_df=enr.results,
            overlap_thresh=0.6,
        )
        # 适合在绘制 GO barplot 前先减少高度重复的 terms
    """
    if not isinstance(res_df, pd.DataFrame):
        raise TypeError("Argument `res_df` must be a pandas DataFrame.")

    required_cols = {"Adjusted P-value", "Genes"}
    missing_cols = required_cols - set(res_df.columns)
    if missing_cols:
        raise KeyError(f"Required columns are missing in `res_df`: {sorted(missing_cols)}.")
    if not 0 <= overlap_thresh <= 1:
        raise ValueError("Argument `overlap_thresh` must be between 0 and 1.")

    res_df = res_df.sort_values("Adjusted P-value").reset_index(drop=True)
    keep_indices = []
    gene_sets = [set(str(genes).split(";")) for genes in res_df["Genes"]]

    for index in range(len(gene_sets)):
        is_redundant = False
        for kept_index in keep_indices:
            intersection = len(gene_sets[index].intersection(gene_sets[kept_index]))
            union = len(gene_sets[index].union(gene_sets[kept_index]))
            jaccard = intersection / union if union > 0 else 0
            if jaccard > overlap_thresh:
                is_redundant = True
                break

        if not is_redundant:
            keep_indices.append(index)

    return res_df.iloc[keep_indices].copy()


@logged
def run_go_enrichment(
    result: pd.DataFrame,
    save_addr,
    filename,
    dataset_paths: Mapping[str, str],
    go_types: Sequence[str] = ("BP",),
    overlap_thresh: float = 0.6,
    p_thr: float = 0.001,
    logFC_thr: float = 4,
    target_sig: str = "Up",
    topN: int = 10,
    organism: str = "Human",
    font_family: str = "DejaVu Sans",
    term_of_interest=None,
):
    """执行 GO enrichment、去冗余并绘制 barplot。

    该函数会先从 DEG 结果中筛选 `Up` 或 `Down` genes，再按给定 GO type
    运行 enrichr，去除高度冗余的 terms，并输出 CSV 与条形图。

    Args:
        result: DEG 结果表，至少需要包含 `names`、`pvals_adj`、`logfoldchanges` 列。
        save_addr: 输出目录。
        filename: 输出文件名前缀。
        dataset_paths: 不同 GO type 到基因集文件路径的映射。
        go_types: 需要运行的 GO 类型列表，例如 `["BP", "MF", "CC"]`。
        overlap_thresh: 去冗余时的 overlap 阈值。
        p_thr: 筛选 DEG 的校正 p 值阈值。
        logFC_thr: 筛选 DEG 的 logFC 阈值。
        target_sig: 选择 `Up` 或 `Down` 基因做富集。
        topN: 绘图时保留的 term 数量。
        organism: 传递给 gseapy 的 organism 参数。
        font_family: 绘图字体。
        term_of_interest: 感兴趣 term 列表或按 GO type 分组的字典。

    Returns:
        一个字典，键为 GO type，值为去冗余后的结果表。

    Example:
        go_results = run_go_enrichment(
            result=deg_result,
            save_addr=save_addr,
            filename="Guanylins_GO",
            dataset_paths={
                "BP": "C:/path/to/GO_BP_2023.gmt",
                "MF": "C:/path/to/GO_MF_2023.gmt",
            },
            go_types=["BP", "MF"],
            target_sig="Up",
            topN=12,
            term_of_interest={
                "BP": ["epithelial cell differentiation", "ion transport"],
                "MF": ["guanylate cyclase activator activity"],
            },
        )
        # 若 `term_of_interest` 不足 `topN`，函数会自动按显著性补齐
    """
    if not isinstance(result, pd.DataFrame):
        raise TypeError("Argument `result` must be a pandas DataFrame.")
    if not isinstance(dataset_paths, Mapping) or len(dataset_paths) == 0:
        raise ValueError("Argument `dataset_paths` must be a non-empty mapping.")
    if target_sig not in {"Up", "Down"}:
        raise ValueError("Argument `target_sig` must be either 'Up' or 'Down'.")
    if topN <= 0:
        raise ValueError("Argument `topN` must be greater than 0.")

    required_cols = {"names", "pvals_adj", "logfoldchanges"}
    missing_cols = required_cols - set(result.columns)
    if missing_cols:
        raise KeyError(f"Required columns are missing in `result`: {sorted(missing_cols)}.")

    matplotlib.rcParams["font.family"] = font_family
    os.makedirs(save_addr, exist_ok=True)

    df = result.copy()
    if target_sig == "Up":
        gene_list = df[(df["pvals_adj"] < p_thr) & (df["logfoldchanges"] > logFC_thr)]["names"].dropna().tolist()
    else:
        gene_list = df[(df["pvals_adj"] < p_thr) & (df["logfoldchanges"] < -logFC_thr)]["names"].dropna().tolist()

    if len(gene_list) < 5:
        print(f"[run_go_enrichment] Warning! Too few `{target_sig}` genes were selected for enrichment.")
        return {}

    output_results = {}
    for go_type in go_types:
        if go_type not in dataset_paths:
            print(f"[run_go_enrichment] Warning! GO type '{go_type}' was not found in `dataset_paths`. Skip it.")
            continue

        dataset_path = dataset_paths[go_type]
        if not os.path.exists(dataset_path):
            print(f"[run_go_enrichment] Warning! Dataset file was not found for GO type '{go_type}': '{dataset_path}'.")
            continue

        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=dataset_path,
            organism=organism,
            outdir=None,
        )
        res = enr.results
        if res.empty:
            print(f"[run_go_enrichment] Warning! No significant GO terms were found for GO type: '{go_type}'.")
            continue

        res = res.copy()
        res["Term"] = res["Term"].str.replace(f"^GO{go_type}_", "", regex=True)
        res["Term"] = res["Term"].str.split(" \\(GO:").str[0]
        res["Term"] = res["Term"].apply(lambda x: x[:57] + "..." if len(x) > 60 else x)
        res["Term"] = res["Term"].str.lower().str.replace("_", " ")

        res_trimmed = trim_redundant_terms(res, overlap_thresh=overlap_thresh)
        res_trimmed = res_trimmed.sort_values("Adjusted P-value").reset_index(drop=True)
        output_results[go_type] = res_trimmed

        if term_of_interest is not None:
            if isinstance(term_of_interest, dict):
                toi = term_of_interest.get(go_type, [])
            else:
                toi = term_of_interest
            toi = [str(term).lower().replace("_", " ") for term in toi]

            selected = res_trimmed[res_trimmed["Term"].isin(toi)]
            if len(selected) < topN:
                remaining = res_trimmed[~res_trimmed.index.isin(selected.index)]
                selected = pd.concat([selected, remaining.head(topN - len(selected))], axis=0)
            top_terms = selected.head(topN).copy()
        else:
            top_terms = res_trimmed.head(topN).copy()

        if top_terms.empty:
            print(f"[run_go_enrichment] Warning! No terms remain for plotting after filtering GO type: '{go_type}'.")
            continue

        top_terms["nlog10"] = -np.log10(top_terms["Adjusted P-value"] + 1e-10)

        csv_file = os.path.join(save_addr, f"{filename}_{target_sig}_{go_type}.csv")
        res_trimmed.to_csv(csv_file, index=False)
        print(f"[run_go_enrichment] GO term table was saved to: '{csv_file}'.")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("white")
        sns.barplot(data=top_terms, x="nlog10", y="Term", palette="magma", ax=ax)
        ax.set_title(f"Top {topN} {go_type} terms ({target_sig} regulated)", fontsize=14, pad=15)
        ax.set_xlabel("-log10(Adjusted P-value)", fontsize=12)
        ax.set_ylabel("")
        sns.despine(fig=fig, ax=ax)
        plt.tight_layout()

        abs_path = os.path.join(save_addr, f"{filename}_{target_sig}_{go_type}")
        matplotlib_savefig(fig, abs_path, close_after=False)
        plt.close(fig)
        print(f"[run_go_enrichment] Refined GO barplot was saved for GO type '{go_type}' and target set '{target_sig}'.")

    return output_results
