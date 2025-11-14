import pandas as pd
import numpy as np
import os, gc, sys, re
from typing import Dict, List, Any
# from MulticoreTSNE import MulticoreTSNE as TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import h5py
import statsmodels.api as sm
from adjustText import adjust_text

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)


######################################
@logged
def _make_palette(cell_identity_list, seed=42):
    """
    自动根据细胞类型数量选择合适的调色板：
      - <10：使用 Seaborn "Set2"（柔和可区分）
      - 10–30：使用 HLS 色环，色差大且均匀
      - >30：使用 Glasbey（最大化颜色区分度）

    Parameters
    ----------
    cell_identity_list : pandas.Series or list-like
        包含细胞类型名
    seed : int
        随机数种子，确保颜色可复现

    Returns
    -------
    palette_dict : dict
        {cell_type: RGB_color_tuple}
    """
    unique_idents = pd.Series(cell_identity_list).unique()
    n = len(unique_idents)
    np.random.seed(seed)

    if n <= 10:
        # 柔和可区分
        palette_idents = sns.color_palette("Set2", n)
        mode = "Set2"
    elif n <= 30:
        # 色相均匀分布
        palette_idents = sns.color_palette("hls", n)
        mode = "hls"
    else:
        # 尝试使用 glasbey，若版本不支持则退化到随机HSV
        try:
            palette_idents = sns.color_palette("glasbey", n)
            mode = "glasbey"
        except Exception:
            hues = np.linspace(0, 1, n, endpoint=False)
            np.random.shuffle(hues)
            palette_idents = [
                mcolors.hsv_to_rgb((h, 0.6 + np.random.rand() * 0.3,
                                       0.8 + np.random.rand() * 0.2))
                for h in hues
            ]
            mode = "HSV-random"

    logger.info(f"{n} unique identities → using '{mode}' palette.")
    return dict(zip(unique_idents, palette_idents))

@logged
def pyscenic_pheatmap(tf_data: pd.DataFrame,
                      metadata: pd.DataFrame,
                      plt_savedir: str,
                      plt_name: str,
                      obs_key: str | list = "Cell_Identity",
                      **kwargs: object) -> None:
    """
    绘制pySCENIC转录因子活性热图，并根据metadata信息上色。
    支持单个或多个obs_key上色。

    Examples
    --------
    # 对标准化的全部转录因子 AUC
    pyscenic_pheatmap(tf_all_scaled,
                      meta,
                      plt_savedir=f"{output_dir}/{file}",
                      plt_name=f"{file}_by_disease_tf_all_heatmap",
                      obs_key=["disease","Cell_Identity"])


    Parameters
    ----------
    tf_data : pd.DataFrame
        行为TF或基因，列为细胞（列名必须与metadata中Barcodes匹配）
    metadata : pd.DataFrame
        包含细胞注释信息，需有 'Barcodes' 列
    plt_savedir : str
        输出目录
    plt_name : str
        输出文件名前缀
    obs_key : str or list
        metadata中用于着色的列名，可以是单个或多个
    **kwargs :
        传递给 sns.clustermap 的其他参数
    """

    # --- 安全检查 ---
    if "Barcodes" not in metadata.columns:
        raise ValueError("metadata 必须包含 'Barcodes' 列。")
    if not isinstance(tf_data, pd.DataFrame):
        raise TypeError("tf_data 必须是 pandas.DataFrame。")

    os.makedirs(plt_savedir, exist_ok=True)

    # --- 处理obs_key ---
    if isinstance(obs_key, list) and len(obs_key) > 1:
        set_flag = True
        col_colors_dict = {}
        palette_list = ["Set1", "Set2", "Set3", "Spectral", "Pastel1", "Paired"]
        for i, key in enumerate(obs_key):
            color_idents = metadata.set_index("Barcodes").loc[tf_data.columns, key]
            palette_name = palette_list[i % len(palette_list)]
            lut = _make_palette(color_idents, palette_name)
            col_colors_dict[key] = color_idents.map(lut)
        col_colors = pd.DataFrame(col_colors_dict)
    else:
        set_flag = False
        if isinstance(obs_key, list):
            obs_key = obs_key[0]
        color_idents = metadata.set_index("Barcodes").loc[tf_data.columns, obs_key]
        lut = _make_palette(color_idents)
        col_colors = color_idents.map(lut).to_frame(name=obs_key)

    # --- 默认参数 ---
    bl_yel_red = LinearSegmentedColormap.from_list(
        "bl_yel_red", ["navy", "lightyellow", "maroon"]
    )
    default_pars = {
        "row_cluster": True,
        "col_cluster": False,
        "figsize": (12, 6),
        "cmap": bl_yel_red,
        "vmin": -2, "vmax": 2,
        "xticklabels": False,
        "yticklabels": True
    }
    default_pars.update(kwargs)

    # --- 按 obs_key 排序列顺序（如果不聚类列） ---
    if not default_pars["col_cluster"]:
        if set_flag:
            # ⚠️ 原来这里的 obs_key 不能直接用于 sort_values(by=obs_key)
            # 因为 obs_key 是列表
            sorted_idx = col_colors.sort_values(by=obs_key).index
            tf_data = tf_data[sorted_idx]
            col_colors = col_colors.loc[sorted_idx]
        else:
            sorted_idx = col_colors.sort_values(by=obs_key).index
            tf_data = tf_data[sorted_idx]
            col_colors = col_colors.loc[sorted_idx]

    # --- 绘图 ---
    g = sns.clustermap(
        tf_data,
        col_colors=col_colors if set_flag else col_colors[obs_key],
        **default_pars
    )

    # --- 添加图例 ---
    if set_flag:
        for i, key in enumerate(obs_key):
            color_idents = metadata.set_index("Barcodes").loc[tf_data.columns, key]
            palette_name = palette_list[i % len(palette_list)]
            lut = _make_palette(color_idents, palette_name)
            for label in color_idents.unique():
                g.ax_col_dendrogram.bar(0, 0, color=lut[label], label=label, linewidth=0)
    else:
        for label in color_idents.unique():
            g.ax_col_dendrogram.bar(0, 0, color=lut[label], label=label, linewidth=0)

    g.ax_col_dendrogram.legend(
        title=", ".join(obs_key) if set_flag else obs_key,
        loc="center", ncol=3
    )

    # --- 保存 ---
    outpath = os.path.join(plt_savedir, f"{plt_name}.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure: {outpath}")

@logged
def plot_regulon_variability(
    mean_auc: pd.Series,
    cv2: pd.Series,
    fit_model: pd.Series,
    fit_thr: float = 1.5,
    hv_regulons: list | None = None,
    plt_savedir: str | None = None,
    plt_name: str = "CV2_summary"
):
    """
    Plot CV² vs mean activity and fitted model.
    考察在相同均值水平上变异度（CV^2）异常偏高的 Regulon，思路与高变基因（HVG）一致。

    - Brennecke et al. (2013) Accounting for technical noise in single-cell RNA-seq experiments. Nature Methods, 10(11):1093–1095.
    - Satija et al. (2015) Spatial reconstruction of single-cell gene expression data. Nature Biotechnology.
    Examples
    --------
    # 首先计算高变调控子 (Regulon)
    hv_data, mean_auc, cv2, fit_model = get_most_var_regulon(regulon_auc_matrix)
    # 绘图
    plot_regulon_variability(mean_auc, cv2, fit_model,
                             fit_thr=1.5,
                             hv_regulons=hv_data.index.tolist(),
                             plt_savedir="./figures",
                             plt_name="AUC_CV2_summary")


    Parameters
    ----------
    mean_auc : pd.Series
        Mean regulon activity per regulon
    cv2 : pd.Series
        Coefficient of variation squared
    fit_model : pd.Series
        Fitted CV² values from GLM
    fit_thr : float
        Threshold multiplier for HV regulon selection
    hv_regulons : list or None
        Optional list of HV regulon names for highlighting
    plt_savedir : str
        Directory to save the figure
    plt_name : str
        File name (without extension)
    """
    eps = 1e-8
    plot_df = pd.DataFrame({
        "log10_mean": np.log10(mean_auc + eps),
        "log10_cv2": np.log10(cv2 + eps),
    }).loc[fit_model.index]

    plot_df["log10_fit"] = np.log10(fit_model + eps)
    plot_df["log10_thr"] = np.log10(fit_model * fit_thr + eps)

    if plt_savedir is None:
        plt_savedir = os.getcwd()
    os.makedirs(plt_savedir, exist_ok=True)

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x="log10_mean", y="log10_cv2", data=plot_df, s=15, color="gray", alpha=0.6)
    sns.lineplot(x="log10_mean", y="log10_fit", data=plot_df, color="red", label="Fitted model")
    sns.lineplot(x="log10_mean", y="log10_thr", data=plot_df, color="blue", label=f"{fit_thr}× threshold")

    if hv_regulons is not None:
        sns.scatterplot(
            x=np.log10(mean_auc[hv_regulons] + eps),
            y=np.log10(cv2[hv_regulons] + eps),
            color="#cf5c60", s=20, label="HV regulons"
        )

    plt.title(f"Highly variable regulons (n={len(hv_regulons) if hv_regulons is not None else 0})")
    plt.xlabel("Mean regulon activity (log10)")
    plt.ylabel("CV² (log10)")
    plt.legend(frameon=False)
    plt.grid(True)
    plt.tight_layout()

    outfile = os.path.join(plt_savedir, f"{plt_name}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved figure: {outfile}")

@logged
def plot_rss_heatmap(rss: pd.DataFrame,
                     plt_savedir: object,
                     plt_name: object,
                     thr: float = None,
                     order_rows: bool = True,
                     cluster_rows: bool = False,
                     figsize: tuple = (10, 8),
                     cmap: str = "Reds",
                     verbose: bool = True) -> None:
    '''
    Plot a heatmap of RSS scores, thresholded and optionally ordered.

    Examples
    --------
    # 在绘制前需要首先根据 AUC 矩阵计算 RSS。
    rss = calc_rss(regulon_AUC,
                   cell_annotation=meta["Cell_Identity"])
    # 阈值设置为 0.01，用来尽量打印非 0 值
    plot_rss_heatmap(rss,
                     plt_savedir=f"{output_dir}/{file}",plt_name=f"{file}_RSS",
                     thr=0.01)


    Parameters
    ----------
    :param rss:
    :param plt_savedir:
    :param plt_name:
    :param thr:
    :param order_rows:
    :param cluster_rows:
    :param figsize:
    :param cmap:
    :param verbose:
    :return:
    '''
    if thr is None:
        thr = round(np.quantile(rss.values.flatten(), 0.90), 2)

    rss_subset = rss.loc[(rss > thr).any(axis=1), (rss > thr).any(axis=0)]

    if verbose:
        logger.info(f"Showing regulons and cell types with any RSS > {thr} (dim: {rss_subset.shape})")

    if rss_subset.shape[0] == 0 or rss_subset.shape[1] == 0:
        logger.info("No regulons or cell types exceed the threshold.")
        return

    if order_rows:
        max_val_idx = rss_subset.idxmax(axis=1)
        order = []
        for col in rss_subset.columns:
            sub = rss_subset[max_val_idx == col].sort_values(by=col, ascending=False)
            order.append(sub)
        rss_subset = pd.concat(order)
        cluster_rows = False

    g = sns.clustermap(rss_subset,
                       row_cluster=cluster_rows,
                       col_cluster=False,
                       cmap=cmap,
                       figsize=figsize,
                       xticklabels=True,
                       yticklabels=True)

    g.savefig(f"{plt_savedir}/{plt_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

@logged
def plot_rss_one_set(rss_df: pd.DataFrame,
                     plt_savedir, plt_name, set_name,
                     n: int = 5):
    """
    Plot RSS values for a given cell type, highlighting top n regulons.

    Examples
    --------
    # 在绘制前需要首先根据 AUC 矩阵计算 RSS。
    rss = calc_rss(regulon_AUC,
                   cell_annotation=meta["Cell_Identity"])
    # 绘图
    for set in rss.columns:
        plot_rss_one_set(rss,
                         plt_savedir=f"{output_dir}/{file}",
                         plt_name=f"{file}_{set}_RSS_Rank",
                         set_name=set)

    Parameters
    ----------
    rss_df : pd.DataFrame
        RSS matrix with regulons as rows and cell types as columns.
    set_name : str
        Cell type (column name in rss_df) to plot.
    n : int
        Number of top regulons to label.
    """
    if set_name not in rss_df.columns:
        raise ValueError(f"'{set_name}' not found in RSS dataframe columns.")

    rss_this_type = rss_df[set_name].sort_values(ascending=False)
    df = pd.DataFrame({
        "regulon": rss_this_type.index,
        "rank": range(1, len(rss_this_type) + 1),
        "rss": rss_this_type.values
    })

    # Only keep top `n` for labeling
    df["label"] = df["regulon"]
    df.loc[n:, "label"] = None

    plt.figure(figsize=(6, 4))
    plt.scatter(df["rank"], df["rss"], color="blue", s=10)
    texts = [plt.text(x, y, label, fontsize=8)
             for x, y, label in zip(df["rank"], df["rss"], df["label"]) if label is not None]
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='grey', lw=0.5))

    plt.xlabel("Regulon Rank")
    plt.ylabel("RSS")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(f"{plt_savedir}/{plt_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

