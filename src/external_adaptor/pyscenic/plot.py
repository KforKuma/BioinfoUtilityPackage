"""pySCENIC 可视化函数。"""

import logging
import os
from typing import Iterable, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from src.core.plot.utils import matplotlib_savefig
from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


def _ensure_save_dir(plt_savedir: str) -> str:
    """检查并创建输出目录。"""
    if not isinstance(plt_savedir, str) or plt_savedir.strip() == "":
        raise ValueError("Argument `plt_savedir` must be a non-empty string.")
    plt_savedir = plt_savedir.strip()
    os.makedirs(plt_savedir, exist_ok=True)
    return plt_savedir


@logged
def _make_palette(cell_identity_list, *, seed=42):
    """根据类别数自动生成调色板。

    Args:
        cell_identity_list: 类别名称序列。
        seed: 随机种子。

    Returns:
        字典，键为类别名，值为颜色。
    """
    unique_idents = pd.Series(cell_identity_list).astype(str).unique()
    n = len(unique_idents)

    if seed is not None and not isinstance(seed, (int, np.integer)):
        raise TypeError(f"Argument `seed` must be an integer or `None`, but got: {type(seed)}.")

    rng = np.random.default_rng(seed)
    if n <= 10:
        palette_idents = sns.color_palette("Set2", n)
        mode = "Set2"
    elif n <= 30:
        palette_idents = sns.color_palette("hls", n)
        mode = "hls"
    else:
        try:
            palette_idents = sns.color_palette("glasbey", n)
            mode = "glasbey"
        except Exception:
            hues = np.linspace(0, 1, n, endpoint=False)
            rng.shuffle(hues)
            palette_idents = [
                mcolors.hsv_to_rgb((hue, 0.6 + rng.random() * 0.3, 0.8 + rng.random() * 0.2))
                for hue in hues
            ]
            mode = "HSV-random"

    logger.info(f"[_make_palette] {n} unique identities were detected and palette mode '{mode}' was used.")
    return dict(zip(unique_idents, palette_idents))


@logged
def pyscenic_pheatmap(
    tf_data: pd.DataFrame,
    metadata: pd.DataFrame,
    plt_savedir: str,
    plt_name: str,
    obs_key="Cell_Identity",
    **kwargs,
):
    """绘制 pySCENIC 转录因子活性热图。

    支持单个或多个 `obs_key` 作为列颜色注释。若未启用列聚类，则会按注释颜色顺序重排列，
    以保证同类 cell subtype/subpopulation 更容易聚在一起观察。

    Args:
        tf_data: 行为 regulon/TF、列为 Barcodes 的矩阵。
        metadata: 至少包含 `Barcodes` 和 `obs_key` 列的元数据表。
        plt_savedir: 图像输出目录。
        plt_name: 输出文件名主体，不带扩展名。
        obs_key: 单个或多个元数据分组列名。
        **kwargs: 透传给 `sns.clustermap()` 的附加参数。

    Returns:
        `None`。

    Example:
        pyscenic_pheatmap(
            tf_data=tf_matrix,
            metadata=meta_df,
            plt_savedir=save_addr,
            plt_name="TF_activity_heatmap",
            obs_key=["Cell_Identity", "disease"],
        )
    """
    if not isinstance(tf_data, pd.DataFrame):
        raise TypeError("Argument `tf_data` must be a pandas DataFrame.")
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("Argument `metadata` must be a pandas DataFrame.")
    if "Barcodes" not in metadata.columns:
        raise ValueError("Column `Barcodes` was not found in `metadata`.")
    if not isinstance(plt_name, str) or plt_name.strip() == "":
        raise ValueError("Argument `plt_name` must be a non-empty string.")

    plt_savedir = _ensure_save_dir(plt_savedir)
    obs_keys = [obs_key] if isinstance(obs_key, str) else list(obs_key)
    missing = [key for key in obs_keys if key not in metadata.columns]
    if missing:
        raise KeyError(f"Required metadata columns were not found: {missing}.")

    metadata_indexed = metadata.set_index("Barcodes")
    missing_barcodes = [barcode for barcode in tf_data.columns if barcode not in metadata_indexed.index]
    if missing_barcodes:
        logger.info(
            f"[pyscenic_pheatmap] Warning! {len(missing_barcodes)} barcodes were not found in `metadata` and will be dropped."
        )
        tf_data = tf_data.loc[:, [barcode for barcode in tf_data.columns if barcode in metadata_indexed.index]]

    if tf_data.empty:
        raise ValueError("No columns remained in `tf_data` after barcode matching.")

    col_colors_dict = {}
    lut_dict = {}
    for key in obs_keys:
        color_idents = metadata_indexed.loc[tf_data.columns, key].astype(str)
        lut = _make_palette(color_idents)
        col_colors_dict[key] = color_idents.map(lut)
        lut_dict[key] = lut
    col_colors = pd.DataFrame(col_colors_dict)

    bl_yel_red = LinearSegmentedColormap.from_list("bl_yel_red", ["navy", "lightyellow", "maroon"])
    default_params = {
        "row_cluster": True,
        "col_cluster": False,
        "figsize": (12, 6),
        "cmap": bl_yel_red,
        "vmin": -2,
        "vmax": 2,
        "xticklabels": False,
        "yticklabels": True,
    }
    default_params.update(kwargs)

    if not default_params["col_cluster"]:
        sort_order = metadata_indexed.loc[tf_data.columns, obs_keys].astype(str).apply(tuple, axis=1).sort_values().index
        tf_data = tf_data.loc[:, sort_order]
        col_colors = col_colors.loc[sort_order]

    g = sns.clustermap(
        tf_data,
        col_colors=col_colors if len(obs_keys) > 1 else col_colors[obs_keys[0]],
        **default_params,
    )

    handles = []
    labels = []
    for key in obs_keys:
        for label, color in lut_dict[key].items():
            handles.append(plt.Line2D([0], [0], marker="s", color=color, linestyle="", markersize=8))
            labels.append(f"{key}: {label}")
    g.ax_col_dendrogram.legend(handles, labels, title=", ".join(obs_keys), loc="center", ncol=3, frameon=False)

    abs_path = os.path.join(plt_savedir, plt_name.strip())
    matplotlib_savefig(g.fig, abs_path)
    logger.info(f"[pyscenic_pheatmap] Figure was saved with base filename: '{plt_name.strip()}'.")


@logged
def plot_regulon_variability(
    mean_auc: pd.Series,
    cv2: pd.Series,
    fit_model: pd.Series,
    fit_thr: float = 1.5,
    hv_regulons=None,
    plt_savedir=None,
    plt_name: str = "CV2_summary",
):
    """绘制 regulon variability summary 图。

    Args:
        mean_auc: 每个 regulon 的均值。
        cv2: 每个 regulon 的 `CV^2`。
        fit_model: 拟合得到的理论 `CV^2` 曲线。
        fit_thr: 高变阈值倍数。
        hv_regulons: 需要高亮的高变 regulon 列表。
        plt_savedir: 输出目录。
        plt_name: 输出文件名主体，不带扩展名。

    Returns:
        `None`。

    Example:
        plot_regulon_variability(
            mean_auc=mean_auc,
            cv2=cv2,
            fit_model=fit_model,
            fit_thr=1.5,
            hv_regulons=hv_data.index.tolist(),
            plt_savedir=save_addr,
            plt_name="AUC_CV2_summary",
        )
    """
    plt_savedir = _ensure_save_dir(plt_savedir or os.getcwd())
    eps = 1e-8
    plot_df = pd.DataFrame({"log10_mean": np.log10(mean_auc + eps), "log10_cv2": np.log10(cv2 + eps)}).loc[fit_model.index]
    plot_df["log10_fit"] = np.log10(fit_model + eps)
    plot_df["log10_thr"] = np.log10(fit_model * fit_thr + eps)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x="log10_mean", y="log10_cv2", data=plot_df, s=15, color="gray", alpha=0.6, ax=ax)
    sns.lineplot(x="log10_mean", y="log10_fit", data=plot_df, color="red", label="Fitted model", ax=ax)
    sns.lineplot(x="log10_mean", y="log10_thr", data=plot_df, color="blue", label=f"{fit_thr}x threshold", ax=ax)

    if hv_regulons is not None and len(hv_regulons) > 0:
        hv_regulons = [regulon for regulon in hv_regulons if regulon in mean_auc.index]
        sns.scatterplot(
            x=np.log10(mean_auc[hv_regulons] + eps),
            y=np.log10(cv2[hv_regulons] + eps),
            color="#cf5c60",
            s=20,
            label="HV regulons",
            ax=ax,
        )

    num_hv = len(hv_regulons) if hv_regulons is not None else 0
    ax.set_title(f"Highly Variable Regulons (n={num_hv})")
    ax.set_xlabel("Mean Regulon Activity (log10)")
    ax.set_ylabel("CV^2 (log10)")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()

    abs_path = os.path.join(plt_savedir, plt_name.strip())
    matplotlib_savefig(fig, abs_path)
    logger.info(f"[plot_regulon_variability] Figure was saved with base filename: '{plt_name.strip()}'.")


@logged
def plot_rss_heatmap(
    rss: pd.DataFrame,
    plt_savedir,
    plt_name,
    thr: float = None,
    order_rows: bool = True,
    cluster_rows: bool = False,
    figsize: tuple = (10, 8),
    cmap: str = "Reds",
    verbose: bool = True,
) -> None:
    """绘制 RSS heatmap。

    Args:
        rss: 行为 regulon、列为 cell subtype 的 RSS 矩阵。
        plt_savedir: 输出目录。
        plt_name: 输出文件名主体。
        thr: 过滤阈值；若不提供，则自动使用 90% 分位数。
        order_rows: 是否按 peak cell type 重排行。
        cluster_rows: 是否对行做聚类。
        figsize: 图像大小。
        cmap: 热图配色。
        verbose: 是否打印过程日志。

    Returns:
        `None`。

    Example:
        plot_rss_heatmap(
            rss=rss,
            plt_savedir=save_addr,
            plt_name="rss_heatmap",
            thr=0.01,
        )
    """
    if not isinstance(rss, pd.DataFrame):
        raise TypeError("Argument `rss` must be a pandas DataFrame.")
    if rss.empty:
        raise ValueError("Argument `rss` must not be empty.")

    plt_savedir = _ensure_save_dir(plt_savedir)
    if thr is None:
        thr = round(float(np.nanquantile(rss.values.flatten(), 0.90)), 2)

    rss_subset = rss.loc[(rss > thr).any(axis=1), (rss > thr).any(axis=0)]
    if verbose:
        logger.info(
            f"[plot_rss_heatmap] Showing regulons and cell subtypes with any RSS > {thr} "
            f"(shape: {rss_subset.shape})."
        )

    if rss_subset.shape[0] == 0 or rss_subset.shape[1] == 0:
        logger.info("[plot_rss_heatmap] Warning! No regulons or cell subtypes exceeded the threshold.")
        return

    if order_rows:
        max_val_idx = rss_subset.idxmax(axis=1)
        ordered_groups = []
        for column in rss_subset.columns:
            subset = rss_subset[max_val_idx == column].sort_values(by=column, ascending=False)
            if not subset.empty:
                ordered_groups.append(subset)
        rss_subset = pd.concat(ordered_groups) if ordered_groups else rss_subset
        cluster_rows = False

    g = sns.clustermap(
        rss_subset,
        row_cluster=cluster_rows,
        col_cluster=False,
        cmap=cmap,
        figsize=figsize,
        xticklabels=True,
        yticklabels=True,
    )
    abs_path = os.path.join(plt_savedir, plt_name.strip())
    matplotlib_savefig(g.fig, abs_path)
    plt.close(g.fig)
    logger.info(f"[plot_rss_heatmap] Figure was saved with base filename: '{plt_name.strip()}'.")


@logged
def plot_rss_one_set(rss_df: pd.DataFrame, plt_savedir, plt_name, set_name, n: int = 5):
    """绘制单个 cell subtype 的 RSS 排名图，并标注 Top regulons。

    Args:
        rss_df: RSS 矩阵。
        plt_savedir: 输出目录。
        plt_name: 输出文件名主体。
        set_name: 目标 cell subtype 名称。
        n: 需要标注的 Top regulon 数量。

    Returns:
        `None`。

    Example:
        plot_rss_one_set(
            rss_df=rss,
            plt_savedir=save_addr,
            plt_name="Treg_RSS_rank",
            set_name="Treg",
            n=10,
        )
    """
    if set_name not in rss_df.columns:
        raise ValueError(f"Cell subtype `{set_name}` was not found in `rss_df.columns`.")

    try:
        from adjustText import adjust_text
    except ImportError as exc:
        raise ImportError(
            "Function `plot_rss_one_set` requires package `adjustText`. "
            "Please install it in the target environment."
        ) from exc

    plt_savedir = _ensure_save_dir(plt_savedir)
    rss_this_type = rss_df[set_name].sort_values(ascending=False)
    df = pd.DataFrame({"regulon": rss_this_type.index, "rank": range(1, len(rss_this_type) + 1), "rss": rss_this_type.values})
    df["label"] = df["regulon"]
    df.loc[n:, "label"] = None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["rank"], df["rss"], color="blue", s=10)
    texts = [
        ax.text(x, y, label, fontsize=8)
        for x, y, label in zip(df["rank"], df["rss"], df["label"])
        if label is not None
    ]
    adjust_text(texts, ax=ax, arrowprops={"arrowstyle": "-", "color": "grey", "lw": 0.5})

    ax.set_xlabel("Regulon Rank")
    ax.set_ylabel("RSS")
    ax.set_title(f"RSS Rank for '{set_name}'")
    ax.grid(True, linestyle="--", alpha=0.3)

    abs_path = os.path.join(plt_savedir, plt_name.strip())
    matplotlib_savefig(fig, abs_path)
    logger.info(f"[plot_rss_one_set] Figure was saved with base filename: '{plt_name.strip()}'.")
