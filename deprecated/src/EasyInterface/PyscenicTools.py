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

######################################

def regulons_to_gene_lists(incid_mat: pd.DataFrame, omit_empty = True) -> Dict[str, List[Any]]:
    """
    将 incidence matrix 转换为 regulon 对应的基因列表字典。
    
    参数
    ----
    incid_mat : pd.DataFrame
        一个以 regulon 名称为行索引、基因名称为列名的 DataFrame，
        元素为 0 或 1，表示该 regulon 是否包含对应基因。

    返回
    ----
    Dict[str, List[Any]]
        一个字典，键是每个 regulon（行索引），
        值是该行中所有值为 1 的列名列表（基因列表）。
    """
    # 通过布尔索引 + 列名过滤的方式构建字典
    regulon_dict = {regulon: incid_mat.columns[incid_mat.loc[regulon] == 1].tolist()
            for regulon in incid_mat.index}
    if omit_empty:
        empty_keys = [k for k, v in regulon_dict.items() if v==[]]
        for keys in empty_keys:
            regulon_dict.pop(keys)
    return regulon_dict


###

def _make_palette(cell_identity_list, palette="Set2"):
    unique_idents = cell_identity_list.unique()
    palette_idents = sns.color_palette(palette, len(unique_idents))
    return dict(zip(unique_idents, palette_idents))


###

def pyscenic_pheatmap(tf_data, meta, plt_savedir, plt_name, meta_key="Cell_Identity", **kwargs):
    if isinstance(meta_key, list) and len(meta_key) > 1:
        set_flag = 1
        col_colors_dict = {}
        palette_list = ["Set1", "Set2", "Set3", "Spectral", "Pastel1", "Paired"]
        for i, key in enumerate(meta_key):
            color_idents = meta.set_index("Barcodes").loc[tf_data.columns, key]
            palette_name = palette_list[i % len(palette_list)]
            lut = _make_palette(color_idents, palette_name)
            col_colors_dict[key] = color_idents.map(lut)
        col_colors = pd.DataFrame(col_colors_dict)
    else:
        set_flag = 0
        meta_key = meta_key[0] if isinstance(meta_key, list) else meta_key
        color_idents = meta.set_index("Barcodes").loc[tf_data.columns, meta_key]
        lut = _make_palette(color_idents)
        col_colors = color_idents.map(lut).to_frame(name=meta_key)
    
    # 默认参数
    bl_yel_red = LinearSegmentedColormap.from_list("bl_yel_red", ["navy", "lightyellow", "maroon"])
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
    
    # 按 meta_key 排序列顺序（如果不聚类列）
    if not default_pars["col_cluster"]:
        if set_flag:
            tf_data = tf_data[col_colors.sort_values(by=meta_key).index]
            col_colors = col_colors.loc[tf_data.columns]
        else:
            sorted_idx = col_colors.sort_values(by=meta_key).index
            tf_data = tf_data[sorted_idx]
            col_colors = col_colors.loc[sorted_idx]
    
    # 绘图
    g = sns.clustermap(
        tf_data,
        col_colors=col_colors if set_flag else col_colors[meta_key],
        **default_pars
    )
    
    # 添加图例
    if set_flag:
        for i, key in enumerate(meta_key):
            annot = meta.set_index("Barcodes").loc[tf_data.columns, key]
            color_idents = meta.set_index("Barcodes").loc[tf_data.columns, key]
            palette_name = palette_list[i % len(palette_list)]
            lut = _make_palette(color_idents, palette_name)
            for label in annot.unique():
                g.ax_col_dendrogram.bar(0, 0, color=lut[label], label=label, linewidth=0)
    else:
        for label in color_idents.unique():
            g.ax_col_dendrogram.bar(0, 0, color=lut[label], label=label, linewidth=0)
    
    g.ax_col_dendrogram.legend(title=", ".join(meta_key) if set_flag else meta_key, loc="center", ncol=3)
    plt.savefig(f"{plt_savedir}/{plt_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


###

def get_regulons_incid_matrix(loom_file: str) -> pd.DataFrame:
    import h5py
    import pandas as pd
    import numpy as np
    
    with h5py.File(loom_file, "r") as f:
        regulon_struct = f["row_attrs/Regulons"][()]  # structured array
        genes = f["row_attrs/Gene"][()].astype(str)
        
        regulon_names = regulon_struct.dtype.names
        data = np.vstack([regulon_struct[name] for name in regulon_names]).T
        
        df = pd.DataFrame(data=data, index=genes, columns=regulon_names)
        return df


###


def get_regulons_auc_from_h5(loom_file: str, col_attr_name: str = "RegulonsAUC") -> pd.DataFrame:
    """
    从 pyscenic 生成的 loom 文件中读取结构化列属性（如 RegulonsAUC）并转为 DataFrame。
    """
    with h5py.File(loom_file, "r") as f:
        # 取出结构化字段
        auc_struct = f[f"col_attrs/{col_attr_name}"][()]
        cells = f["col_attrs/CellID"][()].astype(str)
        
        # regulon 名称是 dtype 中的字段名
        regulons = auc_struct.dtype.names
        data = []
        for reg in regulons:
            data.append(auc_struct[reg])
        
        df = pd.DataFrame(data=data, index=regulons, columns=cells)
        df.index.name = "regulons"
        df.columns.name = "cells"
    
    return df


###

def get_most_var_regulon(data: pd.DataFrame,
                         fit_thr: float = 1.5,
                         min_mean_for_fit: float = 0.05,
                         return_names=False,
                         picture: bool = False,
                         plt_savedir=None,
                         plt_name=None
                         ) -> pd.DataFrame:
    """
    Select highly variable regulons based on coefficient of variation model.

    Parameters
    ----------
    data : pd.DataFrame
        Regulon activity matrix (rows: regulons, columns: cells)
    fit_thr : float
        Threshold multiplier above the fitted CV2 line.
    min_mean_for_fit : float
        Minimum mean AUC for a regulon to be included in the model fitting.
    picture : bool
        Whether to plot the CV2 vs mean expression and fitted model.

    Returns
    -------
    pd.DataFrame
        Subset of input data with highly variable regulons.
    """
    # Remove regulons not expressed in any cell
    data_no0 = data.loc[data.sum(axis=1) > 0].copy()
    
    # Mean and CV2：计算平均表达 & CV²（变异系数平方）
    mean_auc = data_no0.mean(axis=1)
    var_auc = data_no0.var(axis=1, ddof=1)
    cv2 = var_auc / (mean_auc ** 2)
    
    # Regulons above mean threshold：选择平均值足够大的 regulon
    use_for_fit = mean_auc >= min_mean_for_fit
    x = 1 / mean_auc[use_for_fit].values
    y = cv2[use_for_fit].values
    
    # Fit GLM (gamma family with identity link)：拟合 CV² ~ 1/均值的广义线性模型（GLM）
    X = sm.add_constant(x)
    glm_gamma = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.identity()))
    fit_res = glm_gamma.fit()
    a0, a1 = fit_res.params
    
    # Fitted model values
    fit_model = pd.Series(fit_res.fittedvalues, index=mean_auc[use_for_fit].index)
    hv_regulons = fit_model[cv2[use_for_fit] > fit_model * fit_thr]
    print(f"{len(hv_regulons)} highly variable regulons selected.")
    
    if picture:
        eps = 1e-8
        plot_df = pd.DataFrame({
            "log10_mean": np.log10(mean_auc[use_for_fit] + eps),
            "log10_cv2": np.log10(cv2[use_for_fit] + eps),
            "log10_fit": np.log10(fit_model + eps),
            "log10_thr": np.log10(fit_model * fit_thr + eps)
        })
        if plt_savedir == None:
            plt_savedir = os.getcwd()
        if plt_name == None:
            plt_name = "CV2_summary"
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x="log10_mean", y="log10_cv2", data=plot_df, s=15)
        sns.lineplot(x="log10_mean", y="log10_fit", data=plot_df, color="red")
        sns.lineplot(x="log10_mean", y="log10_thr", data=plot_df, color="blue")
        plt.title(f"{len(hv_regulons)} selected regulons")
        plt.xlabel("Mean expression (log10)")
        plt.ylabel("CV2 (log10)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fname=f"{plt_savedir}/{plt_name}.png")
    if return_names:
        return hv_regulons.index.tolist()
    else:
        print(hv_regulons.index.tolist())
        return data_no0.loc[hv_regulons.index]


###

def scale_tf_matrix(data):
    return data.sub(data.mean(axis=1), axis=0).div(data.std(axis=1), axis=0)


###

def calc_rss_one_regulon(p_regulon, p_celltype):
    jsd = jensenshannon(p_regulon, p_celltype, base=2) ** 2
    return 1 - np.sqrt(jsd)


def calc_rss(auc: pd.DataFrame, cell_annotation: pd.Series, cell_types: list = None) -> pd.DataFrame:
    import warnings
    from scipy.spatial.distance import jensenshannon
    
    if cell_annotation.isna().any():
        raise ValueError("NAs in annotation")
    
    auc = auc.copy()
    
    # Remove regulons with all-zero AUC
    auc = auc.loc[auc.sum(axis=1) > 0]
    
    # Normalize AUC by row
    norm_auc = auc.div(auc.sum(axis=1), axis=0)
    
    if cell_types is None:
        cell_types = cell_annotation.unique()
    
    rss_dict = {}
    
    for this_type in cell_types:
        # 找出该细胞类型的掩码
        mask = (cell_annotation == this_type).values
        if mask.sum() == 0:
            warnings.warn(f"No cells found for cell type '{this_type}', skipping.")
            continue
        
        p_celltype = mask.astype(float)
        p_celltype /= p_celltype.sum()
        
        rss_scores = []
        
        for reg_name, row in norm_auc.iterrows():
            p_regulon = row.values
            if np.sum(p_regulon) == 0:
                rss_scores.append(np.nan)
                continue
            try:
                jsd = jensenshannon(p_regulon, p_celltype, base=2) ** 2
                rss = 1 - np.sqrt(jsd)
                rss_scores.append(rss)
            except Exception as e:
                warnings.warn(f"JSD failed for regulon {reg_name}: {e}")
                rss_scores.append(np.nan)
        
        rss_dict[this_type] = pd.Series(rss_scores, index=norm_auc.index)
    
    rss_df = pd.DataFrame(rss_dict)
    return rss_df


def plot_rss_heatmap(rss: pd.DataFrame,
                     plt_savedir, plt_name,
                     thr: float = None,
                     order_rows: bool = True,
                     cluster_rows: bool = False,
                     figsize=(10, 8),
                     cmap="Reds",
                     verbose=True):
    """
    Plot a heatmap of RSS scores, thresholded and optionally ordered.
    """
    if thr is None:
        thr = round(np.quantile(rss.values.flatten(), 0.90), 2)
    
    rss_subset = rss.loc[(rss > thr).any(axis=1), (rss > thr).any(axis=0)]
    
    if verbose:
        print(f"Showing regulons and cell types with any RSS > {thr} (dim: {rss_subset.shape})")
    
    if rss_subset.shape[0] == 0 or rss_subset.shape[1] == 0:
        print("No regulons or cell types exceed the threshold.")
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
    plt.savefig(f"{plt_savedir}/{plt_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_rss_one_set(rss_df: pd.DataFrame, plt_savedir, plt_name, set_name,
                     n: int = 5):
    """
    Plot RSS values for a given cell type, highlighting top n regulons.

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


