import numpy as np
import pandas as pd
import inspect

import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import pdist

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)

@logged
def prep_query_group(means: pd.DataFrame,
                     custom_dict: dict[str, list[str]] | None = None) -> dict:
    """Return gene family query groups.

    Parameters
    ----------
    means : pd.DataFrame
        Means table.
    custom_dict : dict[str, list[str]] | None, optional
        If provided, will update the query groups with the custom list of genes.

    Returns
    -------
    dict
        Dictionary of gene families.
    """
    chemokines = [i for i in means.interacting_pair if re.search(r"^CXC|CCL|CCR|CX3|XCL|XCR", i)]
    th1 = [
        i
        for i in means.interacting_pair
        if re.search(
            r"IL2|IL12|IL18|IL27|IFNG|IL10|TNF$|TNF |LTA|LTB|STAT1|CCR5|CXCR3|IL12RB1|IFNGR1|TBX21|STAT4",
            i,
        )
    ]
    th2 = [i for i in means.interacting_pair if re.search(r"IL4|IL5|IL25|IL10|IL13|AREG|STAT6|GATA3|IL4R", i)]
    th17 = [
        i
        for i in means.interacting_pair
        if re.search(
            r"IL21|IL22|IL24|IL26|IL17A|IL17A|IL17F|IL17RA|IL10|RORC|RORA|STAT3|CCR4|CCR6|IL23RA|TGFB",
            i,
        )
    ]
    treg = [i for i in means.interacting_pair if re.search(r"IL35|IL10|FOXP3|IL2RA|TGFB", i)]
    costimulatory = [
        i
        for i in means.interacting_pair
        if re.search(
            r"CD86|CD80|CD48|LILRB2|LILRB4|TNF|CD2|ICAM|SLAM|LT[AB]|NECTIN2|CD40|CD70|CD27|CD28|CD58|TSLP|PVR|CD44|CD55|CD[1-9]",
            i,
        )
    ]
    coinhibitory = [i for i in means.interacting_pair if
                    re.search(r"SIRP|CD47|ICOS|TIGIT|CTLA4|PDCD1|CD274|LAG3|HAVCR|VSIR", i)]

    query_dict = {
        "chemokines": chemokines,
        "th1": th1,
        "th2": th2,
        "th17": th17,
        "treg": treg,
        "costimulatory": costimulatory,
        "coinhibitory": coinhibitory,
    }
    if custom_dict is not None:
        for k, r in custom_dict.items():
            query_dict.update({k: [i for i in means.interacting_pair if re.search(r"|".join(r), i)]})
    return query_dict

@logged
def hclust(
        data: pd.DataFrame,
        axis: int = 0,
        method: str = "average",
        metric: str = "euclidean",
        optimal_ordering: bool = True,
) -> list: # 目前被 _safe_hclust 代替
    """
    Perform hierarchical clustering on rows or columns of a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Input data to perform clustering on.
    axis : int, optional
        0 = cluster rows, 1 = cluster columns. Default is 0.
    method : str, optional
        Linkage method passed to scipy.cluster.hierarchy.linkage.
    metric : str, optional
        Distance metric passed to scipy.spatial.distance.pdist.
    optimal_ordering : bool, optional
        Whether to reorder linkage matrix for minimal distance distortion.

    Returns
    -------
    list
        Ordered labels after hierarchical clustering.
    """
    if data.empty:
        raise ValueError("Input DataFrame is empty.")

    # transpose if clustering columns
    matrix = data.T if axis == 1 else data

    # 若仅有一行或一列，直接返回
    if matrix.shape[0] <= 1:
        return list(matrix.index)

    # 检查是否存在NaN，否则 linkage 会报错
    if matrix.isnull().values.any():
        matrix = matrix.fillna(matrix.mean(numeric_only=True))

    # 计算距离矩阵（pdist 比直接传 data 到 linkage 更安全）
    dist = pdist(matrix, metric=metric)
    linkage = shc.linkage(dist, method=method, optimal_ordering=optimal_ordering)

    dendro = shc.dendrogram(linkage, no_plot=True, labels=matrix.index)
    return dendro["ivl"]

def split_kwargs(*funcs, strict=False, **kwargs):
    """
    根据函数签名拆分 kwargs，但不执行函数。

    Parameters
    ----------
    *funcs : callable
        要分析参数的函数列表。
    strict : bool, optional
        若为 True，则遇到未匹配参数时报错。
    **kwargs : dict
        待拆分的关键字参数。

    Returns
    -------
    dict
        {函数名: {参数名: 参数值}}
    """
    results = {}
    unused = set(kwargs.keys())

    for func in funcs:
        sig = inspect.signature(func)
        func_kwargs = {}

        for name in sig.parameters:
            if name in kwargs:
                func_kwargs[name] = kwargs[name]
                unused.discard(name)

        results[func.__name__] = func_kwargs

    if strict and unused:
        raise ValueError(f"Unrecognized arguments: {unused}")

    return results

def contained_in(lst, sub):
    n = len(sub)
    return any(sub == lst[i:i + n] for i in range(len(lst) - n + 1))
