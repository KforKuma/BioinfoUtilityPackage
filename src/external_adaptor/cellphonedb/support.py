import inspect
import logging
import re
from typing import Any, Callable, Optional

import pandas as pd
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import pdist

from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


@logged
def prep_query_group(means: pd.DataFrame, custom_dict: Optional[dict[str, list[str]]] = None) -> dict:
    """构建内置与自定义基因家族查询字典。

    Args:
        means: CellPhoneDB 的 means 表，至少需要包含 `interacting_pair` 列。
        custom_dict: 额外的自定义基因家族字典，格式为
            `{family_name: [regex_or_gene_pattern, ...]}`。

    Returns:
        一个字典，键为基因家族名称，值为匹配到的 `interacting_pair` 列表。

    Example:
        query_groups = prep_query_group(
            means=ci.means,
            custom_dict={"scavenger_rec": ["^ACKR", "^SR-"]},
        )
        # 后续可以将 `query_groups["th17"]` 或自定义 key 继续用于 prepare_gene_query
    """
    if not isinstance(means, pd.DataFrame):
        raise TypeError("Argument `means` must be a pandas DataFrame.")
    if "interacting_pair" not in means.columns:
        raise KeyError("Column `interacting_pair` was not found in `means`.")

    interacting_pairs = means["interacting_pair"].astype(str).tolist()
    chemokines = [x for x in interacting_pairs if re.search(r"^CXC|CCL|CCR|CX3|XCL|XCR", x)]
    th1 = [
        x for x in interacting_pairs
        if re.search(
            r"IL2|IL12|IL18|IL27|IFNG|IL10|TNF$|TNF |LTA|LTB|STAT1|CCR5|CXCR3|IL12RB1|IFNGR1|TBX21|STAT4",
            x,
        )
    ]
    th2 = [x for x in interacting_pairs if re.search(r"IL4|IL5|IL25|IL10|IL13|AREG|STAT6|GATA3|IL4R", x)]
    th17 = [
        x for x in interacting_pairs
        if re.search(
            r"IL21|IL22|IL24|IL26|IL17A|IL17F|IL17RA|IL10|RORC|RORA|STAT3|CCR4|CCR6|IL23RA|TGFB",
            x,
        )
    ]
    treg = [x for x in interacting_pairs if re.search(r"IL35|IL10|FOXP3|IL2RA|TGFB", x)]
    costimulatory = [
        x for x in interacting_pairs
        if re.search(
            r"CD86|CD80|CD48|LILRB2|LILRB4|TNF|CD2|ICAM|SLAM|LT[AB]|NECTIN2|CD40|CD70|CD27|CD28|CD58|TSLP|PVR|CD44|CD55|CD[1-9]",
            x,
        )
    ]
    coinhibitory = [
        x for x in interacting_pairs
        if re.search(r"SIRP|CD47|ICOS|TIGIT|CTLA4|PDCD1|CD274|LAG3|HAVCR|VSIR", x)
    ]

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
        for family_name, patterns in custom_dict.items():
            query_dict[family_name] = [x for x in interacting_pairs if re.search(r"|".join(patterns), x)]
    return query_dict


@logged
def hclust(
    data: pd.DataFrame,
    axis: int = 0,
    method: str = "average",
    metric: str = "euclidean",
    optimal_ordering: bool = True,
) -> list:
    """对 DataFrame 的行或列执行层次聚类。

    Args:
        data: 需要聚类的数据表。
        axis: `0` 表示聚类行，`1` 表示聚类列。
        method: linkage 方法。
        metric: 距离度量方式。
        optimal_ordering: 是否启用 optimal ordering。

    Returns:
        聚类后的标签顺序列表。

    Example:
        row_order = hclust(
            data=means_matrix,
            axis=0,
            method="average",
            metric="euclidean",
        )
        # 返回值可直接用于对 means / pvals / scores 矩阵同步重排
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Argument `data` must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("Argument `data` must not be empty.")
    if axis not in (0, 1):
        raise ValueError("Argument `axis` must be either 0 or 1.")

    matrix = data.T if axis == 1 else data
    if matrix.shape[0] <= 1:
        logger.warning("[hclust] Warning! Too few rows are available for clustering. Return the original order.")
        return list(matrix.index)

    if matrix.isnull().values.any():
        logger.warning("[hclust] Warning! NaN values were detected and will be filled by column mean before clustering.")
        matrix = matrix.fillna(matrix.mean(numeric_only=True))

    distances = pdist(matrix, metric=metric)
    linkage = shc.linkage(distances, method=method, optimal_ordering=optimal_ordering)
    dendro = shc.dendrogram(linkage, no_plot=True, labels=matrix.index)
    return dendro["ivl"]


def split_kwargs(*funcs: Callable[..., Any], strict: bool = False, **kwargs) -> dict:
    """按函数签名拆分 kwargs。

    Args:
        *funcs: 需要分析参数签名的函数列表。
        strict: 若为 True，则对无法匹配的参数报错。
        **kwargs: 待拆分的关键字参数。

    Returns:
        字典，格式为 `{function_name: {matched_kwargs}}`。

    Example:
        split = split_kwargs(
            CellphoneInspector.prepare_cpdb_tables,
            CellphoneInspector.prepare_cell_metadata,
            add_meta=True,
            celltype_key="Subset_Identity",
        )
        # 适合在批量封装多个步骤时，把一组 kwargs 自动分发到不同函数
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
        raise ValueError(f"Unrecognized arguments were found: {sorted(unused)}.")
    return results


def contained_in(lst, sub):
    """判断 `sub` 是否作为连续子列表出现在 `lst` 中。"""
    n = len(sub)
    return any(sub == lst[i:i + n] for i in range(len(lst) - n + 1))
