import pandas as pd
import anndata
import numpy as np
import sklearn
import anndata
import os
import gc
import scanpy as sc
import sys
import os, re
from typing import List, Dict, Optional

# 迁移完成（仅data_split）

from ktplotspy.utils.support import (
    ensure_categorical, # 强制categorical化pd.Dataframe
    filter_interaction_and_celltype,
    hclust,
    prep_celltype_query,
    prep_query_group,
    prep_table, # 用于格式化均值和 pvalues 表的通用函数
    set_x_stroke,
    sub_pattern,
)

from plotnine import *

# 鉴于 https://github.com/zktuong/ktplotspy/blob/master/ktplotspy/utils/settings.py 硬编码了CPDB数据处理的必要信息
# 我们直接把它们拷贝过来比较好debug
DEFAULT_SEP = ">@<"
DEFAULT_SPEC_PAT = "/|:|\\?|\\*|\\+|\\(|\\)|\\/|\\[|\\]\\-"
DEFAULT_CELLSIGN_ALPHA = 0.5
DEFAULT_COLUMNS = ["interaction_group", "celltype_group"]
DEFAULT_V5_COL_START = 13
DEFAULT_COL_START = 11
DEFAULT_CLASS_COL = 12
DEFAULT_CPDB_SEP = "|"

def filter_by_frequency(data, column, min_count):
    """
    根据特定列中值的频率过滤 DataFrame 或 AnnData 对象中的行。

    Parameters:
    ----------
    data : pd.DataFrame or AnnData
        The data object to filter. For AnnData, filtering is applied on `data.obs`.
    column : str
        The name of the column to calculate value frequencies.
    min_count : int
        The minimum frequency required to keep a value.

    Returns:
    -------
    filtered_data : same type as input (pd.DataFrame or AnnData)
        The filtered data object containing only rows with values meeting the frequency threshold.
    """
    # Determine the value counts
    value_counts = data.obs[column].value_counts() if hasattr(data, "obs") else data[column].value_counts()
    
    # Get values meeting the frequency threshold
    keep_values = value_counts[value_counts > min_count].index
    
    # Filter the data based on the threshold
    if hasattr(data, "obs"):  # If the input is AnnData
        return data[data.obs[column].isin(keep_values)]
    else:  # If the input is a DataFrame
        return data[data[column].isin(keep_values)]


def data_split(adata, disease, data_path, downsample_by_key,min_count=30,
               downsample=True, max_cells=2000,random_state=0,
               use_raw=False):
    '''
    用来按照疾病拆分adata文件、去除低频亚群，并生成cellphonedb所需文件

    :param adata: 输入 AnnData 对象
    :param disease: 要筛选的疾病组名
    :param data_path: 输出文件夹的根路径
    :param min_count: 低于此数量的 Subset_Identity 会被移除
    :param use_raw: 是否使用 adata.raw.X 来保存表达矩阵
    :return: 输出 counts.h5ad 和 metadata.tsv，并保存 adata_subset
    '''
    print(f"\n➡️ 当前处理疾病组: {disease}")
    
    np.random.seed(random_state)
    
    # 1. 根据疾病名称筛选数据
    adata_subset = adata[adata.obs["disease"] == disease].copy()
    print(f"原始细胞数: {adata_subset.shape[0]}")
    
    # 2. 检查 Subset_Identity 列是否存在
    if 'Subset_Identity' not in adata_subset.obs.columns:
        raise KeyError("'Subset_Identity' column not found in adata.obs")
    
    # 3. 根据频率过滤亚群
    subset_counts = adata_subset.obs["Subset_Identity"].value_counts()
    valid_subsets = subset_counts[subset_counts >= min_count].index
    adata_subset = adata_subset[adata_subset.obs["Subset_Identity"].isin(valid_subsets)].copy()
    print(f"过滤后细胞数: {adata_subset.shape[0]}")
    
    if downsample:
        selected_indices = []
        for group, idx in adata_subset.obs.groupby(downsample_by_key).indices.items():
            if len(idx) > max_cells:
                sampled = np.random.choice(idx, max_cells, replace=False)
            else:
                sampled = idx
            selected_indices.extend(sampled)
        
        adata_subset = adata_subset[selected_indices].copy()
        print(f"下采样后细胞数: {adata_subset.shape[0]}")
    
    # 🔍 打印保留下来的 cluster
    print("✅ 保留的 Subset_Identity:")
    print(adata_subset.obs["Subset_Identity"].value_counts())
    
    # 4. 准备输出路径
    disease_dir = os.path.join(data_path, disease)
    os.makedirs(disease_dir, exist_ok=True)
    
    # 5. 保存 metadata.tsv
    meta_file = pd.DataFrame({
        'Cell': adata_subset.obs.index,
        'cell_type': adata_subset.obs["Subset_Identity"]
    })
    meta_file_path = os.path.join(disease_dir, "metadata.tsv")
    meta_file.to_csv(meta_file_path, index=False, sep="\t")
    
    # 6. 选择表达矩阵：raw or not
    if use_raw:
        if adata.raw is None:
            raise ValueError("adata.raw is None, cannot use raw matrix.")
        X = adata.raw[adata_subset.obs_names].X
        var = adata.raw.var.copy()
    else:
        X = adata_subset.X
        var = adata_subset.var.copy()
    
    # 添加基因名字段，CellPhoneDB 可能需要
    var["gene_name"] = var.index
    
    # 7. 创建新 AnnData 并保存f
    adata_out = sc.AnnData(
        X=X,
        obs=adata_subset.obs.copy(),
        var=var
    )
    count_file_path = os.path.join(disease_dir, "counts.h5ad")
    adata_out.write(count_file_path)
    
    print(f"📁 文件保存至: {count_file_path} 与 {meta_file_path}")


#
def find_file(file_dir, pattern):
    """用来简单快速地用文档开头来匹配文档"""
    files = os.listdir(file_dir)
    for file in files:
        if re.search(pattern, file):
            return os.path.join(file_dir, file)
    raise FileNotFoundError(f"No file matching pattern '{pattern}' found in {file_dir}")


def extract_cpdb_result(file_dir):
    '''
    从输出目录下读取CPDB结果，实际上也就是恢复cpdb_analysis_method直接返回的对象 （检查可用）
    
    使用例：result = extract_cpdb_result('/path/to/cpdb/results')
    :param file_dir: 输出文件
    :return: 返回包含输出内容的字典
    '''
    
    # Define the patterns for each type of result
    file_patterns = {
        'deconvoluted_percents': "analysis_deconvoluted_percents",
        'deconvoluted': "analysis_deconvoluted",
        'means': "analysis_means",
        'pvalues': "analysis_pvalues|analysis_relevant_interactions",
        'interaction_scores': "analysis_interaction_scores",
        'significant_means': "analysis_significant_means",
    }
    
    # Load the files into a dictionary
    cpdb_results = {}
    for key, pattern in file_patterns.items():
        file_path = find_file(file_dir, pattern)
        cpdb_results[key] = pd.read_table(file_path, delimiter='\t')
    
    return cpdb_results


def retrieve_name(var):
    '''
    获取变量的名称
    更新：检查输入变量类型是否是哈希不可变类型（如整数、字符串），避免常量池的问题。
    :param var: 变量名
    :return: "var"
    '''
    var_id = id(var)
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if id(var_val) == var_id]




## 以下都是extract_cpdb_table的辅助函数，大部分会在将来内部化或放入CPDB_util.py中
def align_pvals_to_means(means_mat, pvals_mat, degs_analysis, col_start = DEFAULT_V5_COL_START):
    """
    准备p-values matrix，以便于后面和means matrix对齐。
    """
    if pvals_mat.shape != means_mat.shape:
        tmp_pvals = pd.DataFrame(index=means_mat.index, columns=means_mat.columns)
        tmp_pvals.iloc[:, :col_start] = means_mat.iloc[:, :col_start]
        tmp_pvals.update(pvals_mat)
        tmp_pvals.fillna(0 if degs_analysis else 1, inplace=True)
        return tmp_pvals
    return pvals_mat
    
def prepare_input(cpdb_outcome_dict, degs_analysis, col_start = DEFAULT_V5_COL_START,cellsign=None):
    means = prep_table(data=cpdb_outcome_dict["means"])
    pvals = prep_table(data=cpdb_outcome_dict["pvalues"])
    interaction_scores = cpdb_outcome_dict["interaction_scores"]
    if interaction_scores is not None:
        interaction_scores_mat = prep_table(data=interaction_scores)
        cellsign_mat = []
    else:
        interaction_scores_mat = []
        if cellsign is not None:
            cellsign_mat = prep_table(data=cellsign)
        else:
            cellsign_mat = []
    
    if pvals.shape != means.shape:
        # 这一步大部分情况是符合的，反而不太清楚不符合的情况，因此调用这个函数的时候我们主动报告一下
        print("Warning: P-val and Means matrices are not in the same shape.")
        pvals = align_pvals_to_means(means_mat=means, pvals_mat=pvals, col_start=col_start,
                                     degs_analysis=degs_analysis)
    
    return means, pvals, interaction_scores, interaction_scores_mat, cellsign_mat
    
def prepare_metadata(adata_obs, celltype_key, splitby_key):
    """
    处理adata.obs中的metadata，确保celltype_key和splitby_key列为分类变量（categorical）。
    并根据splitby_key是否存在，生成组合标签'_labels'用于后续分析。

    参数：
    - adata_obs: pandas.DataFrame，adata.obs的数据副本
    - celltype_key: str，细胞类型所在的列名
    - splitby_key: str或None，用于拆分的列名，如果为None，则只用celltype_key

    返回：
    - metadata: 处理后带有'_labels'列的DataFrame
    """

    # 复制adata.obs，避免修改原始数据
    metadata = adata_obs.copy()

    # 确保celltype_key对应的列为分类变量
    metadata = ensure_categorical(metadata, celltype_key)

    # 如果指定了拆分列splitby_key
    if splitby_key:
        # 确保splitby_key对应的列为分类变量
        metadata = ensure_categorical(metadata, splitby_key)
        
        # 生成新列'_labels'，值为splitby_key与celltype_key的字符串拼接，中间用下划线连接
        metadata["_labels"] = metadata[splitby_key] + "_" + metadata[celltype_key]
        
        # 将'_labels'列转为分类变量
        metadata["_labels"] = metadata["_labels"].astype("category")
        
        # 重新定义分类顺序，顺序为所有splitby_key分类 * 所有celltype_key分类的组合
        # 仅保留实际存在于'_labels'中的组合
        cat_orders = [
            f"{s}_{c}"
            for s in metadata[splitby_key].cat.categories
            for c in metadata[celltype_key].cat.categories
            if f"{s}_{c}" in metadata._labels.values
        ]
        
        # 按照cat_orders顺序重新排序'_labels'分类
        metadata["_labels"] = metadata["_labels"].cat.reorder_categories(cat_orders)
    
    else:
        # 如果没有splitby_key，则直接将celltype_key列赋值给'_labels'
        metadata["_labels"] = metadata[celltype_key]
    
    return metadata

def validate_inputs(interaction_scores, cellsign, genes, gene_family):
    if interaction_scores is not None and cellsign is not None:
        raise KeyError("Please specify either interaction scores or cellsign, not both.")
    if genes is not None and gene_family is not None:
        raise KeyError("Please specify either genes or gene_family, not both.")

def get_gene_query(means_mat: pd.DataFrame, genes: Optional[List[str]] = None, gene_family: Optional[str] = None,
                   custom_gene_family: Optional[Dict[str, List[str]]] = None, debug: Optional[bool]=False) -> List[str]:
    """
    Get a list of genes based on either 'genes' or 'gene_family'.

    Parameters
    ----------
    means_mat : pd.DataFrame
        DataFrame containing at least the 'interacting_pair' column.
    genes : Optional[List[str]], optional
        List of genes to query.
    gene_family : Optional[str], optional
        Predefined gene family to query.
    custom_gene_family : Optional[Dict[str, List[str]]], optional
        Custom gene family definitions.

    Returns
    -------
    List[str]
        List of genes matching the query.
    """
    print("Starting generate ligand-receptor pair query.")
    
    def validate_gene_family(gene_family: str, query_group: Dict) -> List[str]:
        """Validate and retrieve genes for a given gene family."""
        if gene_family.lower() in query_group:
            return query_group[gene_family.lower()]
        raise KeyError(f"gene_family must be one of the following: {list(query_group.keys())}")
    
    if genes is None:
        if gene_family is not None:
            query_group = prep_query_group(means_mat, custom_gene_family)
            if isinstance(gene_family, list):
                query = []
                for gf in gene_family:
                    query += validate_gene_family(gf, query_group)
                query = list(set(query))  # Remove duplicates
            else:
                query = validate_gene_family(gene_family, query_group)
        else:
            # Default: Return all genes
            query = [i for i in means_mat.interacting_pair if re.search("", i)]
    elif genes is not None:
        if gene_family is not None:
            raise KeyError("Please specify either 'genes' or 'gene_family', not both.")
        else:
            # Match genes in the interacting_pair column
            query = [i for i in means_mat.interacting_pair if re.search("|".join(genes), i)]
    print(f"Found {len(query)} different gene queires.")
    if debug:
        print("#" * 16)
        print("Head 10s of all gene queries:")
        print(query[0:10])
        print("#" * 16)
    return query

def generate_celltype_patterns(c_type1: List[str], c_type2: List[str], lock_celltype_direction: bool,
                               splitby_key: Optional[str], groups: List[str],debug) -> List[str]:
    """
    生成细胞类型的正则表达式模式，根据锁定方向和分组条件筛选组合。

    Parameters
    ----------
    c_type1 : List[str]
        第一组细胞类型。
    c_type2 : List[str]
        第二组细胞类型。
    lock_celltype_direction : bool
        是否锁定细胞类型的方向性。
    splitby_key : Optional[str]
        分组的键值，若为 None 则不进行分组筛选。
    groups : List[str]
        分组的列表；只有当splitby_key给定参数时才会产生
    DEFAULT_SEP : str
        细胞类型之间的分隔符。

    Returns
    -------
    List[str]
        生成的正则表达式模式列表。
    """
    celltype_patterns = []
    
    # 生成基础组合模式
    for cell1 in c_type1:
        cq = [
            f"^{cell1}{DEFAULT_SEP}{cell2}$" for cell2 in c_type2
        ]
        if not lock_celltype_direction:  # 双向组合
            cq.extend(
                f"^{cell2}{DEFAULT_SEP}{cell1}$" for cell2 in c_type2
            )
        combined_patterns = "|".join(cq)
        
        if splitby_key is not None:  # 按分组过滤
            for group in groups:
                filtered_patterns = [
                    pattern for pattern in cq
                    if re.search(f"{group}.*{DEFAULT_SEP}{group}", pattern)
                ]
                if filtered_patterns:
                    celltype_patterns.append("|".join(filtered_patterns))
        else:  # 无分组时直接添加
            celltype_patterns.append(combined_patterns)
    
    if debug:
        print("#" * 16)
        print("Celltype_1:")
        print(*c_type1,sep=",")
        print("#" * 16)
        print("Celltype_2:")
        print(*c_type2,sep=",")
        print("#" * 16)
    return celltype_patterns

def get_cell_query(metadata, means_mat, cell_type1, cell_type2, lock_celltype_direction, splitby_key,
                   special_character_regex_pattern,debug):
    print("Starting generate cell pair query.")
    All_labels = list(metadata._labels.cat.categories)
    
    # 将cell_type可能包含的特殊字符转义
    if special_character_regex_pattern is None:
        special_character_regex_pattern = DEFAULT_SPEC_PAT
    
    All_labels = [sub_pattern(cell_type=labels, pattern=special_character_regex_pattern) for labels in All_labels]
    cell_type1 = sub_pattern(cell_type=cell_type1, pattern=special_character_regex_pattern)
    cell_type2 = sub_pattern(cell_type=cell_type2, pattern=special_character_regex_pattern)
    c_type1 = cell_type1 if cell_type1 != "." else All_labels
    c_type2 = cell_type2 if cell_type2 != "." else All_labels
    groups = list(metadata[splitby_key].cat.categories) if splitby_key else None
    
    # 生成所有符合条件的细胞对
    All_valid_cellpair = generate_celltype_patterns(c_type1, c_type2, lock_celltype_direction, splitby_key,
                                                    groups=groups,debug=debug)
    if debug:
        print("#" * 16)
        print("Head of All_valid_cellpair:")
        print(All_valid_cellpair[0:10])
        print("#" * 16)
    # All_valid_cellpair = "|".join(All_valid_cellpair)
    compiled_patterns = [re.compile(pattern) for pattern in All_valid_cellpair]
    print(f"There are {len(All_valid_cellpair)} possible celltype interactions.")
    
    # 从结果中抽出符合条件的，即：取交集
    # ct_columns = [ct for ct in means_mat.columns if re.search(All_valid_cellpair, ct)]
    ct_columns = [
        ct for ct in means_mat.columns
        if any(pattern.search(ct) for pattern in compiled_patterns)
    ]
    print(f"Found {len(ct_columns)} different celltype queires.")
    return ct_columns, groups

def get_col_order(columns, groups=None):
    """
    获取列的顺序，根据 groups 进行筛选。
    Parameters
    ----------
    columns : pd.Index
        数据矩阵的列名。
    groups : List[str], optional
        分组名称列表，用于正则匹配列名。
    Returns
    -------
    List[str]
        筛选后的列顺序。
    """
    if groups:
        return [c for g in groups for c in columns if re.search(g, c)]
    return list(columns)

def filter_data(means_mat, pvals_mat, interaction_scores,interaction_scores_mat, cellsign,
                gene_query, cell_query,
                keep_significant_only,groups,alpha,debug):
    '''核心是包装的filter_interaction_and_celltype函数'''
    if debug:
        print("#" * 16)
        print("Gene queries:")
        print(gene_query)
        print("#" * 16)
        print("Cell queries:")
        print(cell_query)
        print("#" * 16)
    # 过滤主要矩阵
    means_matx = filter_interaction_and_celltype(data=means_mat, genes=gene_query, celltype_pairs=cell_query)
    pvals_matx = filter_interaction_and_celltype(data=pvals_mat, genes=gene_query, celltype_pairs=cell_query)
    # 过滤其他矩阵
    if interaction_scores is not None:
        interaction_scores_matx = filter_interaction_and_celltype(data=interaction_scores_mat, genes=gene_query,
                                                                  celltype_pairs=cell_query)
    elif cellsign is not None:
        cellsign_matx = filter_interaction_and_celltype(data=cellsign_mat, genes=gene_query, celltype_pairs=cell_query)
    
    # 调整列的顺序，以对齐矩阵
    col_order = get_col_order(means_matx.columns, groups)
    if debug:
        print("#" * 16)
        print(col_order)
        print("#" * 16)
    # 应用列顺序到矩阵
    means_matx = means_matx[col_order]
    pvals_matx = pvals_matx[col_order]
    # 更新其他矩阵
    if interaction_scores is not None:
        interaction_scores_matx = interaction_scores_matx[col_order]
        cellsign_matx = []
    elif cellsign is not None:
        interaction_scores_matx = []
        cellsign_matx = cellsign_matx[col_order]
    else:
        interaction_scores_matx = []
        cellsign_matx = []
    if keep_significant_only:
        # 筛选出任意列中 p 值小于 alpha 的行
        keep_rows = pvals_matx.index[pvals_matx.lt(alpha).any(axis=1)]
        
        if keep_rows.size > 0:
            print(f"After significance filtering, there are {keep_rows.size} lines remain.")
            # 更新主要矩阵
            pvals_matx = pvals_matx.loc[keep_rows]
            means_matx = means_matx.loc[keep_rows]
            if debug:
                print("#" * 16)
                print("Kept rows are:")
                print(keep_rows)
                print("#" * 16)
            # 更新interaction_scores和cellsign
            if interaction_scores is not None:
                interaction_scores_matx = interaction_scores_matx.loc[keep_rows]
            
            if cellsign is not None:
                cellsign_rows = keep_rows.intersection(cellsign_matx.index)
                if cellsign_rows.size > 0:
                    cellsign_matx = cellsign_matx.loc[cellsign_rows]
                else:
                    raise ValueError("Your cellsign data may not contain significant hits.")
        else:
            raise ValueError("No significant rows found in the data.")
    return means_matx, pvals_matx, interaction_scores_matx, cellsign_matx

def cluster_rows_by_means(means_matx, pvals_matx, interaction_scores, interaction_scores_matx, cellsign,
                          cellsign_matx):
    if means_matx.shape[0] > 2:
        # 行聚类获取顺序
        h_order = hclust(means_matx, axis=0)
        print("Hclust algorithm processed successfully.")
        # Index = 0 and columns = 1，对行进行层次聚类
        # 对主要矩阵重新排序
        means_matx = means_matx.loc[h_order]
        pvals_matx = pvals_matx.loc[h_order]
        # 对 interaction_scores 和 cellsign 数据处理
        if interaction_scores is not None:
            interaction_scores_matx = interaction_scores_matx.loc[h_order]
        elif cellsign is not None:
            # 仅保留 cellsign_matx 中存在的行
            valid_h_order = [h for h in h_order if h in cellsign_matx.index]
            if valid_h_order:
                cellsign_matx = cellsign_matx.loc[valid_h_order]
            else:
                raise ValueError(
                    "No significant hits found in cellsign data after clustering. "
                    "Ensure cellsign_matx contains matching rows."
                )
    return means_matx, pvals_matx, interaction_scores_matx, cellsign_matx

def standardize_means(means_matx):
    # 矢量化操作进行行标准化
    row_min = means_matx.min(axis=1)  # 按行求最小值
    row_max = means_matx.max(axis=1)  # 按行求最大值
    range_diff = row_max - row_min  # 计算范围
    
    # 避免分母为 0 的行
    range_diff[range_diff == 0] = 1
    
    means_matx = (means_matx.sub(row_min, axis=0)).div(range_diff, axis=0)
    return means_matx

def finalize_output(means_matx, pvals_matx, interaction_scores, interaction_scores_matx, cellsign, cellsign_matx,
                    col_start,standard_scale, additional_grouping, exclude_interactions, keep_significant_only,alpha,
                    keep_id_cp_interaction=False):
    # 确定列名
    colm = "scaled_means" if standard_scale else "means"
    
    # 转换为长格式并设置索引
    df = means_matx.melt(ignore_index=False).reset_index()
    df.index = df["index"] + DEFAULT_SEP * 3 + df["variable"]
    df.columns = DEFAULT_COLUMNS + [colm]
    
    df_pvals = pvals_matx.melt(ignore_index=False).reset_index()
    df_pvals.index = df_pvals["index"] + DEFAULT_SEP * 3 + df_pvals["variable"]
    df_pvals.columns = DEFAULT_COLUMNS + ["pvals"]
    
    # 合并 pvals
    df["pvals"] = df_pvals["pvals"]
    
    # 合并 interaction_scores 或 cellsign
    if interaction_scores is not None:
        df_interaction_scores = interaction_scores_matx.melt(ignore_index=False).reset_index()
        df_interaction_scores.index = df_interaction_scores["index"] + DEFAULT_SEP * 3 + df_interaction_scores[
            "variable"]
        df_interaction_scores.columns = DEFAULT_COLUMNS + ["interaction_scores"]
        df["interaction_scores"] = df_interaction_scores["interaction_scores"]
    elif cellsign is not None:
        df_cellsign = cellsign_matx.melt(ignore_index=False).reset_index()
        df_cellsign.index = df_cellsign["index"] + DEFAULT_SEP * 3 + df_cellsign["variable"]
        df_cellsign.columns = DEFAULT_COLUMNS + ["cellsign"]  # same as above.
        df["cellsign"] = df_cellsign["cellsign"]
    
    # 分类变量转换
    df["celltype_group"] = df["celltype_group"].str.replace(DEFAULT_SEP, "-").astype("category")
    df[["cell_left", "cell_right"]] = df["celltype_group"].str.split("-", expand=True)
    
    # 标准化与显著性过滤
    df[colm] = df[colm].replace(0, np.nan)  # 替换为 NaN
    df["x_means"] = df[colm]
    df["y_means"] = df[colm]
    
    # 矢量化操作处理显著性
    df.loc[df["pvals"] < alpha, "x_means"] = np.nan
    df.loc[df["pvals"] == 0, "pvals"] = 0.001
    if keep_significant_only:
        # 仅保留显著性较高的行
        df = df[df["pvals"] < alpha]
    
    
    if interaction_scores is not None:
        df.loc[df["interaction_scores"] < 1, "x_means"] = np.nan
    elif cellsign is not None:
        df.loc[df["cellsign"] < 1, "cellsign"] = DEFAULT_CELLSIGN_ALPHA
    
    # 额外元数据添加
    if col_start == DEFAULT_V5_COL_START and additional_grouping:
        df["is_integrin"] = df.index.map(is_int)
        df["directionality"] = df.index.map(direc)
        df["classification"] = df.index.map(classif)
    
    # 显著性标记等
    df["neglog10p"] = -np.log10(df["pvals"].clip(lower=alpha))
    df["significant"] = np.where(df["pvals"] < alpha, "yes", "no")
        # 原函数返回nan的行为令人迷惑……除了方便检查是否都isnull到底还有什么用
    
    # 排除指定interaction_group
    if exclude_interactions:
        exclude_interactions = exclude_interactions if isinstance(exclude_interactions, list) else [
            exclude_interactions]
        df = df[~df["interaction_group"].isin(exclude_interactions)]
    
    # 重写interaction_group的名字
        # 拆分返回可能是两列或者三列
    if keep_id_cp_interaction:
        df.interaction_group = [re.sub(DEFAULT_SEP * 3, "_", c) for c in df.interaction_group]
            # 间隔符为 id_a-b-c， 不进行拆分
    else:
        df.interaction_group = [c.split(DEFAULT_SEP * 3)[1] for c in df.interaction_group]
        
        
        
        
    # Acetylcholine-byCHAT-CHRM3
        
    
    
    # 输出结果
    if df.empty:
        print("The result is empty in this case.")
    else:
        return df

def extract_cpdb_table(
        metadata,
        cell_type1: str,
        cell_type2: str,
        cpdb_outcome_dict,
        cellsign=None,
        degs_analysis: bool = False,
        splitby_key=None,
        alpha: float = 0.05,
        min_interaction_score = 0.01,
        keep_significant_only: bool = True,
        genes=None, gene_family=None,
        additional_grouping=False,
        custom_gene_family=None,
        standard_scale: bool = True,
        cluster_rows: bool = True,
        special_character_regex_pattern=None, # 指定需要额外转义的特殊字符，一般我们用默认即可
        exclude_interactions=None,
        lock_celltype_direction: bool =True,
        keep_id_cp_interaction: bool =False,
        debug: bool =False
):
    '''
    从读取的cpdb_outcome_dict中读出符合条件的相互作用表格。
    参考的源文件是https://github.com/zktuong/ktplotspy/blob/master/ktplotspy/plot/plot_cpdb.py的plot_cpdb函数
    :param adata: 应当是有完整metadata的adata，以便接下来从中析出信息。
        当给splitby_key赋值时，adata.obs应具有分离的"celltype_key"和"splitby_key"两列。
    :param cell_type1/cell_type2: 搜索的细胞类型1和细胞类型2
    :param cpdb_outcome_dict: extract_cpdb_result的输出。
    :param celltype_key:细胞类型列
    :param cellsign:
    :param degs_analysis:
    :param splitby_key:给出adata.obs里对应的另一列作为细胞的分类变量；
        为了使其正常工作，用于 CellPhoneDB 的输入“meta.txt”的第二列必须采用这种格式：{splitby}_{celltype}。
    :param alpha: p值阈值
    :param keep_significant_only:
    :param genes:
    :param gene_family:
    :param additional_grouping: 返回的表格是否需要后面增加三列：directionality, classification, is_integrin；这一步非常耗时
    :param custom_gene_family:
    :param standard_scale:
    :param cluster_rows:
    :param highlight_size:
    :param special_character_regex_pattern:
    :param exclude_interactions:
    :param lock_celltype_direction:锁定细胞类型1在左侧、细胞类型2在右侧，不进行交换
    :return:
    '''
    
    # Step 1: 数据准备
    means, pvals, interaction_scores, interaction_scores_mat, cellsign_mat = prepare_input(cpdb_outcome_dict, degs_analysis)
    
    col_start = (
        DEFAULT_V5_COL_START if pvals.columns[DEFAULT_CLASS_COL] == "classification" else DEFAULT_COL_START)
    
    
    # Step 2: 验证输入参数
    validate_inputs(interaction_scores, cellsign, genes, gene_family)
    
    # Step 3: 获取查询基因或基因家族
    gene_queries = get_gene_query(means, genes, gene_family, custom_gene_family,debug)
    
    # Step 4: 准备细胞类型查询
    celltype_queries, groups = get_cell_query(metadata=metadata,means_mat=means,
                                              cell_type1=cell_type1,cell_type2=cell_type2,
                                              lock_celltype_direction=lock_celltype_direction,
                                              splitby_key=splitby_key,
                                              special_character_regex_pattern=special_character_regex_pattern,
                                              debug=debug)
    
    # Step 5: 筛选数据
    means_matx, pvals_matx, interaction_scores_matx, cellsign_matx = filter_data(means_mat=means, pvals_mat=pvals,
                                                                                 interaction_scores=interaction_scores,interaction_scores_mat=interaction_scores_mat,cellsign=cellsign,
                                                                                 gene_query=gene_queries, cell_query=celltype_queries,
                                                                                 keep_significant_only=keep_significant_only,
                                                                                 groups=groups, # if splitby_key
                                                                                 alpha=alpha,
                                                                                 debug=debug)
    
    # Step 6: 行聚类
    if cluster_rows:
        means_matx, pvals_matx, interaction_scores_matx, cellsign_matx = cluster_rows_by_means(means_matx, pvals_matx,
                                                                                               interaction_scores, interaction_scores_matx, cellsign, cellsign_matx)
    
    # Step 7: 标准化
    if standard_scale:
        means_matx = standardize_means(means_matx)
    
    # 填充 NaN 值为 0
    means_matx.fillna(0, inplace=True)
    
    # Step 8: 生成最终表格
    result = finalize_output(means_matx=means_matx, pvals_matx=pvals_matx,
                             interaction_scores=interaction_scores, interaction_scores_matx=interaction_scores_matx, cellsign=cellsign, cellsign_matx=cellsign_matx,
                             col_start=col_start,standard_scale=standard_scale,additional_grouping=additional_grouping,
                             exclude_interactions=exclude_interactions,keep_significant_only=keep_significant_only,alpha=alpha,keep_id_cp_interaction=keep_id_cp_interaction)
    result = result[result.interaction_scores >= min_interaction_score]
    return result



## 以上都是extract_cpdb_table
# 目前的提取出的table还有一些可读性问题，我们先在代码里做一些调整


def draw_cpdb_plot(
        dataframe,                # 输入的 Pandas DataFrame，包含要绘制的数据
        cmap_name: str = "viridis",  # 配色方案名称
        max_size: int = 8,        # 点的最大尺寸
        highlight_size=None,      # 如果指定，则用于高亮点的尺寸
        max_highlight_size: int = 3,  # 高亮点的最大尺寸
        interaction_scores=None,  # 可选的交互评分列，用于调整透明度
        default_style: bool = True,  # 是否应用默认样式
        highlight_col: str = "#080000",  # 高亮点的颜色
        title: str = "",          # 图表标题
        cellsign=None,            # 可选的 cellsign 数据列
        figsize=(6.4, 4.8),       # 图表尺寸
        min_interaction_score: int = 0,  # 最小交互评分
        scale_alpha_by_interaction_scores: bool = False,  # 是否根据交互评分调整透明度
        gene_family=None,         # 基因家族信息（用于标题）
        alpha: float = 0.05,      # 显著性水平的阈值
        scale_alpha_by_cellsign: bool = False,  # 是否根据 cellsign 调整透明度
        filter_by_cellsign: bool = False,  # 是否根据 cellsign 过滤数据
        standard_scale: bool = True  # 是否使用标准化均值列
):
    """
    绘制基于输入 DataFrame 的细胞相互作用图表。

    Args:
        dataframe: 包含交互数据的 Pandas DataFrame，直接使用extract_cpdb_table的的输出结果
        cmap_name: 指定颜色映射名称。
        max_size: 设置点的最大尺寸。
        highlight_size: 指定用于高亮点的尺寸。
        max_highlight_size: 高亮点的最大尺寸（默认 3）。
        interaction_scores: 用于控制点透明度的交互评分列。
        default_style: 是否启用默认样式（影响颜色映射和填充）。
        highlight_col: 高亮点的颜色。
        title: 图表的标题。
        cellsign: cellsign 数据列，用于过滤或调整透明度。
        figsize: 图表的宽度和高度。
        min_interaction_score: 筛选的最小交互评分。
        scale_alpha_by_interaction_scores: 是否基于交互评分调整透明度。
        gene_family: 基因家族名称，用于图表标题。
        alpha: 显著性水平的阈值。
        scale_alpha_by_cellsign: 是否基于 cellsign 调整透明度。
        filter_by_cellsign: 是否根据 cellsign 的阈值过滤数据。
        standard_scale: 是否使用标准化的均值列（"scaled_means"）。
        keep_id_cp_interaction: 是否保留完整的交互组标识符。

    Returns:
        一个 ggplot 图对象，用于可视化细胞相互作用。
    """
    
    def configure_ggplot(df, colm, stroke, default_style, alpha_by_interaction = False, alpha_by_cellsign = False):
        aes_params = {
            "x": "celltype_group",
            "y": "interaction_group",
            "size": colm,
            "stroke": stroke
        }
        if alpha_by_interaction:
            aes_params["alpha"] = "interaction_scores"
        elif alpha_by_cellsign:
            aes_params["alpha"] = "cellsign"
        else:
            return None
        
        if default_style:
            aes_params.update({"colour": "significant", "fill": colm})
        else:
            aes_params.update({"colour": colm, "fill": "significant"})
        return ggplot(df, aes(**aes_params))
    
    
    df = dataframe.copy()
    colm = "scaled_means" if standard_scale else "means"

    # set global figure size
    options.figure_size = figsize
    if highlight_size is not None:
        max_highlight_size = highlight_size
        stroke = "x_stroke"
    else:
        stroke = "neglog10p"
    
    # plotting
    print(highlight_size);
    print(stroke)
    if interaction_scores is not None:
        df = df[df.interaction_scores >= min_interaction_score]
    elif cellsign is not None and filter_by_cellsign:
        df = df[df.cellsign >= DEFAULT_CELLSIGN_ALPHA]
    else:
        print("Skip filtering by interaction score or cellsign.")
    
    if interaction_scores and scale_alpha_by_interaction_scores:
        print("Set alpha by interaction_scores.")
        if default_style or all(df["significant"] == "no"):
            default_style = True
            g = configure_ggplot(df, colm, stroke, default_style, alpha_by_interaction=True)
        else:
            highlight_col = "#FFFFFF"  # enforce this
            g = configure_ggplot(df, colm, stroke, default_style, alpha_by_interaction=True)
    elif interaction_scores and not scale_alpha_by_interaction_scores: # 冲突情况
        print("Skip transparency setting: interaction_scores setted but not scale_alpha_by_interaction_scores.")
        g = None
    elif cellsign and scale_alpha_by_cellsign:
        print("Set alpha by cellsign.")
        if default_style or all(df["significant"] == "no"):
            default_style = True
            g = configure_ggplot(df, colm, stroke, default_style, alpha_by_cellsign=True)
        else:
            highlight_col = "#FFFFFF"  # enforce this
            g = configure_ggplot(df, colm, stroke, default_style, alpha_by_cellsign=True)
    elif cellsign and not scale_alpha_by_cellsign:
        print("Skip transparency setting: cellsign setted but not scale_alpha_by_cellsign.")
        g = None
    elif interaction_scores is None and cellsign is None:
        print("Skip transparency setting: both interaction score and cellsign is none.")
        g = None
    
    if g is None:
        if default_style or all(df["significant"] == "no"):
            g = configure_ggplot(df, colm, stroke, default_style) # 不设置alpha
        else:
            highlight_col = "#FFFFFF"  # enforce this
            g = configure_ggplot(df, colm, stroke, default_style) # 不设置alpha，同时也不是defualt style
    
    g = (
            g
            + geom_point(
        na_rm=True,
    )
            + theme_bw()
            + theme(
        axis_text_x=element_text(angle=90, hjust=0, colour="#000000"),
        axis_text_y=element_text(colour="#000000"),
        axis_ticks=element_blank(),
        axis_title_x=element_blank(),
        axis_title_y=element_blank(),
        legend_key=element_rect(alpha=0, width=0, height=0),
        legend_direction="vertical",
        legend_box="horizontal",
    )
            + scale_size_continuous(range=(0, max_size), aesthetics=["size"])
            + scale_size_continuous(range=(0, max_highlight_size), aesthetics=["stroke"])
    )
    if default_style:
        g = (
                g
                + scale_colour_manual(values=highlight_col, na_translate=False)
                + guides(
            fill=guide_colourbar(barwidth=4, label=True, ticks=True, draw_ulim=True, draw_llim=True, order=1),
            size=guide_legend(
                reverse=True,
                order=2,
            ),
            stroke=guide_legend(
                reverse=True,
                order=3,
            ),
        )
                + scale_fill_continuous(cmap_name=cmap_name)
        )
    else:
        g = (
                g
                + scale_fill_manual(values=highlight_col, na_translate=False)
                + guides(
            colour=guide_colourbar(barwidth=4, label=True, ticks=True, draw_ulim=True, draw_llim=True, order=1),
            size=guide_legend(
                reverse=True,
                order=2,
            ),
            stroke=guide_legend(
                reverse=True,
                order=3,
            ),
        )
        )
        df2 = df.copy()
        for i in df2.index:
            if df2.at[i, "pvals"] < alpha:
                df2.at[i, colm] = np.nan
        g = (
                g
                + geom_point(aes(x="celltype_group", y="interaction_group", colour=colm, size=colm), df2,
                             inherit_aes=False, na_rm=True)
                + scale_colour_continuous(cmap_name=cmap_name)
        )
    
    if highlight_size is not None:
        g = g + guides(stroke=None)
    if (interaction_scores is not None) and scale_alpha_by_interaction_scores:
        g = g + scale_alpha_continuous(breaks=(0, 25, 50, 75, 100))
    if (cellsign is not None) and scale_alpha_by_cellsign:
        g = g + scale_alpha_continuous(breaks=(0, 1))
    if title != "":
        g = g + ggtitle(title)
    elif gene_family is not None:
        if isinstance(gene_family, list):
            gene_family = ", ".join(gene_family)
        g = g + ggtitle(gene_family)
    return g



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    







     

  

     

   

      

   

  

   

   

   

   

   

   

   

   

   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   