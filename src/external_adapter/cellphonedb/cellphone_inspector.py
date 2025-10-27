import inspect

import numpy as np
import pandas as pd

# 直接使用 ktplotspy 的常数列表可以减少一些维护的痛苦
from ktplotspy.utils.settings import (
    DEFAULT_V5_COL_START,
    DEFAULT_COL_START,
    DEFAULT_CLASS_COL,
    DEFAULT_SEP,
    DEFAULT_CELLSIGN_ALPHA,
    DEFAULT_COLUMNS,
)
from ktplotspy.utils.support import (
    filter_interaction_and_celltype,
    hclust,
    prep_query_group,
    prep_table,
    set_x_stroke,
)
import os, re
from plotnine import *



# 对接 cellphonedb v5
class CellphoneInspector():

    def __init__(self, cpdb_outfile):
        self.file_path = cpdb_outfile
        self.data = self._load_file(cpdb_outfile) # 格式为字典
        self._data_integrity_check()


    @staticmethod
    def _log(msg):
        print(f"[CellphoneInspector Message] {msg}")

    def _load_file(self, path):
        cpdb_results = {
            'deconvoluted_percents': pd.read_table(
                path + [s for s in os.listdir(path) if bool(re.search("analysis_deconvoluted_percents", s))][0],
                delimiter='\t'),
            'deconvoluted': pd.read_table(
                path + [s for s in os.listdir(path) if bool(re.search("analysis_deconvoluted", s))][0],
                delimiter='\t'),
            'means': pd.read_table(
                path + [s for s in os.listdir(path) if bool(re.search("analysis_means", s))][0],
                delimiter='\t'),
            'pvalues': pd.read_table(path + [s for s in os.listdir(path) if
                                                 bool(re.search("analysis_pvalues|analysis_relevant_interactions", s))][
                0],
                                     delimiter='\t'),
            'interaction_scores': pd.read_table(
                path + [s for s in os.listdir(path) if bool(re.search("analysis_interaction_scores", s))][0],
                delimiter='\t'),
            'significant_means': pd.read_table(
                path + [s for s in os.listdir(path) if bool(re.search("analysis_significant_means", s))][0],
                delimiter='\t')
            # relevant_interactions - this is statistical_analysis_significant_means file in the case of a statistical analysis,
            # and degs_analysis_relevant_interactions file in the case of DEG analysis.
        }
        return cpdb_results

    def _data_integrity_check(self):


    def meta_prepare(self, metadata, celltype_key, groupby_key=None):
        '''

        Example
        -------
        ci.meta_prepare(adata.obs, celltype_key = "Subset_Identity", groupby_key = "disease") # 通常直接输入就好

        :param metadata: pd.DataFrame
        :param celltype_key: str，用于指定 meta 的哪一列储存细胞身份信息
        :param groupby_key: str，如果存在一个独立于细胞身份的分组信息列则填写，如 gender, disease_type, disease_state
        :return:
        '''
        # 确保为 category
        if not metadata[celltype_key].dtype.name == "category":
            metadata[celltype_key] = metadata[celltype_key].astype("category")
        if groupby_key is not None:
            # 确保为 category
            if not metadata[groupby_key].dtype.name == "category":
                metadata[groupby_key] = metadata[groupby_key].astype("category")

             # 构造“_labels”列
            metadata["_labels"] = [split + "_" + celltype for split, celltype in zip(metadata[groupby_key],
                                                                                     metadata[celltype_key])]
            metadata["_labels"] = metadata["_labels"].astype("category")
        else:
            metadata["_labels"] = metadata[celltype_key]

    def prepare_gene_query(self):
        means_mat = prep_table(data=self.data["means"])
        pvals_mat = prep_table(data=self.data["pvalues"])



def extract_cpdb_table(
        adata: "AnnData",
        cell_type1: str,
        cell_type2: str,
        cpdb_outcome_dict,
        celltype_key: str,
        cellsign=None,
        degs_analysis: bool = False,
        splitby_key=None,
        alpha: float = 0.05,
        keep_significant_only: bool = True,
        genes=None, gene_family=None,
        additional_grouping=False,
        custom_gene_family=None,
        standard_scale: bool = True,
        cluster_rows: bool = True,
        highlight_size=None,
        special_character_regex_pattern=None,
        exclude_interactions=None,
        lock_celltype_direction=True,
        debug=False
):
    '''
    从读取的cpdb_outcome_dict中读出符合条件的相互作用表格。

    :param adata:
    :param cell_type1:
    :param cell_type2:
    :param cpdb_outcome_dict:
    :param celltype_key:
    :param cellsign:
    :param degs_analysis:
    :param splitby_key:
    :param alpha: p值阈值
    :param keep_significant_only:
    :param genes:
    :param gene_family:
    :param additional_grouping: 返回的表格是否需要后面增加三列：directionality, classification, is_integrin
    :param custom_gene_family:
    :param standard_scale:
    :param cluster_rows:
    :param highlight_size:
    :param special_character_regex_pattern:
    :param exclude_interactions:
    :param lock_celltype_direction:
    :param debug:
    :return:
    '''
    means = cpdb_outcome_dict["means"];
    pvals = cpdb_outcome_dict["pvalues"];
    interaction_scores = cpdb_outcome_dict["interaction_scores"]
    # prepare data
    metadata = adata.obs.copy()
    means_mat = prep_table(data=means);
    pvals_mat = prep_table(data=pvals)
    col_start = (
        DEFAULT_V5_COL_START if pvals_mat.columns[DEFAULT_CLASS_COL] == "classification" else DEFAULT_COL_START)
    # 魔法数字，把行列数硬编码进去；在 cellphonedb v5 版本, 具体的数值前有12列 meta 信息列；因此 col_start 是 13

    if pvals_mat.shape != means_mat.shape:
        tmp_pvals_mat = pd.DataFrame(index=means_mat.index, columns=means_mat.columns)
        # Copy the values from means_mat to new_df
        tmp_pvals_mat.iloc[:, :col_start] = means_mat.iloc[:, :col_start]
        tmp_pvals_mat.update(pvals_mat)
        if degs_analysis:
            tmp_pvals_mat.fillna(0, inplace=True)
        else:
            tmp_pvals_mat.fillna(1, inplace=True)
        pvals_mat = tmp_pvals_mat.copy()

    if (interaction_scores is not None) & (cellsign is not None):
        raise KeyError("Please specify either interaction scores or cellsign, not both.")

    if interaction_scores is not None:
        interaction_scores_mat = prep_table(data=interaction_scores)
    elif cellsign is not None:
        cellsign_mat = prep_table(data=cellsign)

    if degs_analysis:
        pvals_mat.iloc[:, col_start: pvals_mat.shape[1]] = 1 - pvals_mat.iloc[:, col_start: pvals_mat.shape[1]]
    # front load the dictionary construction here
    if col_start == DEFAULT_V5_COL_START:  # very time-consuming!
        if additional_grouping:
            # 宽转长：前 col_start 列保留为标识列（id_vars），其余列被转换为行
            tmp = means_mat.melt(id_vars=means_mat.columns[:col_start])
            direc, classif, is_int = {}, {}, {}
            for _, r in tmp.iterrows():
                key = r.id_cp_interaction + DEFAULT_SEP * 3 + r.interacting_pair.replace("_",
                                                                                         "-") + DEFAULT_SEP * 3 + r.variable
                direc[key] = r.directionality
                classif[key] = r.classification
                is_int[key] = r.is_integrin
    # 转义特殊字符以避免regex出错
    if special_character_regex_pattern is None:
        special_character_regex_pattern = DEFAULT_SPEC_PAT

    cell_type1 = sub_pattern(cell_type=cell_type1, pattern=special_character_regex_pattern)
    cell_type2 = sub_pattern(cell_type=cell_type2, pattern=special_character_regex_pattern)

    # 生成基因对query
    if genes is None:
        if gene_family is not None:
            # 这将会返回一组query，包含内置的chemokines, th1, th2, th17, treg, costimulatory, coinhibitory基因
            query_group = prep_query_group(means_mat, custom_gene_family)
            if isinstance(gene_family, list):  # 是列表，逐一查找并合并相关基因。
                query = []
                for gf in gene_family:
                    if gf.lower() in query_group:
                        for gfg in query_group[gf.lower()]:
                            query.append(gfg)
                    else:
                        raise KeyError("gene_family needs to be one of the following: {}".format(query_group.keys()))
                query = list(set(query))
            else:  # 是单个字符串，直接提取内置的对应的基因组
                if gene_family.lower() in query_group:
                    query = query_group[gene_family.lower()]
                else:
                    raise KeyError("gene_family needs to be one of the following: {}".format(query_group.keys()))
        else:  # 构造默认查询，提取 means_mat.interacting_pair 中所有基因（通过空正则表达式匹配）*
            query = [i for i in means_mat.interacting_pair if re.search(pattern="", string=i)]
    elif genes is not None:
        if gene_family is not None:
            raise KeyError("Please specify either genes or gene_family, not both.")
        else:  # 筛选 means_mat.interacting_pair 中匹配 genes 列表的基因
            query = [i for i in means_mat.interacting_pair if re.search("|".join(genes), i)]

    metadata = ensure_categorical(meta=metadata, key=celltype_key)
    # prepare regex query for celltypes
    if splitby_key is not None:
        metadata = ensure_categorical(meta=metadata, key=splitby_key)
        groups = list(metadata[splitby_key].cat.categories)
        # 构造“_labels”列
        metadata["_labels"] = [split + "_" + celltype for split, celltype in
                               zip(metadata[splitby_key], metadata[celltype_key])]
        metadata["_labels"] = metadata["_labels"].astype("category")
        # 生成所有可能的分类组合
        all_combinations = {f"{s}_{c}" for s in split_categories for c in celltype_categories}
        # 过滤实际存在的组合
        valid_combinations = [label for label in all_combinations if label in metadata["_labels"].tolist()]
        # 重新设置分类顺序
        metadata["_labels"] = metadata["_labels"].cat.reorder_categories(valid_combinations)
    else:
        metadata["_labels"] = metadata[celltype_key]

    # ------------- 单独处理一下prep_celltype_query
    # 全部的细胞类别
    labels = list(metadata._labels.cat.categories)
    # print(labels)
    c_type1 = cell_type1 if cell_type1 != "." else labels
    c_type2 = cell_type2 if cell_type2 != "." else labels
    celltype = []
    # 生成全部的细胞-细胞对组合
    for i in range(0, len(c_type1)):
        cq = []
        for cx2 in c_type2:
            cq.append("^" + c_type1[i] + DEFAULT_SEP + cx2 + "$")  # 无论是否锁定，我们都有c_type1在左边，ctype_2在右边的情况
            if not lock_celltype_direction:  # 当锁定的时候，不出现交换
                cq.append("^" + cx2 + DEFAULT_SEP + c_type1[i] + "$")
        cq = "|".join(cq)
        if splitby_key is not None:
            for g in groups:
                ctx = cq.split("|")
                ctx = [x for x in ctx if re.search(g + ".*" + DEFAULT_SEP + g, x)]
                cqi = "|".join(ctx)
                if cqi != "":
                    celltype.append(cqi)
        else:
            celltype.append(cq)

    # if debug:
    #     print("Cell-cell relation including:")
    #     print(celltype)
    cell_type = "|".join(celltype)

    # keep cell types
    ct_columns = [ct for ct in means_mat.columns if re.search(cell_type, ct)]

    # filter
    means_matx = filter_interaction_and_celltype(data=means_mat, genes=query, celltype_pairs=ct_columns)
    pvals_matx = filter_interaction_and_celltype(data=pvals_mat, genes=query, celltype_pairs=ct_columns)

    if interaction_scores is not None:
        interaction_scores_matx = filter_interaction_and_celltype(data=interaction_scores_mat, genes=query,
                                                                  celltype_pairs=ct_columns)
    elif cellsign is not None:
        cellsign_matx = filter_interaction_and_celltype(data=cellsign_mat, genes=query, celltype_pairs=ct_columns)

    # reorder the columns
    col_order = []
    if splitby_key:
        for g in groups:
            for c in means_matx.columns:
                if re.search(g, c):
                    col_order.append(c)
    else:
        col_order = means_matx.columns

    means_matx = means_matx[col_order]
    pvals_matx = pvals_matx[col_order]
    if interaction_scores is not None:
        interaction_scores_matx = interaction_scores_matx[col_order]
    elif cellsign is not None:
        cellsign_matx = cellsign_matx[col_order]

    # whether or not to filter to only significant hits
    # return pvals_matx
    if keep_significant_only:
        # 筛选出任意列中 p 值小于 alpha 的行
        keep_rows = pvals_matx.index[pvals_matx.lt(alpha).any(axis=1)]
        # 用.lt (lower than)方法，比lamda方法可读可维护性更好
        if keep_rows.size > 0:
            print(f"{keep_rows.size} different cell-cell interaction pairs are found in the output.")
            # 更新主要矩阵
            pvals_matx = pvals_matx.loc[keep_rows]
            means_matx = means_matx.loc[keep_rows]

            if interaction_scores is not None:
                interaction_scores_matx = interaction_scores_matx.loc[keep_rows]

            if cellsign is not None:
                # cellsign data is actually a subset so let's do
                cellsign_rows = keep_rows.intersection(cellsign_matx.index)
                if cellsign_rows.size > 0:
                    cellsign_matx = cellsign_matx.loc[cellsign_rows]
                else:
                    raise ValueError("Your cellsign data may not contain significant hits.")
        else:
            raise ValueError("No significant rows found in the data.")
    # run hierarchical clustering on the rows based on interaction value.
    if cluster_rows:
        if means_matx.shape[0] > 2:
            # 行聚类获取顺序
            h_order = hclust(means_matx, axis=0)
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

    if standard_scale:
        means_matx = means_matx.apply(lambda r: (r - np.min(r)) / (np.max(r) - np.min(r)), axis=1)
    means_matx.fillna(0, inplace=True)

    # prepare final table
    colm = "scaled_means" if standard_scale else "means"
    df = means_matx.melt(ignore_index=False).reset_index()  # 宽变长
    df.index = df["index"] + DEFAULT_SEP * 3 + df["variable"]
    df.columns = DEFAULT_COLUMNS + [colm]
    df_pvals = pvals_matx.melt(ignore_index=False).reset_index()
    df_pvals.index = df_pvals["index"] + DEFAULT_SEP * 3 + df_pvals["variable"]
    df_pvals.columns = DEFAULT_COLUMNS + ["pvals"]
    df.celltype_group = [re.sub(DEFAULT_SEP, "-", c) for c in df.celltype_group]
    df["pvals"] = df_pvals["pvals"]
    if debug:
        print(df)
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
    # set factors
    df.celltype_group = df.celltype_group.astype("category")
    # prepare for non-default style plotting
    for i in df.index:
        if df.at[i, colm] == 0:
            df.at[i, colm] = np.nan
    df["x_means"] = df[colm]
    df["y_means"] = df[colm]
    for i in df.index:
        if df.at[i, "pvals"] < alpha:
            df.at[i, "x_means"] = np.nan
            if df.at[i, "pvals"] == 0:
                df.at[i, "pvals"] = 0.001
        if df.at[i, "pvals"] >= alpha:
            if keep_significant_only:
                df.at[i, "y_means"] = np.nan
        if interaction_scores is not None:
            if df.at[i, "interaction_scores"] < 1:
                df.at[i, "x_means"] = np.nan
        elif cellsign is not None:
            if df.at[i, "cellsign"] < 1:
                df.at[i, "cellsign"] = DEFAULT_CELLSIGN_ALPHA
    df["x_stroke"] = df["x_means"]
    set_x_stroke(df=df, isnull=False, stroke=0)
    set_x_stroke(df=df, isnull=True, stroke=highlight_size)
    if exclude_interactions is not None:
        if not isinstance(exclude_interactions, list):
            exclude_interactions = [exclude_interactions]
        df = df[~df.interaction_group.isin(exclude_interactions)]
    # return df
    df.pvals = df.pvals.astype(np.float64)
    df["neglog10p"] = abs(-1 * np.log10(df.pvals))
    df["neglog10p"] = [0 if x >= 0.05 else j for x, j in zip(df["pvals"], df["neglog10p"])]
    df["significant"] = ["yes" if x < alpha else np.nan for x in df.pvals]
    # highlight_col = "#FFFFFF"
    # append the initial data
    if col_start == DEFAULT_V5_COL_START:  # 这一步实际上倒是非常的快
        if additional_grouping:
            df["is_integrin"] = [is_int[i] for i in df.index]
            df["directionality"] = [direc[i] for i in df.index]
            df["classification"] = [classif[i] for i in df.index]
    if df.shape[0] == 0:
        # raise ValueError("The result is empty in this case.")
        print("The result is empty in this case.")
    return df


def draw_cpdb_plot(
        dataframe,  # 输入的 Pandas DataFrame，包含要绘制的数据
        cmap_name: str = "viridis",  # 配色方案名称
        max_size: int = 8,  # 点的最大尺寸
        highlight_size=None,  # 如果指定，则用于高亮点的尺寸
        max_highlight_size: int = 3,  # 高亮点的最大尺寸
        interaction_scores=None,  # 可选的交互评分列，用于调整透明度
        default_style: bool = True,  # 是否应用默认样式
        highlight_col: str = "#080000",  # 高亮点的颜色
        title: str = "",  # 图表标题
        cellsign=None,  # 可选的 cellsign 数据列
        figsize=(6.4, 4.8),  # 图表尺寸
        min_interaction_score: int = 0,  # 最小交互评分
        scale_alpha_by_interaction_scores: bool = False,  # 是否根据交互评分调整透明度
        gene_family=None,  # 基因家族信息（用于标题）
        alpha: float = 0.05,  # 显著性水平的阈值
        scale_alpha_by_cellsign: bool = False,  # 是否根据 cellsign 调整透明度
        filter_by_cellsign: bool = False,  # 是否根据 cellsign 过滤数据
        standard_scale: bool = True,  # 是否使用标准化均值列
        keep_id_cp_interaction: bool = False  # 是否保留交互组标识符
):
    df = dataframe.copy()
    colm = "scaled_means" if standard_scale else "means"
    # change the labelling of interaction_group
    print(keep_id_cp_interaction)
    if keep_id_cp_interaction:
        df.interaction_group = [re.sub(DEFAULT_SEP * 3, "_", c) for c in df.interaction_group]
    else:
        df.interaction_group = [c.split(DEFAULT_SEP * 3)[1] for c in df.interaction_group]
    # print(df.interaction_group[1:5])
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
        if scale_alpha_by_interaction_scores:
            if default_style:
                g = ggplot(
                    df,
                    aes(
                        x="celltype_group",
                        y="interaction_group",
                        colour="significant",
                        fill=colm,
                        size=colm,
                        stroke=stroke,
                        alpha="interaction_scores",
                    ),
                )
            else:
                if all(df["significant"] == "no"):
                    g = ggplot(
                        df,
                        aes(
                            x="celltype_group",
                            y="interaction_group",
                            colour="significant",
                            fill=colm,
                            size=colm,
                            stroke=stroke,
                            alpha="interaction_scores",
                        ),
                    )
                    default_style = True
                else:
                    highlight_col = "#FFFFFF"  # enforce this
                    g = ggplot(
                        df,
                        aes(
                            x="celltype_group",
                            y="interaction_group",
                            colour=colm,
                            fill="significant",
                            size=colm,
                            stroke=stroke,
                            alpha="interaction_scores",
                        ),
                    )
        else:
            g = None
    else:
        if cellsign is not None:
            if filter_by_cellsign:
                df = df[df.cellsign >= DEFAULT_CELLSIGN_ALPHA]
            if scale_alpha_by_cellsign:
                if default_style:
                    g = ggplot(
                        df,
                        aes(
                            x="celltype_group",
                            y="interaction_group",
                            colour="significant",
                            fill=colm,
                            size=colm,
                            stroke=stroke,
                            alpha="cellsign",
                        ),
                    )
                else:
                    if all(df["significant"] == "no"):
                        g = ggplot(
                            df,
                            aes(
                                x="celltype_group",
                                y="interaction_group",
                                colour="significant",
                                fill=colm,
                                size=colm,
                                stroke=stroke,
                                alpha="cellsign",
                            ),
                        )
                        default_style = True
                    else:
                        highlight_col = "#FFFFFF"  # enforce this
                        g = ggplot(
                            df,
                            aes(
                                x="celltype_group",
                                y="interaction_group",
                                colour=colm,
                                fill="significant",
                                size=colm,
                                stroke=stroke,
                                alpha="cellsign",
                            ),
                        )
            else:
                g = None
        else:
            g = None

    if g is None:
        if default_style:
            g = ggplot(
                df,
                aes(
                    x="celltype_group",
                    y="interaction_group",
                    colour="significant",
                    fill=colm,
                    size=colm,
                    stroke=stroke,
                ),
            )
        else:
            if all(df["significant"] == "no"):
                g = ggplot(
                    df,
                    aes(
                        x="celltype_group",
                        y="interaction_group",
                        colour="significant",
                        fill=colm,
                        size=colm,
                        stroke=stroke,
                    ),
                )
                default_style = True
            else:
                highlight_col = "#FFFFFF"  # enforce this
                g = ggplot(
                    df,
                    aes(
                        x="celltype_group",
                        y="interaction_group",
                        colour=colm,
                        fill="significant",
                        size=colm,
                        stroke=stroke,
                    ),
                )

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


def extract_cpdb_table_list(adata_list, cpdb_results_list, cell_type1, cell_type2, genes_list=None, lock=True,
                            keep_significant=False, alpha=0.05,
                            celltype_key="Subset_Identity", degs_analysis=False,
                            disease_sequence=["BS", "CD", "Control", "Colitis", "UC"], debug=False):
    df_list = []
    for i in range(0, len(disease_sequence)):
        print(disease_sequence[i])
        adata = adata_list[i]
        cpdb_results_dict = cpdb_results_list[i]
        dfx = extract_cpdb_table(adata=adata,
                                 additional_grouping=False,
                                 cell_type1=cell_type1, cell_type2=cell_type2, celltype_key=celltype_key,
                                 cpdb_outcome_dict=cpdb_results_dict,
                                 cellsign=None, degs_analysis=False, splitby_key=None,
                                 keep_significant_only=keep_significant,
                                 genes=genes_list, alpha=alpha,
                                 lock_celltype_direction=lock,
                                 debug=debug)
        # print(dfx['significant'].unique())
        dfx["disease"] = disease_sequence[i]
        dfx.index = dfx.index + DEFAULT_SEP * 3 + dfx["disease"]
        df_list.append(dfx);
        del dfx
    dff = pd.concat(df_list)
    if all(pd.isnull(dff["significant"])):
        dff["significant"] = "no"
    else:
        print("Significant Outcome Detected.")
    del df_list
    print(dff)
    # dff["celltype_group"] = [item.replace("Mph-1", "Mph_1").replace("Mph-2", "Mph_2") for item in dff["celltype_group"]]
    dff["Cell_left"] = [item.split("-")[0] for item in dff["celltype_group"]]
    dff["Cell_right"] = [item.split("-")[1] for item in dff["celltype_group"]]
    return dff


def cpdb_table_list_trim(cpdb_table_list, trim_recog):
    int_list = cpdb_table_list["interaction_group"].unique().tolist()
    trim = [s for s in int_list if re.findall(trim_recog, s)]
    int_list = [s for s in int_list if s not in trim]
    cpdb_table_list = cpdb_table_list[cpdb_table_list["interaction_group"].isin(int_list)]
    return cpdb_table_list


def easy_stack_violin(adata, cell_list, cell_name, gene_list, gene_name, output_dir, celltype_key="Subset",
                      split=False):
    filename = (output_dir + '_'.join([cell_name, "_", gene_name, "StViolin(split)."])) if split else (
            output_dir + '_'.join([cell_name, "_", gene_name, "StViolin."]))
    if split:
        if "Tmp" not in adata.obs.columns:
            adata.obs["Tmp"] = adata.obs[["disease", celltype_key]].agg('>>'.join, axis=1)
        group_by = 'Tmp'
    else:
        group_by = 'disease'
    with plt.rc_context():
        sc.pl.stacked_violin(
            adata[adata.obs[celltype_key].isin(cell_list)], gene_list,
            groupby=group_by, swap_axes=False, cmap="viridis_r", use_raw=False, layer="log1p_norm",
            show=False)
        plt.savefig(filename + "png", bbox_inches="tight")
        plt.savefig(filename + "pdf", bbox_inches="tight")


def contained_in(lst, sub):
    n = len(sub)
    return any(sub == lst[i:i + n] for i in range(len(lst) - n + 1))


def easy_stack_barplot(adata, cell_list, cell_name, outputdir,
                       celltype_key, compare_by='disease', is_immune=True, signote=True):
    adata_ss = adata[
        adata.obs["Celltype_Identity"].isin(['T Cell', 'B Cell', 'Plasma', 'Myeloid Cell'])] if is_immune else adata[
        adata.obs["Celltype_Identity"].isin(['Epithelium', 'Endothelium', 'Fibroblast'])]
    if contained_in(adata_ss.obs[celltype_key].unique().tolist(), cell_list):
        all_prop = adata_ss.obs[compare_by].value_counts().sort_index()
        all_prop_perc = [i / sum(all_prop) for i in all_prop]
        adata_ss = adata_ss[adata_ss.obs[celltype_key].isin(cell_list)]
        cell_prop = adata_ss.obs[compare_by].value_counts().sort_index()
        cell_prop_perc = [i / sum(cell_prop) for i in cell_prop]
        df = pd.DataFrame([['All'] + list(all_prop_perc),
                           [celltype_key] + list(cell_prop_perc)],
                          columns=[celltype_key] + all_prop.index.tolist())
        ax = df.plot(x=celltype_key, kind='bar', stacked=True, title='Stacked Bar of ' + "_".join(cell_list), rot=90,
                     ylabel='Percentage')
        for c in ax.containers:
            labels = [v.get_height() if v.get_height() > 0 else '' for v in c]
            labels = [round(m * 100, 2) for m in labels]
            if signote:
                labels[1] = str(labels[1]) if (labels[1] < labels[0] * 1.1) & (
                        labels[1] > labels[0] * 0.9) else " ".join([str(labels[1]), "+"]) if (
                        labels[1] >= labels[0] * 1.1) else " ".join([str(labels[1]), "-"])
            ax.bar_label(c, labels=labels, label_type='center')
        filename = "/".join([outputdir, "_".join([cell_name, "CompareProp", "StBarplot."])])
        plt.savefig(filename + "png", bbox_inches="tight")
        plt.savefig(filename + "pdf", bbox_inches="tight")
    else:
        raise ValueError("Cell list are mixed with immune and non-immune cell, please recheck.")


def print_cpdb_data_table(cpdb_outcome_dict_list, disease_type, output_dir, filename,
                          query_ct1,
                          query_ct2=['Absorp.Colonocyte', 'Tuft Cell', 'Secret.Colonocyte_AQP8', 'Goblet', 'AEC',
                                     'Goblet_KLF4', 'AC', 'pre_Absorp.Colonocyte', 'CD8 Tem', 'CD4 Tfh',
                                     'Secret.Colonocyte_mature', 'B memory', 'Mitotic Mix', 'Fibroblast',
                                     'Undiff_SLC2A13', 'DN T', 'Colonocyte_BEST4', 'NK.gdT1', 'CD8 Trm',
                                     'Epi.Stem_CD24', 'Epi.Stem_FOXP2', 'Fb.Inf', 'Epi.Stem_OLFM4', 'Plasma',
                                     'Monocyte', 'Neutrophil', 'Goblet_SOX4', 'CD4 Tem', 'Tcm', 'GC B Cell', 'pDC',
                                     'cDC3', 'Secret.Colonocyte_ALDOA', 'Mph-1', 'CD4 Treg_CXCR4', 'CD4 Treg_CD27',
                                     'CD4 Treg_CXCR6', 'ILC3.gdT17', 'CD4 Treg_Th17', 'cDC2', 'Mast Cell',
                                     'CD4 Treg_cycling', 'CD4 Treg_BCL11B', 'Mph-2', 'CD4 Treg_PKM',
                                     'Secret.Colonocyte_TFF3', 'CD4 Treg_Th1_Th17', 'Ent.Endocrine',
                                     'pre_Secret.Colonocyte', 'Tuft_precursor', 'Paneth Cell', 'LEC', 'Undiff_MUC2',
                                     'cDC1'],
                          query_min_score=0.5, return_result=False, print_result=True):
    index = disease_sequence.index(disease_type)
    search_results = search_utils.search_analysis_results(
        query_cell_types_1=query_ct1, query_cell_types_2=query_ct2,
        significant_means=cpdb_outcome_dict_list[index]['significant_means'],
        deconvoluted=cpdb_outcome_dict_list[index]['deconvoluted'],
        interaction_scores=cpdb_outcome_dict_list[index]['interaction_scores'],
        query_minimum_score=query_min_score)
    if search_results.empty:
        print("search_result is empty.")
        return None
    if print_result:
        excel_name = output_dir + filename + "(" + disease_type + ")_min_" + str(query_min_score) + ".xlsx"
        search_results.to_excel(excel_name)
    if return_result:
        return search_results
    else:
        print("Searched successfully finished.")
