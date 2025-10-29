import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import pdist
import numpy as np
import pandas as pd

from src.external_adapter.cellphonedb.settings import (
    DEFAULT_V5_COL_START,
    DEFAULT_COL_START,
    DEFAULT_CLASS_COL,
    DEFAULT_SEP,
    DEFAULT_SPEC_PAT,
    DEFAULT_V5_COL_NAMES,
    DEFAULT_CELLSIGN_ALPHA,
    DEFAULT_COLUMNS,
)

from ktplotspy.utils.support import (
    set_x_stroke,
)
import os, re
from plotnine import *


# 对接 cellphonedb v5
class CellphoneInspector():
    '''
    Example
    -------
    一个完整的最简流程如下
    # 生成基本信息
    ci = CellphoneInspector(cpdb_outfile = save_dir)
    ci.prepare_cpdb_tables(degs_analysis=True,cellsign=False,add_meta=True)
    ci.prepare_cell_metadata(metadata=adata.obs, celltype_key = "Subset_Identity", groupby_key = "disease") # 通常直接输入就好

    # 准备查询
    ## 输出是可读的，所以这一步可以手动修改
    gene_query = ci.prepare_gene_query(gene_family=“th17”)
    celltype_pairs = ci.prepare_celltype_pairs(cell_type1=["Th17","Tfh"], cell_type2="B cell", lock_celltype_direction=True)

    # 查询
    ci.filter_and_cluster(gene_query=gene_query, celltype_pairs=celltype_pairs,keep_significant_only=False)


    '''

    def __init__(self, cpdb_outfile):
        self.file_path = cpdb_outfile
        self.data = self._load_file(cpdb_outfile)  # 格式为字典
        self._log(f"Initialized with CPDB output directory: {cpdb_outfile}")

    @staticmethod
    def _log(msg):
        print(f"[CellphoneInspector Message] {msg}")

    def _load_file(self, path):
        self._log("Loading CellPhoneDB output files...")
        def find_file(pattern):
            matches = [s for s in os.listdir(path) if re.search(pattern, s)]
            if not matches:
                raise FileNotFoundError(f"No file matching pattern '{pattern}' in {path}")
            return os.path.join(path, matches[0])

        result = {
            "deconvoluted_percents": pd.read_table(find_file("analysis_deconvoluted_percents"), delimiter="\t"),
            "deconvoluted": pd.read_table(find_file("analysis_deconvoluted"), delimiter="\t"),
            "means": pd.read_table(find_file("analysis_means"), delimiter="\t"),
            "pvalues": pd.read_table(find_file("analysis_pvalues|analysis_relevant_interactions"), delimiter="\t"),
            "interaction_scores": pd.read_table(find_file("analysis_interaction_scores"), delimiter="\t"),
            "significant_means": pd.read_table(find_file("analysis_significant_means"), delimiter="\t"),
        }
        self._log(f"Loaded tables: {list(result.keys())}")
        for k, v in result.items():
            self._log(f"  {k}: {v.shape[0]} rows × {v.shape[1]} cols")
        return result

    def _prep_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generic function to format the means and pvalues tables.
        (This part completely from ktplotspy)

        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe. Either pandas DataFrame for means or pvalues.

        Returns
        -------
        pd.DataFrame
            Table ready for further analysis.
        """
        dat = data.copy()
        dat.index = [x + DEFAULT_SEP * 3 + y for x, y in zip(dat.id_cp_interaction, dat.interacting_pair)]
        dat.columns = [re.sub(f"\\{DEFAULT_CPDB_SEP}", DEFAULT_SEP, col) for col in dat.columns]
        dat.index = [re.sub("_", "-", row) for row in dat.index]
        dat.index = [re.sub("[.]", " ", row) for row in dat.index]

        return dat

    def prepare_cpdb_tables(self,
                            degs_analysis=False, cellsign=False,
                            add_meta=False):
        '''
        完成读取后，进行的第一步操作，进行数据准备

        Example
        -------
        ci.prepare_cpdb_tables(degs_analysis=True,cellsign=False,add_meta=True)

        Parameters
        ----------
        :param degs_analysis: bool，是否在 CPDB 分析时采用 degs_analysis 模式
        :param cellsign: bool，是否在 CPDB 分析时采用 CellSign 转录因子分析模式
        :param add_meta: 返回的表格是否需要后面增加三列 meta 信息：directionality, classification, is_integrin
        :return: 为 ci 类增加四个属性：ci.pvals, ci.means, ci.matrix, ci.meta
        '''
        self._log("Preparing CPDB matrices...")

        means_mat = self._prep_table(data=self.data["means"])
        pvals_mat = self._prep_table(data=self.data["pvalues"])
        interaction_scores = self.data["interaction_scores"]
        self._log(f"Means table: {means_mat.shape}, P-values table: {pvals_mat.shape}")

        if means_mat.shape[0] == 0:
            self._log("[Warning] Empty means table — skipping further processing.")
            return

        col_start = (
            DEFAULT_V5_COL_START if pvals_mat.columns[DEFAULT_CLASS_COL] == "classification" else DEFAULT_COL_START)

        if means_mat.shape[1] <= col_start:
            self._log("[Warning] No sample columns detected after metadata — check input file.")
            return

        # 对齐 pvals 和 means 矩阵
        if pvals_mat.shape != means_mat.shape:
            tmp_pvals_mat = pd.DataFrame(index=means_mat.index,
                                         columns=means_mat.columns)

            # 把 means_mat 复制到 new_df
            tmp_pvals_mat.iloc[:, :col_start] = means_mat.iloc[:, :col_start]
            tmp_pvals_mat.update(pvals_mat)

            if degs_analysis:
                tmp_pvals_mat.fillna(0, inplace=True)
            else:
                tmp_pvals_mat.fillna(1, inplace=True)

            pvals_mat = tmp_pvals_mat.copy()

        if degs_analysis:
            pvals_mat.iloc[:, col_start: pvals_mat.shape[1]] = 1 - pvals_mat.iloc[:, col_start: pvals_mat.shape[1]]

        # 处理 interaction 评分矩阵
        if (interaction_scores is not None) and (cellsign is not None):
            raise KeyError("Please specify either interaction scores or cellsign, not both.")

        if interaction_scores is not None:
            interaction_scores_mat = self._prep_table(data=interaction_scores)
            self.matrix = interaction_scores_mat
            self.mode = "interaction_scores"
        elif cellsign is not None:
            cellsign_mat = self._prep_table(data=cellsign)
            self.matrix = cellsign_mat
            self.mode = "cellsign"

        self.pvals = pvals_mat
        self.means = means_mat
        self._log("Finished preparing CPDB tables.")

        # 处理额外添加的 meta 信息
        if col_start == DEFAULT_V5_COL_START and add_meta:
            id_cols = DEFAULT_V5_COL_NAMES
            # melt 时只提取必要列
            melted = pd.melt(means_mat, id_vars=id_cols, var_name="variable")

            # 构造 key 向量
            keys = (
                    melted["id_cp_interaction"] + DEFAULT_SEP * 3 +
                    melted["interacting_pair"].str.replace("_", "-") + DEFAULT_SEP * 3 +
                    melted["variable"]
            )

            # 转为 numpy 数组以加速 zip + dict 构造
            direc = dict(zip(keys, melted["directionality"].to_numpy()))
            classif = dict(zip(keys, melted["classification"].to_numpy()))
            is_int = dict(zip(keys, melted["is_integrin"].to_numpy()))
            self.meta = {"directionality": direc,
                         "classification": classif,
                         "is_integrin": is_int}
            self._log("Constructed mapping dictionaries for directionality, classification, and integrin.")

    def prepare_cell_metadata(self, metadata, celltype_key, groupby_key=None):
        '''

        Example
        -------
        ci.prepare_cell_metadata(adata.obs, celltype_key = "Subset_Identity", groupby_key = "disease") # 通常直接输入就好

        :param metadata: pd.DataFrame
        :param celltype_key: str，用于指定 meta 的哪一列储存细胞身份信息
        :param groupby_key: str，如果存在一个独立于细胞身份的分组信息列则填写，如 gender, disease_type, disease_state
        :return: 为 ci 对象增加一个属性 ci.cell_metadata；存在 group 时增加 ci.group_info
        '''
        self._log("Preparing cell information...")
        # 确保为 category
        if not metadata[celltype_key].dtype.name == "category":
            metadata[celltype_key] = metadata[celltype_key].astype("category")
        if groupby_key is not None:
            self._log(f"Grouping by '{groupby_key}' within cell type '{celltype_key}'")
            # 确保为 category
            if not metadata[groupby_key].dtype.name == "category":
                metadata[groupby_key] = metadata[groupby_key].astype("category")

            # 构造“_labels”列
            metadata["_labels"] = [split + "_" + celltype for split, celltype in zip(metadata[groupby_key],
                                                                                     metadata[celltype_key])]
            metadata["_labels"] = metadata["_labels"].astype("category")
            self.group_info = list(metadata[groupby_key].cat.categories)

        elif groupby_key is None:
            self._log("No groupby key provided, using cell type only")
            metadata["_labels"] = metadata[celltype_key]
            self.group_info = None

        self.cell_metadata = metadata
        self._log(f"Prepared metadata with {metadata.shape[0]} cells and columns {list(metadata.columns)}")

    def prepare_gene_query(self, genes=None, gene_family=None, custom_gene_family=None):
        '''

        Example
        -------
        gene_query = ci.prepare_gene_query(genes=[]) # 默认查询全部基因

        gene_query = ci.prepare_gene_query(gene_family=“th17”) # 内置正则，自动搜索相关基因

        gene_query = ci.prepare_gene_query(custom_gene_family={“scavenger_rec”:["^ACKR","^SR-"]})

        当然，也可以和 geneset 类联用
        my_markers = Geneset(save_addr + "Markers-updated.xlsx")
        th17_sigs = my_markers.get(siganature='Th17',sheet_name="Immunocyte")
        ci.prepare_gene_query(genes=th17_sigs)


        :param genes:
        :param gene_family: 自动查询，可选参数 chemokines, th1, th2, th17, treg, costimulatory, coinhibitory
        :param custom_gene_family: [name: [list of genes]] 格式的列表，进行简单的字符串搜索，支持正则
        :return:
        '''
        if genes is None:
            if gene_family is not None:
                # 这将会返回一组基因列表
                query_group = prep_query_group(self.means, custom_gene_family)
                if isinstance(gene_family, list):  # 是列表，逐一查找并合并相关基因。
                    query = []
                    for gf in gene_family:
                        if gf.lower() in query_group:
                            for gfg in query_group[gf.lower()]:
                                query.append(gfg)
                        else:
                            raise KeyError(
                                "gene_family needs to be one of the following: {}".format(query_group.keys()))
                    query = list(set(query))
                else:  # 是单个字符串，直接提取内置的对应的基因组
                    if gene_family.lower() in query_group:
                        query = query_group[gene_family.lower()]
                    else:
                        raise KeyError("gene_family needs to be one of the following: {}".format(query_group.keys()))
            else:  # 构造默认查询，提取 means_mat.interacting_pair 中所有基因（通过空正则表达式匹配）*
                query = [i for i in self.means.interacting_pair if re.search(pattern="", string=i)]
        elif genes is not None:
            if gene_family is not None:
                raise KeyError("Please specify either genes or gene_family, not both.")
            else:  # 筛选 self.means.interacting_pair 中匹配 genes 列表的基因
                query = [i for i in self.means.interacting_pair if re.search("|".join(genes), i)]
        return query

    def prepare_celltype_pairs(self, cell_type1, cell_type2, lock_celltype_direction=False):
        '''
        Example
        -------
        celltype_pairs = ci.prepare_celltype_pairs(cell_type1="Th17", cell_type2="Epithelium", lock_celltype_direction=True)

        :param cell_type1:
        :param cell_type2:
        :param lock_celltype_direction: 是否锁定 cell_type1 和 cell_type2 的左位和右位
        :return:
        '''
        # 全部的细胞类别
        labels = list(self.cell_metadata._labels.cat.categories)
        c_type1 = cell_type1 if cell_type1 != "." else labels
        c_type2 = cell_type2 if cell_type2 != "." else labels

        # 生成全部的细胞-细胞对组合
        celltype = []
        for i in range(0, len(c_type1)):
            cq = []
            for cx2 in c_type2:
                # 无论是否锁定，我们都有c_type1在左边，ctype_2在右边的情况
                cq.append("^" + c_type1[i] + DEFAULT_SEP + cx2 + "$")
                if not lock_celltype_direction:
                    # 当锁定的时候，不进行交换
                    cq.append("^" + cx2 + DEFAULT_SEP + c_type1[i] + "$")
            cq = "|".join(cq)
            if self.group_info is not None:
                for g in self.group_info:
                    ctx = cq.split("|")
                    ctx = [x for x in ctx if re.search(g + ".*" + DEFAULT_SEP + g, x)]
                    cqi = "|".join(ctx)
                    if cqi != "":
                        celltype.append(cqi)
            else:
                celltype.append(cq)

        cell_type = "|".join(celltype)

        # keep cell types
        ct_columns = [ct for ct in self.means.columns if re.search(cell_type, ct)]

        return ct_columns

    def filter_and_cluster(self, gene_query, celltype_pairs,
                           alpha=0.05, keep_significant_only=True, cluster_rows=True, standard_scale=True):
        '''
        Example
        -------
        gene_query = ci.prepare_gene_query(genes=[])
        celltype_pairs = ci.prepare_celltype_pairs(cell_type1="Th17", cell_type2="B cell", lock_celltype_direction=True)
        ci.filter_and_cluster(gene_query=gene_query, celltype_pairs=celltype_pairs,keep_significant_only=False)


        :param gene_query:
        :param celltype_pairs:
        :param alpha:
        :param keep_significant_only:
        :param cluster_rows:
        :param standard_scale:
        :return:
        '''
        self._log("Filtering and hierarchical clustering matrices...")
        means_matx = self.means[self.means.interacting_pair.isin(gene_query)][celltype_pairs]
        pvals_matx = self.pvals[self.pvals.interacting_pair.isin(gene_query)][celltype_pairs]
        interact_matx = self.matrix[self.matrix.interacting_pair.isin(gene_query)][celltype_pairs]

        # 重新对列排序
        col_order = []
        if self.group_info is not None:
            for g in self.group_info:
                for c in means_matx.columns:
                    if re.search(g, c):
                        col_order.append(c)
        else:
            col_order = means_matx.columns

        shared_cols = [c for c in col_order if c in means_matx.columns]
        means_matx, pvals_matx, interact_matx = (
            means_matx[shared_cols],
            pvals_matx[shared_cols],
            interact_matx[shared_cols]
        )

        # 处理显著过滤
        if keep_significant_only:
            # 筛选出任意列中 p 值小于 alpha 的行
            keep_rows = pvals_matx.index[pvals_matx.lt(alpha).any(axis=1)]
            # 用.lt (lower than)方法，比lamda方法可读可维护性更好
            if keep_rows.size > 0:
                print(f"{keep_rows.size} different cell-cell interaction pairs are found in the output.")
                # 更新主要矩阵
                pvals_matx = pvals_matx.loc[keep_rows]
                means_matx = means_matx.loc[keep_rows]
                keep_rows = keep_rows.intersection(interact_matx.index)
                interact_matx = interact_matx.loc[keep_rows]

                if interact_matx.size > 0:
                    self._log(f"Totally {interact_matx.size} reads found available after significance filtering.")
                else:
                    raise ValueError("Your data may not contain significant hits.")
            else:
                raise ValueError("No significant rows found in the data.")

        # 处理层次聚类
        if cluster_rows:
            if means_matx.shape[0] > 2:
                # 行聚类获取顺序
                self._log("Performing hierarchical clustering...")
                h_order = hclust(means_matx, axis=0)  # axis Index = 0 and columns = 1，对行进行层次聚类

                # 对主要矩阵重新排序
                means_matx = means_matx.loc[h_order]
                pvals_matx = pvals_matx.loc[h_order]

                # 对 interaction_scores 和 cellsign 数据处理，记住 cellsign 只是个子集，为了兼容：
                valid_h_order = [h for h in h_order if h in interact_matx.index]
                if valid_h_order:
                    interact_matx = interact_matx.loc[valid_h_order]
                else:
                    raise ValueError("No significant hits found after clustering. ")
        else:
            self._log("Skipping scaling step.")

        def safe_scale(r):
            denom = np.max(r) - np.min(r)
            return (r - np.min(r)) / denom if denom != 0 else r - np.min(r)

        if standard_scale:
            means_matx = means_matx.apply(safe_scale, axis=1)

        means_matx.fillna(0, inplace=True)

        self.outcome = {"means_matx": means_matx,
                        "pvals_matx": pvals_matx,
                        "interact_matx": interact_matx,
                        "scale": standard_scale,
                        "alpha": alpha,
                        "significance": keep_significant_only}

    def format_outcome(self, exclude_interactions):
        '''

        :param exclude_interactions: list, 包含想要手动排除的作用对，不支持自动检索
        :return: 返回一个包含主要输出的 pd.Dataframe， 对其进行手动修改也是容易的
        '''
        self._log("Formatting output table...")
        colm = "scaled_means" if self.outcome["scale"] else "means"

        df = self.outcome["means_matx"].melt(ignore_index=False).reset_index()  # 宽变长

        df.index = df["index"] + DEFAULT_SEP * 3 + df["variable"]
        df.columns = DEFAULT_COLUMNS + [colm]
        df.celltype_group = [re.sub(DEFAULT_SEP, "-", c) for c in df.celltype_group]

        df_pvals = self.outcome["pvals_matx"].melt(ignore_index=False).reset_index()
        df_pvals.index = df_pvals["index"] + DEFAULT_SEP * 3 + df_pvals["variable"]
        df_pvals.columns = DEFAULT_COLUMNS + ["pvals"]
        # df["pvals"] = df_pvals["pvals"]
        df["pvals"] = pd.to_numeric(df_pvals["pvals"], errors="coerce") # invalid parsing will be set as NaN

        df_matrix = self.outcome["interact_matx"].melt(ignore_index=False).reset_index()
        df_matrix.index = df_matrix["index"] + DEFAULT_SEP * 3 + df_matrix["variable"]
        df_matrix.columns = DEFAULT_COLUMNS + ["scores"]
        df = df.join(df_pvals["pvals"]).join(df_matrix["scores"]) # 对齐索引

        # set factors
        df.celltype_group = df.celltype_group.astype("category")

        # 为了画图做一些设置
        ## 平均值
        for i in df.index:
            if df.at[i, colm] == 0:
                df.at[i, colm] = np.nan

        df["x_means"] = df[colm]
        df["y_means"] = df[colm]

        ## 对边缘值进行处理
        for i in df.index:
            if df.at[i, "pvals"] < self.outcome["alpha"]:
                df.at[i, "x_means"] = np.nan
                if df.at[i, "pvals"] == 0:
                    df.at[i, "pvals"] = 0.001

            if df.at[i, "pvals"] >= self.outcome["alpha"]:
                if self.outcome["significance"]:
                    df.at[i, "y_means"] = np.nan
            if self.mode == "interaction_scores":
                if df.at[i, "interaction_scores"] < 1:
                    df.at[i, "x_means"] = np.nan
            elif self.mode == "cellsign":
                if df.at[i, "cellsign"] < 1:
                    df.at[i, "cellsign"] = DEFAULT_CELLSIGN_ALPHA

        # 排除项
        if exclude_interactions is not None:
            if not isinstance(exclude_interactions, list):
                exclude_interactions = [exclude_interactions]
            df = df[~df.interaction_group.isin(exclude_interactions)]

        # 补充计算显著性
        df.pvals = df.pvals.astype(np.float64)
        df["neglog10p"] = abs(-1 * np.log10(df.pvals))
        df["neglog10p"] = [0 if x >= 0.05 else j for x, j in zip(df["pvals"], df["neglog10p"])]
        df["neglog10p"] = df["neglog10p"].astype(float)
        df["significant"] = ["yes" if x < self.outcome["alpha"] else np.nan for x in df.pvals]

        # 增添 meta 信息列
        if self.meta is not None:
            df["is_integrin"] = [self.meta["is_integrin"].get(i, np.nan) for i in df.index]
            df["directionality"] = [self.meta["directionality"].get(i, np.nan) for i in df.index]
            df["classification"] = [self.meta["classification"].get(i, np.nan) for i in df.index]

        if df.shape[0] == 0:
            self._log("The result is empty in this case.")
        else:
            self._log(f"Formatted output: {df.shape} rows × {df.shape[1]} cols")

        return df


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


def prep_query_group(means: pd.DataFrame, custom_dict: dict[str, list[str]] | None = None) -> dict:
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


def hclust(
        data: pd.DataFrame,
        axis: int = 0,
        method: str = "average",
        metric: str = "euclidean",
        optimal_ordering: bool = True,
) -> list:
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
