import numpy as np
import pandas as pd

from src.external_adapter.cellphonedb.settings import (
    DEFAULT_V5_COL_START,
    DEFAULT_COL_START,
    DEFAULT_CLASS_COL,
    DEFAULT_SEP,
    INTERACTION_COLUMNS,
    DEFAULT_V5_COL_NAMES,
    DEFAULT_CELLSIGN_ALPHA,
    DEFAULT_COLUMNS,
)
from src.external_adapter.cellphonedb.support import (
    prep_query_group,hclust,split_kwargs
)

import os, re

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)


# 对接 cellphonedb v5
class CellphoneInspector():
    '''
    Example
    -------
    一个完整的最简流程如下
    # 生成基本信息
    ci = CellphoneInspector(cpdb_outfile = save_dir,degs_analysis=False,cellsign=False)
    ci.prepare_cpdb_tables(add_meta=True)
    ci.prepare_cell_metadata(metadata=adata.obs, celltype_key = "Subset_Identity", groupby_key = "disease") # 通常直接输入就好

    # 准备查询
    ## 输出是可读的，所以这一步可以手动修改
    gene_query = ci.prepare_gene_query(gene_family=“th17”)
    celltype_pairs = ci.prepare_celltype_pairs(cell_type1=["Th17","Tfh"], cell_type2="B cell", lock_celltype_direction=True)

    # 查询
    ci.filter_and_cluster(gene_query=gene_query, celltype_pairs=celltype_pairs,keep_significant_only=False)

    # 输出结果表格
    ci.format_outcome()

    Instance Variables
    ------------------
    ci.file_path
    ci.data
    ci.means, ci.pvalues, ci.matrix, ci.mode, ci.meta
    ci.cell_metadata, ci.group_info (可选)
    ci.outcome：dict，keys 包括 "means_matx", "pvals_matx", "interact_matx", "scale", "alpha", "significance", "final")

    '''

    def __init__(self, cpdb_outfile, degs_analysis=False,cellsign=False):
        self.file_path = cpdb_outfile
        self.deg = degs_analysis
        self.cellsign = cellsign
        self.data = self._load_file(cpdb_outfile)  # 格式为字典
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized with CPDB output directory: {cpdb_outfile}")
    
    def _load_file(self, path):
        self.logger.info("Loading CellPhoneDB output files...")
        def find_file(pattern):
            matches = [s for s in os.listdir(path) if re.search(pattern, s)]
            if not matches:
                raise FileNotFoundError(f"No file matching pattern '{pattern}' in {path}")
            return os.path.join(path, matches[0])

        # cpdb_analysis_method
        basic_file_keys = {
            "means":'means_result',
            "deconvoluted":'deconvoluted',
            "deconvoluted_percents":'deconvoluted_percents',
            "interaction_scores":'interaction_scores',
            "pvals":'pvalues' # 默认 cpdb_analysis_method 不含此方法
        }

        # 在 cpdb_statistical_analysis_method 和 cpdb_degs_analysis_method 中
        # 添加 active_tfs_file_path = active_tf.txt，即可采用 CellSign Module
        if self.deg:
            basic_file_keys.update({
                "pvals": 'relevant_interactions'
            })

        if self.cellsign:
            basic_file_keys.update({
                "cellsign_interactions": 'CellSign_active_interactions',
                "cellsign_deconvoluted":"CellSign_active_interactions_deconvoluted"
            })

        result = {}
        for k, v in basic_file_keys.items():
            matches = [s for s in os.listdir(path) if re.search(v + r'\.txt$', s)]
            if not matches:
                raise FileNotFoundError(f"No file matching pattern '{v}.txt' in {path}")
            if len(matches) > 1:
                self.logger.info(f"Warning: multiple files match pattern '{v}.txt', using the first one: {matches[0]}")
            df = pd.read_csv(os.path.join(path, matches[0]), sep="\t", low_memory=False)
            result[k] = df

        if "cellsign_interactions" not in result.keys() and "cellsign_deconvoluted" not in result.keys():
            self.logger.info("Cannot detect cellsign files, fallback to self.cellsign=False")

        self.logger.info(f"Loaded tables: {list(result.keys())}")
        for k, v in result.items():
            self.logger.info(f"  {k}: {v.shape[0] if v is not None else 0} rows × {v.shape[1] if v is not None else 0} cols")

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
                            add_meta=False,use_cellsign=False):
        '''
        完成读取后，进行的第一步操作，进行数据准备

        Example
        -------
        ci.prepare_cpdb_tables(add_meta=True)

        Parameters
        ----------
        :param add_meta: 返回的表格是否需要后面增加三列 meta 信息：directionality, classification, is_integrin
        :param use_cellsign: 当且仅当 self.cellsign = True 时可用，否则自动忽略
        :return: 为 ci 类增加四个属性：ci.pvals, ci.means, ci.matrix, ci.meta
        '''
        self.logger.info("Preparing CPDB matrices...")

        means_mat = self._prep_table(data=self.data["means"])
        pvals_mat = self._prep_table(data=self.data["pvalues"])
        interaction_scores = self.data["interaction_scores"]
        self.logger.info(f"Means table: {means_mat.shape}, P-values table: {pvals_mat.shape}")
        self.means = means_mat
        self.pvals = pvals_mat

        if means_mat.shape[0] == 0:
            self.logger.info("[Warning] Empty means table — skipping further processing.")
            return

        col_start = (
            DEFAULT_V5_COL_START if pvals_mat.columns[DEFAULT_CLASS_COL] == "classification" else DEFAULT_COL_START)

        if means_mat.shape[1] <= col_start:
            self.logger.info("[Warning] No sample columns detected after metadata — check input file.")
            return

        # 对齐 pvals 和 means 矩阵
        if pvals_mat.shape != means_mat.shape:
            tmp_pvals_mat = pd.DataFrame(index=means_mat.index,
                                         columns=means_mat.columns)

            # 把 means_mat 复制到 new_df
            tmp_pvals_mat.iloc[:, :col_start] = means_mat.iloc[:, :col_start]
            tmp_pvals_mat.update(pvals_mat)

            if self.deg:
                tmp_pvals_mat.fillna(0, inplace=True)
            else:
                tmp_pvals_mat.fillna(1, inplace=True)

            pvals_mat = tmp_pvals_mat.copy()

        if self.deg:
            pvals_mat.iloc[:, col_start: pvals_mat.shape[1]] = 1 - pvals_mat.iloc[:, col_start: pvals_mat.shape[1]]


        # 处理 interaction 评分矩阵
        if (interaction_scores is not None) and (use_cellsign is True):
            raise KeyError("Please specify either using interaction scores or cellsign interaction, not both.")

        if use_cellsign and self.cellsign:
            cellsign_mat = self._prep_table(data=self.data["cellsign_interactions"])
            self.matrix = cellsign_mat
            self.mode = "cellsign"
        else:
            interaction_scores_mat = self._prep_table(data=interaction_scores)
            self.matrix = interaction_scores_mat
            self.mode = "interaction_scores"


        self.logger.info("Finished preparing CPDB tables.")

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
            self.logger.info("Constructed mapping dictionaries for directionality, classification, and integrin.")
    
    @logged
    def prepare_cell_metadata(self, metadata, celltype_key, groupby_key=None):
        '''

        Example
        -------
        ci.prepare_cell_metadata(adata.obs, celltype_key = "Subset_Identity", groupby_key = "disease") # 通常直接输入就好

        :param metadata: pd.DataFrame
        :param celltype_key: str，用于指定 meta 的哪一列储存细胞身份信息
        :param groupby_key: str，
            当进行 CPDB 分析的数据混合了多个控制条件时，如 gender, disease_type, disease_state，
            为了保留实际上的分组信息，同时探索一些不平衡组别潜在的细胞通讯（如，某组别 A 纳入的上皮过少），
            建议进行：
                adata.obs["combined_identity"] = adata.obs["celltype"] + "_" + adata.obs["control_condition"]
                df = adata.obs["combined_identity"]
                df = df.reset_index()
                df.columns = ["Cell", "cell_type"]
                df.write_csv(meta_file_path, sep="\t")
            这里的 meta_file_path 参数就是要填入 `cpdb_statistical_analysis_method.call()` 方法中的参数。
            在进行下游分析的时候，如果希望仅保留需要的 grouping 信息，即可将 "control_condition" 列填入 groupby_key
            函数会仅保留同一样本内的分组。当采用此参数时，推荐接下来 keep_significant_only = False 进行观察。
        :return: 为 ci 对象增加一个属性 ci.cell_metadata；存在 group 时增加 ci.group_info
        '''
        self.logger.info("Preparing cell information...")
        # 确保为 category
        if not metadata[celltype_key].dtype.name == "category":
            metadata[celltype_key] = metadata[celltype_key].astype("category")
        if groupby_key is not None:
            self.logger.info(f"Grouping by '{groupby_key}' within cell type '{celltype_key}'")
            # 确保为 category
            if not metadata[groupby_key].dtype.name == "category":
                metadata[groupby_key] = metadata[groupby_key].astype("category")

            # 构造“_labels”列
            metadata["_labels"] = [split + "_" + celltype for split, celltype in zip(metadata[groupby_key],
                                                                                     metadata[celltype_key])]
            metadata["_labels"] = metadata["_labels"].astype("category")
            self.group_info = list(metadata[groupby_key].cat.categories)

        elif groupby_key is None:
            self.logger.info("No groupby key provided, using cell type only")
            metadata["_labels"] = metadata[celltype_key]
            self.group_info = None

        self.cell_metadata = metadata
        self.logger.info(f"Prepared metadata with {metadata.shape[0]} cells and columns {list(metadata.columns)}")
    
    @logged
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
    
    @logged
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
        self.logger.info("Filtering and hierarchical clustering matrices...")
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
                    self.logger.info(f"Totally {interact_matx.size} reads found available after significance filtering.")
                else:
                    raise ValueError("Your data may not contain significant hits.")
            else:
                raise ValueError("No significant rows found in the data.")

        # 处理层次聚类
        if cluster_rows:
            if means_matx.shape[0] > 2:
                # 行聚类获取顺序
                self.logger.info("Performing hierarchical clustering...")
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
            self.logger.info("Skipping scaling step.")

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
    
    @logged
    def format_outcome(self, exclude_interactions=None):
        '''

        :param exclude_interactions: list, 包含想要手动排除的作用对，不支持自动检索
        :return: 返回一个包含主要输出的 pd.Dataframe， 对其进行手动修改也是容易的
        '''
        self.logger.info("Formatting output table...")
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
        df["cell_left"] = [item.split("-")[0] for item in df["celltype_group"]]
        df["cell_right"] = [item.split("-")[1] for item in df["celltype_group"]]

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
        else:
            self.logger.info("No outcome excluded.")

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
            self.logger.info("The result is empty in this case.")
        else:
            self.logger.info(f"Formatted output: {df.shape} rows × {df.shape[1]} cols")

        self.outcome.update({"final":df})

def aggregate_cpdb_results(cpdb_outfile_list, meta_list, name_list,gene_query,celltype_pairs,**kwargs):
    '''

    整合多个 cpdb 结果并对齐
    需要一系列对应的 anndata.obs/metadata 和 cpdb 输入

    Example
    -------
    # 准备查询
    gene_query = ci.prepare_gene_query(gene_family=“th17”)
    celltype_pairs = ci.prepare_celltype_pairs(cell_type1=["Th17","Tfh"], cell_type2="B cell", lock_celltype_direction=True)
    combined_df = aggregate_cpdb_results(cpdb_outfile_list, meta_list, name_list,gene_query,celltype_pairs)

    :return:
    '''
    # 检查格式
    if not len(cpdb_outfile_list) == len(meta_list) and len(meta_list) == len(name_list):
        raise ValueError("cpdb_outfile_list, meta_list and name_list must have the same length")

    default_param = {"prepare_cpdb_tables":{"degs_analysis":False, "cellsign":False,"add_meta":False},
                     "prepare_cell_metadata": {"celltype_key":"Subset_Identity", "groupby_key":None},
                     "filter_and_cluster": {"alpha":0.05, "keep_significant_only":True,
                                            "cluster_rows":True, "standard_scale":True},
                     "format_outcome": {"exclude_interactions":None},

    }

    split = split_kwargs(CellphoneInspector.prepare_cpdb_tables,
                          CellphoneInspector.prepare_cell_metadata,
                          CellphoneInspector.filter_and_cluster,
                          CellphoneInspector.format_outcome,
                         **kwargs)
    print("[aggregate_cpdb_results] Split kwargs:", split)

    final_param = {}
    for k, v in split.items():
        default = default_param[k]
        param = default.update(v)
        final_param[k] = param

    final_param["filter_and_cluster"].update({"gene_query": gene_query,
                                              "celltype_pairs": celltype_pairs})

    print(f"[aggregate_cpdb_results] final_param[{k}]:", final_param[k])

    df_list = []
    for i in range(0, len(cpdb_outfile_list)):
        cpdb_outfile = cpdb_outfile_list[i]
        meta = meta_list[i]
        name = name_list[i]
        print("[aggregate_cpdb_results] Processing batch:", name)

        batch_param = {k: v.copy() for k, v in final_param.items()}
        batch_param["prepare_cell_metadata"].update({"meta":meta})
        print("[aggregate_cpdb_results] Params:", batch_param)

        # 构造对象
        ci = CellphoneInspector(cpdb_outfile=cpdb_outfile)
        # 准备基本信息
        ci.prepare_cpdb_tables(**batch_param["prepare_cpdb_tables"])
        ci.prepare_cell_metadata(**batch_param["prepare_cell_metadata"])
        # 查询
        ci.filter_and_cluster(**batch_param["filter_and_cluster"])
        # 输出结果表格
        ci.format_outcome(**batch_param["format_outcome"])

        dfx = ci.outcome["final"]
        dfx["batch"] = name
        dfx.index = dfx.index + DEFAULT_SEP * 3 + dfx["batch"]
        df_list.append(dfx)
        del dfx

    dff = pd.concat(df_list)

    return dff



def cpdb_results_trim(df, to_left=None,to_right=None,
                      interaction_remove=None,celltype_remove=None,
                      sep=None):
    '''
    用来修整标准输出或整合的输出.
    主要用来将配体-受体顺序错位的进行更正。
    能够自行检查冲突的情况。

    Example
    -------
    将具有 LG/L 结尾的基因放到左边（作为配体）
    df = cpdb_results_trim(df, to_left ="*LG$|*L$")


    :param df: CI.format_outcome() 或 aggregate_cpdb_results 的输出
    :param LR_switch: 用来填写希望被锁在左侧或右的配体、受体，用“|”分隔并支持正则
    :param interaction_remove:
    :return:
    '''
    if sep is None:
        sep = DEFAULT_SEP * 3
    # 整齐化 significant，避免稍后绘图产生 NaN 问题
    if df["significant"].isnull().all():
        df["significant"] = "no"
        print("[cpdb_results_trim] No significant outcome detected.")
    else:
        print("[cpdb_results_trim] Significant outcome detected.")

    df[["Ligand", "Receptor"]] = df["interaction_group"].str.split(sep, n=1, expand=True)

    if to_left is None and to_right is None:
        print("[cpdb_results_trim] Skip ligand-receptor revision.")
    else:
        print("[cpdb_results_trim] Revising ligand-receptor direction...")

        # 构建正则
        pattern_left = re.compile(to_left, flags=re.IGNORECASE) if to_left is not None else None
        pattern_right = re.compile(to_right, flags=re.IGNORECASE) if to_right is not None else None

        # 找出哪些行方向反了
        mask_wrong_direction = (
                df["Ligand"].str.contains(pattern_right, na=False) &
                df["Receptor"].str.contains(pattern_left, na=False)
        )
        potential_conflicted = (
                df["Ligand"].str.contains(pattern_left, na=False) &
                df["Receptor"].str.contains(pattern_right, na=False)
        )
        conflicted = mask_wrong_direction & potential_conflicted # 两个 bool 向量相加

        if conflicted.any():
            print("[cpdb_results_trim] Possible conflict in ligand-receptor direction.")
            interaction_conflicted = df["interaction_group"]
            print(f"[cpdb_results_trim] Interaction conflict: {interaction_conflicted}]")
            return


        wrong_idx = df.index[mask_wrong_direction]
        print(f"[cpdb_results_trim] {len(wrong_idx)} reversed interactions detected.")

        # 对这些行进行纠正
        if len(wrong_idx) > 0:
            df.loc[wrong_idx, ["Ligand", "Receptor"]] = df.loc[wrong_idx, ["Receptor", "Ligand"]].values

        # interaction_group 通常是 “A-B” 格式
        df.loc[wrong_idx, "interaction_group"] = df.loc[wrong_idx, "interaction_group"].apply(
            lambda x: (sep).join(x.split(sep)[::-1]) if isinstance(x, str) else x
        )

        print("[cpdb_results_trim] Ligand-Receptor direction corrected.")

    if interaction_remove is not None:
        print(f"[cpdb_results_trim] Remove interactions: {interaction_remove}.")
        int_list = df["interaction_group"].unique().tolist()
        trim = [s for s in int_list if re.search(interaction_remove, s)]
        print(f"[cpdb_results_trim] Matched interactions: {trim}.")
        int_list = [s for s in int_list if s not in trim]
        df = df[df["interaction_group"].isin(int_list)]
        print(f"[cpdb_results_trim] Remained intereactions: {len(int_list)}.")
    else:
        print("[cpdb_results_trim] Not remove any interactions.")

    if celltype_remove is not None:
        print(f"[cpdb_results_trim] Remove celltype: {celltype_remove}.")
        ct_list = df["celltype_group"].unique().tolist()
        trim = [s for s in ct_list if re.search(celltype_remove, s)]
        print(f"[cpdb_results_trim] Matched celltype: {trim}.")
        ct_list = [s for s in ct_list if s not in trim]
        df = df[df["celltype_group"].isin(ct_list)]
        print(f"[cpdb_results_trim] Remained celltype: {len(ct_list)}.")
    else:
        print("[cpdb_results_trim] Not remove any celltypes.")

    return df


# 来自 Cellphonedb v5，私有化便于维护
def collect_celltype_pairs(
        significant_means: pd.DataFrame,
        query_cell_types_1: list,
        query_cell_types_2: list,
        separator: str) -> list:
    if query_cell_types_1 is None or query_cell_types_2 is None:
        cols_filter = significant_means.filter(regex="\\{}".format(separator)).columns
        all_cts = set([])
        for ct_pair in [i.split(separator) for i in cols_filter.tolist()]:
            all_cts |= set(ct_pair)
        all_cell_types = list(all_cts)
        if query_cell_types_1 is None:
            query_cell_types_1 = all_cell_types
        if query_cell_types_2 is None:
            query_cell_types_2 = all_cell_types
    cell_type_pairs = []
    for ct in query_cell_types_1:
        for ct1 in query_cell_types_2:
            cell_type_pairs += ["{}{}{}".format(ct, separator, ct1), "{}{}{}".format(ct1, separator, ct)]
    return cell_type_pairs

# 来自 Cellphonedb v5，私有化便于维护
def _search_analysis_results(
        query_cell_types_1: list = None,
        query_cell_types_2: list = None,
        query_genes: list = None,
        query_interactions: list = None,
        query_classifications: list = None,
        query_minimum_score: int = None,
        significant_means: pd.DataFrame = None,
        deconvoluted: pd.DataFrame = None,
        interaction_scores=None,
        separator: str = "|",
        long_format: bool = False
) -> pd.DataFrame:
    """
    搜索统计分析或差异表达基因（DEG）分析中与以下任一相关的相互作用：
        1. query_genes 中的基因
        2. 包含 query_genes 中基因的复合物
        3. query_interactions 中的相互作用名称（例如 12oxoLeukotrieneB4_byPTGR1）
    其中至少存在一对细胞类型，包含 query_cell_types_1 中的一个细胞类型和 query_cell_types_2 中的一个细胞类型，且二者均值显著。

    注意：
        如果 query_cell_types_1、query_cell_types_2、query_genes 和 query_interactions 全部设置为 None，
        则返回所有相关的相互作用。
    Parameters
    ----------
    query_cell_types_1: list
        A list of cell types
    query_cell_types_2: list
        A list of cell types
    query_genes: list
        A list of gene names
    query_interactions: list
        A list of interactions
    query_classifications: list
        A list of query classifications
    query_minimum_score: int
        Find all interactions with at least minimum_score across the selected cell types
    significant_means: pd.DataFrame
    deconvoluted: pd.DataFrame
    interaction_scores: pd.DataFrame
        Files output by either (by the same) statistical or DEG analysis
    separator: str
        Separator used in cell type pair column names in significant_means dataFrame
    long_format: bool
        Return the search result DataFrame in long format (while dropping rows with NaN in significant_mean column)
    Returns
    -------
    pd.DataFrame
        Relevant interactions from significant_means that match query criteria
    """

    if significant_means is None or deconvoluted is None:
        print("[_search_analysis_results] ERROR: Both significant_means and deconvoluted dataframes need to be provided")
        return

    # Collect all combinations of cell types (disregarding the order) from query_cell_types_1 and query_cell_types_2
    cell_type_pairs = collect_celltype_pairs(significant_means, query_cell_types_1, query_cell_types_2, separator)
    cols_filter = significant_means.columns[significant_means.columns.isin(cell_type_pairs)]

    # Collect all interactions from query_genes and query_interactions
    interactions = set([])
    if query_genes:
        interactions = interactions.union(frozenset(deconvoluted[deconvoluted['gene_name']
                                                    .isin(query_genes)]['id_cp_interaction'].tolist()))
    if query_interactions:
        interactions = interactions.union(frozenset(significant_means[significant_means['interacting_pair']
                                                    .isin(query_interactions)]['id_cp_interaction'].tolist()))
    if query_classifications:
        interactions = interactions.union(frozenset(significant_means[significant_means['classification']
                                                    .isin(query_classifications)]['id_cp_interaction'].tolist()))
    # If minimum_score was provided, filter interactions to those with at least minimum_score across the selected cell types
    if query_minimum_score is not None and interaction_scores is not None:
        # Filter out interactions which are below query_minimum_score in any cell_type_pair/column in cols_filter
        interactions_filtered_by_minimum_score = interaction_scores[
            interaction_scores[cols_filter].max(axis=1) >= query_minimum_score
        ]['id_cp_interaction'].tolist()
        if query_genes or query_interactions or query_classifications:
            interactions = interactions.intersection(interactions_filtered_by_minimum_score)
        else:
            # Filter all interactions by query_minimum_score
            interactions = set(interactions_filtered_by_minimum_score)
    result_df = significant_means[significant_means['id_cp_interaction'].isin(interactions)]

    # Filter out cell_type_pairs/columns in cols_filter for which no interaction in interactions set is significant
    cols_filter = cols_filter[result_df[cols_filter].notna().any(axis=0)]

    # Filter out interactions which are not significant in any cell_type_pair/column in cols_filter
    result_df = result_df[result_df[cols_filter].notna().any(axis=1)]
    # Select display columns
    result_df = result_df[INTERACTION_COLUMNS + cols_filter.tolist()]

    if long_format:
        # Convert the results DataFrame from (default) wide to long format
        result_df = pd.melt(result_df,
                            id_vars=result_df.columns[0:len(INTERACTION_COLUMNS)],
                            value_vars=result_df.columns[len(INTERACTION_COLUMNS):],
                            value_name='significant_mean',
                            var_name='interacting_cells') \
            .dropna(subset=['significant_mean'])

    print("[_search_analysis_results] Seach finished")
    return result_df

def search_results(CIObject,
                   query_ct1,query_ct2,
                   output_dir=None,file_suffix=None,
                   do_save=False,
                   do_return=True,
                   **kwargs
):
    '''
    对 search_utils.search_analysis_results 进行了简单包装，更方便打印


    :param df: 取
    :param disease_type:
    :param output_dir:
    :param filename:
    :param query_ct1:
    :param query_ct2:
    :param query_min_score:
    :param return_result:
    :param print_result:
    :return:
    '''



    default_param = {"query_cell_types_1":query_ct1, "query_cell_types_2":query_ct2,
                     "query_minimum_score":0.5,
                     "deconvoluted":CIObject.data["deconvoluted"],
                     "significant_means":CIObject.data["significant_means"],
                     "interaction_scores":CIObject.data["interaction_scores"] if CIObject.data["interaction_scores"] is not None else None
                     }

    default_param.update(kwargs)

    search_results = _search_analysis_results(**default_param)

    if search_results.empty:
        print("[search_results] Result is empty.")
        return None

    if do_save and output_dir is not None and file_suffix is not None:
        os.makedirs(output_dir, exist_ok=True)
        excel_name = os.path.join(output_dir + file_suffix + ".xlsx")
        search_results.to_excel(excel_name)
        print(f"[search_results] Xlsx file saved to {excel_name}.")

    if do_return:
        return search_results
    else:
        print("[search_results] Search finished, skip return.")


def prepare_gene_query(self, genes=None, gene_family=None, custom_gene_family=None):
    '''

    Example
    -------
    gene_query = prepare_gene_query(genes=[]) # 默认查询全部基因
    gene_query = prepare_gene_query(genes=["IL2", "IL6"]) # 列表也行
    gene_query = prepare_gene_query(genes="IL2|IL6") # 可正则搜索的字符串也行

    gene_query = prepare_gene_query(gene_family=“th17”) # 内置基因字典，自动搜索相关基因

    gene_query = prepare_gene_query(custom_gene_family={“scavenger_rec”:["^ACKR","^SR-"]})

    # 当然，也可以和 geneset 类联用
    my_markers = Geneset(save_addr + "Markers-updated.xlsx")
    th17_sigs = my_markers.get(siganature='Th17',sheet_name="Immunocyte") # 查询单独 sig，返回一个列表
    prepare_gene_query(genes=th17_sigs)

    # 也可以自定义基因字典；这样，gene_family 就是 custom_gene_family 的 key
    my_marker_dict = ...
    # 或者
    my_marker_dict = my_markers = Geneset(save_addr + "Markers-updated.xlsx")
    thymo_sigs_dict = my_markers.get(siganature='Th17',sheet_name="Immunocyte") # 查询 sheet/多个 sig，返回一个字典
    gene_query = prepare_gene_query(gene_family="Key1", custom_gene_family=my_marker_dict)
    --------------------------------------------------------------------------------------
    :param genes:
    :param gene_family: 自动查询，可选参数 chemokines, th1, th2, th17, treg, costimulatory, coinhibitory
    :param custom_gene_family: [name: [list of genes]] 格式的列表，进行简单的字符串搜索，支持正则
    :return: a list of interacting_pair, which looks like "CDH1_integrin_a2b1_complex"
    '''
    if genes is None:
        if gene_family is not None:
            print("[prepare_gene_query] Mode: Gene_family")
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
            elif isinstance(gene_family, str):  # 是单个字符串，直接提取内置的对应的基因组
                if gene_family.lower() in query_group:
                    query = query_group[gene_family.lower()]
                else:
                    raise KeyError("gene_family needs to be one of the following: {}".format(query_group.keys()))
            else:
                raise ValueError("gene_family needs to be either a list or a string")
        else:  # 构造默认查询，提取 means_mat.interacting_pair 中所有基因（通过空正则表达式匹配）*
            query = [i for i in self.means.interacting_pair if re.search(pattern="", string=i)]
    elif genes is not None:
        print("[prepare_gene_query] Mode: Genes")
        if gene_family is not None:
            raise KeyError("Please specify either genes or gene_family, not both.")
        elif isinstance(genes, list):  # 筛选 self.means.interacting_pair 中匹配 genes 列表的基因
            query = [i for i in self.means.interacting_pair if re.search("|".join(genes), i)]
        elif isinstance(genes, str):
            query = [i for i in self.means.interacting_pair if re.search(genes, i)]
        else:
            raise ValueError("genes needs to be either a list or a string")

    return query

class _prepare_gene_query:
    '''

    Example
    -------
    # 又忘了 gene_family 参数？
    _prepare_gene_query.print()

    '''
    builtin_gene_keys = ["chemokines", "th1", "th2", "th17", "treg", "costimulatory", "coinhibitory"]

    @staticmethod
    def print():
        print("[prepare_gene_query] Allowed keywords for gene_family is:\n "
              + ", ".join(CellphoneInspector._prepare_gene_query.builtin_gene_keys))
