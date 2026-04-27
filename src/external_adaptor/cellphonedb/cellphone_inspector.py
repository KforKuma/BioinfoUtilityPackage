import logging
import os
import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from src.external_adaptor.cellphonedb.settings import (
    DEFAULT_CELLSIGN_ALPHA,
    DEFAULT_CLASS_COL,
    DEFAULT_COL_START,
    DEFAULT_COLUMNS,
    DEFAULT_CPDB_SEP,
    DEFAULT_SEP,
    DEFAULT_V5_COL_NAMES,
    DEFAULT_V5_COL_START,
    INTERACTION_COLUMNS,
)
from src.external_adaptor.cellphonedb.support import prep_query_group, split_kwargs
from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


class CellphoneInspector:
    """用于读取、整理和筛选 CellPhoneDB 输出结果的类对象。

    该对象主要面向 CellPhoneDB v5 输出目录，支持：

    1. 自动读取 means / pvalues / interaction scores 等表。
    2. 结合 `adata.obs` 元数据构造 cell subtype/subpopulation 及分组信息。
    3. 依据基因集合和 celltype pairs 进行筛选、显著性过滤和层次聚类。
    4. 将结果整理成长表，方便后续 dotplot、heatmap 和自定义统计分析。

    Example:
        # Step 1: 初始化并读取 CellPhoneDB 输出目录
        ci = CellphoneInspector(
            cpdb_outfile=save_dir,
            degs_analysis=False,
            cellsign=False,
        )

        # Step 2: 准备内部矩阵
        ci.prepare_cpdb_tables(add_meta=True)

        # Step 3: 准备元数据，通常直接传入 `adata.obs`
        ci.prepare_cell_metadata(
            metadata=adata.obs,
            celltype_key="Subset_Identity",
            groupby_key="disease",
        )

        # Step 4: 准备基因和 celltype 配对查询
        gene_query = prepare_gene_query(ci, gene_family="th17")
        celltype_pairs = ci.prepare_celltype_pairs(
            cell_type1=["Th17", "Tfh"],
            cell_type2="B cell",
            lock_celltype_direction=True,
        )

        # Step 5: 过滤、聚类并输出长表
        ci.filter_and_cluster(
            gene_query=gene_query,
            celltype_pairs=celltype_pairs,
            keep_significant_only=False,
        )
        ci.format_outcome()
        final_df = ci.outcome["final"]

    Notes:
        `ci.outcome` 常包含：
        `means_matx`、`pvals_matx`、`interact_matx`、`scale`、`alpha`、
        `significance` 和 `final`。
    """

    def __init__(self, cpdb_outfile, degs_analysis: bool = False, cellsign: bool = False):
        """初始化 CellPhoneDB 检查器。

        Args:
            cpdb_outfile: CellPhoneDB 输出目录路径。
            degs_analysis: 是否按 DEG analysis 模式读取结果。
            cellsign: 是否同时读取 CellSign 相关输出。
        """
        self.file_path = cpdb_outfile
        self.deg = degs_analysis
        self.cellsign = cellsign
        self.logger = logging.getLogger(self.__class__.__name__)
        self.meta = None
        self.mode = None
        self.outcome = {}
        self.data = self._load_file(cpdb_outfile)
        self.logger.info(f"[__init__] Initialized with CellPhoneDB output directory: '{cpdb_outfile}'.")

    def _load_file(self, path: str) -> dict:
        """读取 CellPhoneDB 输出目录中的核心结果文件。"""
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory was not found for `cpdb_outfile`: '{path}'.")

        self.logger.info(f"[_load_file] Loading CellPhoneDB output files from: '{path}'.")
        basic_file_keys = {
            "means": "means",
            "deconvoluted_percents": "deconvoluted_percents",
            "deconvoluted": "deconvoluted",
            "interaction_scores": "interaction_scores",
            "pvals": "pvalues",
        }
        if self.deg:
            basic_file_keys["pvals"] = "relevant_interactions"
        if self.cellsign:
            basic_file_keys.update({
                "cellsign_interactions": "CellSign_active_interactions",
                "cellsign_deconvoluted": "CellSign_active_interactions_deconvoluted",
            })

        result = {}
        filelist = os.listdir(path)
        for key, pattern in basic_file_keys.items():
            matches = [name for name in filelist if re.search(rf"^.*({pattern}).*\.txt$", name)]
            if not matches:
                raise FileNotFoundError(
                    f"No file matching pattern '{pattern}' with extension '.txt' was found in: '{path}'."
                )
            if len(matches) > 1:
                self.logger.warning(
                    f"[_load_file] Warning! Multiple files matched pattern '{pattern}'. "
                    f"Use the first file: '{matches[0]}'."
                )
            df = pd.read_csv(os.path.join(path, matches[0]), sep="\t", low_memory=False)
            if matches[0] in filelist:
                filelist.remove(matches[0])
            result[key] = df

        if self.cellsign and (
            "cellsign_interactions" not in result or "cellsign_deconvoluted" not in result
        ):
            self.logger.warning(
                "[_load_file] Warning! CellSign files were not detected completely. Fallback to `self.cellsign = False`."
            )
            self.cellsign = False

        self.logger.info(f"[_load_file] Loaded tables: {list(result.keys())}")
        for key, value in result.items():
            self.logger.info(f"[_load_file] Table `{key}`: {value.shape[0]} rows x {value.shape[1]} columns.")
        return result

    def _prep_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """格式化 means / pvalues / interaction scores 表。

        Args:
            data: 输入原始结果表。

        Returns:
            便于后续分析的 DataFrame。
        """
        dat = data.copy()
        if "id_cp_interaction" not in dat.columns or "interacting_pair" not in dat.columns:
            raise KeyError("Columns `id_cp_interaction` and `interacting_pair` must exist in the input table.")

        dat.index = [x + DEFAULT_SEP * 3 + y for x, y in zip(dat.id_cp_interaction, dat.interacting_pair)]
        dat.columns = [re.sub(f"\\{DEFAULT_CPDB_SEP}", DEFAULT_SEP, col) for col in dat.columns]
        dat.index = [re.sub("_", "-", row) for row in dat.index]
        dat.index = [re.sub("[.]", " ", row) for row in dat.index]
        return dat

    @logged
    def prepare_cpdb_tables(self, add_meta: bool = False, use_cellsign: bool = False) -> None:
        """准备 CellPhoneDB 的内部矩阵对象。

        Args:
            add_meta: 是否额外构造方向性、分类、整合素等映射信息。
            use_cellsign: 是否使用 CellSign interaction 作为主矩阵。

        Example:
            ci.prepare_cpdb_tables(add_meta=True)

            # 若输出目录中包含 CellSign 文件，并且后续想以 CellSign 激活结果为主矩阵
            ci.prepare_cpdb_tables(add_meta=True, use_cellsign=True)
        """
        self.logger.info("[prepare_cpdb_tables] Preparing CellPhoneDB matrices...")
        means_mat = self._prep_table(self.data["means"])
        pvals_mat = self._prep_table(self.data["pvals"])
        interaction_scores = self.data.get("interaction_scores")

        self.means = means_mat
        self.pvals = pvals_mat
        self.logger.info(
            f"[prepare_cpdb_tables] Means table shape: {means_mat.shape}; p-values table shape: {pvals_mat.shape}."
        )

        if means_mat.empty:
            self.logger.warning("[prepare_cpdb_tables] Warning! The means table is empty. Skip downstream preparation.")
            return

        class_col_value = pvals_mat.columns[DEFAULT_CLASS_COL] if len(pvals_mat.columns) > DEFAULT_CLASS_COL else None
        col_start = DEFAULT_V5_COL_START if class_col_value == "classification" else DEFAULT_COL_START
        if means_mat.shape[1] <= col_start:
            self.logger.warning("[prepare_cpdb_tables] Warning! No sample columns were detected after metadata columns.")
            return

        if self.pvals.shape != self.means.shape:
            tmp_pvals_mat = means_mat.iloc[:, :col_start].copy()
            pvals_aligned = pvals_mat.reindex(index=means_mat.index, columns=means_mat.columns[col_start:])
            tmp_pvals_mat = pd.concat([tmp_pvals_mat, pvals_aligned], axis=1)
            fill_value = 0 if self.deg else 1
            tmp_pvals_mat.iloc[:, col_start:] = tmp_pvals_mat.iloc[:, col_start:].fillna(fill_value)
            self.pvals = tmp_pvals_mat

        if self.deg:
            self.pvals.iloc[:, col_start:] = 1 - self.pvals.iloc[:, col_start:]

        if interaction_scores is not None and use_cellsign:
            raise KeyError("Please specify either `interaction_scores` or CellSign interaction, not both.")

        if use_cellsign and self.cellsign:
            cellsign_mat = self._prep_table(self.data["cellsign_interactions"])
            self.matrix = cellsign_mat
            self.mode = "cellsign"
        else:
            if interaction_scores is None:
                self.logger.warning(
                    "[prepare_cpdb_tables] Warning! `interaction_scores` was not found. Fallback to a zero matrix aligned to `means`."
                )
                interaction_scores_mat = means_mat.copy()
                interaction_scores_mat.iloc[:, col_start:] = 0
            else:
                interaction_scores_mat = self._prep_table(interaction_scores)
            self.matrix = interaction_scores_mat
            self.mode = "interaction_scores"

        if col_start == DEFAULT_V5_COL_START and add_meta:
            melted = pd.melt(means_mat, id_vars=DEFAULT_V5_COL_NAMES, var_name="variable")
            keys = (
                melted["id_cp_interaction"] + DEFAULT_SEP * 3 +
                melted["interacting_pair"].str.replace("_", "-") + DEFAULT_SEP * 3 +
                melted["variable"]
            )
            self.meta = {
                "directionality": dict(zip(keys, melted["directionality"].to_numpy())),
                "classification": dict(zip(keys, melted["classification"].to_numpy())),
                "is_integrin": dict(zip(keys, melted["is_integrin"].to_numpy())),
            }
            self.logger.info(
                "[prepare_cpdb_tables] Constructed meta dictionaries for directionality, classification, and integrin."
            )
        else:
            self.meta = None

        self.logger.info("[prepare_cpdb_tables] Finished preparing CellPhoneDB tables.")

    @logged
    def prepare_cell_metadata(self, metadata: pd.DataFrame, celltype_key: str, groupby_key: Optional[str] = None) -> None:
        """准备 CellPhoneDB 查询所需的细胞元数据。

        Args:
            metadata: 通常为 `adata.obs` 的 DataFrame。
            celltype_key: 表示 cell subtype/subpopulation 的列名。
            groupby_key: 需要保留的组学/样本条件列名。

        Example:
            ci.prepare_cell_metadata(
                metadata=adata.obs,
                celltype_key="Subset_Identity",
                groupby_key="disease",
            )
        """
        if not isinstance(metadata, pd.DataFrame):
            raise TypeError("Argument `metadata` must be a pandas DataFrame.")
        if celltype_key not in metadata.columns:
            raise KeyError(f"Column `{celltype_key}` was not found in `metadata`.")
        if groupby_key is not None and groupby_key not in metadata.columns:
            raise KeyError(f"Column `{groupby_key}` was not found in `metadata`.")

        self.logger.info("[prepare_cell_metadata] Preparing cell metadata...")
        metadata = metadata.copy()
        self.celltype_key = celltype_key

        if metadata[celltype_key].dtype.name != "category":
            metadata[celltype_key] = metadata[celltype_key].astype("category")

        if groupby_key is not None:
            self.logger.info(
                f"[prepare_cell_metadata] Group by `{groupby_key}` within cell subtype key `{celltype_key}`."
            )
            if metadata[groupby_key].dtype.name != "category":
                metadata[groupby_key] = metadata[groupby_key].astype("category")
            metadata["_labels"] = [
                f"{group}_{celltype}" for group, celltype in zip(metadata[groupby_key], metadata[celltype_key])
            ]
            metadata["_labels"] = metadata["_labels"].astype("category")
            self.group_info = list(metadata[groupby_key].cat.categories)
        else:
            self.logger.info("[prepare_cell_metadata] No `groupby_key` was provided. Use cell subtype only.")
            metadata["_labels"] = metadata[celltype_key]
            self.group_info = None

        self.cell_metadata = metadata
        self.logger.info(
            f"[prepare_cell_metadata] Prepared metadata with {metadata.shape[0]} cells and columns: {list(metadata.columns)}."
        )

    @logged
    def prepare_celltype_pairs(self, cell_type1, cell_type2, lock_celltype_direction: bool = False) -> list:
        """根据 cell subtype/subpopulation 查询构造 celltype pair 列名。

        Args:
            cell_type1: 左侧查询，支持字符串、字符串列表或正则。
            cell_type2: 右侧查询，支持字符串、字符串列表或正则。
            lock_celltype_direction: 是否固定 `cell_type1 -> cell_type2` 的方向。

        Returns:
            匹配到的 `self.means.columns` 中的配对列名列表。

        Example:
            celltype_pairs = ci.prepare_celltype_pairs(
                cell_type1="Th17",
                cell_type2="Epithelium",
                lock_celltype_direction=True,
            )
        """
        if not hasattr(self, "cell_metadata"):
            raise AttributeError("Please call `prepare_cell_metadata()` before `prepare_celltype_pairs()`.")

        if isinstance(cell_type1, str):
            cell_type1 = [cell_type1]
        if isinstance(cell_type2, str):
            cell_type2 = [cell_type2]

        all_labels = list(self.cell_metadata[self.celltype_key].cat.categories)

        def resolve_celltypes(patterns, all_candidates):
            if "." in patterns:
                return all_candidates
            resolved = set()
            for pattern in patterns:
                matched = [label for label in all_candidates if re.search(pattern, label)]
                resolved.update(matched)
            return sorted(list(resolved))

        real_c1_list = resolve_celltypes(cell_type1, all_labels)
        real_c2_list = resolve_celltypes(cell_type2, all_labels)
        if not real_c1_list or not real_c2_list:
            self.logger.warning("[prepare_celltype_pairs] Warning! No cell subtype matched the provided query patterns.")
            return []

        pair_patterns = []
        for c1 in real_c1_list:
            for c2 in real_c2_list:
                p1 = re.escape(c1)
                p2 = re.escape(c2)
                pair_patterns.append(f"^{p1}{DEFAULT_SEP}{p2}$")
                if not lock_celltype_direction and c1 != c2:
                    pair_patterns.append(f"^{p2}{DEFAULT_SEP}{p1}$")

        if not pair_patterns:
            return []

        combined_query = "|".join(pair_patterns)
        ct_columns = [column for column in self.means.columns if re.search(combined_query, column)]
        self.logger.info(f"[prepare_celltype_pairs] Matched {len(ct_columns)} celltype pairs.")
        return ct_columns

    @logged
    def filter_and_cluster(
        self,
        gene_query,
        celltype_pairs,
        alpha: float = 0.05,
        keep_significant_only: bool = True,
        cluster_rows: bool = True,
        standard_scale: bool = True,
    ) -> None:
        """按基因集和 celltype pairs 筛选并聚类 CPDB 结果矩阵。

        Args:
            gene_query: 目标 `interacting_pair` 列表。
            celltype_pairs: 目标 celltype pair 列名列表。
            alpha: 显著性阈值。
            keep_significant_only: 是否只保留显著结果。
            cluster_rows: 是否对 interaction 行做层次聚类。
            standard_scale: 是否按行进行 0-1 标准化。

        Example:
            gene_query = prepare_gene_query(ci, gene_family="th17")
            celltype_pairs = ci.prepare_celltype_pairs(
                cell_type1="Th17",
                cell_type2="B cell",
                lock_celltype_direction=True,
            )
            ci.filter_and_cluster(
                gene_query=gene_query,
                celltype_pairs=celltype_pairs,
                keep_significant_only=False,
                cluster_rows=True,
                standard_scale=True,
            )
        """
        if gene_query is None or len(gene_query) == 0:
            raise ValueError("Argument `gene_query` must not be empty.")
        if celltype_pairs is None or len(celltype_pairs) == 0:
            raise ValueError("Argument `celltype_pairs` must not be empty.")

        self.logger.info("[filter_and_cluster] Filtering and clustering CellPhoneDB matrices...")
        for df_name in ("means", "pvals", "matrix"):
            if not hasattr(self, df_name):
                raise AttributeError(f"Please call `prepare_cpdb_tables()` before `filter_and_cluster()`. Missing `{df_name}`.")
            if "interacting_pair" not in getattr(self, df_name).columns:
                raise KeyError(f"Table `{df_name}` does not contain required column `interacting_pair`.")

        means_matx = self.means[self.means.interacting_pair.isin(gene_query)][celltype_pairs]
        pvals_matx = self.pvals[self.pvals.interacting_pair.isin(gene_query)][celltype_pairs]
        interact_matx = self.matrix[self.matrix.interacting_pair.isin(gene_query)][celltype_pairs]

        if self.group_info:
            col_order = []
            seen = set()
            for group_name in self.group_info:
                for column in means_matx.columns:
                    if column in seen:
                        continue
                    if re.search(group_name, column):
                        col_order.append(column)
                        seen.add(column)
            for column in means_matx.columns:
                if column not in seen:
                    col_order.append(column)
        else:
            col_order = list(means_matx.columns)

        shared_cols = [
            column for column in col_order
            if column in means_matx.columns and column in pvals_matx.columns and column in interact_matx.columns
        ]
        if len(shared_cols) == 0:
            raise ValueError(
                "No shared columns were found between `means`, `pvals`, and `matrix` after filtering."
            )

        means_matx = means_matx[shared_cols]
        pvals_matx = pvals_matx[shared_cols]
        interact_matx = interact_matx[shared_cols]

        if keep_significant_only:
            pvals_numeric = pvals_matx.apply(pd.to_numeric, errors="coerce")
            keep_mask = pvals_numeric.lt(alpha).any(axis=1)
            keep_rows = pvals_matx.index[keep_mask]
            if len(keep_rows) == 0:
                raise ValueError("No significant rows were found after applying the p-value threshold.")

            pvals_matx = pvals_matx.loc[keep_rows]
            means_matx = means_matx.loc[keep_rows]
            common_rows = [row for row in keep_rows if row in interact_matx.index]
            interact_matx = interact_matx.loc[common_rows] if common_rows else interact_matx.iloc[0:0]
            if interact_matx.shape[0] == 0:
                self.logger.warning(
                    "[filter_and_cluster] Warning! No interaction rows remain after significance filtering. Continue with an empty interaction matrix."
                )
            else:
                self.logger.info(
                    f"[filter_and_cluster] Interaction matrix size after significance filtering: {interact_matx.size}."
                )
        else:
            self.logger.info("[filter_and_cluster] Skip significance filtering because `keep_significant_only` is False.")

        if cluster_rows:
            if means_matx.shape[0] < 2:
                self.logger.info("[filter_and_cluster] Fewer than 2 rows are available. Skip clustering.")
                h_order = list(means_matx.index)
            else:
                self.logger.info("[filter_and_cluster] Performing hierarchical clustering with the safe path.")
                try:
                    h_order = _safe_hclust(means_matx)
                except Exception as exc:
                    self.logger.exception(
                        "[filter_and_cluster] Warning! Hierarchical clustering failed; fallback to the original row order. Error: %s",
                        str(exc),
                    )
                    h_order = list(means_matx.index)

            means_matx = means_matx.reindex(h_order)
            pvals_matx = pvals_matx.reindex(h_order)
            valid_h_order = [row for row in h_order if row in interact_matx.index]
            interact_matx = interact_matx.reindex(valid_h_order) if valid_h_order else interact_matx.iloc[0:0]
        else:
            self.logger.info("[filter_and_cluster] Skip clustering because `cluster_rows` is False.")

        def safe_scale_series(series: pd.Series):
            values = pd.to_numeric(series, errors="coerce")
            if values.isna().all():
                return series * 0
            mn = values.min()
            mx = values.max()
            denom = mx - mn
            if denom == 0 or np.isclose(denom, 0):
                return values - mn
            return (values - mn) / denom

        if standard_scale:
            means_matx = means_matx.apply(safe_scale_series, axis=1)

        means_matx = means_matx.fillna(0)
        pvals_matx = pvals_matx.fillna(np.nan)
        interact_matx = interact_matx.fillna(0)

        self.outcome = {
            "means_matx": means_matx,
            "pvals_matx": pvals_matx,
            "interact_matx": interact_matx,
            "scale": standard_scale,
            "alpha": alpha,
            "significance": keep_significant_only,
        }
        self.logger.info("[filter_and_cluster] Finished with %d rows and %d columns.", means_matx.shape[0], means_matx.shape[1])

    @logged
    def format_outcome(self, cell_name_list: Optional[list] = None, exclude_interactions=None) -> None:
        """将筛选后的矩阵整理为长表格式。

        Args:
            cell_name_list: 可选的完整 cell name 列表，用于稳健拆分 `celltype_group`。
            exclude_interactions: 需要排除的 interaction 名称或列表。

        Example:
            ci.format_outcome(
                cell_name_list=list(adata.obs["Subset_Identity"].cat.categories),
                exclude_interactions=["MIF-CD74"],
            )
            final_df = ci.outcome["final"]
        """
        self.logger.info("[format_outcome] Formatting the final long-format outcome table...")
        if not self.outcome:
            raise AttributeError("Please call `filter_and_cluster()` before `format_outcome()`.")

        value_col = "scaled_means" if self.outcome["scale"] else "means"
        alpha = self.outcome["alpha"]

        df = self.outcome["means_matx"].melt(ignore_index=False).reset_index()
        df.index = df["index"] + DEFAULT_SEP * 3 + df["variable"]
        df.columns = DEFAULT_COLUMNS + [value_col]
        df["celltype_group"] = df["celltype_group"].str.replace(DEFAULT_SEP, "-", regex=False)

        df_pvals = self.outcome["pvals_matx"].melt(ignore_index=False).reset_index()
        df_pvals.index = df_pvals["index"] + DEFAULT_SEP * 3 + df_pvals["variable"]
        df_pvals.columns = DEFAULT_COLUMNS + ["pvals"]
        df["pvals"] = pd.to_numeric(df_pvals["pvals"], errors="coerce")

        df_scores = self.outcome["interact_matx"].melt(ignore_index=False).reset_index()
        df_scores.index = df_scores["index"] + DEFAULT_SEP * 3 + df_scores["variable"]
        df_scores.columns = DEFAULT_COLUMNS + ["scores"]
        df["scores"] = pd.to_numeric(df_scores["scores"], errors="coerce")

        if cell_name_list is None:
            split_cells = df["celltype_group"].astype("category").str.split("-", expand=True)
            df["cell_left"] = split_cells[0]
            df["cell_right"] = split_cells[1]
        elif isinstance(cell_name_list, list) and len(cell_name_list) > 1:
            df[["cell_left", "cell_right"]] = (
                df["celltype_group"].apply(lambda s: split_cellpair(s, cell_name_list)).apply(pd.Series)
            )
            bad = df[df["cell_left"].isna()]
            if len(bad) > 0:
                print("[format_outcome] Warning! `cell_name_list` may be incomplete.")
                print(f"[format_outcome] Unparsed rows: {len(bad)}")
                print(f"[format_outcome] Unparsed celltype groups: {bad['celltype_group'].unique().tolist()}")
        else:
            raise ValueError("Argument `cell_name_list` must be a list with at least 2 entries when provided.")

        df.loc[df[value_col] == 0, value_col] = np.nan
        df["x_means"] = df[value_col]
        df["y_means"] = df[value_col]
        df.loc[df["pvals"] == 0, "pvals"] = 0.001

        mask_sig = df["pvals"] < alpha
        df.loc[mask_sig, "x_means"] = np.nan
        if self.outcome["significance"]:
            df.loc[~mask_sig, "y_means"] = np.nan

        if self.mode == "interaction_scores":
            df.loc[df["scores"] < 1, "x_means"] = np.nan
        elif self.mode == "cellsign" and "cellsign" in df.columns:
            df.loc[df["cellsign"] < 1, "cellsign"] = DEFAULT_CELLSIGN_ALPHA

        if exclude_interactions is not None:
            if not isinstance(exclude_interactions, list):
                exclude_interactions = [exclude_interactions]
            df = df[~df["interaction_group"].isin(exclude_interactions)]
        else:
            self.logger.info("[format_outcome] No interactions were excluded from the final output.")

        df["pvals"] = df["pvals"].astype(float)
        df["neglog10p"] = -np.log10(df["pvals"])
        df.loc[df["pvals"] >= 0.05, "neglog10p"] = 0.0
        df["significant"] = "yes"
        df.loc[df["pvals"] >= alpha, "significant"] = np.nan

        if self.meta is not None:
            df["is_integrin"] = df.index.map(lambda x: self.meta["is_integrin"].get(x, np.nan))
            df["directionality"] = df.index.map(lambda x: self.meta["directionality"].get(x, np.nan))
            df["classification"] = df.index.map(lambda x: self.meta["classification"].get(x, np.nan))

        if df.empty:
            self.logger.info("[format_outcome] The formatted result is empty for the current query.")
        else:
            self.logger.info(f"[format_outcome] Formatted output shape: {df.shape[0]} rows x {df.shape[1]} columns.")

        self.outcome.update({"final": df})


@logged
def aggregate_cpdb_results(cpdb_outfile_list, meta_list, name_list, gene_query, celltype_pairs, **kwargs):
    """整合多个 CellPhoneDB 批次结果。

    Args:
        cpdb_outfile_list: 多个 CellPhoneDB 输出目录列表。
        meta_list: 与输出目录一一对应的元数据列表，通常为多个 `adata.obs`。
        name_list: 批次名称列表。
        gene_query: 目标 `interacting_pair` 列表。
        celltype_pairs: 目标 celltype pair 列名列表。
        **kwargs: 将自动分发到 `prepare_cpdb_tables`、`prepare_cell_metadata`、
            `filter_and_cluster` 和 `format_outcome`。

    Returns:
        合并后的长表 DataFrame。

    Example:
        combined_df = aggregate_cpdb_results(
            cpdb_outfile_list=[cpdb_dir_1, cpdb_dir_2],
            meta_list=[adata1.obs, adata2.obs],
            name_list=["batch1", "batch2"],
            gene_query=gene_query,
            celltype_pairs=celltype_pairs,
            celltype_key="Subset_Identity",
            groupby_key="disease",
            keep_significant_only=False,
        )
    """
    if not (len(cpdb_outfile_list) == len(meta_list) == len(name_list)):
        raise ValueError("Arguments `cpdb_outfile_list`, `meta_list`, and `name_list` must have the same length.")

    default_param = {
        "prepare_cpdb_tables": {"add_meta": False, "use_cellsign": False},
        "prepare_cell_metadata": {"celltype_key": "Subset_Identity", "groupby_key": None},
        "filter_and_cluster": {
            "alpha": 0.05,
            "keep_significant_only": True,
            "cluster_rows": True,
            "standard_scale": True,
        },
        "format_outcome": {"exclude_interactions": None},
    }

    split = split_kwargs(
        CellphoneInspector.prepare_cpdb_tables,
        CellphoneInspector.prepare_cell_metadata,
        CellphoneInspector.filter_and_cluster,
        CellphoneInspector.format_outcome,
        **kwargs,
    )
    print(f"[aggregate_cpdb_results] Split kwargs: {split}")

    final_param = {}
    for key, value in split.items():
        merged = default_param[key].copy()
        merged.update(value)
        final_param[key] = merged

    final_param["filter_and_cluster"].update({"gene_query": gene_query, "celltype_pairs": celltype_pairs})
    df_list = []
    for cpdb_outfile, meta, name in zip(cpdb_outfile_list, meta_list, name_list):
        print(f"[aggregate_cpdb_results] Processing batch: '{name}'.")
        ci = CellphoneInspector(cpdb_outfile=cpdb_outfile)
        ci.prepare_cpdb_tables(**final_param["prepare_cpdb_tables"])
        ci.prepare_cell_metadata(metadata=meta, **final_param["prepare_cell_metadata"])
        ci.filter_and_cluster(**final_param["filter_and_cluster"])
        ci.format_outcome(**final_param["format_outcome"])

        dfx = ci.outcome["final"].copy()
        dfx["batch"] = name
        dfx.index = dfx.index + DEFAULT_SEP * 3 + dfx["batch"]
        df_list.append(dfx)

    if not df_list:
        raise ValueError("No batch result was generated during aggregation.")
    return pd.concat(df_list)


def cpdb_results_trim(df: pd.DataFrame, to_left=None, to_right=None, interaction_remove=None, celltype_remove=None, sep=None):
    """修正和裁剪 CellPhoneDB 标准输出或整合输出。

    Args:
        df: `format_outcome()` 或 `aggregate_cpdb_results()` 的输出表。
        to_left: 需要固定在左侧的 ligand 正则。
        to_right: 需要固定在右侧的 receptor 正则。
        interaction_remove: 需要移除的 interaction 正则。
        celltype_remove: 需要移除的 celltype 正则。
        sep: interaction 组名使用的分隔符。

    Returns:
        修正后的 DataFrame。

    Example:
        # 将以 LG / L 结尾的分子优先放到左侧
        trimmed_df = cpdb_results_trim(
            df=combined_df,
            to_left="LG$|L$",
            interaction_remove="MIF",
        )
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Argument `df` must be a pandas DataFrame.")

    df = df.copy()
    sep = DEFAULT_SEP * 3 if sep is None else sep
    if df["significant"].isnull().all():
        df["significant"] = "no"
        print("[cpdb_results_trim] Warning! No significant outcome was detected.")
    else:
        print("[cpdb_results_trim] Significant outcome was detected.")

    df[["Ligand", "Receptor"]] = df["interaction_group"].str.split(sep, n=1, expand=True)
    if to_left is None and to_right is None:
        print("[cpdb_results_trim] Skip ligand-receptor direction revision.")
    else:
        print("[cpdb_results_trim] Revising ligand-receptor direction...")
        pattern_left = re.compile(to_left, flags=re.IGNORECASE) if to_left is not None else re.compile(r"$^")
        pattern_right = re.compile(to_right, flags=re.IGNORECASE) if to_right is not None else re.compile(r"$^")
        mask_wrong_direction = (
            df["Ligand"].str.contains(pattern_right, na=False)
            & df["Receptor"].str.contains(pattern_left, na=False)
        )
        wrong_idx = df.index[mask_wrong_direction]
        print(f"[cpdb_results_trim] Reversed interactions detected: {len(wrong_idx)}.")
        if len(wrong_idx) > 0:
            df.loc[wrong_idx, ["Ligand", "Receptor"]] = df.loc[wrong_idx, ["Receptor", "Ligand"]].values
            df.loc[wrong_idx, "interaction_group"] = df.loc[wrong_idx, "interaction_group"].apply(
                lambda x: sep.join(x.split(sep)[::-1]) if isinstance(x, str) else x
            )
            print("[cpdb_results_trim] Ligand-receptor direction was corrected.")

    if interaction_remove is not None:
        print(f"[cpdb_results_trim] Remove interactions matching: '{interaction_remove}'.")
        interaction_list = df["interaction_group"].unique().tolist()
        trim = [x for x in interaction_list if re.search(interaction_remove, x)]
        print(f"[cpdb_results_trim] Matched interactions: {trim}.")
        interaction_list = [x for x in interaction_list if x not in trim]
        df = df[df["interaction_group"].isin(interaction_list)]
        print(f"[cpdb_results_trim] Remaining interactions: {len(interaction_list)}.")
    else:
        print("[cpdb_results_trim] No interactions were removed.")

    if celltype_remove is not None:
        print(f"[cpdb_results_trim] Remove celltype groups matching: '{celltype_remove}'.")
        celltype_list = df["celltype_group"].unique().tolist()
        trim = [x for x in celltype_list if re.search(celltype_remove, x)]
        print(f"[cpdb_results_trim] Matched celltype groups: {trim}.")
        celltype_list = [x for x in celltype_list if x not in trim]
        df = df[df["celltype_group"].isin(celltype_list)]
        print(f"[cpdb_results_trim] Remaining celltype groups: {len(celltype_list)}.")
    else:
        print("[cpdb_results_trim] No celltype groups were removed.")
    return df


def collect_celltype_pairs(significant_means: pd.DataFrame, query_cell_types_1: list, query_cell_types_2: list, separator: str) -> list:
    """根据两组 cell types 收集所有可能的配对列名。"""
    if query_cell_types_1 is None or query_cell_types_2 is None:
        cols_filter = significant_means.filter(regex=f"\\{separator}").columns
        all_cts = set()
        for ct_pair in [x.split(separator) for x in cols_filter.tolist()]:
            all_cts |= set(ct_pair)
        all_cell_types = list(all_cts)
        if query_cell_types_1 is None:
            query_cell_types_1 = all_cell_types
        if query_cell_types_2 is None:
            query_cell_types_2 = all_cell_types
    cell_type_pairs = []
    for ct in query_cell_types_1:
        for ct1 in query_cell_types_2:
            cell_type_pairs += [f"{ct}{separator}{ct1}", f"{ct1}{separator}{ct}"]
    return cell_type_pairs


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
    long_format: bool = False,
) -> pd.DataFrame:
    """在 CPDB 统计分析结果中搜索满足条件的交互。"""
    if significant_means is None or deconvoluted is None:
        raise ValueError("Arguments `significant_means` and `deconvoluted` must both be provided.")

    cell_type_pairs = collect_celltype_pairs(significant_means, query_cell_types_1, query_cell_types_2, separator)
    cols_filter = significant_means.columns[significant_means.columns.isin(cell_type_pairs)]

    interactions = set()
    if query_genes:
        interactions |= set(deconvoluted[deconvoluted["gene_name"].isin(query_genes)]["id_cp_interaction"].tolist())
    if query_interactions:
        interactions |= set(significant_means[significant_means["interacting_pair"].isin(query_interactions)]["id_cp_interaction"].tolist())
    if query_classifications:
        interactions |= set(significant_means[significant_means["classification"].isin(query_classifications)]["id_cp_interaction"].tolist())

    if query_minimum_score is not None and interaction_scores is not None and len(cols_filter) > 0:
        interactions_filtered_by_minimum_score = interaction_scores[
            interaction_scores[cols_filter].max(axis=1) >= query_minimum_score
        ]["id_cp_interaction"].tolist()
        interactions = interactions.intersection(interactions_filtered_by_minimum_score) if interactions else set(interactions_filtered_by_minimum_score)

    result_df = significant_means[significant_means["id_cp_interaction"].isin(interactions)]
    if len(cols_filter) > 0:
        cols_filter = cols_filter[result_df[cols_filter].notna().any(axis=0)]
        result_df = result_df[result_df[cols_filter].notna().any(axis=1)]
    result_df = result_df[INTERACTION_COLUMNS + cols_filter.tolist()]

    if long_format:
        result_df = pd.melt(
            result_df,
            id_vars=result_df.columns[0:len(INTERACTION_COLUMNS)],
            value_vars=result_df.columns[len(INTERACTION_COLUMNS):],
            value_name="significant_mean",
            var_name="interacting_cells",
        ).dropna(subset=["significant_mean"])

    print("[_search_analysis_results] Search finished.")
    return result_df


def search_results(
    CIObject,
    query_ct1,
    query_ct2,
    output_dir=None,
    file_suffix=None,
    do_save: bool = False,
    do_return: bool = True,
    **kwargs,
):
    """对 `_search_analysis_results` 的轻量包装。

    Args:
        CIObject: `CellphoneInspector` 实例。
        query_ct1: 第一组 celltype 查询。
        query_ct2: 第二组 celltype 查询。
        output_dir: 输出目录。
        file_suffix: 输出文件名后缀。
        do_save: 是否保存结果为 Excel。
        do_return: 是否返回结果。
        **kwargs: 透传给 `_search_analysis_results`。

    Returns:
        搜索结果 DataFrame；若 `do_return=False` 则返回 `None`。

    Example:
        search_df = search_results(
            CIObject=ci,
            query_ct1=["Th17", "Tfh"],
            query_ct2="B cell",
            do_save=True,
            output_dir=save_addr,
            file_suffix="th17_bcell_search",
        )
    """
    default_param = {
        "query_cell_types_1": query_ct1,
        "query_cell_types_2": query_ct2,
        "query_minimum_score": 0.5,
        "deconvoluted": CIObject.data.get("deconvoluted"),
        "significant_means": CIObject.data.get("means"),
        "interaction_scores": CIObject.data.get("interaction_scores"),
    }
    default_param.update(kwargs)

    results_df = _search_analysis_results(**default_param)
    if results_df.empty:
        print("[search_results] Result is empty.")
        return None

    if do_save and output_dir is not None and file_suffix is not None:
        os.makedirs(output_dir, exist_ok=True)
        excel_name = os.path.join(output_dir, f"{file_suffix}.xlsx")
        results_df.to_excel(excel_name, index=False)
        print(f"[search_results] Excel file was saved to: '{excel_name}'.")

    if do_return:
        return results_df
    print("[search_results] Search finished. Skip return.")
    return None


def prepare_gene_query(self, genes=None, gene_family=None, custom_gene_family=None):
    """根据基因列表或基因家族构造 CPDB interaction 查询。

    Args:
        self: `CellphoneInspector` 实例。
        genes: 基因列表或正则字符串。
        gene_family: 内置或自定义基因家族名称。
        custom_gene_family: 形如 `{family_name: [pattern1, pattern2]}` 的自定义字典。

    Returns:
        匹配到的 `interacting_pair` 列表。

    Example:
        gene_query = prepare_gene_query(ci, genes=["IL2", "IL6"])
        gene_query = prepare_gene_query(ci, genes="IL2|IL6")
        gene_query = prepare_gene_query(ci, gene_family="th17")

        my_markers = Geneset(save_addr + "Markers-updated.xlsx")
        th17_sigs = my_markers.get(signature="Th17", sheet_name="Immunocyte")
        gene_query = prepare_gene_query(ci, genes=th17_sigs)
    """
    if not hasattr(self, "means"):
        raise AttributeError("Please call `prepare_cpdb_tables()` before `prepare_gene_query()`.")

    if genes is None:
        if gene_family is not None:
            print("[prepare_gene_query] Mode: gene family.")
            query_group = prep_query_group(self.means, custom_gene_family)
            if isinstance(gene_family, list):
                query = []
                for family_name in gene_family:
                    key = family_name.lower()
                    if key not in query_group:
                        raise KeyError(f"Argument `gene_family` must be one of: {sorted(query_group.keys())}.")
                    query.extend(query_group[key])
                query = sorted(set(query))
            elif isinstance(gene_family, str):
                key = gene_family.lower()
                if key not in query_group:
                    raise KeyError(f"Argument `gene_family` must be one of: {sorted(query_group.keys())}.")
                query = query_group[key]
            else:
                raise ValueError("Argument `gene_family` must be either a string or a list of strings.")
        else:
            query = list(self.means.interacting_pair.astype(str))
    else:
        print("[prepare_gene_query] Mode: genes.")
        if gene_family is not None:
            raise KeyError("Please specify either `genes` or `gene_family`, not both.")
        if isinstance(genes, list):
            query = [x for x in self.means.interacting_pair.astype(str) if re.search("|".join(genes), x)]
        elif isinstance(genes, str):
            query = [x for x in self.means.interacting_pair.astype(str) if re.search(genes, x)]
        else:
            raise ValueError("Argument `genes` must be either a list or a string.")
    return query


class _prepare_gene_query:
    """显示 `prepare_gene_query` 可用的内置基因家族关键字。"""

    builtin_gene_keys = ["chemokines", "th1", "th2", "th17", "treg", "costimulatory", "coinhibitory"]

    @staticmethod
    def print():
        """打印 `gene_family` 可用的内置关键词。

        Example:
            # 如果忘记了 `gene_family` 的可用关键字
            _prepare_gene_query.print()
        """
        print("[prepare_gene_query] Allowed keywords for `gene_family` are:\n " + ", ".join(_prepare_gene_query.builtin_gene_keys))


@logged
def _safe_hclust(means_df, method: str = "average", metric: str = "euclidean", min_rows_for_clustering: int = 3):
    """更安全的层次聚类包装器。"""
    if means_df.shape[0] < min_rows_for_clustering:
        return list(means_df.index)

    numeric = means_df.applymap(lambda x: np.isreal(x) and not (pd.isna(x) or np.isinf(x)))
    if not numeric.all(axis=None):
        bad = np.where(~numeric.values)
        raise ValueError(
            f"Non-finite or non-numeric values were found in `means_df` at positions: {bad}. "
            "Please clean NaN/Inf values before clustering."
        )

    X = means_df.values.astype(float)
    dists = pdist(X, metric=metric)
    if np.any(np.isnan(dists)) or np.any(np.isinf(dists)):
        raise ValueError("Distance computation produced NaN or Inf values.")

    Z = linkage(dists, method=method)
    leaf_idx = leaves_list(Z)
    return [means_df.index[i] for i in leaf_idx]


def combine_outcome(dfs):
    """合并多个格式化结果并补全未标注 classification。"""
    df_merge = pd.concat(dfs)
    mask = df_merge["directionality"].notna() & df_merge["classification"].isna()
    df_merge.loc[mask, "classification"] = "Unlabeled_" + df_merge.loc[mask, "directionality"]
    return df_merge


def split_cellpair(s, cell_type_set):
    """将 `celltype_group` 拆分为 `(cell_left, cell_right)`。"""
    if pd.isna(s):
        return None, None
    if s.count("-") == 1:
        return tuple(part.strip() for part in s.split("-", 1))
    for celltype in cell_type_set:
        if s.startswith(celltype + "-"):
            right = s[len(celltype) + 1:]
            if right in cell_type_set:
                return celltype, right
    return None, None


def split_and_save(df_merge: pd.DataFrame, save_addr, by_key: str = "classification", zero_omit: bool = True):
    """按指定列拆分合并结果并保存为 CSV。

    Args:
        df_merge: 合并后的 CPDB 结果表。
        save_addr: 输出目录。
        by_key: 用于拆分的列名。
        zero_omit: 是否移除 `scores` 总和为 0 的 interaction-celltype 组合。

    Example:
        split_and_save(
            df_merge=combined_df,
            save_addr=save_addr,
            by_key="classification",
            zero_omit=True,
        )
    """
    from src.external_adaptor.cellphonedb.settings import DEFAULT_SEP
    from src.utils.env_utils import sanitize_filename

    if by_key not in df_merge.columns:
        raise KeyError(f"Column `{by_key}` was not found in `df_merge`.")
    if df_merge[by_key].isna().any():
        raise ValueError(f"Please fix missing values in column `{by_key}` before saving.")

    os.makedirs(save_addr, exist_ok=True)
    df_merge = df_merge.copy()
    if not df_merge.empty and DEFAULT_SEP in str(df_merge["interaction_group"].iloc[0]):
        sep3 = DEFAULT_SEP * 3
        df_merge["interaction_group"] = df_merge["interaction_group"].astype(str).str.rsplit(pat=sep3, n=1).str[-1]

    for key in df_merge[by_key].dropna().unique():
        print(f"[split_and_save] Saving subset '{key}' to: '{save_addr}'.")
        key_name = sanitize_filename(str(key))
        filename = os.path.join(save_addr, f"CPDB_combined_{key_name}.csv")
        df_sub = df_merge[df_merge[by_key] == key].copy()

        if zero_omit:
            print("[split_and_save] Mode: omit zero-score interaction-celltype groups.")
            score_sum = (
                df_sub.groupby(["interaction_group", "celltype_group"])["scores"]
                .sum()
                .reset_index()
            )
            zero_pairs = score_sum[score_sum["scores"] == 0]
            if len(zero_pairs) > 0:
                zero_pairs = zero_pairs.copy()
                zero_pairs["to_remove"] = True
                df_sub = df_sub.merge(
                    zero_pairs[["interaction_group", "celltype_group", "to_remove"]],
                    on=["interaction_group", "celltype_group"],
                    how="left",
                )
                df_sub = df_sub[df_sub["to_remove"] != True].drop(columns=["to_remove"])

        df_sub.to_csv(filename, index=False)
        print(f"[split_and_save] Saved CSV file to: '{filename}'.")
