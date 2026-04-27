import gc
import logging
import os
import re
from typing import Any, Iterable

import anndata
import pandas as pd
import scanpy as sc
from anndata import AnnData

from src.utils.env_utils import ensure_package
from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


class ObsEditor:
    """用于编辑 `AnnData.obs` 的代理工具类。

    该对象采用 proxy pattern，将大多数未知属性和切片操作转发给底层
    `AnnData` 对象，便于在保留原始使用习惯的同时补充若干常见的
    `obs` 编辑方法。

    Args:
        adata: 需要包装的 `AnnData` 对象。

    Example:
        editor = ObsEditor(adata)
        adata_cd4 = editor.slice_with_key("Subset_Identity", "CD4+ Th17")

    Notes:
        1. `obs` 中的列如无特别说明，通常理解为 cell subtype 或
           subpopulation。
        2. 本类仍以轻量代理为主，尽量避免改变原始 `AnnData` 的使用方式。
    """

    def __init__(self, adata: AnnData):
        if not isinstance(adata, AnnData):
            raise TypeError("Argument `adata` must be an AnnData object.")
        self._adata = adata
        self.logger = logging.getLogger(self.__class__.__name__)

    def __getattr__(self, attr: str) -> Any:
        """将未知属性转发到底层 `AnnData` 对象。"""
        if attr in self.__dict__:
            return self.__dict__[attr]
        if hasattr(self._adata, attr):
            return getattr(self._adata, attr)
        raise AttributeError(f"'ObsEditor' has no attribute '{attr}'.")

    def __repr__(self) -> str:
        """返回代理对象及底层 `AnnData` 的简要表示。"""
        adata = self.__dict__.get("_adata")
        header = "ObsEditor Proxy (you are working on an AnnData proxy)\n"
        if adata is None:
            return header + "<No underlying AnnData>"
        return header + repr(adata)

    def __setattr__(self, name: str, value: Any) -> None:
        """优先写入底层 `AnnData` 的同名属性。"""
        if name == "_adata":
            super().__setattr__(name, value)
            return

        adata = self.__dict__.get("_adata")
        if adata is not None and hasattr(adata, name):
            setattr(adata, name, value)
        else:
            super().__setattr__(name, value)

    def __getitem__(self, key: Any) -> "ObsEditor":
        """支持切片后返回新的 `ObsEditor` 代理对象。"""
        if isinstance(key, tuple):
            rows, cols = key
            return ObsEditor(self._adata[rows, cols])
        return ObsEditor(self._adata[key])

    def __len__(self) -> int:
        """返回当前 `obs` 的行数。"""
        return len(self.df)

    @property
    def df(self) -> pd.DataFrame:
        """快速访问 `obs` DataFrame。"""
        return self._adata.obs

    @property
    def X(self):
        """返回底层表达矩阵。"""
        return self._adata.X

    @property
    def var_names(self):
        """返回基因名称索引。"""
        return self._adata.var_names

    @property
    def obs_names(self):
        """返回细胞条码索引。"""
        return self._adata.obs_names

    @property
    def columns(self):
        """返回 `obs` 的列名。"""
        return self.df.columns

    @property
    def shape(self):
        """返回 `obs` 的形状。"""
        return self.df.shape

    def subset_cells(self, cells: Iterable[str]) -> "ObsEditor":
        """按细胞集合切片并返回新的代理对象。

        Args:
            cells: 细胞条码列表或可迭代对象。

        Returns:
            切片后的 `ObsEditor` 对象。
        """
        return ObsEditor(self._adata[cells, :])

    def subset_genes(self, genes: Iterable[str]) -> "ObsEditor":
        """按基因集合切片并返回新的代理对象。

        Args:
            genes: 基因名称列表或可迭代对象。

        Returns:
            切片后的 `ObsEditor` 对象。
        """
        return ObsEditor(self._adata[:, genes])

    def add_column(self, col: str, default: Any) -> "ObsEditor":
        """为 `obs` 添加新列。

        Args:
            col: 新列名。
            default: 默认值。

        Returns:
            当前 `ObsEditor` 对象，便于链式调用。
        """
        self._adata.obs[col] = default
        self.logger.info(f"[add_column] Added column `{col}` with default value: '{default}'.")
        return self

    def rename_column(self, old: str, new: str) -> "ObsEditor":
        """重命名 `obs` 列。

        Args:
            old: 原列名。
            new: 新列名。

        Returns:
            当前 `ObsEditor` 对象，便于链式调用。
        """
        if old not in self._adata.obs.columns:
            raise KeyError(f"Column `{old}` was not found in `adata.obs`.")
        self._adata.obs.rename(columns={old: new}, inplace=True)
        self.logger.info(f"[rename_column] Renamed column `{old}` to `{new}`.")
        return self

    def drop_missing(self, col: str) -> AnnData:
        """移除指定列缺失的细胞。

        Args:
            col: 需要检查缺失值的 `obs` 列名。

        Returns:
            过滤后的 `AnnData` 对象。
        """
        if col not in self._adata.obs.columns:
            raise KeyError(f"Column `{col}` was not found in `adata.obs`.")
        mask = self._adata.obs[col].isna() | (self._adata.obs[col] == None)
        self._adata = self._adata[~mask].copy()
        self.logger.info(f"[drop_missing] Dropped {int(mask.sum())} cells with missing value in `{col}`.")
        return self._adata

    def slice_with_key(self, obs_key: str, value: Any, inplace: bool = False) -> AnnData:
        """按 `obs` 列取值过滤细胞。

        Args:
            obs_key: 需要筛选的 `obs` 列名。
            value: 目标取值。支持单个字符串、整数或列表。
            inplace: 是否直接更新当前代理中的 `AnnData`。

        Returns:
            过滤后的 `AnnData` 对象。

        Example:
            adata_subset = ObsEditorClass.slice_with_key("Subset_Identity", "CD4+ Th17")
        """
        if obs_key not in self._adata.obs.columns:
            raise KeyError(f"Column `{obs_key}` was not found in `adata.obs`.")

        if isinstance(value, list):
            adata_tmp = self._adata[self._adata.obs[obs_key].isin(value)].copy()
        elif isinstance(value, (str, int)):
            adata_tmp = self._adata[self._adata.obs[obs_key] == value].copy()
        else:
            raise TypeError("Argument `value` must be a list, string, or integer.")

        if adata_tmp.n_obs == 0:
            self.logger.warning(
                f"[slice_with_key] Warning! No cells were matched for `{obs_key}` with value: '{value}'."
            )
        else:
            self.logger.info(
                f"[slice_with_key] Filtered `{obs_key}` by value: '{value}', matched {adata_tmp.n_obs} cells."
            )

        if inplace:
            self._adata = adata_tmp
        return adata_tmp

    def assign_cluster_identities(self, annotator: list[str] | dict[Any, Any], anno_obs_key: str,
                                  target_obs_key: str) -> None:
        """根据聚类结果写入 cell subtype/subpopulation 注释。

        函数曾用名包括 `make_new_ident` 与 `easy_new_ident`。

        Args:
            annotator: 聚类编号到身份注释的映射，支持 `list` 或 `dict`。
            anno_obs_key: 参考聚类列名，例如 `leiden_res0_5`。
            target_obs_key: 目标输出列名。

        Example:
            ObsEditorClass.assign_cluster_identities(
                annotator=["T_cell", "B_cell", "Mono"],
                anno_obs_key="leiden_res0_5",
                target_obs_key="Subset_Identity"
            )
        """
        if anno_obs_key not in self._adata.obs.columns:
            raise KeyError(f"Column `{anno_obs_key}` was not found in `adata.obs`.")

        cluster_ids = self._adata.obs[anno_obs_key].dropna().unique().tolist()
        try:
            cluster_ids = sorted(cluster_ids, key=lambda x: int(x))
        except Exception:
            cluster_ids = sorted(cluster_ids, key=lambda x: str(x))
            self.logger.warning(
                f"[assign_cluster_identities] Warning! Values in `{anno_obs_key}` are not purely numeric; "
                "fallback to string-based sorting."
            )

        if isinstance(annotator, dict):
            cl_annotation = {str(k): v for k, v in annotator.items()}
            self.logger.info("[assign_cluster_identities] Received `annotator` as a dictionary.")
        elif isinstance(annotator, list):
            if len(annotator) != len(cluster_ids):
                raise ValueError(
                    f"The number of values in `annotator`: {len(annotator)} does not match the number of clusters: "
                    f"{len(cluster_ids)}."
                )
            cl_annotation = dict(zip([str(x) for x in cluster_ids], annotator))
            self.logger.info("[assign_cluster_identities] Received `annotator` as a list.")
            self.logger.info(f"[assign_cluster_identities] Generated mapping: {cl_annotation}")
        else:
            raise TypeError("Argument `annotator` must be a list or dictionary.")

        anno_series = self._adata.obs[anno_obs_key].astype(str)
        self._adata.obs[target_obs_key] = anno_series.map(cl_annotation)

        n_unmapped = int(self._adata.obs[target_obs_key].isna().sum())
        if n_unmapped > 0:
            self.logger.warning(
                f"[assign_cluster_identities] Warning! {n_unmapped} cells were left unmapped in "
                f"`{target_obs_key}`."
            )
        self.logger.info(
            f"[assign_cluster_identities] Identity assignment to `{target_obs_key}` completed successfully."
        )

    def copy_all_ident(self, adata_from: AnnData, from_obs_key: str, to_obs_key: str) -> None:
        """从另一个 `AnnData` 拷贝指定注释列。

        Args:
            adata_from: 来源 `AnnData` 对象，其 `obs.index` 需与当前对象存在交集。
            from_obs_key: 来源列名。
            to_obs_key: 目标列名。

        Example:
            ObsEditor.copy_all_ident(adata_T, "Subset_Identity", "Subset_Identity")
        """
        if not isinstance(adata_from, AnnData):
            raise TypeError("Argument `adata_from` must be an AnnData object.")
        if from_obs_key not in adata_from.obs.columns:
            raise KeyError(f"Column `{from_obs_key}` was not found in source `adata.obs`.")

        obs_data = adata_from.obs[from_obs_key]
        shared_index = obs_data.index.intersection(self._adata.obs.index)
        if len(shared_index) == 0:
            raise ValueError("No shared cell barcodes were found between the two AnnData objects.")

        obs_data = obs_data.loc[shared_index]
        if to_obs_key not in self._adata.obs.columns:
            self._adata.obs[to_obs_key] = None

        for new_ident in obs_data.dropna().unique().tolist():
            index = obs_data[obs_data == new_ident].index
            self._adata.obs.loc[index, to_obs_key] = new_ident
            cell_counts = int((self._adata.obs[to_obs_key] == new_ident).sum())
            self.logger.info(
                f"[copy_all_ident] Updated cell subtype '{new_ident}' into `{to_obs_key}`, total count: "
                f"{cell_counts}."
            )

        n_missing = int(self._adata.obs.loc[shared_index, to_obs_key].isna().sum())
        if n_missing > 0:
            self.logger.warning(
                f"[copy_all_ident] Warning! {n_missing} shared cells still have missing values in "
                f"`{to_obs_key}` after copying."
            )

    def change_one_ident_fast(self, obs_key: str, old: Any, new: Any) -> None:
        """快速替换指定分类列中的取值。

        Args:
            obs_key: 目标 `obs` 列名。
            old: 需要替换的旧值。
            new: 替换后的新值。
        """
        if obs_key not in self._adata.obs.columns:
            raise KeyError(f"Column `{obs_key}` was not found in `adata.obs`.")

        series = self._adata.obs[obs_key]
        if pd.api.types.is_categorical_dtype(series) and new not in series.cat.categories:
            # 仅在必要时扩展类别，避免额外的分类清理开销。
            self._adata.obs[obs_key] = series.cat.add_categories([new])

        mask = self._adata.obs[obs_key] == old
        self._adata.obs.loc[mask, obs_key] = new
        replaced_count = int(mask.sum())
        if replaced_count == 0:
            self.logger.warning(
                f"[change_one_ident_fast] Warning! No cells were matched for `{obs_key}` with old value: '{old}'."
            )
        else:
            self.logger.info(
                f"[change_one_ident_fast] Replaced {replaced_count} cells in `{obs_key}` from '{old}' to '{new}'."
            )

    @logged
    def update_assignment(
        self,
        assignment: str | pd.DataFrame,
        h5ad_dir: str,
        obs_key_col: str = "Obs_key_select",
        subset_file_col: str = "Subset_File",
        subset_no_col: str = "Subset_No",
        identity_col: str = "Identity",
        output_key: str = "Subset_Identity",
        fillna_from_col: str | None = None,
    ) -> None:
        """根据 assignment 表批量更新细胞亚群注释。

        Args:
            assignment: assignment 的 Excel 路径或已经读取好的 DataFrame。
            h5ad_dir: 子集 `.h5ad` 文件所在目录。
            obs_key_col: assignment 中用于指定参考聚类列名的列。
            subset_file_col: assignment 中记录子集文件名的列。
            subset_no_col: assignment 中记录子集聚类编号的列。
            identity_col: assignment 中记录目标身份名称的列。
            output_key: 需要写回当前 `AnnData.obs` 的目标列。
            fillna_from_col: 当 `output_key` 仍有空值时，用于回填的备用列。
        """
        if isinstance(assignment, pd.DataFrame):
            assignment_sheet = assignment.copy()
        elif isinstance(assignment, str):
            if not os.path.exists(assignment):
                raise FileNotFoundError(f"File was not found for `assignment`: '{assignment}'.")
            excel_data = pd.ExcelFile(assignment)
            assignment_sheet = excel_data.parse(excel_data.sheet_names[0])
        else:
            raise TypeError("Argument `assignment` must be a file path or a pandas DataFrame.")

        required_cols = [obs_key_col, subset_file_col, subset_no_col, identity_col]
        missing_cols = [col for col in required_cols if col not in assignment_sheet.columns]
        if missing_cols:
            raise KeyError(f"Required columns are missing in `assignment`: {missing_cols}.")
        if not os.path.isdir(h5ad_dir):
            raise FileNotFoundError(f"Directory was not found for `h5ad_dir`: '{h5ad_dir}'.")

        if output_key not in self._adata.obs.columns:
            self._adata.obs[output_key] = pd.Series(index=self._adata.obs_names, dtype="object")

        subset_filenames = assignment_sheet[subset_file_col].dropna().drop_duplicates().tolist()
        if not subset_filenames:
            self.logger.warning("[update_assignment] Warning! No subset files were found in `assignment`.")
            return

        for subset_filename in subset_filenames:
            self.logger.info(f"[update_assignment] Reading subset file: '{subset_filename}'.")
            input_path = os.path.join(h5ad_dir, subset_filename)
            if not os.path.exists(input_path):
                self.logger.warning(
                    f"[update_assignment] Warning! Subset file was not found and will be skipped: '{input_path}'."
                )
                continue

            adata_subset = anndata.read(input_path)
            subset_data = assignment_sheet[assignment_sheet[subset_file_col] == subset_filename].copy()
            obs_key_series = subset_data[obs_key_col].dropna().drop_duplicates()
            obs_key = obs_key_series.iat[0] if not obs_key_series.empty else None
            if obs_key is None:
                self.logger.warning(
                    f"[update_assignment] Warning! No value was found in `{obs_key_col}` for subset file: "
                    f"'{subset_filename}'."
                )
                del adata_subset
                gc.collect()
                continue
            if obs_key not in adata_subset.obs.columns:
                self.logger.warning(
                    f"[update_assignment] Warning! Column `{obs_key}` was not found in subset file: "
                    f"'{subset_filename}'."
                )
                del adata_subset
                gc.collect()
                continue

            result_dict = subset_data.set_index(subset_no_col)[identity_col].to_dict()
            updated_dict = {str(k): v for k, v in result_dict.items()}
            self.logger.info(
                f"[update_assignment] Built mapping for subset file: '{subset_filename}', total entries: "
                f"{len(updated_dict)}."
            )

            adata_subset.obs[obs_key] = adata_subset.obs[obs_key].astype(str)
            adata_subset.obs["_tmp_identity"] = adata_subset.obs[obs_key].map(updated_dict)
            unique_identities = adata_subset.obs["_tmp_identity"].dropna().unique().tolist()

            if pd.api.types.is_categorical_dtype(self._adata.obs[output_key]):
                existing_categories = set(self._adata.obs[output_key].cat.categories)
                new_categories = set(unique_identities) - existing_categories
                if new_categories:
                    self._adata.obs[output_key] = self._adata.obs[output_key].cat.add_categories(
                        list(new_categories)
                    )

            for cell_identity in unique_identities:
                index = adata_subset.obs_names[adata_subset.obs["_tmp_identity"] == cell_identity]
                self._adata.obs.loc[index, output_key] = cell_identity
                updated_cells = int((self._adata.obs[output_key] == cell_identity).sum())
                self.logger.info(
                    f"[update_assignment] Updated cell subtype '{cell_identity}' into `{output_key}`, total "
                    f"count: {updated_cells}."
                )

            del adata_subset
            gc.collect()

        if fillna_from_col is not None:
            if fillna_from_col not in self._adata.obs.columns:
                self.logger.warning(
                    f"[update_assignment] Warning! Column `{fillna_from_col}` was not found, skip NA filling."
                )
            else:
                n_missing = int(self._adata.obs[output_key].isna().sum())
                self._adata.obs[output_key] = self._adata.obs[output_key].fillna(
                    self._adata.obs[fillna_from_col]
                )
                self.logger.info(
                    f"[update_assignment] Filled {n_missing} missing values in `{output_key}` using "
                    f"`{fillna_from_col}`."
                )

        self.logger.info(f"[update_assignment] All assignments have been applied to `{output_key}`.")


@logged
def gene_annotator(adata: AnnData, chr_annot_path: str | None = None, cyc_annot_path: str | None = None) -> AnnData:
    """为基因与细胞周期信息添加常用注释。

    Args:
        adata: 需要补充注释的 `AnnData` 对象。
        chr_annot_path: 染色体注释文件路径；如为空则尝试在线获取。
        cyc_annot_path: 细胞周期注释文件路径；如为空则尝试在线获取。

    Returns:
        已补充注释的 `AnnData` 对象。

    Example:
        adata = gene_annotator(adata)

    Notes:
        1. 染色体相关注释写入 `adata.var`。
        2. 细胞周期评分写入 `adata.obs`。
        3. 线粒体、核糖体与血红蛋白基因标记写入 `adata.var`。
    """
    if not isinstance(adata, AnnData):
        raise TypeError("Argument `adata` must be an AnnData object.")

    for name, func, path in [
        ("chromosomal", _chr_annotator, chr_annot_path),
        ("cell cycle", _cyc_annotator, cyc_annot_path),
        ("mito/ribo/hb", _mrh_annotator, None),
    ]:
        try:
            func(adata, path) if path is not None or func is not _mrh_annotator else func(adata)
            logger.info(f"[gene_annotator] Successfully annotated {name} information.")
        except Exception as exc:
            logger.warning(f"[gene_annotator] Warning! Failed to annotate {name} information due to: {exc}")

    return adata


@logged
def _chr_annotator(adata: AnnData, chr_annot_path: str | None = None) -> None:
    """为 `adata.var` 添加染色体相关注释。

    Args:
        adata: 需要注释的 `AnnData` 对象。
        chr_annot_path: 染色体注释文件路径；如为空则尝试在线获取。
    """
    if chr_annot_path is None or not os.path.exists(chr_annot_path):
        logger.info("[_chr_annotator] Try retrieving chromosomal information from Ensembl Biomart.")
        try:
            annot = sc.queries.biomart_annotations(
                "hsapiens",
                ["ensembl_gene_id", "external_gene_name", "start_position", "end_position", "chromosome_name"],
            ).set_index("external_gene_name")
            chr_annot_path = (
                os.path.join(os.getcwd(), "HSA_Genome_Annotation.csv")
                if chr_annot_path is None else chr_annot_path
            )
            annot.to_csv(chr_annot_path)
            logger.info(f"[_chr_annotator] Saved chromosomal annotation file to: '{chr_annot_path}'.")
        except Exception as exc:
            logger.warning(
                f"[_chr_annotator] Warning! Failed to retrieve chromosomal information due to: {exc}. "
                "Please download the annotation manually from Ensembl Biomart if needed."
            )
            return
    else:
        annot = pd.read_csv(chr_annot_path).set_index("external_gene_name")
        logger.info(f"[_chr_annotator] Loaded chromosomal annotation file from: '{chr_annot_path}'.")

    adata.var["X"] = adata.var_names.isin(annot.index[annot.chromosome_name == "X"])
    adata.var["Y"] = adata.var_names.isin(annot.index[annot.chromosome_name == "Y"])


@logged
def _cyc_annotator(adata: AnnData, cyc_annot_path: str | None = None) -> None:
    """为 `adata.obs` 添加细胞周期评分。

    Args:
        adata: 需要注释的 `AnnData` 对象。
        cyc_annot_path: 细胞周期注释压缩文件路径；如为空则尝试在线获取。
    """
    if cyc_annot_path is None or not os.path.exists(cyc_annot_path):
        ensure_package("zipfile")
        import zipfile

        logger.info("[_cyc_annotator] Try retrieving cell cycle annotation archive.")
        try:
            ensure_package("pooch")
            import pooch

            path, filename = os.getcwd(), "cell_cycle_vignette_files.zip"
            p_zip = pooch.retrieve(
                "https://www.dropbox.com/s/3dby3bjsaf5arrw/cell_cycle_vignette_files.zip?dl=1",
                known_hash="sha256:6557fe2a2761d1550d41edf61b26c6d5ec9b42b4933d56c3161164a595f145ab",
                path=path,
                fname=filename,
            )
            logger.info(f"[_cyc_annotator] Downloaded cell cycle archive to: '{p_zip}'.")
        except Exception as exc:
            logger.warning(
                f"[_cyc_annotator] Warning! Failed to retrieve cell cycle annotation archive due to: {exc}."
            )
            return
    else:
        import zipfile

        p_zip = cyc_annot_path
        logger.info(f"[_cyc_annotator] Loaded cell cycle archive from: '{p_zip}'.")

    with zipfile.ZipFile(p_zip, "r") as f_zip:
        cell_cycle_genes = zipfile.Path(f_zip, "regev_lab_cell_cycle_genes.txt").read_text().splitlines()

    s_genes = [x for x in cell_cycle_genes[:43] if x in adata.var_names]
    g2m_genes = [x for x in cell_cycle_genes[43:] if x in adata.var_names]
    if len(s_genes) == 0 or len(g2m_genes) == 0:
        logger.warning(
            "[_cyc_annotator] Warning! One or more cell cycle gene sets are empty after matching `adata.var_names`."
        )

    adata.uns["s_genes"] = s_genes
    adata.uns["g2m_genes"] = g2m_genes
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)


@logged
def _mrh_annotator(adata: AnnData) -> None:
    """为 `adata.var` 添加线粒体、核糖体与血红蛋白基因标记。"""
    # 这里使用简单规则快速完成常见基础注释。
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = [bool(re.search(r"^RP[SL]\d|^RPLP\d|^RPSA", x)) for x in adata.var_names]
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
