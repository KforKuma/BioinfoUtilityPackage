import pandas as pd
import anndata
from anndata import AnnData
import scanpy as sc
import os,gc,re

from src.utils.env_utils import ensure_package

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)

'''
主要函数功能来自先前的src.EasyInterface.Anndata_Annotator.py
并整合了src.ScanpyTools.AnnotationMaker.py

增加了 proxy pattern 将剩余功能转发给 anndata
'''

class ObsEditor:
    """工具类：专门用于编辑 AnnData.obs 的各种操作"""

    def __init__(self, adata: AnnData):
        self._adata = adata
        self.logger = logging.getLogger(self.__class__.__name__)
        
    # ------- 代理未知属性 -------
    def __getattr__(self, attr):
        """Forward all unknown attributes to AnnData"""
        return getattr(self._adata, attr)
        
    # ------- 代理 __setattr__ -------
    def __setattr__(self, name, value):
        # 注意：__setattr__方法内不能直接写 self.key = value，这会循环调用
        if name == "_adata":
            super().__setattr__(name, value)
        else:
            # 若 adata 有这个属性，直接写到 adata 上
            if hasattr(self._adata, name):
                setattr(self._adata, name, value)
            else:
                super().__setattr__(name, value)
    
    def add_column(self, col, default):
        """添加一个新的 obs 列"""
        self._adata.obs[col] = default
        self.logger("New category '{}' added".format(col))
        return self

    def rename_column(self, old, new):
        """重命名 obs 列"""
        self._adata.obs.rename(columns={old: new}, inplace=True)
        self.logger("Rename column '{}' to '{}'".format(old, new))
        return self

    def drop_missing(self, col):
        """去掉指定列中缺失值的细胞"""
        mask = self._adata.obs[col].isna() | (self._adata.obs[col] == None)
        self._adata = self._adata[~mask]
        self.logger("Drop missing column '{}'".format(col))
        return self._adata

    def slice_with_key(self, obs_key, value, inplace=False):
        """
        按列值过滤细胞

        Parameters
        ----------
        value ： 接受单独字符串、数字或列表
        使用例：
        adata_subset = ObsEditorClass.slice_with_key("Subset_Identity", "CD4+ Th17")

        """
        if isinstance(value, list):
            adata_tmp = self._adata[self._adata.obs[obs_key].isin(value)]
        elif isinstance(value, str) | isinstance(value, int):
            adata_tmp = self._adata[self._adata.obs[obs_key] == value]
        else:
            raise ValueError("Argument value must be list, str or int.")
        if inplace:
            self._adata = adata_tmp
        self.logger("Filter by value successfully: '{}'".format(value))
        return self._adata

    def assign_cluster_identities(self, annotator, anno_obs_key, target_obs_key):
        """
        将聚类结果的身份注释写入 AnnData.obs 中.
        函数曾用名：make_new_ident, easy_new_ident.

        使用例：
        ObsEditorClass.assign_cluster_identities(annotator = ["T_cell", "B_cell", "Mono", ...],
                                                 anno_obs_key = "leiden_res0_5",
                                                 target_obs_key = "Subset_Identity")

        Parameters
        ----------
        annotator : list/dict
            聚类编号对应的身份注释，如 ["T_cell", "B_cell", "Mono", ...]。
            当然一个标准形态的字典更好，如{"0":"T_cell", "1":"B_cell", "2":"Mono",...}
        anno_obs_key : str
            参考聚类列名，例如 "leiden_0.5"。
        target_obs_key : str
            要新建或更新的 .obs 列名。

        Returns
        -------
        AnnData
            更新后的 AnnData 对象。
        """
        cluster_ids = sorted(self._adata.obs[anno_obs_key].unique(), key=lambda x: int(x))
        if len(annotator) != len(cluster_ids):
            raise ValueError(
                f"The number in new identities: ({len(annotator)}) does not match the number of the reference cluster:  ({len(cluster_ids)})."
            )
        if isinstance(annotator, dict):
            self.logger.info("Received annotator as a dict.")
            cl_annotation = annotator
        elif isinstance(annotator, list):
            self.logger.info("Received annotator as a list.")
            cl_annotation = dict(zip(cluster_ids, annotator))
            self.logger.info(f"Generate the dict for you, as following: \n{cl_annotation}")
        else:
            raise ValueError("Argument annotator must be a list or dict.")

        self._adata.obs[target_obs_key] = self._adata.obs[anno_obs_key].map(cl_annotation)
        
        self.logger.info("Identity assignment done.")

    def copy_all_ident(self, adata_from,from_obs_key, to_obs_key):
        """
        根据另一个 anndata 对象的某一列，更新本 Editor 所包含 anndata 对象的某一列。

        Example
        -------
        ObsEditor.copy_all_ident(adata_T, "Subset_Identity","Subset_Identity")

        Parameters
        :param adata_from: 另一 anndata 对象，其 obs.index 需包含于原对象。
        :param from_obs_key:
        :param to_obs_key:
        """
        # 提取来源列
        obs_data = adata_from.obs[from_obs_key]

        # 检查是否有共享细胞
        shared_index = obs_data.index.intersection(self._adata.obs.index)
        if len(shared_index) == 0:
            raise ValueError("No cell barcodes shared between two AnnData objects.")

        # 截取只包含共享 index 的部分
        obs_data = obs_data.loc[shared_index]

        # 如果目标列不存在则新建
        if to_obs_key not in self._adata.obs.columns:
            self._adata.obs[to_obs_key] = None  # 或者 np.nan

        # 按照来源列内容更新
        for new_ident in obs_data.unique().tolist():
            index = obs_data[obs_data == new_ident].index
            self._adata.obs.loc[index, to_obs_key] = new_ident
            cell_counts = len(self._adata.obs[self._adata.obs[to_obs_key] == new_ident])
            self.logger.info(f"New cell identity '{new_ident}' updated, total count: {cell_counts}")

    def change_one_ident_fast(self, obs_key, old, new):
        """
        更快速地替换分类列中的值，仅当必要时添加类别，忽略执行缓慢的 remove_categories 操作。
        """
        if pd.api.types.is_categorical_dtype(self._adata.obs[obs_key]):
            if new not in self._adata.obs[obs_key].cat.categories:
                self._adata.obs[obs_key] = self._adata.obs[obs_key].cat.add_categories([new])

            # 布尔索引一次完成
            mask = self._adata.obs[obs_key] == old
            self._adata.obs.loc[mask, obs_key] = new
            self.logger.info(f"Replaced {mask.sum()} cells from '{old}' to '{new}'.")
        else:
            mask = self._adata.obs[obs_key] == old
            self._adata.obs.loc[mask, obs_key] = new
            self.logger.info(f"Replaced {mask.sum()} cells from '{old}' to '{new}'.")
    
    @logged
    def update_assignment(self,
                          assignment: str or pd.DataFrame,
                          h5ad_dir: str,
                          obs_key_col: str = "Obs_key_select",
                          subset_file_col: str = "Subset_File",
                          subset_no_col: str = "Subset_No",
                          identity_col: str = "Identity",
                          output_key: str = "Subset_Identity",
                          fillna_from_col: str = None
    ):
        """
        根据 assignment 表格更新主 AnnData 对象中的细胞亚群注释。

        参数:
        - assignment_file: assignment Excel 文件路径
        - h5ad_dir: 子集 h5ad 文件所在目录
        - obs_key_col, subset_file_col, subset_no_col, identity_col: assignment 表格中对应的列名
            - obs_key_col: remap 的 key 的列
            - subset_file_col: 对应的 .h5ad 文件名
            - subset_no_col: obs_key_col 里的值
            - identity_col: 新的定义
        - output_key: 主 AnnData 中需要写入的列名
        - fillna_from_col: 用于填充 output_key 中空值的备用列；比如，当 subset_no_col 不包含所有行，会依赖原 dataframe 的此列进行填充
        """
        if isinstance(assignment, pd.DataFrame):
            assignment_sheet = assignment
        elif isinstance(assignment, str):
            excel_data = pd.ExcelFile(assignment)
            assignment_sheet = excel_data.parse(excel_data.sheet_names[0])
        else:
            raise ValueError("Assignment must either be string or a dataframe.")
        
        for subset_filename in set(assignment_sheet[subset_file_col]):
            self.logger.info(f"Now reading {subset_filename} subset.")
            input_path = f"{h5ad_dir}/{subset_filename}"
            adata_subset = anndata.read(input_path)
            
            # 提取 obs_key
            obs_key_series = assignment_sheet.loc[
                assignment_sheet[subset_file_col] == subset_filename, obs_key_col
            ].dropna().drop_duplicates()
            obs_key = obs_key_series.iat[0] if not obs_key_series.empty else None
            self.logger.info(f"Obs key for {subset_filename}: {obs_key}")
            
            # identity 映射字典
            subset_data = assignment_sheet[assignment_sheet[subset_file_col] == subset_filename]
            result_dict = subset_data.set_index(subset_no_col)[identity_col].to_dict()
            updated_dict = {str(k): v for k, v in result_dict.items()}
            self.logger.info(f"Created identity dictionary for {subset_filename} with {len(updated_dict)} entries")
            adata_subset.obs[obs_key] = adata_subset.obs[obs_key].astype(str) # 确保全都是字符串
            adata_subset.obs["tmp"] = adata_subset.obs[obs_key].map(updated_dict)
            unique_identities = adata_subset.obs["tmp"].dropna().unique()
            
            # 如果 output_key 不存在，则初始化为空列
            if output_key not in self._adata.obs.columns:
                self._adata.obs[output_key] = pd.Series(index=self._adata.obs_names, dtype="str")
            
            # 处理 Categorical 类型的列，扩展类别
            if pd.api.types.is_categorical_dtype(self._adata.obs[output_key]):
                existing_categories = set(self._adata.obs[output_key].cat.categories)
                new_categories = set(unique_identities) - existing_categories
                if new_categories:
                    self._adata.obs[output_key] = self._adata.obs[output_key].cat.add_categories(list(new_categories))
            
            for cell_identity in unique_identities:
                self.logger.info(f"Processing identity: {cell_identity}")
                index = adata_subset.obs_names[adata_subset.obs["tmp"] == cell_identity]
                self._adata.obs.loc[index, output_key] = cell_identity
                updated_cells = (self._adata.obs[output_key] == cell_identity).sum()
                self.logger.info(f"Updated {updated_cells} cells with identity '{cell_identity}'")
            
            del adata_subset
        
        # 用其他列补全缺失值
        if fillna_from_col and fillna_from_col in self._adata.obs.columns:
            n_missing = self._adata.obs[output_key].isna().sum()
            self._adata.obs[output_key] = self._adata.obs[output_key].fillna(self._adata.obs[fillna_from_col])
            self.logger.info(f"Filled {n_missing} missing '{output_key}' values using '{fillna_from_col}'")
        
        self.logger.info("All assignments applied.")

@logged
def gene_annotator(adata: AnnData, chr_annot_path=None, cyc_annot_path=None):
    """
    Annotate genes with chromosomal, cell cycle and mito/ribo/hb information.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to be annotated.
    chr_annot_path : str
        The path to the chromosomal information file. If not given, it will be downloaded from Ensembl Biomart.
    cyc_annot_path : str
        The path to the cell cycle information file. If not given, it will be downloaded from the vignette server.

    Example
    -------
    adata = gene_annotator(adata)

    Returns
    -------
    adata : AnnData
        The AnnData object with chromosomal, cell cycle and mito/ribo/hb information annotated.

    Notes
    -----
    The chromosomal information will be annotated to adata.var. The key "X" will be set to True for genes on the X chromosome, and the key "Y" will be set to True for genes on the Y chromosome.
    The cell cycle information will be annotated to adata.obs. The key "phase" will store the phase information, the key "G2M_score" will store the G2/M score, and the key "S_score" will store the S score.
    The mito/ribo/hb information will be annotated to adata.var. The key "mt" will be set to True for mitochondrial genes, the key "ribo" will be set to True for ribosomal genes, and the key "hb" will be set to True for hemoglobin genes.
    """
    for name, func, path in [
        ("chromosomal", _chr_annotator, chr_annot_path),
        ("cell cycle", _cyc_annotator, cyc_annot_path),
        ("mito/ribo/hb", _mrh_annotator, None),
    ]:
        try:
            func(adata, path)
            logger.info(f"[gene_annotator] {name} info successfully annotated.")
        except Exception as e:
            logger.info(f"[gene_annotator] Failed {name} info annotation due to {e}")

    return adata

@logged
def _chr_annotator(adata: AnnData, chr_annot_path: str = None):
    """
    Annotates chromosomal information to adata.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to be annotated.
    chr_annot_path : str
        The path to the chromosomal information file. If not given, it will be downloaded from Ensembl Biomart.

    Returns
    -------
    None

    Notes
    -----
    The chromosomal information will be annotated to adata.var. The key "X" will be set to True for genes on the X chromosome, and the key "Y" will be set to True for genes on the Y chromosome.
    """
    if chr_annot_path is None or not os.path.exists(chr_annot_path):
        logger.info("Try retrieving chromosomal info from Ensembl Biomart.")
        try:
            annot = sc.queries.biomart_annotations(
                "hsapiens",
                ["ensembl_gene_id", "external_gene_name", "start_position", "end_position", "chromosome_name"],
            ).set_index("external_gene_name")
            chr_annot_path = os.path.join(os.getcwd(),"HSA_Genome_Annotation.csv") if chr_annot_path is None else chr_annot_path
            annot.to_csv(chr_annot_path)
        except Exception as e:
            logger.info(f"Failed to retrieve database because of {e}, \n "
                  f"try downloading manually, from https://www.ensembl.org/biomart/martview")
            return
    else:
        annot = pd.read_csv(chr_annot_path)
        annot = annot.set_index("external_gene_name")

    adata.var["X"] = adata.var_names.isin(annot.index[annot.chromosome_name == "X"])
    adata.var["Y"] = adata.var_names.isin(annot.index[annot.chromosome_name == "Y"])

@logged
def _cyc_annotator(adata: AnnData, cyc_annot_path: str = None):
    '''

    :param adata:
    :param cyc_annot_path:
    :return: adata.obs['phase'],adata.obs['G2M_score'], adata.obs['S_score']
    '''
    if cyc_annot_path is None or not os.path.exists(cyc_annot_path):
        ensure_package("zipfile")
        import zipfile
        logger.info("Try retrieving chromosomal info from Ensembl Biomart.")
        try:
            ensure_package("pooch")
            import pooch
            path, filename = os.getcwd(), "cell_cycle_vignette_files.zip"
            p_zip = pooch.retrieve(
                "https://www.dropbox.com/s/3dby3bjsaf5arrw/cell_cycle_vignette_files.zip?dl=1",
                known_hash="sha256:6557fe2a2761d1550d41edf61b26c6d5ec9b42b4933d56c3161164a595f145ab",
                path=path,fname=filename
            )
        except Exception as e:
            logger.info(f"Failed to retrieve database because of {e}, \n "
                  f"try downloading manually, from https://www.ensembl.org/biomart/martview")
            return
    else:
        p_zip = cyc_annot_path

    with zipfile.ZipFile(p_zip, "r") as f_zip:
        cell_cycle_genes = zipfile.Path(f_zip, "regev_lab_cell_cycle_genes.txt").read_text().splitlines()

    s_genes = [x for x in cell_cycle_genes[:43] if x in adata.var_names]
    g2m_genes = [x for x in cell_cycle_genes[43:] if x in adata.var_names]
    adata.uns["s_genes"] = s_genes
    adata.uns["g2m_genes"] = g2m_genes
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)

@logged
def _mrh_annotator(adata):
    # 基本基因注释
    # mitochondrial genes
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    # adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["ribo"] = [bool(re.search(r"^RP[SL]\d|^RPLP\d|^RPSA", x)) for x in adata.var_names]
    # hemoglobin genes.
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")









