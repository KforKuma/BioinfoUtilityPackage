import pandas as pd
import anndata
import scanpy as sc
import os,gc,re

from src.utils.env_utils import ensure_package

'''
主要函数功能来自先前的src.EasyInterface.Anndata_Annotator.py
'''

class ObsEditor:
    """工具类：专门用于编辑 AnnData.obs 的各种操作"""

    def __init__(self, adata: ad.AnnData):
        self.adata = adata

    @staticmethod
    def _log(msg):
        print(f"[ObsEditor Message] {msg}")

    def add_category(self, col, default):
        """添加一个新的 obs 列"""
        self.adata.obs[col] = default
        self._log("New category '{}' added".format(col))
        return self.adata

    def rename_column(self, old, new):
        """重命名 obs 列"""
        self.adata.obs.rename(columns={old: new}, inplace=True)
        self._log("Rename column '{}' to '{}'".format(old, new))
        return self.adata

    def drop_missing(self, col):
        """去掉指定列中缺失值的细胞"""
        self.adata = self.adata[~self.adata.obs[col].isna()]
        self._log("Drop missing column '{}'".format(col))
        return self.adata

    def filter_by_value(self, col, value):
        """
        按列值过滤细胞

        Parameters
        ----------
        value ： 接受单独字符串、数字或列表
        使用例：
        adata_subset = ObsEditorClass.filter_by_value("Subset_Identity", "CD4+ Th17")

        """
        if isinstance(value, list):
            self.adata = self.adata[self.adata.obs[col].isin(value)]
        elif isinstance(value, str) | isinstance(value, int):
            self.adata = self.adata[self.adata.obs[col] == value]
        else:
            raise ValueError("Argument value must be list, str or int.")
        self._log("Filter by value successfully: '{}'".format(value))
        return self.adata

    def assign_cluster_identities(self, annotator, anno_obs_key, target_obs_keys):
        """
        将聚类结果的身份注释写入 AnnData.obs 中.
        函数曾用名：make_new_ident.

        使用例：
        ObsEditorClass.assign_cluster_identities(annotator = ["T_cell", "B_cell", "Mono", ...],
                                                 anno_obs_key = "leiden_res0_5",
                                                 target_obs_keys = "Subset_Identity")

        Parameters
        ----------
        annotator : list/dict
            聚类编号对应的身份注释，如 ["T_cell", "B_cell", "Mono", ...]。
            当然一个标准形态的字典更好，如{"0":"T_cell", "1":"B_cell", "2":"Mono",...}
        anno_obs_key : str
            参考聚类列名，例如 "leiden_0.5"。
        target_obs_keys : list[str]
            要新建或更新的 .obs 列名。

        Returns
        -------
        AnnData
            更新后的 AnnData 对象。
        """
        cluster_ids = sorted(map(str, self.adata.obs[anno_obs_key].unique()))
        if len(annotator) != len(cluster_ids):
            raise ValueError(
                f"The number in new identities: ({len(annotator)}) does not match the number of the reference cluster:  ({len(cluster_ids)})."
            )
        if isinstance(annotator, dict):
            self._log("Received annotator as a dict.")
            cl_annotation = annotator
        elif isinstance(annotator, list):
            self._log("Received annotator as a list.")
            cl_annotation = dict(zip(cluster_ids, annotator))
            self._log("Generate the dict for you, as following: \n", cl_annotation)
        else:
            raise ValueError("Argument annotator must be a list or dict.")

        for key in target_obs_keys:
            self.adata.obs[key] = self.adata.obs[anno_obs_key].map(cl_annotation)

        print("Identity assignment done.")

    def copy_all_ident(self, adata_from,from_obs_key, to_obs_key):
        """
        根据另一个 anndata 对象的某一列，更新本 Editor 所包含 anndata 对象的某一列。


        Parameters
        :param adata_from: 另一 anndata 对象，其 obs.index 需包含于原对象。
        :param from_obs_key:
        :param to_obs_key:
        """
        # 提取来源列
        obs_data = adata_from.obs[from_obs_key]

        # 检查是否有共享细胞
        shared_index = obs_data.index.intersection(self.adata.obs.index)
        if len(shared_index) == 0:
            raise ValueError("No cell barcodes shared between two AnnData objects.")

        # 截取只包含共享 index 的部分
        obs_data = obs_data.loc[shared_index]

        # 如果目标列不存在则新建
        if to_obs_key not in self.adata.obs.columns:
            self.adata.obs[to_obs_key] = None  # 或者 np.nan

        # 按照来源列内容更新
        for new_ident in obs_data.unique().tolist():
            index = obs_data[obs_data == new_ident].index
            self.adata.obs.loc[index, to_obs_key] = new_ident
            cell_counts = len(self.adata.obs[self.adata.obs[to_obs_key] == new_ident])
            self._log(f"New cell identity '{new_ident}' updated, total count: {cell_counts}")

    def change_one_ident_fast(self, obs_key, old, new):
        """
        更快速地替换分类列中的值，仅当必要时添加类别，忽略执行缓慢的 remove_categories 操作。
        """
        if pd.api.types.is_categorical_dtype(self.adata.obs[obs_key]):
            if new not in self.adata.obs[obs_key].cat.categories:
                self.adata.obs[obs_key] = self.adata.obs[obs_key].cat.add_categories([new])

            # 布尔索引一次完成
            mask = self.adata.obs[obs_key] == old
            self.adata.obs.loc[mask, obs_key] = new
            self._log(f"Replaced {mask.sum()} cells from '{old}' to '{new}'.")
        else:
            mask = self.adata.obs[key] == old
            self.adata.obs.loc[mask, key] = new
            self._log(f"Replaced {mask.sum()} cells from '{old}' to '{new}'.")



def gene_annotator(adata: AnnData, chr_annot_path=None, cyc_annot_path=None):
    for name, func, path in [
        ("chromosomal", _chr_annotator, chr_annot_path),
        ("cell cycle", _cyc_annotator, cyc_annot_path),
        ("mito/ribo/hb", _mrh_annotator, None),
    ]:
        try:
            func(adata, path)
            print(f"[gene_annotator] {name} info successfully annotated.")
        except Exception as e:
            print(f"[gene_annotator] Failed {name} info annotation due to {e}")

    return adata


def _chr_annotator(adata: AnnData, chr_annot_path: str = None):

    # 数据库准备
    if chr_annot_path is None or not os.path.exists(chr_annot_path):
        print("[gene_annotator] Try retrieving chromosomal info from Ensembl Biomart.")
        try:
            annot = sc.queries.biomart_annotations(
                "hsapiens",
                ["ensembl_gene_id", "external_gene_name", "start_position", "end_position", "chromosome_name"],
            ).set_index("external_gene_name")
            chr_annot_path = os.path.join(os.getcwd(),"HSA_Genome_Annotation.csv") if chr_annot_path is None else chr_annot_path
            annot.to_csv(chr_annot_path)
        except Exception as e:
            print(f"[gene_annotator] Failed to retrieve database because of {e}, \n "
                  f"try downloading manually, from https://www.ensembl.org/biomart/martview")
            return
    else:
        annot = pd.read_csv(chr_annot_path)
        annot = annot.set_index("external_gene_name")

    adata.var["X"] = adata.var_names.isin(annot.index[annot.chromosome_name == "X"])
    adata.var["Y"] = adata.var_names.isin(annot.index[annot.chromosome_name == "Y"])

def _cyc_annotator(adata: AnnData, cyc_annot_path: str = None):
    '''

    :param adata:
    :param cyc_annot_path:
    :return: adata.obs['phase'],adata.obs['G2M_score'], adata.obs['S_score']
    '''
    if cyc_annot_path is None or not os.path.exists(cyc_annot_path):
        ensure_package("zipfile")
        import zipfile
        print("[gene_annotator] Try retrieving chromosomal info from Ensembl Biomart.")
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
            print(f"[gene_annotator] Failed to retrieve database because of {e}, \n "
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

def _mrh_annotator(adata):
    # 基本基因注释
    # mitochondrial genes
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    # adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["ribo"] = [bool(re.search(r"^RP[SL]\d|^RPLP\d|^RPSA", x)) for x in adata.var_names]
    # hemoglobin genes.
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")









