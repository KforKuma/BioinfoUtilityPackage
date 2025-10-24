import pandas as pd
import anndata
import os,gc

'''
主要函数功能来自先前的src.EasyInterface.Anndata_Annotator.py
'''

class ObsEditor:
    """工具类：专门用于编辑 AnnData.obs 的各种操作"""

    def __init__(self, adata: ad.AnnData):
        self.adata = adata

    def add_category(self, col, values):
        """添加一个新的 obs 列"""
        self.adata.obs[col] = values
        return self.adata

    def rename_column(self, old, new):
        """重命名 obs 列"""
        self.adata.obs.rename(columns={old: new}, inplace=True)
        return self.adata

    def drop_missing(self, col):
        """去掉指定列中缺失值的细胞"""
        self.adata = self.adata[~self.adata.obs[col].isna()]
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
            print("Received annotator as a dict.")
            cl_annotation = annotator
        elif isinstance(annotator, list):
            print("Received annotator as a list.")
            cl_annotation = dict(zip(cluster_ids, annotator))
            print("Generate the dict for you, as following: \n", cl_annotation)

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
            print(new_ident)
            index = obs_data[obs_data == new_ident].index
            self.adata.obs.loc[index, to_obs_key] = new_ident
            print(len(self.adata.obs[self.adata.obs[to_obs_key] == new_ident]))

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

            print(f"Replaced {mask.sum()} cells from '{old}' to '{new}'.")
        else:
            mask = self.adata.obs[key] == old
            self.adata.obs.loc[mask, key] = new
            print(f"Replaced {mask.sum()} cells from '{old}' to '{new}'.")











