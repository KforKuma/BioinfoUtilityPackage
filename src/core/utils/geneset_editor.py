import pandas as pd

import logging
from src.utils.hier_logger import logged
logger = logging.getLogger(__name__)

class Geneset():
    """
    用于读写基因标志物，并用于绘图的 class。
    构建 Geneset 可读取的文件时，需要时 xlsx, xls
    """
    def __init__(self, file_path, version=None):
        '''
        Example
        -------
        my_markers = Geneset(save_addr + "Markers-updated.xlsx")

        需要加个版本识别：
        v0: cell_type        gene_set
        v1: cell_type        gene_set        disabled        facet        remark(sheet_name)
        v2: signature_id        genes        status        facet        description        source        species(sheet_name)
        '''
        self.file_path = file_path
        self.data = self._load_file(file_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        if version is not None:
            self.version = version
        else:
            self._detect_version()
            self.logger.info(f"Detected file version: {version}")
        if self.version != "unknown":
            self._migrate_to_current()
        self.data = self._clear_format()
    
    @logged
    def _load_file(self, path):
        if path.endswith(".xlsx") or path.endswith(".xls"):
            sheets = pd.read_excel(path, sheet_name=None) # sheet_name=None for all worksheets.
            df = pd.concat([v.assign(sheet_name=k) for k, v in sheets.items()], ignore_index=True)
            return df
        elif path.endswith(".csv"):
            excel_data = pd.read_csv(path)
            excel_data["sheet_name"] = "default"
            return  excel_data
        else:
            raise ValueError(f"Unsupported file format {path}, must be .xlsx, .xls or .csv")

    def _detect_version(self):
        cols = set(self.data.columns)
        if {"cell_type", "gene_set"}.issubset(cols):
            if {"disabled", "facet", "remark"}.issubset(cols):
                return "v1"
            else:
                return "v0"
        elif {"signature_id", "genes", "status"}.issubset(cols):
            return "v2"
        return "unknown" # unknown is acceptable, for user who only use this wrapper only as a file manager
    
    @logged
    def _migrate_to_current(self):
        if self.version != "v2":
            self.logger.info("Detected v0 or v1 format, migrating to v2...")
            df = self.data.copy()

            # 更新列名
            colname_remap = {
                "cell_type": "signature_id",
                "gene_set": "genes",
                "disabled": "status",
                "remark": "description"
            }
            colname_remap = {k: v for k, v in colname_remap.items() if k in df.columns}
            df = df.rename(colname_remap)

            # 进行默认值处理
            if "status" in df.columns:
                df["status"] = df['status'].replace({'True': True, 'False': False,
                                                     'true': True, 'false': False}).astype(bool)
                df["status"] = df["status"].apply(lambda x: "active" if not x else "archived")
                    # 把先前的 False 转换为 active，其余备用
            else:
                df["status"] = "active"
            df["source"] = None
            df["species"] = None

            # 调整顺序
            desired_cols = ["signature_id", "genes", "status", "species", "source",
                            "facet", "description", "sheet_name"]
            df = df.reindex(columns=[c for c in desired_cols if c in df.columns])

            self.data = df
            self.version = "v2"
            self.logger.info("Migration complete. Upgraded to v2 format.")
    
    @logged
    def _clear_format(self):
        df = self.data.copy()
        df["species"] = df['species'].replace({'human': "Human",
                                               'mouse': "Mouse", 'mice': "Mouse"})
        try:
            df['genes'] = (
                df['gene_set']
                .astype(str)
                .str.strip('[]')
                .str.split(',')
                .apply(lambda lst: [x.strip().strip("'").strip('"') for x in lst])
            )
        except Exception as e:
            self.logger.info(f"Error while parsing gene_set: {e}")
            raise e

        # 基因格式标准化
        df.loc[df["species"] == "Human", "genes"] = df.loc[df["species"] == "Human", "genes"].apply(
            lambda g: [x.upper() for x in g]
        )
        df.loc[df["species"] == "Mouse", "genes"] = df.loc[df["species"] == "Mouse", "genes"].apply(
            lambda g: [x.capitalize() for x in g]
        )

        df_uni = pd.DataFrame()

        for sheet in df["sheet_name"].unique():
            df_s = df[df["sheet_name"] == sheet].copy()
            df_s["uniqueness"] = df_s["signature_id"] + "_" + df_s["status"] + "_" + df_s["species"]

            df_sheet_clean = pd.DataFrame()

            for key, group in df_s.groupby('signature_id', group_keys=False):
                if len(group) == 1:
                    row = group.copy()
                    df_sheet_clean = pd.concat([df_sheet_clean, row], ignore_index=True)
                    continue

                for unique_id, subgroup in group.groupby('uniqueness', group_keys=False):
                    if len(subgroup) > 1:
                        # 完全重复：合并
                        row = subgroup.iloc[0].to_frame().T
                        row['genes'] = ','.join(subgroup['genes'].astype(str).unique())
                        row['description'] = ';'.join(subgroup['description'].astype(str).unique())
                        row['source'] = ';'.join(subgroup['source'].astype(str).unique())
                        row['facet'] = subgroup['facet'].iloc[0]
                        df_sheet_clean = pd.concat([df_sheet_clean, row], ignore_index=True)
                    else:
                        # 半重复：改名
                        row = subgroup.copy()
                        row["signature_id"] = unique_id
                        df_sheet_clean = pd.concat([df_sheet_clean, row], ignore_index=True)

            self.logger.info(f"Sheet [{sheet}] cleaned: {len(df_s)} → {len(df_sheet_clean)} rows.")
            df_sheet_clean = df_sheet_clean.drop("uniqueness", axis=1)
            df_uni = pd.concat([df_uni, df_sheet_clean], ignore_index=True)

        df_uni.sort_values(by=["sheet_name", "facet", "signature_id"], ascending=True, inplace=True)
        return df_uni
    
    @logged
    def save(self,file_name=None):
        '''

        Example
        -------
        gs.save(file_name=f"{save_addr}/gene_markers_updated_251026.xlsx")

        :param file_name:
        :return:
        '''
        if file_name is None:
            file_path = self.file_path
        else:
            file_path = file_name

        if file_path.endswith((".xlsx", ".xls")):
            if "sheet_name" not in self.data.columns:
                self.logger.info("No 'sheet_name' column found, saving all data to single sheet 'default'.")
                with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                    self.data.to_excel(writer, sheet_name="default", index=False)
            else:
                with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                    for value in self.data["sheet_name"].unique():
                        df_sheet = self.data[self.data["sheet_name"] == value]
                        df_sheet.to_excel(writer, sheet_name=value, index=False)
        elif file_path.endswith(".csv"):
            df = self.data.drop("sheet_name", axis=1)
            df.to_csv(file_name, sep='\t', encoding='utf-8', index=False, header=True)
        else:
            raise ValueError(f"Unsupported file format {file_path}, must be .xlsx, .xls or .csv")

        self.logger.info(f"Saved file to {file_path}")
    
    @logged
    def set_value(self, by_dict, value, target_col="status"):
        """
        按条件批量修改 self.data 中的指定列（默认是 'status'）

        Example
        -------
        # 单条件
        gs.set_value({'species': 'Mouse'}, 'archived')

        # 多条件
        gs.set_value({'species': 'Mouse', 'signature_id': ['T Cell','B Cell']}, 'archived')

        # 修改 remark 列
        gs.set_value({'signature_id': 'T Cell'}, 'Not found in all samples', target_col='description')


        Parameters
        ----------
        by_dict : dict
            key 为列名，value 为匹配值（可为 str 或 list）
        value : any
            要设置的新值
        target_col : str, default 'status'
            要更新的列名
        """
        df = self.data

        if target_col not in df.columns:
            raise KeyError(f"Column '{target_col}' not found in data.")

        # 构建综合筛选 mask
        mask = pd.Series(True, index=df.index)
        for k, v in by_dict.items():
            if k not in df.columns:
                raise KeyError(f"Column '{k}' not found in data.")
            if isinstance(v, (list, tuple, set)):
                mask &= df[k].isin(v)
            else:
                mask &= df[k] == v

        count = mask.sum()
        df.loc[mask, target_col] = value
        self.logger.info(f"Set {count} rows where {by_dict} → {target_col} = '{value}'")
    
    @logged
    def get(self, signature=None, sheet_name=None, active_only=True, facet_split=False):
        '''
        本函数的功能比较复杂，在示例中进行详细介绍。

        Example
        -------
        为了服务于 scanpy.pl 可视化方法，其中多个函数都接受 var_names 把单独列表或字典作为参数。
        1. 可以直接用于画图的输出如：
        # 1.1 查询一个 sig，返回一个包含 [gene1, gene2] 的列表
        signatures=gs.get(siganature='T Cell')
        # 1.2 查询多个 sig，返回一个包含 {sig:[genes]} 的字典
        signatures=gs.get(siganature=['B Cell','T Cell'])
        # 1.3 查询一整个 sheet，就算只有一行，默认返回也是一个字典，类似于多个 sig；也可以同时给出两个参数
        signatures=gs.get(siganature=['B Cell','T Cell'], sheet_name="Immunocyte")
        ...对于下游的可视化，通常只需要
        sc.pl.dotplot(
                adata=adata, groupby=groupby_key, layer="log1p_norm", standard_scale="var",
                var_names=signatures
            )

        2. 如果按照 facet_split进行拆分，会返回一个 {facet:[genes]}的一重字典，或 {facet:{sig:[genes]}}的双重字典
        # 2.1 查询一个 sig，但是按 facet 拆分
        gs.get(siganature='T Cell', facet_split=True) # 目前很少有一个 sig 占多行的情况，也不推荐
        # 2.2 查询多个 sig，并按 facet 拆分
        gs.get(siganature=['B Cell','T Cell'], facet_split=True) # 目前很少有一个 sig 占多行的情况，也不推荐
        ...对于下游的可视化，则需要
        for facet_num, facet_markers in gene_dicts.items():
            sc.pl.dotplot(
                adata=adata_subset, groupby=groupby_key, layer="log1p_norm", standard_scale="var",
                var_names=facet_markers
            )


        :param signature:
        :param sheet_name:
        :param active_only:
        :param facet_split:
        :return:
        '''
        df = self.data

        def build_gene_dict(dataframe):
            gene_dict = dict(zip(dataframe['signature_id'], dataframe['genes']))
            return gene_dict

        if active_only:
            df = df[df["status"] == "active"]

        if sheet_name:
            df = df[df["sheet_name"] == sheet_name]

        if isinstance(signature, str):
            # 必须考虑 signature_id 允许重名的情况
            df = df[df["signature_id"] == signature]
            self.logger.info("Retrieved genes in signature id '%s'" % signature)
            return df.tolist()

        elif isinstance(signature, list):
            df = df[df["signature_id"].isin(signature)]
            if facet_split:
                # 按 'facet' 列进行分组，并对每组应用 build_gene_dict
                gene_dicts = {facet: build_gene_dict(sub_df) for facet, sub_df in df.groupby('facet')}
                self.logger.info("Retrieved genes in signature id '%s'" % signature)
                return gene_dicts
            else:
                # 不分组，直接构建 gene_dict
                self.logger.info("Retrieved genes in signature id '%s'" % signature)
                return build_gene_dict(df)
        elif signature is None: # 如果什么都不填，那你肯定是想要整个表格；当然，也有可能已经填了 sheet_name 了，总之 return 一个字典给你
            self.logger.info("No signature ids given.")
            if facet_split:
                # 按 'facet' 列进行分组，并对每组应用 build_gene_dict
                gene_dicts = {facet: build_gene_dict(sub_df) for facet, sub_df in df.groupby('facet')}
                self.logger.info("Retrieved genes, without signature assigned.")
                return gene_dicts
            else:
                self.logger.info("Retrieved genes, without signature assigned.")
                return build_gene_dict(df)
        else:
            raise ValueError("signature must be str or list, or None when sheet_name is provided.")
    
    @logged
    def update(self, gene_dict, inplace=False,
               sheet_name=None, species=None, status=None, facet=None):
        """
        如果手写了一个字典临时使用，可以快速保存在数据中以便复用。
        字典格式应为：{sig1:[genes1], sig2:[genes2]}
        description 和 source 默认不填。

        :param gene_dict:
        :param inplace: 同名的 key (signature_id) 是否替代旧行，否则自动改名
        :param sheet_name: 默认 'default'
        :param species: 默认 'Human'
        :param status: 默认 'active'
        :param facet: 默认 1
        """

        df = self.data.copy()

        # 检查重复
        repeated_name_list = [key for key in gene_dict.keys() if key in df["signature_id"].unique()]

        if not inplace and repeated_name_list:
            from datetime import datetime
            suffix = datetime.now().strftime("%m%d%H%M%S")
            for name in repeated_name_list:
                new_name = f"{name}_{suffix}"
                gene_dict[new_name] = gene_dict[name]
                del gene_dict[name]

        elif inplace:
            df_s = df[df["signature_id"].isin(repeated_name_list)]
            if df_s["signature_id"].value_counts().max() > 1:
                raise ValueError(
                    "Multiple repeated signatures found. "
                    "Please run Geneset._clear_format() first."
                )
            # 删除旧行
            df = df[~df["signature_id"].isin(repeated_name_list)]

        # 统一默认参数
        status = "active" if status is None else status
        sheet_name = "default" if sheet_name is None else sheet_name
        species = "Human" if species is None else species
        facet = 1 if facet is None else int(facet)
        source = ""
        description = ""

        # 写入新行
        for sig, gen in gene_dict.items():
            if isinstance(gen, list):
                gen = ','.join(map(str, gen))
            new_row = pd.DataFrame(
                [[sig, gen, status, species, source, facet, description, sheet_name]],
                columns=["signature_id", "gene_set", "status", "species", "source",
                         "facet", "description", "sheet_name"]
            )
            df = pd.concat([df, new_row], ignore_index=True)

        self.logger.info(f"Geneset updated: {len(self.data)} → {len(df)} rows "
                  f"({len(gene_dict)} signatures added, inplace={inplace})")

        self.data = df

