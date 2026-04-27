import os
import sys
from datetime import datetime
from typing import Any, Optional

import pandas as pd

import logging
from src.utils.hier_logger import logged

sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


def _parse_gene_list(value: Any) -> list[str]:
    """将基因列表字段安全解析为字符串列表。"""
    if isinstance(value, list):
        return [str(gene).strip() for gene in value if str(gene).strip()]
    text = str(value).strip().strip("[]")
    if not text or text.lower() == "nan":
        return []
    return [item.strip().strip("'").strip('"') for item in text.split(",") if item.strip()]


def _normalize_status(value: Any) -> str:
    """将不同格式的状态字段统一为 v2 使用的文本状态。"""
    if pd.isna(value):
        return "active"

    text = str(value).strip().lower()
    if text in {"active", "enabled"}:
        return "active"
    if text in {"archived", "disabled"}:
        return "archived"
    if text in {"true", "1"}:
        return "archived"
    if text in {"false", "0"}:
        return "active"
    return str(value).strip()


class Geneset:
    """用于读取、整理、查询和保存基因集表的类对象。

    当前统一内部格式为 v2，核心字段包括：
    `signature_id`、`genes`、`status`、`species`、`source`、
    `facet`、`description`、`sheet_name`。
    """

    def __init__(self, file_path: str, version: Optional[str] = None):
        """初始化 Geneset 对象。

        Args:
            file_path: 基因集文件路径，支持 `.xlsx`、`.xls`、`.csv`。
            version: 手动指定版本；若为空则自动检测。

        Example:
            ```python
            my_markers = Geneset(save_addr + "Markers-updated.xlsx")
            ```

        Notes:
            需要识别的版本格式包括：
            `v0`: `cell_type`, `gene_set`
            `v1`: `cell_type`, `gene_set`, `disabled`, `facet`, `remark`, `sheet_name`
            `v2`: `signature_id`, `genes`, `status`, `facet`, `description`, `source`, `species`, `sheet_name`
        """
        self.file_path = file_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data = self._load_file(file_path)

        if version is not None:
            self.version = version
            self.logger.info(f"[Geneset.__init__] Using user-provided `version`: '{version}'.")
        else:
            self.version = self._detect_version()
            self.logger.info(f"[Geneset.__init__] Detected geneset file `version`: '{self.version}'.")

        if self.version != "v2":
            self._migrate_to_current()

        self.data = self._clear_format()

    @logged
    def _load_file(self, path: str) -> pd.DataFrame:
        """读取基因集文件。"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file does not exist: {path}")

        if path.endswith((".xlsx", ".xls")):
            sheets = pd.read_excel(path, sheet_name=None)
            return pd.concat(
                [sheet_df.assign(sheet_name=sheet_name) for sheet_name, sheet_df in sheets.items()],
                ignore_index=True,
            )

        if path.endswith(".csv"):
            csv_data = pd.read_csv(path)
            csv_data["sheet_name"] = "default"
            return csv_data

        raise ValueError(f"Unsupported file format: {path}. Only '.xlsx', '.xls', and '.csv' are supported.")

    def _detect_version(self) -> str:
        """自动识别输入文件版本。"""
        cols = set(self.data.columns)
        if {"cell_type", "gene_set"}.issubset(cols):
            if {"disabled", "facet", "remark"}.issubset(cols):
                return "v1"
            return "v0"
        if {"signature_id", "genes", "status"}.issubset(cols):
            return "v2"
        return "unknown"

    @logged
    def _migrate_to_current(self) -> None:
        """将旧版基因集表迁移为统一的 v2 结构。"""
        if getattr(self, "version", None) == "v2":
            return

        self.logger.info("[Geneset._migrate_to_current] Warning! Detected legacy geneset format. Starting migration to v2.")
        df = self.data.copy()

        colname_remap = {
            "cell_type": "signature_id",
            "gene_set": "genes",
            "disabled": "status",
            "remark": "description",
        }
        colname_remap = {old: new for old, new in colname_remap.items() if old in df.columns}
        df = df.rename(columns=colname_remap)

        if "genes" not in df.columns:
            df["genes"] = [[] for _ in range(len(df))]
        else:
            df["genes"] = df["genes"].map(_parse_gene_list)

        if "status" not in df.columns:
            df["status"] = "active"
        else:
            df["status"] = df["status"].map(_normalize_status)

        if "sheet_name" not in df.columns:
            df["sheet_name"] = "Sheet1"
        if "facet" not in df.columns:
            df["facet"] = ""
        if "description" not in df.columns:
            df["description"] = ""

        df["source"] = df.get("source", "")
        df["species"] = df.get("species", "Unknown")

        desired_cols = [
            "signature_id",
            "genes",
            "status",
            "species",
            "source",
            "facet",
            "description",
            "sheet_name",
        ]
        self.data = df.reindex(columns=desired_cols)
        self.version = "v2"
        self.logger.info("[Geneset._migrate_to_current] Migration to v2 completed.")

    @logged
    def _clear_format(self) -> pd.DataFrame:
        """清理并标准化 v2 格式基因集表。"""
        df = self.data.copy()

        for col in ["signature_id", "species", "source", "facet", "description", "sheet_name", "status"]:
            if col not in df.columns:
                df[col] = ""

        df["species"] = (
            df["species"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({
                "human": "Human",
                "homo sapiens": "Human",
                "mouse": "Mouse",
                "mice": "Mouse",
            })
        )
        df["species"] = df["species"].replace({"": "Unknown"})
        df["genes"] = df["genes"].map(_parse_gene_list)
        df["status"] = df["status"].map(_normalize_status)

        df.loc[df["species"] == "Human", "genes"] = df.loc[df["species"] == "Human", "genes"].apply(
            lambda genes: [gene.upper() for gene in genes]
        )
        df.loc[df["species"] == "Mouse", "genes"] = df.loc[df["species"] == "Mouse", "genes"].apply(
            lambda genes: [gene.capitalize() for gene in genes]
        )

        for col in ["description", "source", "facet", "signature_id", "sheet_name"]:
            df[col] = df[col].fillna("").astype(str).str.strip()

        df_clean_all = []
        for sheet in df["sheet_name"].unique():
            df_sheet = df[df["sheet_name"] == sheet].copy()
            df_sheet["uniq"] = (
                df_sheet["signature_id"]
                + "|||"
                + df_sheet["status"]
                + "|||"
                + df_sheet["species"]
            )

            cleaned_rows = []
            for signature_id, group in df_sheet.groupby("signature_id", sort=False):
                if len(group) == 1:
                    cleaned_rows.append(group.iloc[0])
                    continue

                for uniq_id, sub in group.groupby("uniq", sort=False):
                    if len(sub) > 1:
                        merged = sub.iloc[0].copy()
                        merged["genes"] = sorted({gene for genes in sub["genes"] for gene in genes})

                        def merge_text(col: str) -> str:
                            unique_values = {str(value).strip() for value in sub[col] if str(value).strip()}
                            return ";".join(sorted(unique_values))

                        merged["description"] = merge_text("description")
                        merged["source"] = merge_text("source")
                        merged["facet"] = sub["facet"].iloc[0]
                        cleaned_rows.append(merged)
                    else:
                        row = sub.iloc[0].copy()
                        row["signature_id"] = uniq_id
                        cleaned_rows.append(row)

            cleaned_df = pd.DataFrame(cleaned_rows)
            cleaned_df.drop(columns=["uniq"], inplace=True, errors="ignore")
            self.logger.info(
                f"[Geneset._clear_format] Sheet '{sheet}' cleaned from {len(df_sheet)} rows to {len(cleaned_df)} rows."
            )
            df_clean_all.append(cleaned_df)

        df_final = pd.concat(df_clean_all, ignore_index=True)
        df_final.sort_values(by=["sheet_name", "facet", "signature_id"], inplace=True)
        return df_final

    @logged
    def save(self, file_name: Optional[str] = None) -> None:
        """保存当前基因集表。

        Args:
            file_name: 输出文件路径；若为空则覆盖原始输入文件。

        Example:
            ```python
            gs.save(file_name=f"{save_addr}/gene_markers_updated_251026.xlsx")
            ```
        """
        file_path = self.file_path if file_name is None else file_name

        if file_path.endswith((".xlsx", ".xls")):
            if "sheet_name" not in self.data.columns:
                self.logger.info("[Geneset.save] Warning! `sheet_name` does not exist. Saving all data into a single sheet named 'default'.")
                with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                    self.data.to_excel(writer, sheet_name="default", index=False)
            else:
                with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
                    for sheet_name in self.data["sheet_name"].unique():
                        df_sheet = self.data[self.data["sheet_name"] == sheet_name].copy()
                        df_sheet.to_excel(writer, sheet_name=str(sheet_name), index=False)
        elif file_path.endswith(".csv"):
            df_export = self.data.drop(columns=["sheet_name"], errors="ignore")
            df_export.to_csv(file_path, sep="\t", encoding="utf-8", index=False, header=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path}. Only '.xlsx', '.xls', and '.csv' are supported.")

        self.logger.info(f"[Geneset.save] Geneset file was saved to: {file_path}")

    @logged
    def set_value(self, by_dict: dict, value: Any, target_col: str = "status") -> None:
        """按条件批量更新指定列的值。

        Args:
            by_dict: 条件字典，键为列名，值为匹配值或匹配值列表。
            value: 需要写入的新值。
            target_col: 需要更新的目标列名。

        Example:
            ```python
            gs.set_value({"species": "Mouse"}, "archived") # 按单一条件，禁用所有应用于小鼠的基因标志物
            gs.set_value({"species": "Mouse", "signature_id": ["T Cell", "B Cell"]}, "archived") # 按多条件进行禁用
            gs.set_value({"signature_id": "T Cell"}, "Not found in all samples", target_col="description") # 修改 description 列
            ```
        """
        df = self.data
        if target_col not in df.columns:
            raise KeyError(f"`target_col`: '{target_col}' does not exist in the geneset table.")

        mask = pd.Series(True, index=df.index)
        for key, match_value in by_dict.items():
            if key not in df.columns:
                raise KeyError(f"Column does not exist in the geneset table: '{key}'.")
            if isinstance(match_value, (list, tuple, set)):
                mask &= df[key].isin(match_value)
            else:
                mask &= df[key] == match_value

        count = int(mask.sum())
        df.loc[mask, target_col] = value
        self.logger.info(
            f"[Geneset.set_value] Updated {count} rows where {by_dict}. "
            f"Set `target_col`: '{target_col}' to value: '{value}'."
        )

    @logged
    def get(
            self,
            signature: Optional[str | list[str]] = None,
            sheet_name: Optional[str] = None,
            active_only: bool = True,
            facet_split: bool = False,
    ):
        """查询基因集并按下游绘图需要返回列表或字典。

        Args:
            signature: 目标 `signature_id`；可为字符串、字符串列表或 None。
            sheet_name: 仅查询指定 sheet。
            active_only: 是否仅保留 `status == 'active'` 的记录。
            facet_split: 是否按 `facet` 字段拆分返回结果。

        Returns:
            根据查询条件返回基因列表、基因字典或 facet 分组字典。

        Example:
            ```python
            ## 基本用法
            # 查询一个 sig，返回一个包含 [gene1, gene2] 的列表
            signatures = gs.get(signature="T Cell")
            # 查询多个 sig，返回一个包含 {sig:[genes]} 的字典
            signatures = gs.get(signature=["B Cell", "T Cell"])
            # 查询一整个 sheet，就算只有一行，默认返回也是一个字典
            signatures = gs.get(signature=["B Cell", "T Cell"], sheet_name="Immunocyte")
            
            # 对于下游的可视化，通常只需要调用 scanpy.pl.dotplot，它兼容列表或字典
            sc.pl.dotplot(
                adata=adata,
                groupby=groupby_key,
                layer="log1p_norm",
                standard_scale="var",
                var_names=signatures,
            )
            
            ## 按照 facet_split 进行拆分，情况略微复杂
            # 返回一个 {facet:[genes]}的一重字典，或 {facet:{sig:[genes]}}的双重字典
            # 取决于 signatures 参数的情况，具体见前述
            gene_dicts = gs.get(signature=["B Cell", "T Cell"], facet_split=True)
            
            # 那么可视化的时候则需要
            for facet_num, facet_markers in gene_dicts.items():
                sc.pl.dotplot(
                    adata=adata_subset,
                    groupby=groupby_key,
                    layer="log1p_norm",
                    standard_scale="var",
                    var_names=facet_markers,
                )
            ```
        """
        df = self.data.copy()

        def build_gene_dict(dataframe: pd.DataFrame) -> dict:
            dataframe = dataframe.sort_index()
            return dict(zip(dataframe["signature_id"], dataframe["genes"]))

        if active_only and "status" in df.columns:
            df = df[df["status"] == "active"]
        if sheet_name is not None:
            df = df[df["sheet_name"] == sheet_name]

        if isinstance(signature, str):
            df = df[df["signature_id"] == signature]
            if df.empty:
                self.logger.info(f"[Geneset.get] Warning! No geneset was found for `signature`: '{signature}'.")
                return []
            self.logger.info(f"[Geneset.get] Retrieved genes for `signature`: '{signature}'.")
            return df["genes"].iloc[0]

        if isinstance(signature, list):
            df = df[df["signature_id"].isin(signature)]
            if facet_split:
                gene_dicts = {facet: build_gene_dict(sub_df) for facet, sub_df in df.groupby("facet", sort=False)}
                self.logger.info("[Geneset.get] Retrieved genes for `signature` list with `facet_split`: True.")
                return gene_dicts
            self.logger.info("[Geneset.get] Retrieved genes for `signature` list.")
            return build_gene_dict(df)

        if signature is None:
            self.logger.info("[Geneset.get] No `signature` was provided. Returning all matched genesets.")
            if facet_split:
                gene_dicts = {facet: build_gene_dict(sub_df) for facet, sub_df in df.groupby("facet", sort=False)}
                return gene_dicts
            return build_gene_dict(df)

        raise ValueError("`signature` must be a string, a list of strings, or `None`.")

    @logged
    def update(
            self,
            gene_dict: dict,
            inplace: bool = False,
            sheet_name: Optional[str] = None,
            species: Optional[str] = None,
            status: Optional[str] = None,
            facet: Optional[int] = None,
    ) -> None:
        """将手写基因字典快速并入当前基因集表。

        Args:
            gene_dict: 形如 `{signature_id: [genes]}` 的字典。
            inplace: 若为 True，则同名 signature 会覆盖旧记录；否则自动重命名。
            sheet_name: 新增记录的 sheet 名称。
            species: 新增记录的物种。
            status: 新增记录的状态。
            facet: 新增记录的 facet 编号。

        Example:
            ```python
            gs.update(
                gene_dict={"sig1": ["GeneA", "GeneB"],
                           "sig2": ["GeneC"]},
                inplace=False,
                sheet_name="default",
                species="Human",
                status="active",
                facet=1,
            )
            ```
        """
        if not isinstance(gene_dict, dict) or not gene_dict:
            raise ValueError("`gene_dict` must be a non-empty dictionary.")

        old_len = len(self.data)
        df = self.data.copy()
        repeated_name_list = [key for key in gene_dict.keys() if key in df["signature_id"].unique()]

        if not inplace and repeated_name_list:
            suffix = datetime.now().strftime("%m%d%H%M%S")
            gene_dict = gene_dict.copy()
            for name in repeated_name_list:
                new_name = f"{name}_{suffix}"
                gene_dict[new_name] = gene_dict.pop(name)
            self.logger.info(
                f"[Geneset.update] Warning! Repeated `signature_id` was detected. "
                f"Auto-renamed {len(repeated_name_list)} signatures."
            )
        elif inplace and repeated_name_list:
            df_existing = df[df["signature_id"].isin(repeated_name_list)]
            if df_existing["signature_id"].value_counts().max() > 1:
                raise ValueError(
                    "Multiple repeated signatures were found. Please run `Geneset._clear_format()` before `inplace=True` update."
                )
            df = df[~df["signature_id"].isin(repeated_name_list)]

        status = "active" if status is None else status
        sheet_name = "default" if sheet_name is None else sheet_name
        species = "Human" if species is None else species
        facet = 1 if facet is None else int(facet)

        new_rows = []
        for signature_id, genes in gene_dict.items():
            genes_list = _parse_gene_list(genes)
            new_rows.append({
                "signature_id": signature_id,
                "genes": genes_list,
                "status": status,
                "species": species,
                "source": "",
                "facet": facet,
                "description": "",
                "sheet_name": sheet_name,
            })

        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        self.data = df
        self.data = self._clear_format()
        self.logger.info(
            f"[Geneset.update] Geneset table updated from {old_len} rows "
            f"to {len(self.data)} rows. Added signatures: {len(new_rows)}. `inplace`: {inplace}."
        )
