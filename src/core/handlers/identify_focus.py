import gc
import os
import sys
from typing import Any

import anndata
import pandas as pd
from anndata import AnnData

from src.core.adata.subset_analyze_pipeline import adata_subset_analyze_pipeline

import logging
from src.utils.hier_logger import logged

sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


def _parse_subset_value(value: Any) -> list[str]:
    """将 focus 文件中的 `Subsets` 字段解析为字符串列表。"""
    if isinstance(value, list):
        return [str(item).strip().strip("'").strip('"') for item in value if str(item).strip()]
    if pd.isna(value):
        return []
    text = str(value).strip().strip("[]")
    if not text:
        return []
    return [item.strip().strip("'").strip('"') for item in text.split(",") if item.strip()]


def _parse_bool_like(value: Any) -> bool:
    """将常见的布尔表达转换为 `bool`。"""
    if isinstance(value, str):
        value_lower = value.strip().lower()
        if value_lower in ("true", "t", "1", "yes", "y"):
            return True
        if value_lower in ("false", "f", "0", "no", "n", ""):
            return False
        return False
    return bool(value)


@logged
def make_a_focus(
        adata: AnnData,
        filename: str,
        cat_key: str = "Celltype",
        type_key: str = "Subset_Identity",
        resubset: bool = False,
) -> pd.DataFrame:
    """根据 `adata.obs` 中的分类关系生成 focus 表。

    Args:
        adata: 输入的 AnnData 对象。
        filename: 输出 CSV 文件路径。
        cat_key: `adata.obs` 中表示大类群的列名。
        type_key: `adata.obs` 中表示 cell subtype/subpopulation 的列名。
        resubset: 是否在 focus 表中将 `Resubset` 设为 True。

    Returns:
        生成后的 focus DataFrame。

    Example:
        ```python
        focus_df = make_a_focus(
            adata=adata,
            filename="focus.csv",
            cat_key="Celltype",
            type_key="Subset_Identity",
            resubset=False,
        )
        ```
    """
    if not filename.endswith(".csv"):
        raise ValueError("`filename` must end with '.csv'.")
    if cat_key not in adata.obs.columns:
        raise KeyError(f"`cat_key`: '{cat_key}' does not exist in `adata.obs`.")
    if type_key not in adata.obs.columns:
        raise KeyError(f"`type_key`: '{type_key}' does not exist in `adata.obs`.")

    df = adata.obs.astype(object).copy()[[cat_key, type_key]]
    df[cat_key] = df[cat_key].tolist()
    df[type_key] = df[type_key].tolist()

    df_grouped = (
        df.groupby(cat_key, as_index=False)
        .agg(lambda x: list(dict.fromkeys(value for value in x if pd.notna(value))))
    )
    df_grouped.rename(columns={cat_key: "Name", type_key: "Subsets"}, inplace=True)
    df_grouped["Subsets"] = df_grouped["Subsets"].apply(
        lambda values: ",".join(map(str, values)) if isinstance(values, list) else values
    )
    df_grouped["Resubset"] = resubset
    df_grouped["Marker_class"] = df_grouped["Name"]
    df_grouped.to_csv(filename, index=False)

    logger.info(f"[make_a_focus] Focus table was saved to: {filename}")
    return df_grouped


class IdentifyFocus:
    """根据 focus 表对 AnnData 进行子集拆分和自动化分析。"""

    def __init__(self, focus_file: str, adata: AnnData):
        """初始化 IdentifyFocus 对象。

        Args:
            focus_file: focus 文件路径，支持 `.csv`、`.xlsx`、`.xls`。
            adata: 待处理的 AnnData 对象。

        Example:
            ```python
            focus = IdentifyFocus(focus_file="focus.csv", adata=adata)
            ```

        Notes:
            focus 文件需要至少包含四列：
            `Name`、`Subsets`、`Resubset`、`Marker_class`。
        """
        self.adata = adata
        self.logger = logging.getLogger(self.__class__.__name__)
        self.focus = self._load_focus_file(focus_file)

    @logged
    def _load_focus_file(self, focus_file: str) -> pd.DataFrame:
        """读取并标准化 focus 文件。"""
        if focus_file.endswith(".csv"):
            focus_sheet = pd.read_csv(focus_file)
        elif focus_file.endswith((".xlsx", ".xls")):
            excel_file = pd.ExcelFile(focus_file)
            focus_sheet = excel_file.parse(excel_file.sheet_names[0])
        else:
            raise ValueError("`focus_file` must end with '.csv', '.xlsx', or '.xls'.")

        required_columns = {"Name", "Subsets", "Resubset", "Marker_class"}
        missing_columns = required_columns - set(focus_sheet.columns)
        if missing_columns:
            raise KeyError(f"Required columns are missing from the focus file: {sorted(missing_columns)}.")

        focus_sheet = focus_sheet.copy()
        focus_sheet["Subsets"] = focus_sheet["Subsets"].apply(_parse_subset_value)
        self.logger.info(
            f"[IdentifyFocus._load_focus_file] Focus file was loaded with {len(focus_sheet)} rows."
        )
        return focus_sheet

    @logged
    def filter_and_save_subsets(
            self,
            h5ad_prefix: str,
            save_addr: str,
            obs_key: str = "Subset_Identity",
    ) -> None:
        """按 focus 表中定义的 cell subtype/subpopulation 拆分并保存 h5ad 子集。

        Args:
            h5ad_prefix: 输出 h5ad 文件前缀。
            save_addr: 输出目录。
            obs_key: `adata.obs` 中用于筛选子集的列名。

        Example:
            ```python
            focus.filter_and_save_subsets(
                h5ad_prefix="SubsetSplit",
                save_addr="./subsets",
                obs_key="Subset_Identity",
            )
            ```
        """
        if obs_key not in self.adata.obs.columns:
            raise KeyError(f"`obs_key`: '{obs_key}' does not exist in `adata.obs`.")

        self.obs_key = obs_key
        os.makedirs(save_addr, exist_ok=True)

        for _, row in self.focus.iterrows():
            name = row["Name"]
            subsets = row["Subsets"]

            if not subsets:
                self.logger.info(
                    f"[IdentifyFocus.filter_and_save_subsets] Warning! `Subsets` is empty for focus name: '{name}'. Skipping."
                )
                continue

            index_list = self.adata.obs[obs_key].isin(subsets)
            adata_subset = self.adata[index_list].copy()
            self.logger.info(
                f"[IdentifyFocus.filter_and_save_subsets] Saving subset '{name}' with "
                f"{adata_subset.n_obs} cells from `obs_key`: '{obs_key}'."
            )

            output_path = os.path.join(save_addr, f"{h5ad_prefix}_{name}.h5ad")
            try:
                adata_subset.write(output_path)
                self.logger.info(f"[IdentifyFocus.filter_and_save_subsets] Subset was written to: {output_path}")
            except Exception as exc:
                self.logger.info(
                    f"[IdentifyFocus.filter_and_save_subsets] Warning! Failed to save subset '{name}' "
                    f"to: {output_path}. Details: {exc}"
                )

    @logged
    def process_filtered_files(
            self,
            Geneset_class,
            save_addr: str,
            h5ad_prefix: str,
            **kwargs,
    ) -> None:
        """对已保存的 h5ad 子集执行自动化分析流程。

        Args:
            Geneset_class: 传递给分析流程的 Geneset 对象。
            save_addr: 子集 h5ad 所在目录，也是结果输出根目录。
            h5ad_prefix: 子集 h5ad 文件前缀。
            **kwargs: 透传给 `adata_subset_analyze_pipeline` 的参数。

        Example:
            ```python
            focus.process_filtered_files(
                Geneset_class=my_markers,
                save_addr="./subsets",
                h5ad_prefix="SubsetSplit",
                use_rep="X_scVI",
            )
            ```
        """
        if not hasattr(self, "obs_key"):
            self.obs_key = "Subset_Identity"
            self.logger.info(
                "[IdentifyFocus.process_filtered_files] Warning! `obs_key` is not set via "
                "`filter_and_save_subsets()`. Falling back to 'Subset_Identity'."
            )

        for _, row in self.focus.iterrows():
            name = row["Name"]
            resubset = _parse_bool_like(row["Resubset"])

            self.logger.info(
                f"[IdentifyFocus.process_filtered_files] Processing focus name: '{name}' with "
                f"`Resubset`: {resubset}."
            )

            input_path = os.path.join(save_addr, f"{h5ad_prefix}_{name}.h5ad")
            if not os.path.exists(input_path):
                self.logger.info(
                    f"[IdentifyFocus.process_filtered_files] Warning! Input file does not exist: {input_path}. Skipping."
                )
                continue

            adata_subset = anndata.read_h5ad(input_path)
            output_dir = os.path.join(save_addr, name)
            os.makedirs(output_dir, exist_ok=True)

            default_pars = {
                "resolutions_list": None,
                "use_rep": "X_scVI",
                "use_raw": True,
                "do_DEG_enrich": True,
                "DEG_enrich_key": self.obs_key,
                "do_subcluster": False,
            }
            default_pars.update(**kwargs)

            adata_subset_analyze_pipeline(
                adata_subset=adata_subset,
                filename_prefix=name,
                my_markers=Geneset_class,
                marker_sheet=row["Marker_class"],
                save_addr=output_dir,
                **default_pars,
            )

            if resubset:
                default_pars.update({
                    "DEG_enrich_key": "leiden_res",
                    "do_subcluster": True,
                })
                adata_subset_analyze_pipeline(
                    adata_subset=adata_subset,
                    filename_prefix=name,
                    my_markers=Geneset_class,
                    marker_sheet=row["Marker_class"],
                    save_addr=output_dir,
                    **default_pars,
                )

            adata_subset.write_h5ad(input_path)
            self.logger.info(f"[IdentifyFocus.process_filtered_files] Updated h5ad was written back to: {input_path}")
            del adata_subset
            gc.collect()
