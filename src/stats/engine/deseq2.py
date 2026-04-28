from __future__ import annotations
from typing import Dict, Any
import logging

import hashlib
import numpy as np
import pandas as pd

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

from src.stats.support import make_result, parse_formula_columns

from src.utils.hier_logger import logged_class

logger = logging.getLogger(__name__)

@logged_class
class PyDESeq2Manager:
    """缓存式 PyDESeq2 丰度统计管理器。

    PyDESeq2 一次会对所有 cell subtype/subpopulation 建模，因此本类在首次调用时
    全量拟合并缓存结果，后续不同 ``cell_type`` 的请求只从缓存中提取对应行。

    Example:
        >>> res = run_PyDESeq2(
        ...     df_all=abundance_df,
        ...     cell_type="CD4_Tcm",
        ...     formula="disease + tissue",
        ...     main_variable="disease",
        ...     ref_label="HC",
        ...     group_label="sample_id",
        ... )
        >>> res["contrast_table"][["Coef.", "P>|z|", "direction"]]
        # Coef. 已从 log2FoldChange 转换到自然对数尺度，便于和 CLR 方法比较。
    """

    def __init__(self):
        """初始化缓存容器。"""
        self.last_data_hash = None
        self.cached_results = {}  # 存储 {other_label: results_df}
        self.current_cell_type = None
        self.ref_label = "HC"
        self.method_name = "PyDESeq2"
    
    def _get_data_hash(self, df_all, formula):
        """根据输入数据和公式生成缓存键。"""
        # 使用 pandas 行哈希是为了同时感知元数据和 count 改动，避免复用旧模型。
        content_hash = hashlib.md5(pd.util.hash_pandas_object(df_all).values).hexdigest()
        return f"{content_hash}_{formula}"
    
    def __call__(self, df_all: pd.DataFrame, cell_type: str, formula: str = "disease",
                 main_variable: str = None, ref_label: str = "HC",
                 group_label: str = "sample_id", alpha: float = 0.05, **kwargs) -> Dict[str, Any]:
        """运行或复用 PyDESeq2 全量拟合，并提取目标亚群结果。

        Args:
            df_all: 长表丰度数据，至少包含 ``group_label``、``cell_type``、``count``
                和公式中的设计列。
            cell_type: 目标 cell subtype/subpopulation。
            formula: PyDESeq2 设计公式的右侧变量，例如 ``"disease + tissue"``。
            main_variable: 主要解释变量。多因素公式中必须指定。
            ref_label: ``main_variable`` 的参考组。
            group_label: 样本唯一标识列。
            alpha: 显著性阈值。
            **kwargs: 预留兼容参数。

        Returns:
            标准 ``make_result`` 字典。
        """
        
        self.ref_label = ref_label
        design_cols = parse_formula_columns(f"y ~ {formula}")
        if main_variable is None:
            if len(design_cols) > 1:
                raise KeyError("Main explanatory variable must be specified when `formula` contains more than one variable.")
            main_variable = design_cols[0]
        current_hash = self._get_data_hash(df_all, formula)
        
        # PyDESeq2 全量拟合较重，缓存可以避免逐个 cell subtype/subpopulation 重复跑模型。
        if current_hash != self.last_data_hash:
            self.cached_results = self._run_full_deseq2(
                df_all, formula, main_variable, ref_label, group_label, alpha
            )
            self.last_data_hash = current_hash
        
        return self._extract_result(cell_type, alpha)
    
    def _run_full_deseq2(self, df_all, formula, main_variable, ref_label, group_label, alpha):
        """准备 count/metadata 宽表并运行 PyDESeq2 全量拟合。"""
        design_cols = parse_formula_columns(f"y ~ {formula}")
        required_cols = {group_label, "cell_type", "count", main_variable, *design_cols}
        missing_cols = required_cols - set(df_all.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {sorted(missing_cols)}")
        
        # 聚合数据，确保每个 sample_id 只有一行（避免 MultiIndex 冲突）
        # 先提取元数据映射表 (sample_id -> disease/tissue 等)
        meta_map = df_all[[group_label] + design_cols].drop_duplicates().set_index(group_label)
        
        # 生成 Count 宽表：只用 group_label 做 index，避免产生 MultiIndex
        pivot = df_all.pivot_table(
            index=group_label,
            columns="cell_type",
            values="count",
            aggfunc="sum",
            fill_value=0
        )
        
        # 严格对齐 Metadata 和 Counts
        counts_df = pivot.astype(int)  # DESeq2 需要整数
        metadata = meta_map.loc[counts_df.index].copy()
        
        # --- 关键修复：确保 metadata 的所有列都是简单的 object 或 category 类型 ---
        for col in metadata.columns:
            metadata[col] = metadata[col].astype(str)
        
        # 2. 全量拟合
        # quiet=True 停止打印所有细胞类型的进度
        design_factors = design_cols
        dds = DeseqDataSet(
            counts=counts_df,
            metadata=metadata,
            design_factors=design_factors,
            refit_cooks=True,
            quiet=True
        )
        dds.deseq2()
        print(f"[PyDESeq2Manager._run_full_deseq2] LFC columns: {list(dds.varm['LFC'].columns)}")
        
        # 3. 提取所有对比组
        # full_cache = {k: {} for k in design_cols} # TODO: 改的兼容一点
        full_cache = {"disease": {}, "tissue": {}}
        
        # --- 提取 Disease 对比 (对比各 labels vs HC) ---
        clean_main = main_variable
        unique_disease = [l for l in metadata[clean_main].unique() if l != ref_label]
        for other_label in unique_disease:
            stat_res = DeseqStats(dds, contrast=[clean_main, other_label, ref_label], quiet=True)
            stat_res.summary()
            full_cache["disease"][other_label] = stat_res.results_df
        
        # --- 提取 Tissue 对比 (if vs nif) ---
        # 假设你的 tissue 列名叫 'tissue'，对照组叫 'nif'
        if 'tissue' in metadata.columns:
            # 自动寻找非 nif 的 label (通常是 'if')
            tissue_labels = [l for l in metadata['tissue'].unique() if l != 'nif']
            for t_label in tissue_labels:
                stat_res_t = DeseqStats(dds, contrast=['tissue', t_label, 'nif'], quiet=True)
                stat_res_t.summary()
                full_cache["tissue"][t_label] = stat_res_t.results_df
        
        return full_cache
    
    def _extract_result(self, cell_type: str, alpha: float) -> Dict[str, Any]:
        """从缓存结果中提取目标 cell subtype/subpopulation 的对比表。"""
        try:
            contrast_rows = []
            
            # 遍历所有已缓存的对比组结果
            for other_label, res_df in self.cached_results["disease"].items():
                if cell_type in res_df.index:
                    row = res_df.loc[cell_type]
                    
                    # PyDESeq2 使用 log2，为了和 CLR (ln) 对应，建议转换：
                    # ln(x) = log2(x) * ln(2)
                    coef_ln = row["log2FoldChange"] * np.log(2)
                    
                    contrast_rows.append({
                        "other": other_label,
                        "ref": self.ref_label,
                        "Coef.": coef_ln,
                        "Std.Err.": row["lfcSE"] * np.log(2),
                        "z": row["stat"],
                        "P>|z|": row["pvalue"],
                        "significant": row["pvalue"] < alpha,
                        "direction": "other_greater" if coef_ln > 0 else "ref_greater"
                    })
            
            # 提取 Tissue 的表格 (新增)
            tissue_rows = []
            for t_label, res_df in self.cached_results.get("tissue", {}).items():
                if cell_type in res_df.index:
                    row = res_df.loc[cell_type]
                    lfc_ln = row["log2FoldChange"] * np.log(2)
                    tissue_rows.append({
                        "other": t_label, "ref": "nif", "Coef.": lfc_ln,
                        "P>|z|": row["pvalue"], "significant": row["pvalue"] < alpha,
                        "direction": "other_greater" if lfc_ln > 0 else "ref_greater"
                    })
            
            if not contrast_rows:
                return make_result(self.method_name, cell_type, np.nan, 'Minimal',
                                   contrast_table=pd.DataFrame(), extra={}, alpha=alpha)
            
            # 构建最终的 contrast_table
            df_contrast = pd.DataFrame(contrast_rows + tissue_rows).set_index("other")
            
            # 选出一个代表性的 P 值和效应值（通常是第一个对比组）
            main_p = df_contrast["P>|z|"].min()
            # main_eff = df_contrast["Coef."].max()
            
            # 调用你定义的标准 make_result
            return make_result(
                method=self.method_name,
                cell_type=cell_type,
                p_val=main_p, p_type='Minimal',
                contrast_table=df_contrast,
                extra={},
                alpha=alpha
            )
        
        except Exception as e:
            return make_result(
                method=self.method_name,
                cell_type=cell_type,
                p_val=np.nan,
                p_type='Minimal',
                extra={
                    "error": repr(e),
                },
                alpha=alpha
            )


run_PyDESeq2 = PyDESeq2Manager()
run_PyDESeq2.__name__ = "run_PyDESeq2"
