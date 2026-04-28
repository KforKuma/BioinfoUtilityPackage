import pandas as pd
import numpy as np
import hashlib
import logging
import re
import scipy.stats as stats
import sccoda.util.comp_ana as ca
import sccoda.util.cell_composition_data as ccd
from typing import Dict, Any

from src.stats.support import make_result, parse_formula_columns

from src.utils.hier_logger import logged_class

logger = logging.getLogger(__name__)

@logged_class
class ScCodaEngine:
    """缓存式 scCODA 丰度统计引擎。

    scCODA 会对完整组成矩阵建模，因此本类先将长表转换为 AnnData，再一次性拟合
    所有 cell subtype/subpopulation，并缓存每个亚群的对比结果。

    Example:
        >>> res = run_scCODA(
        ...     df_all=abundance_df,
        ...     cell_type="CD4_Tcm",
        ...     formula="disease + tissue",
        ...     main_variable="disease",
        ...     ref_label="HC",
        ...     group_label="sample_id",
        ...     num_results=2000,
        ...     num_burnin=1000,
        ... )
        >>> res["contrast_table"][["Coef.", "P>|z|", "direction"]]
        # 查看 scCODA 后验 beta 转换出的 GLM 风格对比表。
    """

    def __init__(self):
        """轻量化初始化，仅准备缓存容器。
        """
        self.method_name = "scCODA"
        self.last_data_hash = None
        self.cached_results = {}  # 存储 {cell_type: result_dict}
    
    def _get_data_hash(self, df_all: pd.DataFrame, formula: str, alpha: float) -> str:
        """计算缓存哈希。

        Args:
            df_all: 输入长表。
            formula: scCODA 公式。
            alpha: 保留参数，当前不写入 hash，避免只改阈值时重复采样。

        Returns:
            数据与公式组合出的缓存键。
        """
        # alpha 不写入 hash，避免只调整显著性阈值时重复执行昂贵的 HMC 采样。
        content_hash = hashlib.md5(pd.util.hash_pandas_object(df_all).values).hexdigest()
        return f"{content_hash}_{formula}"
    
    def _prepare_data(self, df_all: pd.DataFrame, group_label: str):
        """将长表丰度数据转换为 scCODA 所需的 AnnData。

        Args:
            df_all: 长表丰度数据，包含 ``group_label``、``cell_type`` 和 ``count``。
            group_label: 样本唯一标识列。

        Returns:
            带有 ``obs`` 元数据的 AnnData-like 对象。
        """
        # 1. Pivot 宽表
        cell_counts = (
            df_all.pivot_table(
                index=group_label,
                columns="cell_type",
                values="count",
                aggfunc="sum",
                fill_value=0
            ).reset_index()
        )
        
        # 2. 转换对象
        data_test = ccd.from_pandas(cell_counts, covariate_columns=[group_label])
        
        # 合并元数据是为了让 scCODA 公式可以访问 disease/tissue 等样本级协变量。
        meta_cols = [c for c in df_all.columns if c not in ["count", "cell_type"]]
        sample_meta = (
            df_all[meta_cols]
            .groupby(group_label, as_index=False)
            .first()
        )
        
        data_test.obs = data_test.obs.merge(
            sample_meta,
            on=group_label,
            how="left",
            validate="one_to_one"
        )
        return data_test
    
    def _extract_contrast_table(self, sim_results, cell_type: str, ref_label: str, main_variable: str, alpha: float):
        """从 scCODA 后验 beta 中还原 GLM 风格对比表。

        Args:
            sim_results: scCODA 采样结果。
            cell_type: 目标 cell subtype/subpopulation。
            ref_label: 主变量参考组。
            main_variable: 主变量名，当前主要用于保留调用语义。
            alpha: 显著性阈值。

        Returns:
            以 ``other`` 为索引的对比表。
        """
        posterior_beta = sim_results.posterior['beta'].sel(cell_type=cell_type)
        covariates = posterior_beta.coords['covariate'].values
        
        # 定义参考系映射，用于处理复杂的 formula 提取后的回填
        # 比如在公式中，tissue 的参考系是 nif，disease 的参考系是 HC
        ref_map = {"disease": "HC", "tissue": "nif"}
        
        rows = []
        for cov in covariates:
            if cov == "Intercept":
                continue
            
            samples = posterior_beta.sel(covariate=cov).values.flatten()
            coef = np.mean(samples)
            std_err = np.std(samples)
            
            # 用后验均值和标准差近似 z 检验，便于与其他 engine 的输出列对齐。
            if std_err > 1e-10:
                z_val = coef / std_err
                p_val = 2 * (1 - stats.norm.cdf(abs(z_val)))
            else:
                z_val, p_val = 0.0, 1.0
            
            # --- 核心改进：解析 patsy 复杂标签 ---
            # 匹配逻辑：支持 "disease[T.CD]" 或 "C(tissue, Treatment(reference='nif'))[T.if]"
            # 正则解析：查找 [T. 之后，] 之前的内容作为 other_label
            match_other = re.search(r"\[T\.(.*?)\]", cov)
            
            if match_other:
                other_label = match_other.group(1)
                
                # 自动推断该项所属的原始变量名（用于确定 ref）
                # 如果 cov 包含 'tissue'，则对应的 ref 是 'nif'
                current_ref = ref_label  # 默认值
                for var_key, r_val in ref_map.items():
                    if var_key in cov:
                        current_ref = r_val
                        break
            else:
                # 兜底逻辑：如果没有 [T.xxx]，保留原样
                other_label = cov
                current_ref = ref_label
            
            # 剔除自比较行
            if other_label == current_ref:
                continue
            
            rows.append({
                "other": other_label,
                "ref": current_ref,
                "Coef.": coef,
                "Std.Err.": std_err,
                "z": z_val,
                "P>|z|": p_val,
                "[0.025": np.percentile(samples, 2.5),
                "0.975]": np.percentile(samples, 97.5),
                "significant": p_val < alpha,
                "direction": "other_greater" if coef > 0 else "ref_greater"
            })
        
        df_res = pd.DataFrame(rows)
        return df_res.set_index("other") if not df_res.empty else pd.DataFrame()
    
    def __call__(self, df_all: pd.DataFrame, cell_type: str, formula: str = "disease",
                 main_variable: str = None, ref_label: str = "HC",
                 group_label: str = "sample_id", alpha: float = 0.05,
                 num_results=2000,num_burnin=1000,
                 **kwargs) -> Dict[str, Any]:
        """运行或复用 scCODA 全量采样，并返回目标亚群结果。

        Args:
            df_all: 长表丰度数据。
            cell_type: 目标 cell subtype/subpopulation。
            formula: scCODA 公式。
            main_variable: 主要解释变量，保留给对比表解析。
            ref_label: 主变量参考组。
            group_label: 样本唯一标识列。
            alpha: 显著性阈值。
            num_results: HMC 采样结果数量。
            num_burnin: burn-in 数量。
            **kwargs: 预留兼容参数。

        Returns:
            标准 ``make_result`` 字典。
        """
        current_hash = self._get_data_hash(df_all, formula, alpha)
        
        # 1. 检查缓存是否失效
        if current_hash != self.last_data_hash:
            print("[ScCodaEngine.__call__] Cache miss or expired. Running scCODA for all cell types.")
            self.cached_results = {}  # 清空旧缓存
            
            # 跑全量逻辑
            data_ready = self._prepare_data(df_all, group_label)
            model = ca.CompositionalAnalysis(data_ready, formula=formula,
                                             reference_cell_type="automatic",
                                             automatic_reference_absence_threshold=0.5)
            sim_results = model.sample_hmc(num_results=num_results,num_burnin=num_burnin)
            
            # 提取所有细胞类型并缓存
            all_cts = sim_results.posterior.coords['cell_type'].values
            for ct in all_cts:
                df_contrast = self._extract_contrast_table(sim_results, ct, ref_label, main_variable, alpha)
                main_p = df_contrast['P>|z|'].min() if not df_contrast.empty else 1.0
                # 假设 make_result 已经在你的全局命名空间中
                self.cached_results[ct] = make_result(
                    method=self.method_name,
                    cell_type=ct,
                    p_val=main_p,
                    p_type='Minimal',
                    contrast_table=df_contrast,
                    extra={'sccoda_summary':sim_results.summary()},
                    alpha=alpha
                )
            
            self.last_data_hash = current_hash
        
        # 2. 从缓存中返回特定细胞类型的结果
        if cell_type not in self.cached_results:
            # 万一请求了一个模型中不存在的细胞类型（可能被过滤了）
            return make_result(method=self.method_name,
                               cell_type=cell_type,
                               p_val=1.0,
                               p_type='Minimal',
                               contrast_table=pd.DataFrame(),
                               extra={},
                               alpha=alpha)
        
        return self.cached_results[cell_type]
    

run_scCODA = ScCodaEngine()
run_scCODA.__name__ = "run_scCODA"
