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
    def __init__(self):
        """
        轻量化初始化，仅准备容器。
        """
        self.method_name = "scCODA"
        self.last_data_hash = None
        self.cached_results = {}  # 存储 {cell_type: result_dict}
    
    def _get_data_hash(self, df_all: pd.DataFrame, formula: str, alpha: float) -> str:
        """
        计算哈希，包含数据、公式及关键超参数。
        """
        content_hash = hashlib.md5(pd.util.hash_pandas_object(df_all).values).hexdigest()
        return f"{content_hash}_{formula}"
            # alpha 不用硬编码到 hash 里
    
    def _prepare_data(self, df_all: pd.DataFrame, group_label: str):
        """
        利用你提供的逻辑，将长表转换为 scCODA 所需的 AnnData。
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
        
        # 3. 合并元数据
        # 提取除 count 和 cell_type 以外的所有元数据列
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
        """
        还原 GLM 风格的对照表，动态匹配参考系。
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
            
            # 统计推断
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
        """
        入口：先看缓存，没有就跑全量，最后只取你要的那个 cell_type。
        """
        current_hash = self._get_data_hash(df_all, formula, alpha)
        
        # 1. 检查缓存是否失效
        if current_hash != self.last_data_hash:
            print(f"Cache miss or expired. Running scCODA for all cell types...")
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