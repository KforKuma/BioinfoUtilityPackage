# conda activate sc-min
##################################
import os, gc, sys
import numpy as np
import pandas as pd
import anndata

sys.path.append('/public/home/xiongyuehan/data/BioinfoUtilityPackage')


from src.stats import *
from src.stats.engine.sccoda import run_scCODA

####################################
# 重新加载
# import importlib
# importlib.reload(sys.modules['src.core.utils.geneset_editor'])

# 删除模块缓存
for module_name in list(sys.modules.keys()):
    if module_name.startswith('src.stats'):
        del sys.modules[module_name]

# 重新读入
from src.stats.engine import *
####################################
# 路径初始化
save_addr = "/public/home/xiongyuehan/data/IBD_analysis/output/Step08_Diff_Abundance"
save_fig_addr = f"{save_addr}/fig_0228"
os.makedirs(save_fig_addr, exist_ok=True)
####################################
# 读取准备好的数据
adata_obs = pd.read_csv(f"{save_addr}/Celltype_meta(stratified_clean).csv")

import json
with open(f"{save_addr}/stratified_config.json") as f:
    ct_stratified_dict = json.load(f)

count_df = make_input(adata_obs)

count_df["tissue"] = count_df["tissue"].cat.remove_unused_categories()
print(count_df.tissue)

count_df["presort"].unique()

count_df1 = count_df[count_df["presort"].isin(['CD3+CD19-', 'CD45+', 'intact'])];
count_df1 = count_df1[count_df1["cell_type"].isin(ct_stratified_dict['CD3+CD19-'])]

count_df2 = count_df[count_df["presort"].isin(['CD45+', 'intact'])];
count_df2 = count_df2[count_df2["cell_type"].isin(ct_stratified_dict['CD45+'])]

count_df3 = count_df[count_df["presort"].isin(['CD45-', 'intact'])];
count_df3 = count_df3[count_df3["cell_type"].isin(ct_stratified_dict['CD45-'])]

count_df_sep_ls = [count_df1, count_df2, count_df3]

##############################################################
# 统计 sim 函数所需要的参数
# 1）运行收集函数
from src.stats.evaluation.evaluator import *
collected_results = collect_DM_results(
    df_count=count_df1,
    cell_types_list=count_df.cell_type.unique().tolist(),
    run_DM_func=run_Dirichlet_Multinomial_Wald,
    formula="disease + C(tissue, Treatment(reference=\"nif\"))",
    debug=False
)
# 2） 汇总为 simulate_DM_data 所需的模拟参数
param_dict = get_all_simulation_params(df_real=count_df1,
                                       collected_results=collected_results,
                                       ref_disease="HC", ref_tissue="nif")

param_list = [param_dict['dm_params'],param_dict['ln_params'],param_dict['resample_params']]
for i in range(len(param_list)):
    print(print(param_list[i]))
    param_list[i].update({
        'n_donors': 40, 'n_samples_per_donor': 5,'inflamed_cell_frac':0.4,
        'n_celltypes':50
    })
    print(print(param_list[i]))
    


# func_ls = [run_scCODA, run_Meta_Ensemble, run_PyDESeq2,run_CLR_LMM,run_Dirichlet_Wald, run_Dirichlet_Multinomial_Wald,
#            ]

# func_ls = [run_ANOVA_naive,run_ANOVA_transformed,run_DKD,run_LMM,run_Perm_Mixed]
#
func_ls = [run_pCLR_LMM,run_pCLR_OLS]

df_plot_all_list=[]
for n, sim_func in enumerate([
    simulate_DM_data,
    simulate_LogisticNormal_hierarchical,
    # simulate_CLR_resample_data
]):
    metrics_list = []
    base_params = param_list[n]
    
    for func in func_ls:
        print(f"Now processing func {func.__name__} for simulation {sim_func.__name__}")
        metrics = evaluate_effect_size_scaling(
            # scale_factors=[],
            scale_factors=[0.1, 0.2,1.0,0.5, 0.8, 1.0, 1.5, 2.0, 5.0,8.0,10.0],
            sim_func=sim_func,
            base_params={**base_params,
                         "count_df": count_df1,
                         "random_state": 2026
                         },
            run_stats_func=func,
            formula="disease + C(tissue, Treatment(reference='nif'))",
            n_samples=128,
            main_variable='disease',
        )
        metrics['Method'] = func.__name__
        metrics['Simulation'] = sim_func.__name__  # 新增列标记 sim_func
        metrics_list.append(metrics)
    
    # 合并当前 sim_func 的所有 method
    df_sim = pd.concat(metrics_list, ignore_index=True)
    df_plot_all_list.append(df_sim)  # 保存到总列表
    
    # 合并所有 sim_func 的结果
    df_plot_all = pd.concat(df_plot_all_list, ignore_index=True)
    
    # 保存 CSV
    df_plot_all.to_csv(f"{save_fig_addr}/0228_SimBy_{sim_func.__name__}_Methodology.csv", index=False)

# 可视化
plot_simulation_benchmarks(df_plot_all,
                           save_addr=save_fig_addr,
                           filename=f"0228_Perform_Compare(SimBy_{sim_func.__name__})")
plot_simulation_benchmarks(combine_metrics(df_plot_all),
                           save_addr=save_fig_addr,
                           filename=f"0228_Perform_Compare_Combine(SimBy_{sim_func.__name__})")


# 合并
df_combine = pd.concat([pd.read_csv(f"{save_fig_addr}/0228_SimBy_{sim_func.__name__}_Methodology_test-META.csv"),
                        pd.read_csv(f"{save_fig_addr}/0228_SimBy_{sim_func.__name__}_Methodology_test-META_supp.csv"),
                        pd.read_csv(f"{save_fig_addr}/0228_SimBy_{sim_func.__name__}_Methodology_test_supp.csv")])
plot_simulation_benchmarks(df_combine,
                           save_addr=save_fig_addr,
                           filename=f"0228_Perform_Compare(SimBy_{sim_func.__name__})-META")
plot_simulation_benchmarks(combine_metrics(df_combine),
                           save_addr=save_fig_addr,
                           filename=f"0228_Perform_Compare_Combine(SimBy_{sim_func.__name__})-META")

##############################################################
# 尝试设计 meta 方法
celltype_test = count_df1.cell_type[10]
print(celltype_test)

from src.utils.env_utils import call_with_compatible_args

# 因为 PyDESeq2 一次演算会计算全部细胞类型并储存在类中，每次使用前需要将类实例化
res_ls = []
common_kwargs = dict(
    df_all=count_df1,
    formula="disease + C(tissue, Treatment(reference='nif'))",
    main_variable="disease"
)

res_dmw = call_with_compatible_args(run_Dirichlet_Multinomial_Wald,
                                    cell_type=count_df1.cell_type.iloc[1],  **common_kwargs)

res_clr = call_with_compatible_args(run_CLR_LMM,cell_type=count_df1.cell_type.iloc[1],  **common_kwargs)

res_deseq2 = call_with_compatible_args(run_PyDESeq2,cell_type=count_df1.cell_type.iloc[1],  **common_kwargs)



def run_Meta_Ensemble(res_dmw, res_clr, res_deseq2, alpha=0.05):
    # 1. 提取并对齐索引 (确保三者只对比共同存在的 row)
    df_dmw = res_dmw['contrast_table'].copy()
    df_clr = res_clr['contrast_table'].copy()
    df_dsq = res_deseq2['contrast_table'].copy()
    
    # 获取三者共同的 Index (通常是 'CD', 'UC' 等对比项)
    common_idx = df_dmw.index.intersection(df_clr.index).intersection(df_dsq.index)
    
    if len(common_idx) == 0:
        raise ValueError("三个结果表的索引完全没有交集，请检查 res['contrast_table'].index")
    
    # 统一过滤并排序，确保顺序完全一致
    df_dmw = df_dmw.loc[common_idx]
    df_clr = df_clr.loc[common_idx]
    df_dsq = df_dsq.loc[common_idx]
    
    # 映射方向：other_greater -> 1, ref_greater -> -1
    dir_map = {'other_greater': 1, 'ref_greater': -1}
    rev_map = {1: 'other_greater', -1: 'ref_greater'}
    
    # 使用 .reindex 或直接 loc 已经保证了对齐，现在可以安全比较
    d1 = df_dmw['direction'].map(dir_map)
    d2 = df_clr['direction'].map(dir_map)
    d3 = df_dsq['direction'].map(dir_map)
    
    # 2. 确定 Meta Direction (多数表决)
    # fillna(0) 处理可能存在的 NaN 方向
    dir_sum = d1.fillna(0) + d2.fillna(0) + d3.fillna(0)
    # 方向取和的符号：正数为 other_greater，负数为 ref_greater
    meta_direction = dir_sum.apply(lambda x: rev_map[1] if x > 0 else rev_map[-1] if x < 0 else "Ambiguous")
    
    # 3. 确定 Meta Significance (锚点逻辑)
    # 现在索引已对齐，(d2 == d1) 不再报错
    sig1 = df_dmw['significant'].astype(bool)
    sig2 = df_clr['significant'].astype(bool)
    sig3 = df_dsq['significant'].astype(bool)
    
    # 核心判断：DMW必须显著 且 (CLR同向显著 或 DESeq2同向显著)
    # 注意：这里增加了对方向一致性的显式校验
    match_clr = sig2 & (d2 == d1)
    match_dsq = sig3 & (d3 == d1)
    
    meta_significant = sig1 & (match_clr | match_dsq)
    
    # 4. 统计合并
    meta_table = pd.DataFrame({
        'ref': df_dmw['ref'],
        'Coef.': df_clr['Coef.'],  # 采用 CLR 的系数
        'P_dmw': df_dmw['P>|z|'],
        'P_clr': df_clr['P>|z|'],
        'P_meta': pd.concat([df_dmw['P>|z|'], df_clr['P>|z|']], axis=1).max(axis=1),
        'direction': meta_direction,
        'significant': meta_significant,
        'consistency_score': dir_sum.abs().astype(int)
    }, index=common_idx)
    
    # 5. 质量分级
    def tag_quality(row):
        if row['significant']:
            return 'High-Confidence'
        if (row['P_dmw'] < 0.1) and (row['consistency_score'] == 3):
            return 'Strong-Candidate'
        return 'Noise/Weak'
    
    meta_table['evidence_level'] = meta_table.apply(tag_quality, axis=1)
    
    return {
        'contrast_table': meta_table,
        'common_features': list(common_idx),
        'summary': f"Input features: {len(common_idx)}, Meta-Significant: {meta_significant.sum()}"
    }


res_meta = run_Meta_Ensemble(res_dmw, res_clr, res_deseq2)
print(res_meta['contrast_table'])