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
    # print(print(param_list[i]))
    param_list[i].update({
        'n_donors': 40, 'n_samples_per_donor': 5,'n_celltypes':30,
    })
    print(print(param_list[i]))

# 小测试
df_long, df_true_effect = simulate_DM_data(**param_dict['dm_params'],random_state=2026)
print(df_long.shape)
print(df_true_effect['Is_Detectable_True'].value_counts())

res_meta_orig = call_with_compatible_args(run_Meta_Ensemble,
                                    cell_type="CT3",  **common_kwargs)
print(res_meta_orig['contrast_table'])

res_meta = call_with_compatible_args(run_Meta_Ensemble_adaptive,
                                    cell_type="CT3",  **common_kwargs)
print(res_meta['contrast_table'])

print(df_true_effect[df_true_effect["cell_type"]=="CT3"])


from src.stats.outcome_process import _collect_simulation_results
results_df = _collect_simulation_results(
    df_sim=df_long,
    df_true_effect=df_true_effect,
    run_stats_func=run_Meta_Ensemble,
    formula="disease + C(tissue, Treatment(reference='nif'))"
)

results_df[results_df["True_Significant"]==True]
##############################################################
# func_ls = [run_Meta_Ensemble]
#
# func_ls = [run_Perm_Mixed, run_pCLR_LMM,run_pCLR_OLS]

# func_ls = [run_scCODA]

# func_ls = [run_Dirichlet_Wald, run_ANOVA_naive,run_ANOVA_transformed,run_DKD,run_LMM]

meta_func_dict = {
    "statistical":[run_Dirichlet_Wald, run_ANOVA_naive,run_ANOVA_transformed,run_DKD,run_LMM],
    "bayesian":[run_Perm_Mixed, run_pCLR_LMM,run_pCLR_OLS],
    "sccoda":[run_scCODA]
}
for k, v in meta_func_dict.items():
    df_plot_all_list=[]
    for n, sim_func in enumerate([
        simulate_DM_data,
        simulate_LogisticNormal_hierarchical,
        simulate_CLR_resample_data
    ]):
        base_params = param_list[n]
        metrics_list = []
        
        for func in v:
            print(f"Now processing func {func.__name__} for simulation {sim_func.__name__}")
            metrics = evaluate_effect_size_scaling(
                # scale_factors=[],
                scale_factors=[0.1,0.178, 0.316, 0.562, 1.0, 1.495, 2.236, 3.343, 5.0],
                # scale_factors=[1.0],
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
        df_plot_all.to_csv(f"{save_fig_addr}/0304_SimBy_All_{k}.csv", index=False)


df_plot_all = pd.read_csv(f"{save_fig_addr}/0304_SimBy_All_statistical.csv")
print(combine_metrics(df_plot_all))
for i in df_plot_all["Simulation"].unique():
    df_sub = df_plot_all[df_plot_all["Simulation"]==i]
    print(combine_metrics(df_sub))

##############################################################

df_plot_all_list = []
for n, sim_func in enumerate([
    simulate_DM_data,
    simulate_LogisticNormal_hierarchical,
    simulate_CLR_resample_data
]):
    base_params = param_list[n]
    print(f"Now processing func run_Meta_Ensemble for simulation {sim_func.__name__}")
    metrics_dict = evaluate_effect_size_meta_scaling(
        # scale_factors=[],
        scale_factors=[0.1,0.178, 0.316, 0.562, 1.0, 1.495, 2.236, 3.343, 5.0],
        # scale_factors=[1.0],
        sim_func=sim_func,
        base_params={**base_params,
                     "count_df": count_df1,
                     "random_state": 2026
                     },
        run_meta_func=run_Meta_Ensemble_adaptive,
        formula="disease + C(tissue, Treatment(reference='nif'))",
        n_samples=128,
        main_variable='disease',
    )
    metrics_list=[]
    for key in metrics_dict.keys():
        metrics_dict[key]['Method'] = key
        metrics_dict[key]['Simulation'] = sim_func.__name__
        
        metrics_list.append(metrics_dict[key])
    
    # 合并当前 sim_func 的所有 method
    df_sim = pd.concat(metrics_list, ignore_index=True)
    df_plot_all_list.append(df_sim)  # 保存到总列表
    
    # 合并所有 sim_func 的结果
    df_plot_all = pd.concat(df_plot_all_list, ignore_index=True)
    
    # 保存 CSV
    df_plot_all.to_csv(f"{save_fig_addr}/0304_SimBy_All_meta_adaptive(new_scaling)(post_adj).csv", index=False)


df_plot_all = pd.read_csv(f"{save_fig_addr}/0304_SimBy_All_meta.csv")
print(combine_metrics(df_plot_all))

for i in df_plot_all["Simulation"].unique():
    df_sub = df_plot_all[df_plot_all["Simulation"] == i]
    print(combine_metrics(df_sub))

df_plot_all = pd.read_csv(f"{save_fig_addr}/0304_SimBy_All_meta_adaptive(new_scaling).csv")
df_plot_all['scale_factor'] = 'mask'
print(combine_metrics(df_plot_all))

##############################################################
df_plot_all = pd.read_csv(f"{save_fig_addr}/0304_SimBy_All_meta_adaptive(new_scaling).csv")

# 可视化
for i in df_plot_all["Simulation"].unique():
    df_sub = df_plot_all[df_plot_all["Simulation"] == i]
    plot_simulation_benchmarks(df_sub,
                               save_addr=save_fig_addr,
                               filename=f"0304_Perform_Compare(SimBy_{i})(All_Meta_Adaptive)")
    plot_simulation_benchmarks(combine_metrics(df_sub),
                               save_addr=save_fig_addr,
                               filename=f"0304_Perform_Compare(SimBy_{i})(All_Meta_Adaptive)[Combined]")
plot_simulation_benchmarks(combine_metrics(df_plot_all),
                           save_addr=save_fig_addr,
                           filename=f"0304_Perform_Compare(All_Meta_Adaptive)[Combined]")
##############################################################
# 合并读取数据
import glob

# 找到所有以 0228 开头的 csv 文件
files = glob.glob(os.path.join(save_fig_addr, "0228*.csv"))

# 读取并合并
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

print(df.shape)
df = df.drop_duplicates()
print(df.shape)

print(df['Method'].value_counts())
print(df['Simulation'].value_counts())
# 合并
plot_simulation_benchmarks(df,
                           save_addr=save_fig_addr,
                           filename=f"0228_Perform_Compare_test")
plot_simulation_benchmarks(combine_metrics(df),
                           save_addr=save_fig_addr,
                           filename=f"0228_Perform_Compare_test_Combine")

combined_df = combine_metrics(df)
combined_df.to_csv(f"{save_fig_addr}/0228_Combined_test.csv", index=False)

combined_df = combine_metrics(pd.read_csv(f"{save_fig_addr}/0228_SimBy_simulate_LogisticNormal_hierarchical_Methodology.csv"))
combined_df.to_csv(f"{save_fig_addr}/0228_Combined_test2.csv", index=False)


