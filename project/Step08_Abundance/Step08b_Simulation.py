# conda activate sccoda-2025
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

# count_df_filtered, zero_summary = filter_rare_celltypes(count_df1, zero_threshold=0.25)

collected_results = collect_DM_results(
    df_count=count_df1,
    cell_types_list=count_df1.cell_type.unique().tolist(),
    run_DM_func=run_Dirichlet_Multinomial_Wald,
    formula="disease + C(tissue, Treatment(reference=\"nif\"))"
)
# 2） 汇总为 simulate_DM_data 所需的模拟参数
param_dict = get_all_simulation_params(df_real=count_df1,
                                       collected_results=collected_results,
                                       ref_disease="HC", ref_tissue="nif")

param_list = [param_dict['dm_params'], param_dict['ln_params'], param_dict['resample_params']]

for i in range(len(param_list)):
    # print(print(param_list[i]))
    param_list[i].update({
        'n_donors': 40, 'n_samples_per_donor': 5, 'n_celltypes': 30,
    })
    print(print(param_list[i]))

##############################################################
# 小测试
df_long, df_true_effect = simulate_DM_data(**param_dict['dm_params'], random_state=2026)
print(df_long.shape)
print(df_true_effect['Is_Detectable_True'].value_counts())

res_meta_orig = call_with_compatible_args(run_Meta_Ensemble,
                                          cell_type="CT3", **common_kwargs)
print(res_meta_orig['contrast_table'])

res_meta = call_with_compatible_args(run_Meta_Ensemble_adaptive,
                                     cell_type="CT3", **common_kwargs)
print(res_meta['contrast_table'])

print(df_true_effect[df_true_effect["cell_type"] == "CT3"])

from src.stats.outcome_process import _collect_simulation_results

results_df = _collect_simulation_results(
    df_sim=df_long,
    df_true_effect=df_true_effect,
    run_stats_func=run_Meta_Ensemble,
    formula="disease + C(tissue, Treatment(reference='nif'))"
)

print(results_df[results_df["True_Significant"] == True])
##############################################################
from joblib import Parallel, delayed


def run_one_method(func, sim_func, base_params, count_df1, scale_factors):
    print(f"Now processing func {func.__name__} for simulation {sim_func.__name__}")
    
    metrics = evaluate_effect_size_scaling(
        scale_factors=scale_factors,
        sim_func=sim_func,
        base_params={
            **base_params,
            "count_df": count_df1,
            "random_state": 2026
        },
        run_stats_func=func,
        formula="disease + C(tissue, Treatment(reference='nif'))",
        n_samples=128,
        main_variable='disease',
    )
    
    metrics["Method"] = func.__name__
    metrics["Simulation"] = sim_func.__name__
    
    return metrics


meta_func_dict = {
    # "01statistical":[run_Dirichlet_Wald, run_ANOVA_naive,run_ANOVA_transformed,run_DKD,run_LMM],
    # "02perm":[run_Perm_Mixed],
    # '03pCLR_LMM':[run_pCLR_LMM],
    # '04pCLR_OLS':[run_pCLR_OLS],
    "05sccoda": [run_scCODA]
}

scale_factors = [0.1, 0.178, 0.316, 0.562, 1.0, 1.495, 2.236, 3.343, 5.0]

for k, method_list in meta_func_dict.items():
    
    df_plot_all_list = []
    
    for n, sim_func in enumerate([
        simulate_DM_data,
        simulate_LogisticNormal_hierarchical,
        simulate_CLR_resample_data
    ]):
        base_params = param_list[n]
        
        metrics_list = Parallel(n_jobs=4)(
            delayed(run_one_method)(
                func,
                sim_func,
                base_params,
                count_df,
                scale_factors,
            )
            for func in method_list
        )
        
        df_sim = pd.concat(metrics_list, ignore_index=True)
        
        df_plot_all_list.append(df_sim)
    
    df_plot_all = pd.concat(df_plot_all_list, ignore_index=True)
    
    df_plot_all.to_csv(
        f"{save_fig_addr}/0305_SimBy_All_{k}.csv",
        index=False
    )

df_plot_all = pd.read_csv(f"{save_fig_addr}/0304_SimBy_All_statistical.csv")
print(combine_metrics(df_plot_all))
for i in df_plot_all["Simulation"].unique():
    df_sub = df_plot_all[df_plot_all["Simulation"] == i]
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
        scale_factors=[0.1, 0.178, 0.316, 0.562, 1.0, 1.495, 2.236, 3.343, 5.0],
        # scale_factors=[1.0],
        sim_func=sim_func,
        base_params={**base_params,
                     "count_df": count_df1,
                     "random_state": 2026
                     },
        run_meta_func=run_Meta_Ensemble_adaptive,
        formula="disease + C(tissue, Treatment(reference='nif'))",
        main_variable='disease',
    )
    metrics_list = []
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
    df_plot_all.to_csv(f"{save_fig_addr}/0305_SimBy_All_meta_adaptive(new_scaling).csv", index=False)


df_plot_all = pd.read_csv(f"{save_fig_addr}/0305_SimBy_All_meta_adaptive(new_scaling).csv")
print(combine_metrics(df_plot_all))
for i in df_plot_all["Simulation"].unique():
    print(i)
    df_sub = df_plot_all[df_plot_all["Simulation"] == i]
    print(combine_metrics(df_sub))
##############################################################
# 收集不同细胞比例的数据，验证其稳健性
df_plot_all_list = []

for inflam_prop in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for n, sim_func in enumerate([
        simulate_DM_data,
        simulate_LogisticNormal_hierarchical,
        simulate_CLR_resample_data
    ]):
        
        base_params = param_list[n].copy()
        base_params.update({"inflamed_cell_frac": inflam_prop})
        
        print(f"Now processing func run_Meta_Ensemble for simulation {sim_func.__name__}")
        
        metrics_dict = evaluate_effect_size_meta_scaling(
            scale_factors=[0.1, 0.178, 0.316, 0.562, 1.0, 1.495, 2.236, 3.343, 5.0],
            sim_func=sim_func,
            base_params={
                **base_params,
                "count_df": count_df1,
                "random_state": 2026
            },
            run_meta_func=run_Meta_Ensemble_adaptive,
            formula="disease + C(tissue, Treatment(reference='nif'))",
            main_variable='disease',
        )
        
        metrics_list = []
        for key in metrics_dict.keys():
            metrics_dict[key]['Method'] = key
            metrics_dict[key]['Simulation'] = sim_func.__name__
            metrics_dict[key]['inflam_prop'] = inflam_prop
            metrics_list.append(metrics_dict[key])
        
        df_sim = pd.concat(metrics_list, ignore_index=True)
        
        # 保存当前 simulation 结果（最安全）
        save_path = f"{}/temp_{sim_func.__name__}_inflam_prop={inflam_prop}.csv"
        df_sim.to_csv(save_path, index=False)
        
        print(f"Saved temp result to {save_path}")
        
        # 加入总结果
        df_plot_all_list.append(df_sim)

#####################################
# 提取保存的所有文件，整合绘制 inflam_prop
from glob import glob

# 找到所有 temp 开头、csv 结尾的文件
files = glob(os.path.join(save_fig_addr, "temp*.csv"))

# 读取并 concat
dfs = []

for f in files:
    try:
        # 提取 inflam_prop 数字，并去掉末尾的点
        match = re.search(r'inflam_prop=([0-9.]+)', os.path.basename(f))
        inflam_str = match.group(1) if match else None
        inflam = float(inflam_str.rstrip('.')) if inflam_str else None
        
        df = pd.read_csv(f)
        df["inflam"] = inflam
        dfs.append(df)
    
    except Exception as e:
        print(f"Skip {f}: {e}")

df_plot_all = pd.concat(dfs, ignore_index=True)

print(f"Loaded {len(files)} files.")
print(df_plot_all.shape)

df_plot_all.to_csv(
    f"{save_fig_addr}/0307_SimBy_All_meta_adaptive(new_scaling)_FULL.csv",
    index=False
)

plot_simulation_with_inflam_marginalized(df_plot_all, save_fig_addr, filename="0307_Inflam_Prop_META")

for i in df_plot_all["Simulation"].unique():
    print(i)
    df_sub = df_plot_all[df_plot_all["Simulation"] == i]
    plot_simulation_with_inflam_marginalized(df_sub, save_fig_addr, filename=f"0307_Inflam_Prop_META_SimBy{i}")

##############################################################
# 可视化
# meta 方法
for i in df_plot_all["Simulation"].unique():
    df_sub = df_plot_all[df_plot_all["Simulation"] == i]
    plot_simulation_benchmarks(df_sub,
                               save_addr=save_fig_addr,
                               filename=f"0305_Perform_Compare(SimBy_{i})(All_Meta_Adaptive)")
    plot_simulation_benchmarks(combine_metrics(df_sub),
                               save_addr=save_fig_addr,
                               filename=f"0305_Perform_Compare(SimBy_{i})(All_Meta_Adaptive)[Combined]")

plot_simulation_benchmarks(combine_metrics(df_plot_all),
                           save_addr=save_fig_addr,
                           filename=f"0305_Perform_Compare(All_Meta_Adaptive)[Combined]")

# 其他方法
df_plot_ls = []
for i in ["0305_SimBy_All_01statistical.csv",
          "0305_SimBy_All_02perm.csv",
          "0305_SimBy_All_05sccoda.csv"]:
    print(i)
    df_plot = pd.read_csv(f"{save_fig_addr}/{i}")
    df_plot_ls.append(df_plot)

df_plot_all = pd.concat(df_plot_ls, ignore_index=True)
for i in df_plot_all["Simulation"].unique():
    df_sub = df_plot_all[df_plot_all["Simulation"] == i]
    plot_simulation_benchmarks(df_sub,
                               save_addr=save_fig_addr,
                               filename=f"0305_Perform_Compare(SimBy_{i})(Other_Method)")
    plot_simulation_benchmarks(combine_metrics(df_sub),
                               save_addr=save_fig_addr,
                               filename=f"0305_Perform_Compare(SimBy_{i})(Other_Method)[Combined]")

plot_simulation_benchmarks(combine_metrics(df_plot_all),
                           save_addr=save_fig_addr,
                           filename=f"0305_Perform_Compare(Other_Method)[Combined]")
