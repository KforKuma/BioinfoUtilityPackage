# conda activate sccoda-2025
##################################
import os, gc, sys
import numpy as np
import pandas as pd
import anndata

sys.path.append('/public/home/xiongyuehan/data/BioinfoUtilityPackage')


from src.stats import *
# from src.stats.engine.sccoda import run_scCODA

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
# 分层计算
results_df_ls = []
for i in count_df_sep_ls:
    results_df = collect_real_data_results(
        count_df=i,
        stats_func=run_Meta_Ensemble_adaptive,
        formula="disease + C(tissue, Treatment(reference='nif'))"
    )
    results_df_ls.append(results_df)

# 整合保存
results_df = pd.concat(results_df_ls, ignore_index=False)
results_df.to_csv(f"{save_addr}/0306_Realdata(separated)_Output(no_filter).csv")
print(results_df['significant'].value_counts())
print(max(results_df['Coef.']))
print(min(results_df['Coef.']))

# 可视化
results_df = pd.read_csv(f"{save_addr}/0306_Realdata(separated)_Output(no_filter).csv", index_col=0)

# 火山图：不太清晰，后续不使用
plot_volcano_stratified_label(results_df, save_path=f"{save_fig_addr}/0306_Realdata_volcano_plot(no_filter).png",
                              p_threshold=0.05, coef_threshold=1.0)
plot_volcano_stratified_label(results_df, save_path=f"{save_fig_addr}/0306_Realdata_volcano_plot(no_filter).pdf",
                              p_threshold=0.05, coef_threshold=1.0)
# 热图
# 由于使用了多重 p-值 校验方法，这里不应该注入 p-值 阈值，从而使用默认的结果
plot_significance_heatmap(results_df, term_order=['Colitis', 'BD', 'CD', 'UC', 'if'],
                          save_path=f"{save_fig_addr}/0306_Realdata(separated)_heatmap(no_filter).png")
plot_significance_heatmap(results_df, term_order=['Colitis', 'BD', 'CD', 'UC', 'if'],
                          save_path=f"{save_fig_addr}/0306_Realdata(separated)_heatmap(no_filter).pdf")


###################
results_df = pd.read_csv(f"{save_addr}/0306_Realdata(separated)_Output(no_filter).csv", index_col=0)
adata = anndata.read_h5ad(
    "/public/home/xiongyuehan/data/IBD_analysis/output/Step07_Summary/Step07_DR_clustered_clean_20260210.h5ad")

# 为 adata.uns 增加一个全状态细胞比例-权重表
results_df["term"] = results_df.index
pivot_coef = results_df.pivot(index='cell_type', columns='term', values='Coef.')

# 通过未经分选的“完整健康组织”获得跨越三层次的基准
adata_base_line = adata[adata.obs['tissue-origin']=='colon']
adata_base_line = adata_base_line[adata_base_line.obs['orig.ident'].isin()]
test = adata_base_line.obs["orig.ident"].value_counts()
mask = test.index[test > 500]
adata_base_line = adata_base_line[adata_base_line.obs["orig.ident"].isin(mask)]

adata_base_line = adata_base_line[adata_base_line.obs['presorted'].isin(['intact','DGC_only'])]

hc_nif_prop = (
    adata_base_line.obs
    .query("disease == 'HC' & `tissue-type` == 'normal'")
    .groupby(["orig.ident", "Subset_Identity"])
    .size()
    .groupby(level=0)
    .apply(lambda x: x / x.sum())
    .groupby("Subset_Identity")
    .mean()
    .reset_index(name="weight")
)

# 检查
hc_nif_prop_sorted = hc_nif_prop.sort_values("weight", ascending=False)


pivot_with_weight = pivot_coef.merge(
    hc_nif_prop,
    left_index=True,
    right_on="Subset_Identity",
    how="left"
)
adata.uns["weighted_cell_prop"] = pivot_with_weight

adata.uns['weighted_cell_prop'] = adata.uns['weighted_cell_prop'].set_index("Subset_Identity")

print(adata.uns['weighted_cell_prop'])

adata.write_h5ad("/public/home/xiongyuehan/data/IBD_analysis/output/Step07_Summary/Step07_DR_clustered_clean_20260210.h5ad")
##############################################################
# PPV 的计算
##############################################################
# 统计 sim 函数所需要的参数
# 1）运行收集函数
from src.stats.evaluation.evaluator import *

all_layer_param_ls = []
for i,df in enumerate(count_df_sep_ls):
    count_df_filtered, zero_summary = filter_rare_celltypes(df, zero_threshold=0.25)
    
    collected_results = collect_DM_results(
        df_count=count_df_filtered,
        cell_types_list=count_df_filtered.cell_type.unique().tolist(),
        run_DM_func=run_Dirichlet_Multinomial_Wald,
        formula="disease + C(tissue, Treatment(reference=\"nif\"))"
    )
    # 2） 汇总为 simulate_DM_data 所需的模拟参数
    param_dict = get_all_simulation_params(df_real=count_df_filtered,
                                           collected_results=collected_results,
                                           ref_disease="HC", ref_tissue="nif")
    
    # 将细胞表达矩阵硬编码进去
    param_dict['resample_params'].update({"count_df": df[(df['disease'] == 'HC') & (df['tissue'] == 'nif')],
                                          "n_donors": 20,
                                          "n_samples_per_donor": 4})
    
    all_layer_param_ls.append(param_dict['resample_params'])
    del param_dict

date = "0306"
master_seed = 2026
for n, param in enumerate(all_layer_param_ls):
    # 1. 设定主种子，确保实验可重复且无偏
    rng = np.random.default_rng(master_seed)
    # 预先生成 10 个随机种子，避免在循环中手动干预
    seeds = rng.integers(low=0, high=100000, size=15)
    
    # 用于存储每次循环的 raw_df
    all_raw_results = []
    
    print(f"Starting simulation with seeds: {seeds}")
    
    
    # 2. 开始 15 次循环
    for i, current_seed in enumerate(seeds):
        print(f"Running iteration {i + 1}/15 with seed: {current_seed}...")
        
        # 浅拷贝参数字典并更新随机状态
        temp_params = param.copy()
        temp_params['random_state'] = int(current_seed)
        
        # 执行模拟
        # 注意：我们主要关注 raw_df 的累积，因为 PPV 需要大量样本
        _, iteration_raw_df = evaluate_effect_size_scaling_with_raw(
            scale_factors=[0.1,0.178,0.316,0.562,1.0,1.495,2.236,3.343,5.0],
            sim_func=simulate_CLR_resample_data,
            run_stats_func=run_Meta_Ensemble,
            formula="disease + C(tissue, Treatment(reference='nif'))",
            sim_params=temp_params,
            stats_params={"main_variable": 'disease'},
        )
        
        # 添加一个列记录这是第几次循环（可选）
        iteration_raw_df['iteration'] = i
        iteration_raw_df.to_csv(f"{save_fig_addr}/PPV_stats_Layer{n}_Seed{i}.csv")
        all_raw_results.append(iteration_raw_df)

##############################################################
# 读取输出
import pandas as pd
from pathlib import Path

# 1. 定义文件夹路径
folder_path = Path('/public/home/xiongyuehan/data/IBD_analysis/output/Step08_Diff_Abundance/fig_0228')

# 2. 获取所有符合条件的文件
file_list = list(folder_path.glob('PPV*.csv'))

if not file_list:
    print("未找到符合条件的文件！")
else:
    df_list = []
    for file in file_list:
        # file.stem 获取不带后缀的文件名，例如 "PPV_stats_Layer2_Seed14"
        filename = file.stem
        # 按照下划线切割：['PPV', 'stats', 'Layer2', 'Seed14']
        parts = filename.split('_')
        # 提取 Layer 信息（假设格式固定，Layer2 在第3位）
        # 如果只想保留数字 "2"，可以用 parts[2].replace('Layer', '')
        layer_val = parts[2]
        # 读取并添加新列
        temp_df = pd.read_csv(file)
        temp_df['Layer'] = layer_val
        # 如果你也需要 Seed 信息，可以顺便加上：
        # temp_df['Seed'] = parts[3]
        df_list.append(temp_df)
    
    # 3. 纵向合并
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    print(f"成功合并 {len(file_list)} 个文件。")
    print(combined_df[['Layer']].value_counts())  # 快速检查各 Layer 的样本量
    # 预览结果
    print(combined_df.head())
    
    
from src.stats.simulation.ppv_computation import *
combined_df['layer'] = combined_df['Layer'].str.extract(r'(\d+)')
ppv_table_ls = []
for n,i in enumerate(combined_df['Layer'].unique()):
# 基于汇总后的数据计算 PPV
# 数据量大了 15 倍，分箱后的 PPV 会非常平滑
    df = combined_df[combined_df['Layer']==i]
    ppv_table = calculate_ppv_by_coef(df, bin_size=0.2)
    ppv_table['layer'] = n
    ppv_table_ls.append(ppv_table)
    print("Combined PPV table generated.")
    print(ppv_table)
    ppv_table.to_csv(f"{save_fig_addr}/0307_ppv_table_{i}.csv")
    plot_ppv_with_counts(ppv_table, save_addr=save_fig_addr, filename=f"0307_ppv_table{i}")

# 汇总

df_all = pd.concat(ppv_table_ls)
plot_multi_layer_ppv(df_all, save_addr=save_fig_addr, filename="0307_ppv_table[Combined]",max_x=5.0)


# 计算 75% 比例 Coef. 的ppv
import pandas as pd

df = df_all.copy()

# 取 coef_bin 左端点用于排序（coef_bin 是 Interval 类型）
df["coef_left"] = df["coef_bin"].apply(lambda x: x.left)

df = df.sort_values("coef_left")

# 总 TP
total_tp = df["tp_count"].sum()
target = 0.75 * total_tp

# 累计 TP
df["cum_tp"] = df["tp_count"].cumsum()

# 找到达到 75% TP 的 bin
q75_row = df[df["cum_tp"] >= target].iloc[0]

print("75% TP reached at bin:", q75_row["coef_bin"])
print("avg coef:", q75_row["avg_est_coef"])
print("PPV at this bin:", q75_row["PPV"])
print("Cumulative TP ratio:", q75_row["cum_tp"] / total_tp)
