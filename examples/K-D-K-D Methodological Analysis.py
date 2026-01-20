# 在协和医院高算上测试
# conda activate sc-min
##################################
import os, gc, sys
import numpy as np
import pandas as pd
import inspect
import anndata

sys.path.append('/public/home/xiongyuehan/data/BioinfoUtilityPackage')


from src.core.kdk_methodology import *
from src.core.kdk_vis import plot_simulation_benchmarks,plot_volcano_stratified_label,plot_significance_heatmap

from src.core.kdk_method_utils import *
from src.core.kdk_vis import plot_ppv_with_counts,plot_multi_layer_ppv

save_addr = "/public/home/xiongyuehan/data/IBD_analysis/output/Step07/Step07c_KDKD_methodology"
###########################################################
# 读取和准备数据
# adata = anndata.read_h5ad("/public/home/xiongyuehan/data/IBD_analysis/output/Step06/Step06_final_identified.h5ad")
adata = anndata.read_h5ad("/public/home/xiongyuehan/data/IBD_analysis/output/Step07/Step07a_Summary/Step07_DR_clustered_clean_20260108.h5ad")
adata_obs = adata.obs
del adata; gc.collect()
###########################################################
# 微调和保存
adata_obs[adata_obs["tissue-origin"]=="rectum"]["tissue-origin"] = "colon"
adata_obs = adata_obs[adata_obs["tissue-origin"]!="blood"]
adata_obs["tissue-origin"] = adata_obs["tissue-origin"].tolist()
adata_obs["disease_group"] = (adata_obs["disease"].astype(str) + "_" + adata_obs["tissue-type"].astype(str))
adata_obs.to_csv(f"{save_addr}/Celltype_meta.csv")
###########################################################
# 后续的 readin，默认格式为这个 count_df
adata_obs = pd.read_csv(f"{save_addr}/Celltype_meta.csv")
count_df = make_input(adata_obs)

count_df.presort.value_counts()
# 对数据进行一定处理，已经去除了血液来源
count_df.presort[count_df.presort=="CD45+CD3+"] = "CD3+CD19-"
count_df.tissue[count_df.tissue=="mixed"] = "if"
count_df.tissue[count_df.tissue=="normal"] = "nif"

test = count_df.groupby("sample_id")['count'].sum()
mask = test.index[test>500]
count_df = count_df[count_df["sample_id"].isin(mask)]
###########################################################
cell_type_inclusion = {"CD3+CD19-": ['CD4 Tnaive', 'CD4 Tmem', 'CD4 Tmem GZMK+', 'CD4 Tfh', 'CD4 Treg', 'CD4 Th17',
                                     'CD8 Tnaive', 'CD8 Tmem', 'CD8 Tmem GZMK+', 'CD8 Trm', 'CD8 Trm GZMA+',
                                     'CD8 NKT FCGR3A+', 'CD8aa IEL',
                                     'gdTnaive', 'g9d2T cytotoxic', 'gdTrm',
                                      'MAIT TRAV1-2+',
                                     'Mitotic T cell'],
                       "CD45+": ['B cell IL6+', 'B cell kappa', 'B cell lambda', 'Germinal center B cell',
                                 'Plasma IgA+', 'Plasma IgG+', 'Mitotic plasma cell',
                                 'Natural killer cell FCGR3A+', 'Natural killer cell NCAM1+', 'ILC1', 'ILC3',
                                 'Classical monocyte CD14+', 'Nonclassical monocyte CD16A+', 'cDC1 CLEC9A+',
                                 'cDC2 CD1C+', 'pDC GZMB+',
                                 'Macrophage', 'Macrophage M1', 'Macrophage M2', 'Neutrophil CD16B+',
                                 'Mast cell'],
                       "CD45-": ['Intestinal stem cell OLFM4+LGR5+',
                                 'pre-TA cell', 'Transit amplifying cell', 'Quiescent stem cell LEFTY1+',
                                 'CAAP epithelium HLA-DR+',
                                 'Goblet', 'Paneth cell', 'Tuft cell', 'Enteroendocrine',
                                 'Ion-sensing colonocyte BEST4+', 'Microfold cell',
                                 'Early absorptive progenitor', 'Absorptive colonocyte',
                                 'Absorptive colonocyte Guanylins+',
                                 'Endothelium', 'Fibroblast', 'Fibroblast ADAMDEC1+'],
                       }

count_df["presort"].unique()

count_df1 = count_df[count_df["presort"].isin(['CD3+CD19-', 'CD45+', 'intact'])];count_df1 = count_df1[count_df1["cell_type"].isin(cell_type_inclusion['CD3+CD19-'])]

count_df2 = count_df[count_df["presort"].isin(['CD45+', 'intact'])];count_df2 = count_df2[count_df2["cell_type"].isin(cell_type_inclusion['CD45+'])]

count_df3 = count_df[count_df["presort"].isin(['CD45-', 'intact'])];count_df3 = count_df3[count_df3["cell_type"].isin(cell_type_inclusion['CD45-'])]

count_df_sep_ls = [count_df1, count_df2, count_df3]
###########################################################
# 测试函数
celltype_test = count_df.cell_type[10]

# 因为 PyDESeq2 一次演算会计算全部细胞类型并储存在类中，每次使用前需要将类实例化
run_PyDESeq2_cached = PyDESeq2Manager()
DESeq_res = run_PyDESeq2_cached(count_df, count_df.cell_type[10],
                                formula="disease")
DESeq_res["extra"]["contrast_table"]
# 不改变参数复用，可以看到计算速度非常快
DESeq_res = run_PyDESeq2_cached(count_df, count_df.cell_type[11],
                                formula="disease")
DESeq_res["extra"]["contrast_table"]


###########################################################
# 测试模拟数据函数
df_sim,df_true_effect = simulate_CLR_resample_data(count_df,
                                                   disease_effect_size=2.0691,
                                                   tissue_effect_size=0.7250,
                                                   interaction_effect_size=0.00,
                                                   donor_noise_sd=0,sample_noise_sd=0,
                                                   disease_levels= ["HC", "BD", "CD", "Colitis", "UC"],
                                                   tissue_levels=("nif", "if"),random_state=710)
df_true_effect[df_true_effect["True_Significant"]==True]
df_true_effect[df_true_effect["cell_type"]=="CT24"]

# 用模拟数据进行统计测试
test = run_CLR_LMM_with_LFC(df_sim, "CT24",
                            formula="disease + C(tissue, Treatment(reference=\"nif\"))",
                            main_variable = "disease",
                            coef_threshold=1.0)
test["extra"]["contrast_table"]
###########################################################
# 其他统计方法的示例
# 测试模拟输入数据
df_sim,df_true_effect = simulate_DM_data(
    n_donors=8,n_samples_per_donor=4,
    disease_levels=["HC", "CD","UC"],tissue_levels=("nif", "if"),
    sampling_bias_strength=2,
    random_state=710
)

df_sim,df_true_effect = simulate_LogisticNormal_hierarchical(
    n_donors=8,n_samples_per_donor=4,
    disease_levels=["HC", "CD","UC"],tissue_levels=("nif", "if"),
    random_state=710
)


df_sim,df_true_effect = simulate_CLR_resample_data(count_df,
                                                   disease_levels=["HC", "CD","UC"])

print(df_true_effect[df_true_effect.True_Significant==True])
##############################################################
# 统计 sim 函数所需要的参数
# 1）运行收集函数
collected_results = collect_DM_results(
    df_count=count_df,
    cell_types_list=count_df.cell_type.unique().tolist(),
    run_DM_func=run_Dirichlet_Multinomial_Wald,
    formula="disease + C(tissue, Treatment(reference=\"nif\"))"
)
print("--- 收集到的所有 LogFC 系数 ---")
print(collected_results['all_coefs'].head())
# 2） 汇总为 simulate_DM_data 所需的模拟参数
final_simulation_params = summarize_DM_parameters(collected_results,0.05)
print("\n--- 最终模拟参数估计 ---")
print(pd.Series(final_simulation_params))
# 3） 计算 simulate_LogisticNormal_hierarchical 所需的模拟参数
est_params = estimate_simulation_parameters(
    df_real=count_df,
    dm_results=collected_results,
    ref_disease="HC",
    ref_tissue="nif",
    alpha=0.05
)
# 4）计算 simulate_CLR_resample_data 所需的模拟参数，建议分层
count_df_CD45 = count_df[count_df["presort"].isin(["intact","CD45+"])]
count_df_CD45 = count_df_CD45[count_df_CD45["cell_type"].isin(cell_type_inclusion["CD45+"]+cell_type_inclusion["CD3+CD19-"])]
count_df_clean = count_df_CD45[(count_df_CD45['disease'] == 'HC') & (count_df_CD45['tissue'] == 'nif')]
# 4.1 重新运行第一步
collected_results = collect_DM_results(
    df_count=count_df_CD45,
    cell_types_list=count_df.cell_type.unique().tolist(),
    run_DM_func=run_Dirichlet_Multinomial_Wald,
    formula="disease + C(tissue, Treatment(reference=\"nif\"))"
)
# 4.2 运行 simulate_CLR_resample_data 参数收集
est_params = estimate_CLR_params_hierarchical(
    df_real=count_df_CD45,       # 原始计数表
    collected_results=collected_results, # 统计结果
    alpha=0.5
)
##############################################################
# 参数扫描过程
run_PyDESeq2_cached = PyDESeq2Manager()
run_PyDESeq2_cached.__name__ = "run_PyDESeq2_cached"

funcoi = [run_CLR_LMM,run_Dirichlet_Wald]#,run_PyDESeq2_cached,,run_Dirichlet_Multinomial_Wald]



metrics_list = []


for func in funcoi:
    print(f"Now processing func {func.__name__}")
    metrics  = evaluate_effect_size_scaling(
        scale_factors=[0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 5.0, 8.0, 10.0],
        sim_func=simulate_CLR_resample_data,
        base_params= {"count_df":count_df_clean,
                      "n_donors": 20,
                      "n_samples_per_donor": 4,
                      "disease_effect_size":2.0691,
                      "tissue_effect_size":0.7250,
                      "interaction_effect_size":0.00,
                      "inflamed_cell_frac":0.5,
                      "donor_noise_sd":0.05,  # 新增：供体随机效应 (Logit 空间)
                      "sample_noise_sd":0.05,  # 对应原 latent_axis_sd，建议设小一点
                      "disease_levels": ["HC", "BD", "CD", "Colitis", "UC"],
                      "tissue_levels": ("nif", "if"),
                      "random_state":710
                      },
        run_stats_func=func,  # 使用新开发的稳健方法
        formula="disease + C(tissue, Treatment(reference='nif'))",
        n_samples=128,
        main_variable='disease',
    )
    metrics ['Method'] = func.__name__
    metrics_list.append(metrics)

# 数据合并，和进一步 combine 三层次因素
df_plot_all = pd.concat(metrics_list,ignore_index=True)
df_plot_combined = combine_metrics(df_plot_all)

# 数据可视化
plot_simulation_benchmarks(df_plot_all, save_addr=save_addr,filename="1220_CR(with_noise)_fdr_power_plot.jpg")
plot_simulation_benchmarks(df_plot_combined, save_addr=save_addr,filename="1220_CR(with_noise)_fdr_power_plot_combine.jpg")

##############################################################
# 进行真实数据的测试：方法：run_CLR_LMM+LFC=1.0
# 方法 1： 直接一体输入
results_df = collect_real_data_results(
    count_df=count_df,
    formula="disease + C(tissue, Treatment(reference='nif'))",
    coef_threshold=1.0
)

results_df.to_csv(f"{save_addr}/1220_Realdata(one_step)_Output.csv")

##############################################################
# 方法 2： 分层输入
results_df_ls = []
for i in count_df_sep_ls:
    results_df = collect_real_data_results(
        count_df=i,
        formula="disease + C(tissue, Treatment(reference='nif'))",
        coef_threshold=1.0
    )
    results_df_ls.append(results_df)
    

results_df = pd.concat(results_df_ls, ignore_index=False)

results_df.to_csv(f"{save_addr}/0110_Realdata(separated)_Output.csv")
results_df = pd.read_csv(f"{save_addr}/0110_Realdata(separated)_Output.csv", index_col=0)

plot_volcano_stratified_label(results_df,save_path=f"{save_addr}/0110_Realdata(separated)_volcano_plot.png",
                       p_threshold=0.05, coef_threshold=1.0)

# 1. 重置索引以便分析
df_check = results_df.reset_index().rename(columns={'index': 'term', 'other': 'term'})

# 2. 找出重复的 (cell_type, term) 组合
# keep=False 表示标记所有重复行，方便对比差异
duplicates = df_check[df_check.duplicated(subset=['cell_type', 'term'], keep=False)]

if not duplicates.empty:
    print(f"发现 {len(duplicates)} 行数据存在重复组合！")
    # 按 cell_type 和 term 排序，方便肉眼对比到底是哪里不同（比如可能是 Coef 不同）
    print(duplicates.sort_values(['cell_type', 'term']))
else:
    print("没有发现重复组合。")

plot_significance_heatmap(results_df,term_order=['Colitis', 'BD', 'CD', 'UC', 'if'],
                          save_path=f"{save_addr}/0110_Realdata(separated)_heatmap.png",
                          p_threshold=0.05)
##############################################################
##############################################################
# 基于拆分数据计算 PPV
count_df_sep_ls

for i in count_df_sep_ls:
    collected_results = collect_DM_results(
        df_count=i,
        cell_types_list=i.cell_type.unique().tolist(),
        run_DM_func=run_Dirichlet_Multinomial_Wald,
        formula="disease + C(tissue, Treatment(reference=\"nif\"))"
    )
    
    est_params = estimate_CLR_params_hierarchical(
        df_real=i,  # 原始计数表
        collected_results=collected_results,  # 统计结果
        alpha=0.5,
        min_abundance=0.02
    )


layer1_params = {
    "count_df": count_df_sep_ls[0][(count_df_sep_ls[0]['disease'] == 'HC') & (count_df_sep_ls[0]['tissue'] == 'nif')],
    "n_donors": 20,
    "n_samples_per_donor": 4,
    "disease_effect_size": 1.1164,
    "tissue_effect_size": 0.7385,
    "interaction_effect_size": 0.0,
    "inflamed_cell_frac": 0.2778,  # 按照我们之前的讨论，调低比例以获得更真实的 FDR
    "donor_noise_sd": 0.6912,  # 新增：供体随机效应 (Logit 空间)
    "sample_noise_sd": 0.0801,  # 对应原 latent_axis_sd，建议设小一点
    "disease_levels": ["HC", "BD", "CD", "Colitis", "UC"],
    "tissue_levels": ("nif", "if"),
    "random_state": 710
}

layer2_params = {
    "count_df": count_df_sep_ls[0][(count_df_sep_ls[0]['disease'] == 'HC') & (count_df_sep_ls[0]['tissue'] == 'nif')],
    "n_donors": 20,
    "n_samples_per_donor": 4,
    "disease_effect_size": 1.4150,
    "tissue_effect_size": 1.0680,
    "interaction_effect_size": 0.0,
    "inflamed_cell_frac": 0.7143,  # 按照我们之前的讨论，调低比例以获得更真实的 FDR
    "donor_noise_sd": 1.0570,  # 新增：供体随机效应 (Logit 空间)
    "sample_noise_sd": 0.1234,  # 对应原 latent_axis_sd，建议设小一点
    "disease_levels": ["HC", "BD", "CD", "Colitis", "UC"],
    "tissue_levels": ("nif", "if"),
    "random_state": 710
}

layer3_params = {
    "count_df": count_df_sep_ls[0][(count_df_sep_ls[0]['disease'] == 'HC') & (count_df_sep_ls[0]['tissue'] == 'nif')],
    "n_donors": 20,
    "n_samples_per_donor": 4,
    "disease_effect_size": 1.7299,
    "tissue_effect_size": 0.7330,
    "interaction_effect_size": 0.0,
    "inflamed_cell_frac": 0.2353,  # 按照我们之前的讨论，调低比例以获得更真实的 FDR
    "donor_noise_sd": 0.7390,  # 新增：供体随机效应 (Logit 空间)
    "sample_noise_sd": 0.0727,  # 对应原 latent_axis_sd，建议设小一点
    "disease_levels": ["HC", "BD", "CD", "Colitis", "UC"],
    "tissue_levels": ("nif", "if"),
    "random_state": 710
}
all_layer_dict = {"layer1":layer1_params,"layer2":layer2_params,"layer3":layer3_params}
# all_layer_dict = {"layer2":layer2_params,"layer3":layer3_params}

date="0110"
for k, v in all_layer_dict.items():
    # 1. 设定主种子，确保实验可重复且无偏
    master_seed = 2025
    rng = np.random.default_rng(master_seed)
    # 预先生成 10 个随机种子，避免在循环中手动干预
    seeds = rng.integers(low=0, high=100000, size=10)
    
    # 用于存储每次循环的 raw_df
    all_raw_results = []
    
    print(f"Starting simulation with seeds: {seeds}")
    
    # 2. 开始 10 次循环
    for i, current_seed in enumerate(seeds):
        print(f"Running iteration {i + 1}/10 with seed: {current_seed}...")
        
        # 浅拷贝参数字典并更新随机状态
        temp_params = v.copy()
        temp_params['random_state'] = int(current_seed)
        
        # 执行模拟
        # 注意：我们主要关注 raw_df 的累积，因为 PPV 需要大量样本
        _, iteration_raw_df = evaluate_effect_size_scaling_with_raw(
            scale_factors=[0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 5.0, 8.0, 10.0],
            sim_func=simulate_CLR_resample_data,
            run_stats_func=run_CLR_LMM_with_LFC,
            formula="disease + C(tissue, Treatment(reference='nif'))",
            sim_params=temp_params,
            stats_params={"main_variable":'disease','coef_threshold':1.0,"alpha":0.05},
        )
        
        # 添加一个列记录这是第几次循环（可选）
        iteration_raw_df['iteration'] = i
        all_raw_results.append(iteration_raw_df)
    
    # 3. 合并所有结果
    # 现在 final_raw_df_combined 包含了 10 次随机实验的所有点
    final_raw_df_combined = pd.concat(all_raw_results, ignore_index=True)
    
    # 4. 基于汇总后的数据计算 PPV
    # 数据量大了 10 倍，分箱后的 PPV 会非常平滑
    ppv_table = calculate_ppv_by_coef(final_raw_df_combined, bin_size=0.1)
    
    print("Combined PPV table generated.")
    print(ppv_table)
    ppv_table.to_csv(f"{save_addr}/{date}_ppv_table_{k}.csv")
    plot_ppv_with_counts(ppv_table,save_path=f"{save_addr}/{date}_{k}_ppv.png")
    
pppvls = []
for i in all_layer_dict.keys():
    ppv_table = pd.read_csv(f"{save_addr}/{date}_ppv_table_{i}.csv",index_col=0)
    ppv_table["layer"] = i
    pppvls.append(ppv_table)
 