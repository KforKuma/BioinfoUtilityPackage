# 在协和医院高算上测试
# conda activate sc-min
##################################
import os, gc, sys
import numpy as np
import pandas as pd
import anndata

sys.path.append('/public/home/xiongyuehan/data/BioinfoUtilityPackage')


from src.stats import *

####################################
# 重新加载
# import importlib
# importlib.reload(sys.modules['src.core.utils.geneset_editor'])

# 删除模块缓存
for module_name in list(sys.modules.keys()):
    if module_name.startswith('src.stats.engine'):
        del sys.modules[module_name]

# 重新读入
from src.stats.engine import *
####################################
# 路径初始化
save_addr = "/public/home/xiongyuehan/data/IBD_analysis/output/Step08_Diff_Abundance"
save_fig_addr = f"{save_addr}/test"
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

###########################################################
# 测试函数
###########################################################
# 针对单个细胞，进行最小化测试
# 由于 run_PyDESeq2 和 run_scCODA 都是 all in one，不要用全量数据
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

# 循环测试
for func in [run_ANOVA_naive,run_ANOVA_transformed,
             run_CLR_LMM,run_CLR_LMM_with_LFC,#run_pCLR_LMM,run_pCLR_OLS,
             # run_PyDESeq2,
             run_Dirichlet_Wald,run_Dirichlet_Multinomial_Wald,
             run_DKD,
             run_LMM,
             # run_Perm_Mixed
             ]:
    res = call_with_compatible_args(func,cell_type=count_df1.cell_type.iloc[10], **common_kwargs)
    res_ls.append(res)
    print(res["contrast_table"])

## 单独拆分看结果
print(count_df1[count_df1['cell_type']==count_df1.cell_type.iloc[10]])
dfs=count_df1[count_df1['cell_type']==count_df1.cell_type.iloc[11]]
dfs['disease'].value_counts()

res = call_with_compatible_args(run_Dirichlet_Wald,
                                cell_type=count_df1.cell_type.iloc[1],  **common_kwargs)
print(res['contrast_table'])

res = call_with_compatible_args(run_Perm_Mixed,cell_type=count_df1.cell_type.iloc[1],  **common_kwargs)
print(res['contrast_table'])


from src.stats.engine.sccoda import run_scCODA
# 测试时半减 PYMC 参数以节约时间
res = call_with_compatible_args(run_scCODA, num_results=1000,num_burnin=500,
                                cell_type=count_df1.cell_type.iloc[10],
                                **common_kwargs)
print(res['contrast_table'])


# 对结果进行批量的检查
for n,i in enumerate(res_ls):
    print("#"*30 + f"  {n}   " + "#"*30)
    print(i['method'])
    # print(i.keys())
    # print(len(i.keys()))
    # key='effect_size'
    # if key in i.keys():
    #     print(i[key])
    print("#" * 65)
    print(i['extra'].keys())
    print(len(i['extra'].keys()))
    print(i['extra'])
    # print("#"*65)
    # print(i)

for n,i in enumerate(res_ls):
    print("#"*30 + f"  {n}   " + "#"*30)
    print(i['method'])
    print(i['contrast_table'].columns)
    print(i['contrast_table'])

###########################################################
# 生成模拟数据并进行完整的检测
###########################################################
# 测试模拟数据
df_long, df_true_effect = simulate_DM_data(**param_dict['dm_params'],random_state=2026)
print(df_long.shape)
print(df_true_effect['True_Significant'].value_counts())
print(df_true_effect['Is_Detectable_True'].value_counts())
df_true_effect[df_true_effect['Is_Detectable_True']==True]

df_true_effect[
    (df_true_effect["Is_Detectable_True"] == False) &
    (df_true_effect["contrast_factor"] == "interaction")
]


max(df_true_effect[
    (df_true_effect["Is_Detectable_True"] == False) &
    (df_true_effect["contrast_factor"] == "interaction")
]["Observed_LFC"])
# df_true_effect[df_true_effect["Observed_LFC"]>3]

df_long, df_true_effect = simulate_LogisticNormal_hierarchical(**param_dict['ln_params'],random_state=2026)
print(df_long.shape)
print(df_true_effect['True_Significant'].value_counts())
print(df_true_effect['Is_Detectable_True'].value_counts())
df_true_effect[df_true_effect['Is_Detectable_True']==True]

max(df_true_effect[
    (df_true_effect["Is_Detectable_True"] == False) &
    (df_true_effect["contrast_factor"] == "interaction")
]["Observed_LFC"])

df_long, df_true_effect = simulate_CLR_resample_data(**param_dict['resample_params'],
                                                     count_df=count_df1,random_state=2026)
print(df_long.shape)
print(df_true_effect['True_Significant'].value_counts())
# print(df_true_effect[df_true_effect['True_Significant']==True])
# print(df_long[df_long["count"]==0])
print(df_true_effect['Is_Detectable_True'].value_counts())
df_true_effect[df_true_effect['Is_Detectable_True']==True]

df_true_effect[
    (df_true_effect["Is_Detectable_True"] == False) &
    (df_true_effect["True_Significant"] == True)
]

df_true_effect[df_true_effect["Is_Detectable_True"] == True]

# 测试 tri_anchor 的逻辑
res_dw = call_with_compatible_args(run_Dirichlet_Wald,
                                    cell_type=count_df1.cell_type.iloc[1],  **common_kwargs)
print(res_dw['contrast_table'])

res_dmw = call_with_compatible_args(run_Dirichlet_Multinomial_Wald,
                                    cell_type=count_df1.cell_type.iloc[10],  **common_kwargs)
print(res_dmw['contrast_table'])


res_clr = call_with_compatible_args(run_CLR_LMM,
                                    cell_type=count_df1.cell_type.iloc[1],  **common_kwargs)
print(res_clr['contrast_table'])

res_dsq = call_with_compatible_args(run_PyDESeq2,
                                    cell_type=count_df1.cell_type.iloc[1],  **common_kwargs)
print(res_dsq['contrast_table'])

res_meta = call_with_compatible_args(run_Meta_Ensemble,
                                    cell_type=count_df1.cell_type.iloc[1],  **common_kwargs)
print(res_meta['contrast_table'])



res_meta["raw_results"]["dmw"]['contrast_table']
res_meta["raw_results"]["clr"]['contrast_table']
res_meta["raw_results"]["deseq2"]['contrast_table']

