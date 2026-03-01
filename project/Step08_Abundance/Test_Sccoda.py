import importlib
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

from sccoda.util import comp_ana as mod
from sccoda.util import cell_composition_data as dat
from sccoda.util import data_visualization as viz

import sccoda.datasets as scd


cell_counts = scd.haber()

print(cell_counts)

# Convert data to anndata object
data_all = dat.from_pandas(cell_counts, covariate_columns=["Mouse"])

# Extract condition from mouse name and add it as an extra column to the covariates
data_all.obs["Condition"] = data_all.obs["Mouse"].str.replace(r"_[0-9]", "", regex=True)
print(data_all)

# Select control and salmonella data
data_salm = data_all[data_all.obs["Condition"].isin(["Control", "Salm"])]
print(data_salm.obs)

# 测试：参考细胞类型
model_salm = mod.CompositionalAnalysis(data_salm, formula="Condition", reference_cell_type="Goblet")
sim_results = model_salm.sample_hmc()

# 查看结果
print(sim_results.summary())

# 测试：自动选择参考细胞
model_salm = mod.CompositionalAnalysis(data_salm, formula="Condition", reference_cell_type="automatic")

################
# 测试：把标准输入接入 sccoda

# count_df 来自 Statistical Test
# 构建计数表
cell_counts = (
    count_df1
    .pivot_table(
        index="sample_id",
        columns="cell_type",
        values="count",
        aggfunc="sum",     # 防止意外重复
        fill_value=0       # 没有该细胞类型就填 0
    )
    .reset_index()
)
data_test = dat.from_pandas(cell_counts, covariate_columns=["sample_id"])

# 对齐 meta 信息
sample_meta = (
    count_df1[
        ["sample_id", "donor_id", "disease", "tissue", "presort"]
    ]
    .drop_duplicates()
)
data_test.obs = (
    data_test.obs
    .merge(
        sample_meta,
        on="sample_id",
        how="left",
        validate="one_to_one"   # 强烈建议，加保险
    )
)

model_test = mod.CompositionalAnalysis(data_test, formula="disease", reference_cell_type="automatic")
model_test = mod.CompositionalAnalysis(data_test, formula="disease + C(tissue, Treatment(reference='nif'))", reference_cell_type="automatic")

sim_results = model_test.sample_hmc()
sim_results.set_fdr(alpha) # alpha 是一个参数
print(sim_results.summary())



