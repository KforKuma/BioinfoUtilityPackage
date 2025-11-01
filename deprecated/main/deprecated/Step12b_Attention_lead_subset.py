# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 备注：本项目主要将Step12a拆分的小群进行研究，并存在反复研究之可能。
# 运行环境: scvpy10
# 主要输入：拆分的小群，格式如：/data/HeLab/bio/IBD_analysis/tmp/12_adata_En_final.h5ad, .../12_adata_T_final
# 主要输出：
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##——————————————————————————————————————————————————————————————————————————
import scanpy as sc
import anndata
import pandas as pd
import os, gc
import matplotlib
matplotlib.use('Agg')
##——————————————————————————————————————————————————————————————————————————
# 设置scanpy基本属性
import yaml
with open('/data/HeLab/bio/IBD_analysis/assets/io_config.yaml', 'r') as f:
    config = yaml.safe_load(f) # 读取的格式为一系列字典的列表
##——————————————————————————————————————————————————————————————————————————
os.chdir("/data/HeLab/bio/IBD_analysis/")
from src.ScanpyTools.ScanpyTools import ScanpyPlotWrapper, Geneset
from src.ScanpyTools.ScanpyTools import focus_prepare

rank_genes_groups_dotplot = ScanpyPlotWrapper(func = sc.pl.rank_genes_groups_dotplot)
dotplot = ScanpyPlotWrapper(func = sc.pl.dotplot)
umap_plot = ScanpyPlotWrapper(func = sc.pl.umap)
tsne_plot = ScanpyPlotWrapper(func = sc.pl.tsne)
##——————————————————————————————————————————————————————————————————————————
sc.set_figure_params(dpi_save=450, color_map = 'viridis_r',fontsize=6)
sc.settings.verbosity = 1
sc.logging.print_header()
##——————————————————————————————————————————————————————————————————————————

##——————————————————————————————————————————————————————————————————————————
# 第一次处理从这里开始
##——————————————————————————————————————————————————————————————————————————
# 整合和读取
base_path = '/data/HeLab/bio/IBD_analysis/tmp'
sub_paths = [
    '12_adata_T_final.h5ad',
    '12_adata_BP_final.h5ad',
    '12_adata_M_final.h5ad',
    '12_adata_Fb_final.h5ad',
    '12_adata_En_final.h5ad',
    '12_adata_Epi_final.h5ad'
]

# 使用列表生成式生成完整路径
Main_subset_ls = [f"{base_path}/{sub_path}" for sub_path in sub_paths]

# 检查文件是否存在
for file_path in Main_subset_ls:
    if os.path.exists(file_path):
        print(f"File exists: {file_path}")
    else:
        print(f"File does not exist: {file_path}")

adata = anndata.read("/data/HeLab/bio/IBD_analysis/tmp/Step12_cleansed.h5ad")
for addr in Main_subset_ls:
    print(f"Processing {addr}...")  # 使用格式化字符串来增强可读性
    # 读取 subset 数据
    subset = anndata.read(addr)
    # 提取 filtered_celltype_annotation 的唯一值列表
    unique_annotations = subset.obs["manual_cellsubtype_annotation"].unique()
    # 创建一个字典，用于批量更新 adata.obs 的 "Subset_Identity" 列
    update_dict = {}
    for annotation in unique_annotations:
        print(f"Processing annotation: {annotation}")
        if annotation == "Doublet":
            continue
        # 获取 annotation 对应的索引
        index = subset.obs_names[subset.obs["manual_cellsubtype_annotation"] == annotation]
        
        # 将索引和值保存到字典中
        update_dict.update({idx: annotation for idx in index})
    # 批量更新 adata.obs 的 "Subset_Identity" 列
    adata.obs.loc[update_dict.keys(), "Subset_Identity"] = list(update_dict.values())
    # 释放内存
    del subset
    gc.collect()  # 如果确实需要

adata.write("/data/HeLab/bio/IBD_analysis/tmp/Step12_cleansed(updated).h5ad")
##——————————————————————————————————————————————————————————————————————————
adata = anndata.read("/data/HeLab/bio/IBD_analysis/tmp/Step12_cleansed(updated1220_clustered).h5ad")
# 接下来我们按照先前的注意力能力进行一次检查
excel_data = pd.ExcelFile("/data/HeLab/bio/IBD_analysis/assets/1212_subset_focus.xlsx")

focus_sheet = focus_prepare(excel_data)
##——————————————————————————————————————————————————————————————————————————
for index, row in focus_sheet.iterrows():
    name = row['Name']  # 获取 'Name' 列的值
    subsets = row['Subsets']  # 获取 'Subsets' 列的值
    
    # 检查 subsets 是否为空
    if subsets:
        # 根据 subsets 过滤数据
        adata_subset = adata[adata.obs["Subset_Identity"].isin(subsets)]
        
        # 打印或处理每一行的数据
        print(f"Name: {name}, Subsets: {subsets}")
        print(adata_subset.obs["Subset_Identity"].value_counts())
        
        # 写入文件并进行异常处理
        try:
            output_path = f"/data/HeLab/bio/IBD_analysis/tmp/Step12_focus_{name}.h5ad"
            adata_subset.write(output_path)
            print(f"Data for {name} written to {output_path}")
        except Exception as e:
            print(f"Error saving {name} to {output_path}: {e}")
    else:
        print(f"Subsets for {name} is empty, skipping.")
##——————————————————————————————————————————————————————————————————————————
my_markers = Geneset(config['Env']['assets']+"Markers.xlsx")

for index, row in focus_sheet.iterrows():
    name = row['Name']  # 获取 'Name' 列的值
    input_path = f"/data/HeLab/bio/IBD_analysis/tmp/Step12_focus_{name}.h5ad"
    
    # 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"File {input_path} does not exist. Skipping {name}.")
        continue
    
    # 读取 .h5ad 文件
    adata_subset = anndata.read(input_path)
    
    # 设置输出目录并确保其存在
    output_dir = f"/data/HeLab/bio/IBD_analysis/output/Step12b/{name}/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理数据
    process_adata(adata_subset=adata_subset, file_name=name,
                  my_markers=my_markers, marker_sheet=focus_sheet.iloc[index]["Marker_class"],
                  output_dir=output_dir, do_subcluster=True, do_DEG_enrich=True,
                  # obs_subset=True,
                  DEG_enrich_key="leiden_res", resolutions_list=[1, 1.5])
    
    # 写回 .h5ad 文件
    adata_subset.write(input_path)
    
    # 可选：释放内存
    del adata_subset


##——————————————————————————————————————————————————————————————————————————
# 读取注释结果
excel_data = pd.ExcelFile("/data/HeLab/bio/IBD_analysis/assets/0101_0101_subset_assignment.xlsx")

assignment_sheet = excel_data.parse(excel_data.sheet_names[0])

# adata = anndata.read("/data/HeLab/bio/IBD_analysis/tmp/Step12_cleansed(updated1214_clustered).h5ad")

for subset_filename in set(assignment_sheet.Subset_File):
    print(f"Now reading {subset_filename} subset.")
    input_path = f"/data/HeLab/bio/IBD_analysis/tmp/Step12_focus_{subset_filename}.h5ad"
    output_path = f"/data/HeLab/bio/IBD_analysis/tmp/Step12_focus_{subset_filename}.h5ad"
    
    # 读取 .h5ad 文件
    adata_subset = anndata.read(input_path)
    
    obs_key = assignment_sheet.loc[assignment_sheet.Subset_File == subset_filename, "Obs_key_select"].drop_duplicates().iat[0] \
        if not assignment_sheet.loc[assignment_sheet.Subset_File == subset_filename, "Obs_key_select"].empty else None
    print(f"Obs key for {subset_filename}: {obs_key}")
    # 创建字典映射
    result_dict = assignment_sheet[assignment_sheet.Subset_File == subset_filename].set_index('Subset_No')[
        'Identity'].to_dict()
    updated_dict = dict(map(lambda kv: (str(kv[0]) if isinstance(kv[0], int) else kv[0], kv[1]), result_dict.items()))
    print("Updated dictionary:", updated_dict)
    print(f"Created identity dictionary for {subset_filename} with {len(result_dict)} entries")
    adata_subset.obs["manual_cellsubtype_annotation"] = adata_subset.obs[obs_key].map(updated_dict)
    unique_identities = adata_subset.obs["manual_cellsubtype_annotation"].unique()
    for cell_identity in unique_identities:
        print(f"Now synchronizing cell_identity {cell_identity}.")
        # 筛选满足条件的索引
        index = adata_subset.obs_names[adata_subset.obs["manual_cellsubtype_annotation"] == cell_identity]
        # 更新主 adata 对象
        # if type(adata.obs.loc[index, "Subset_Identity"]) !=:
        #     adata.obs.loc[index, "Subset_Identity"] = adata.obs.loc[index, "Subset_Identity"].tolist()
        adata.obs["Subset_Identity"] = adata.obs["Subset_Identity"].astype('str')
        adata.obs.loc[index, "Subset_Identity"] = cell_identity
        updated_cells = len(adata[adata.obs["Subset_Identity"] == cell_identity])
        print(f"Updated {updated_cells} cells with identity '{cell_identity}'")



# 在循环结束后再写入主文件
with Timer("writing adata") as t:
    adata = adata[adata.obs["Subset_Identity"] != "Doublet"] # 优化后直接用布尔掩码筛选，避免创建中间变量。
with Timer("writing adata") as t:
    adata.write("/data/HeLab/bio/IBD_analysis/tmp/Step12_cleansed(updated0101_clustered).h5ad")


# from src.EasyInterface.ScanpyTools import Definition_change
# adata = Definition_change(adata,"Subset_Identity","ILCp","ILCP activated")
# adata = Definition_change(adata,"Subset_Identity","ILC activated","ILCP activated")
#
# adata = Definition_change(adata,"Subset_Identity","ILC naive","ILCP CTLA4+")
# adata = Definition_change(adata,"Subset_Identity","ILC CTLA4+","ILCP CTLA4+")
#
#
# adata = Definition_change(adata,"Subset_Identity","CD8aa CTLA4+","CD8aa")
#
# adata = Definition_change(adata,"Subset_Identity","Epi. Stem cell_Goblet","ESC Goblet")
# adata = Definition_change(adata,"Subset_Identity","Epi. Stem cell_Tuft","ESC Tuft")

# 简易保存
obs_data = adata.obs
obs_data.to_pickle("/data/HeLab/bio/IBD_analysis/tmp/Step12_cleansed(updated0101_clustered)_obs.pkl")


# 读取
adata = anndata.read("/data/HeLab/bio/IBD_analysis/tmp/Step12_cleansed(updated1226_clustered).h5ad")
obs_data = pd.read_pickle("/data/HeLab/bio/IBD_analysis/tmp/Step12_cleansed(updated1229_clustered)_obs.pkl")

obs_data
del adata.obs
adata.obs = obs_data
# index = adata.obs_names[adata.obs["Subset_Identity"]=='nan']
# adata.obs.loc[index,"Subset_Identity"] = "Unnamed"
#
#
adata.obs["Subset_Identity"].value_counts()

adata.obs["Subset_Identity"].value_counts().to_csv("/data/HeLab/bio/IBD_analysis/tmp/0101_value_counts.csv",
                                                   index=True,header=True)
adata[adata.obs["Subset_Identity"]=="ILCp-naive"]
aa = adata.obs["Subset_Identity"].unique()
bb = aa.tolist(); bb.sort()
##——————————————————————————————————————————————————————————————————————————
# 后续分析处理从这里开始
##——————————————————————————————————————————————————————————————————————————
# # 接下来我们按照先前的注意力能力进行一次检查
excel_data = pd.ExcelFile("/data/HeLab/bio/IBD_analysis/assets/0101_subset_focus.xlsx")

focus_sheet = focus_prepare(excel_data)
# ##——————————————————————————————————————————————————————————————————————————
for index, row in focus_sheet.iterrows():
    name = row['Name']  # 获取 'Name' 列的值
    subsets = row['Subsets']  # 获取 'Subsets' 列的值
    
    # 检查 subsets 是否为空
    if subsets:
        # 根据 subsets 过滤数据
        adata_subset = adata[adata.obs["Subset_Identity"].isin(subsets)]
        
        # 打印或处理每一行的数据
        print(f"Name: {name}, Subsets: {subsets}")
        print(adata_subset.obs["Subset_Identity"].value_counts())
        
        # 写入文件并进行异常处理
        try:
            output_path = f"/data/HeLab/bio/IBD_analysis/tmp/Step12_focus_{name}(0101).h5ad"
            adata_subset.write(output_path)
            print(f"Data for {name} written to {output_path}")
        except Exception as e:
            print(f"Error saving {name} to {output_path}: {e}")
    else:
        print(f"Subsets for {name} is empty, skipping.")

# ##——————————————————————————————————————————————————————————————————————————
my_markers = Geneset(config['Env']['assets'] + "Markers-updated.xlsx")
#

for index, row in focus_sheet.iterrows():
    name = row['Name']  # 获取 'Name' 列的值
    input_path = f"/data/HeLab/bio/IBD_analysis/tmp/Step12_focus_{name}(0101).h5ad"
    
    # 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"File {input_path} does not exist. Skipping {name}.")
        continue
    
    # 读取 .h5ad 文件
    adata_subset = anndata.read(input_path)
    
    # 设置输出目录并确保其存在
    output_dir = f"/data/HeLab/bio/IBD_analysis/output/Step12b/{name}(0101)/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理数据
    if len(adata_subset.obs["Subset_Identity"].unique()) > 1:
        process_adata(adata_subset=adata_subset, file_name=f"{name}(original)",
                      my_markers=my_markers, marker_sheet=focus_sheet.iloc[index]["Marker_class"],
                      output_dir=output_dir, do_subcluster=False, do_DEG_enrich=True,
                      # obs_subset=True,
                      DEG_enrich_key="Subset_Identity", resolutions_list=[])
    print("Start Standard analysis.")
    if row['Resubset'] == True:
        process_adata(adata_subset=adata_subset, file_name=name,
                      my_markers=my_markers, marker_sheet=focus_sheet.iloc[index]["Marker_class"],
                      output_dir=output_dir, do_subcluster=True, do_DEG_enrich=True,
                      # obs_subset=True,
                      DEG_enrich_key="leiden_res", resolutions_list=[0.4,0.8,1.2])
    # 写回 .h5ad 文件
    adata_subset.write(input_path)
    # 可选：释放内存
    del adata_subset
