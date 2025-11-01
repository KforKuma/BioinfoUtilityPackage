## 第一部分
import pandas as pd
import os,gc
import re

disease_list = ["BS","CD","UC","Colitis","Control"] # 虽然这一步是固定的，但为了代码的兼容性仍然建议放在函数之外单独运行
all_files = [f"{disease}_Allsig.csv" for disease in disease_list]
folder_path = "/data/HeLab/bio/IBD_analysis/output/Step08_Cellphonedb/cellphonedb_input_0620/output_DEG_0620"
output_file = "/data/HeLab/bio/IBD_analysis/output/Step13_CPDB/output/Merged.csv"
# conda activate cellphonedb
df_list = []
for file in all_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df['source_file'] = file  # 添加表示来源文件的列
        df_list.append(df)

merged_df = pd.concat(df_list, ignore_index=True)
merged_df['interaction_left'] = "Empty"; merged_df['interaction_right'] = "Empty"
merged_df.to_csv(output_file, index=False)
print(f'Merged {len(all_files)} files into {output_file}')


# 处理 interaction_group 列，找出含小写字母的行，进行大致研究
# contains_lowercase = merged_df['interaction_group'].str.contains(r'[a-z]', na=False)
# rows_with_lowercase = merged_df[contains_lowercase]

# 读取 interaction_dict 和 uniprot_dict
interaction_dict = pd.read_csv("/data/HeLab/bio/biosoftware/cpdb/cellphonedb-data-5.0.0/data/complex_input.csv")
uniprot_dict = pd.read_csv("/data/HeLab/bio/biosoftware/cpdb/cellphonedb-data-5.0.0/data/sources/uniprot_synonyms.tsv", sep='\t')
uniprot_dict = uniprot_dict.set_index('Entry')['Gene Names (primary)'].to_dict()

# 处理 interaction_dict 的 Uniprot 转换
uniprot_series = pd.Series(uniprot_dict)
interaction_dict[['uniprot_1', 'uniprot_2', 'uniprot_3', 'uniprot_4']] = (
    interaction_dict[['uniprot_1', 'uniprot_2', 'uniprot_3', 'uniprot_4']]
    .apply(lambda col: col.map(uniprot_series).fillna(col))
)
# 生成 Combined 列，去除 NaN 和空字符串
interaction_dict["Combined"] = (
    interaction_dict[['uniprot_1', 'uniprot_2', 'uniprot_3', 'uniprot_4']]
    .replace('', pd.NA)  # 先将 '' 替换为 pd.NA，保证 dropna() 有效
    .apply(lambda row: '_'.join(row.dropna()), axis=1)  # 只连接非空值
)

interaction_dict.to_csv("/data/HeLab/bio/biosoftware/cpdb/cellphonedb-data-5.0.0/data/complex_dict(by_gene_symbol).csv", index=False)

# 接下来我们处理merged_df
# 鉴于左右两侧可能都有complex，用一个预计10列的来储存有点愚蠢，我们先用两列interaction_left和interaction_right拆分
# 然后用complex_left和complex_right拆分复合物，处理为以“-”分隔的字符串
# 最后用protein_left和protein_right，默认取complex的第一项（我们完全有理由假定一个复合物中第一项和后面的项有相同的KEGG/GO属性）

# 1. interaction_left和interaction_right拆分
merged_df_copy = merged_df.copy()
    # 初始化 interaction_left 和 interaction_right
# 1.1 默认的单一分隔符情况
matches_pattern = merged_df_copy['interaction_group'].str.contains(r'^[^-]*-[^-]*$', na=False)
matching_rows = merged_df_copy.loc[matches_pattern,['interaction_group','interaction_left', 'interaction_right']].copy()  # 复制避免 SettingWithCopyWarning
merged_df_copy.drop(matching_rows.index, inplace=True)
matching_rows = matching_rows.drop_duplicates(subset=['interaction_group'])
matching_rows[['interaction_left', 'interaction_right']] = matching_rows['interaction_group'].str.split("-", expand=True)
split_dict = matching_rows.set_index('interaction_group')[['interaction_left', 'interaction_right']].to_dict(orient='index')


# 1.2 代谢分子-byA-B的格式，如，Cholesterol-byLIPA-RORA, Glutamate-byGLS-and-SLC1A2-GRM5
def split_by(interaction):
    match = re.match(r"^(.*?-by[^-]+(?:-and-[^-]+)*)(?:-(.*))?$",interaction)
    if match:
        protein, receptor = match.groups()
        return pd.Series([protein, receptor])
    else:
        # 兜底方案：如果不匹配，按第一个"-"拆分
        parts = interaction.split('-', 1)
        return pd.Series(parts if len(parts) == 2 else [interaction, None])
matches_pattern = merged_df_copy['interaction_group'].str.contains('by', na=False)
    # 使用正则表达式查找符合 "A-byB-C" 格式的行
matching_rows = merged_df_copy.loc[matches_pattern,['interaction_group','interaction_left', 'interaction_right']].copy()  # 复制避免 SettingWithCopyWarning
    # 根据布尔索引筛选出符合条件的行
merged_df_copy.drop(matching_rows.index, inplace=True)
    # 删除原本
matching_rows = matching_rows.drop_duplicates(subset=['interaction_group'])
    # 去重
matching_rows[['interaction_left', 'interaction_right']] = matching_rows['interaction_group'].apply(lambda x: pd.Series(split_by(x)))
    # 筛选并剪裁为B, C
split_dict.update(matching_rows.set_index('interaction_group')[['interaction_left', 'interaction_right']].to_dict(orient='index'))
    # 生成 split_dict


# 1.3 receptor复合物，如'MICB-NKG2D-II-receptor','TGFB3-TGFbeta-receptor1','IL1B-IL1-receptor-inhibitor',
def split_receptor(interaction):
    # 正则匹配 receptor 及其变体，并确保它作为整体
    match = re.match(r'^(.*?)-(.*?(?:receptor(?:1|2)?(?:-inhibitor)?))$', interaction)
    if match:
        protein, receptor = match.groups()
        return pd.Series([protein, receptor])
    else:
        # 兜底方案：如果不匹配，按第一个"-"拆分
        parts = interaction.split('-', 1)
        return pd.Series(parts if len(parts) == 2 else [interaction, None])

matches_pattern = merged_df_copy['interaction_group'].str.contains(r'receptor', na=False)
matching_rows = merged_df_copy.loc[matches_pattern,['interaction_group','interaction_left', 'interaction_right']].copy()  # 复制避免 SettingWithCopyWarning
merged_df_copy.drop(matching_rows.index, inplace=True)
matching_rows = matching_rows.drop_duplicates(subset=['interaction_group'])
matching_rows[['interaction_left', 'interaction_right']] = matching_rows['interaction_group'].apply(lambda x: pd.Series(split_receptor(x)))
split_dict.update(matching_rows.set_index('interaction_group')[['interaction_left', 'interaction_right']].to_dict(orient='index'))


# 1.4 两个或三个-的情况，如HLA-F-LILRB1；CD47-SIRB1-complex；HLA-E-CD94:NKG2C
def split_trip(interaction):
    if pd.isna(interaction):  # 处理 NaN 情况
        return pd.Series([None, None])
    match = re.match(r"^(HLA-[A-Z]|integrin-[^-]+-complex|[^-]+-complex|[^-]+?)-([^-\n]+(?:-[^-]+)*)(-complex|$)", interaction)
    if match:
        protein, receptor, _ = match.groups()
        return pd.Series([protein, receptor])
    else:
        return pd.Series([interaction, None])

matches_pattern = merged_df_copy['interaction_group'].str.contains(r'^[^-]+(-[^-]+){2,3}$', na=False)
matching_rows = merged_df_copy.loc[matches_pattern,['interaction_group','interaction_left', 'interaction_right']].copy()  # 复制避免 SettingWithCopyWarning
merged_df_copy.drop(matching_rows.index, inplace=True)
matching_rows = matching_rows.drop_duplicates(subset=['interaction_group'])
matching_rows[['interaction_left', 'interaction_right']] = matching_rows['interaction_group'].apply(lambda x: pd.Series(split_trip(x)))
split_dict.update(matching_rows.set_index('interaction_group')[['interaction_left', 'interaction_right']].to_dict(orient='index'))


# **应用 split_dict 到 merged_df**
for index, row in merged_df.iterrows():
    key = row["interaction_group"]
    if key in split_dict:
        merged_df.at[index, "interaction_left"] = split_dict[key]['interaction_left']
        merged_df.at[index, "interaction_right"] = split_dict[key]['interaction_right']
    else:
        print(f"未匹配: {key}")

merged_df.to_csv(output_file, index=False)

# 字典写入 json 文件
import json

# 将字典保存为 JSON 文件
with open('/data/HeLab/bio/biosoftware/cpdb/cellphonedb-data-5.0.0/data/complex_separate.json', 'w') as f:
    json.dump(split_dict, f)
# 读取
with open('/data/HeLab/bio/biosoftware/cpdb/cellphonedb-data-5.0.0/data/complex_separate.json', 'r') as f:
    split_dict = json.load(f)
print(split_dict)

# 2. 拆分complex_left和complex_right
# 核心就是去查表
merged_df = pd.read_csv(output_file)

# 创建 interaction_dict 查找映射
interaction_dict["complex_name"] = interaction_dict["complex_name"].str.replace("-","_")
complex_map = interaction_dict.set_index('complex_name')['Combined'].to_dict()

# 先复制 interaction_left 和 interaction_right 到 complex_left 和 complex_right
merged_df['complex_left'] = merged_df['interaction_left'].str.replace("-","_")
merged_df['complex_right'] = merged_df['interaction_right'].str.replace("-","_")

# 使用 map() 方法进行查表替换
merged_df['complex_left'] = merged_df['complex_left'].map(complex_map).fillna(merged_df['complex_left'])
merged_df['complex_right'] = merged_df['complex_right'].map(complex_map).fillna(merged_df['complex_right'])

# 检查
merged_df['complex_left'].unique().tolist()
merged_df['complex_right'].unique().tolist()

merged_df.to_csv(output_file, index=False)

# 3. 拆分protein_left和protein_right
merged_df['protein_left'] = merged_df['complex_left'].str.split("-").str[0]
merged_df['protein_right'] = merged_df['complex_right'].str.split("-").str[0]

# 检查
merged_df['protein_left'].unique().tolist()
merged_df['protein_right'].unique().tolist()

merged_df.to_csv(output_file, index=False)

## 第二部分
os.chdir("/data/HeLab/bio/IBD_analysis/output/Step13_CPDB/output")

import gseapy as gp
merged_df = pd.read_csv(output_file)


GOBP_dict = {}
with open('/data/HeLab/bio/biosoftware/Enrichr_library/GO_Biological_Process_2023.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()              # 去除换行符及首尾空白
        if not line:
            continue                     # 忽略空行
        parts = line.split('\t')         # 以制表符分割
        key = parts[0]
        value = parts[1:]                # 剩余部分作为 value
        GOBP_dict[key] = value

WIKI_dict = {}
with open('/data/HeLab/bio/biosoftware/Enrichr_library/WikiPathways_2024_Human.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()              # 去除换行符及首尾空白
        if not line:
            continue                     # 忽略空行
        parts = line.split('\t')         # 以制表符分割
        key = parts[0]
        value = parts[1:]                # 剩余部分作为 value
        WIKI_dict[key] = value

KEGG_dict = {}
with open('/data/HeLab/bio/biosoftware/Enrichr_library/KEGG_2021_Human.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()              # 去除换行符及首尾空白
        if not line:
            continue                     # 忽略空行
        parts = line.split('\t')         # 以制表符分割
        key = parts[0]
        value = parts[1:]                # 剩余部分作为 value
        KEGG_dict[key] = value



test_res = go_results.results
test_res.sort_values("Cogo_resultsmbined Score",ascending=False)


os.chdir("/data/HeLab/bio/IBD_analysis/output/Step13_CPDB/output/enrichr_enrichment")
for disease in merged_df["source_file"].unique().tolist():
    print(disease)
    for name, gmt in {"GOBP": GOBP_dict, "Wikidict": WIKI_dict, "KEGG": KEGG_dict}.items():
        print(name)
        # 正确获取 gene_list
        gene_list = merged_df.loc[merged_df["source_file"] == disease, "protein_left"].tolist() + \
                    merged_df.loc[merged_df["source_file"] == disease, "protein_right"].tolist()
        
        if not gene_list:  # 如果 gene_list 为空，则跳过
            print(f"Skipping {disease} - {name} due to empty gene list.")
            continue
        
        go_results = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gmt,
            organism='human',
            outdir=None,
            cutoff=0.05
        )
        
        if go_results is None:
            print(f"No significant enrichment results for {disease} - {name}")
            continue
        
        # 按 "Combined Score" 降序排列
        res = go_results.results.sort_values(by="Combined Score", ascending=False)
        
        # 生成文件名
        filename = "_".join([disease[:2], name, "enrichment.xlsx"])
        res.to_excel(filename)
        print(f"Saved results to {filename}")

go_results = gp.enrichr(gene_list=gene_list,
                            # description='GO_Biological_Process',
                            gene_sets=GOBP_dict,  # 或者使用其他 GO 数据集
                            organism='human',
                            outdir="GOBP_2023",
                            cutoff=0.05)  #