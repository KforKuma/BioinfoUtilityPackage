"""
Step8c_Cellphonedb_plot.py
Author: John Hsiung
Update date: 2025-12-08
Description:
    - 进行 CPDB 下游的绘图
Notes:
    - 使用环境：conda activate scvpy10
"""

import pandas as pd
import anndata
import os
import gc
import scanpy as sc
import sys


sys.stdout.reconfigure(encoding='utf-8')
####################################
sys.path.append('/public/home/xiongyuehan/data/BioinfoUtilityPackage')

from src.external_adaptor.cellphonedb.cellphone_inspector import *
from src.external_adaptor.cellphonedb.plot import *
from src.external_adaptor.cellphonedb.toolkit import *

####################################
parent_dir = "/public/home/xiongyuehan/data/IBD_analysis/output/Step09_Cellphonedb/260125"

analysis_dir = f"{parent_dir}/cpdb_outcome/analysis_thr15percent"
analy_fig_dir = f"{parent_dir}/output/fig_thr15percent"
os.makedirs(analysis_dir, exist_ok=True)
os.makedirs(analy_fig_dir, exist_ok=True)

####################################
####################################
####################################
# 绘制热图
adata = anndata.read_h5ad(
    "/public/home/xiongyuehan/data/IBD_analysis/output/Step07_Summary/Step07_DR_clustered_clean_20260210.h5ad")
obs = adata.obs
del adata;
gc.collect()

disease_list = ['CD', 'HC', 'UC', 'BD', 'Colitis']

for type in disease_list:
    file_dir = f"{output_dir}/{type}/"
    
    ci = CellphoneInspector(cpdb_outfile=file_dir, degs_analysis=False, cellsign=False)
    ci.prepare_cpdb_tables(add_meta=True)
    
    ci.prepare_cell_metadata(metadata=obs, celltype_key="Subset_Identity", groupby_key="disease")  # 通常直接输入就好
    
    cpdb_heatmap(ci, analy_fig_dir, figsize=(15, 15), filename_prefix=type)

####################################
# 改名

print(obs["Subset_Identity"].value_counts())
len(obs["Subset_Identity"].unique())

long_to_short = {
    'Intestinal stem cell OLFM4+LGR5+': "Epi.ISC",
    'Quiescent stem cell LEFTY1+': 'Epi.LEFTY1',
    
    'Absorptive colonocyte Guanylins+': "Col.GUCA",
    'Absorptive colonocyte': "Col.base",
    'Early absorptive progenitor': 'Col.PPAR',
    'Transit amplifying cell': 'Epi.TA',
    'pre-TA cell': 'Epi.pre_TA',
    
    'CAAP epithelium HLA-DR+': 'Col.MHC-II',
    'Ion-sensing colonocyte BEST4+': "Col.BEST4",
    
    'Microfold cell': "Epi.Mcell",
    'Enteroendocrine': "Epi.EEC.",
    'Goblet':"Epi.Goblet", 'Paneth cell':"Epi.Paneth", 'Tuft cell':"Epi.Tuft",
                 
    'Classical monocyte CD14+': "Mono.C.",
    'Nonclassical monocyte CD16A+': "Mono.NC.",
    "Macrophage":"Mph.base",
    'Macrophage M1': 'Mph.M1',
    'Macrophage M2': 'Mph.M2',
    'cDC1 CLEC9A+': 'DC.cDC1', 'cDC2 CD1C+': 'DC.cDC2', 'pDC GZMB+': 'DC.pDC',
    'Neutrophil CD16B+': "Neutro",'Mast cell':"Mast",
    
    'Fibroblast ADAMDEC1+': "Fb.act.",'Fibroblast': "Fb.base","Endothelium":"Endo",
    
    'CD8 NKT FCGR3A+': "T.CD8_NKT",
    'Natural killer cell FCGR3A+': "NK.CD16",
    'Natural killer cell NCAM1+': "NK.CD56",
    'ILC1':'ILC1', 'ILC3':'ILC3',
    'gdTnaive':"T.gdnaive",'g9d2T cytotoxic': "T.g9d2",'gdTrm':"T.gdTrm",
    'CD4 Tmem GZMK+': "T.CD4GZMK",
    'CD8 Tmem GZMK+': "T.CD8GZMK",
    'CD4 Tnaive':"T.CD4naive", 'CD4 Tmem':"T.CD4mem",
    'CD4 Tfh':"T.CD4Tfh", 'CD4 Treg':"T.CD4Treg", 'CD4 Th17':"T.CD4Th17",
    'CD8 Tnaive':"T.CD8naive", 'CD8 Tmem':"T.CD8mem", 'CD8 Trm':"T.CD8Trm", 'CD8 Trm GZMA+':"T.CD8GZMA",
    'CD8aa IEL':"T.CD8aaIEL",
    
    'MAIT TRAV1-2+': 'T.MAIT',
    
    'Germinal center B cell': "B.GC",
    'B cell lambda': 'B.lambda','B cell kappa': 'B.kappa','B cell IL6+':"B.IL6",
    'Plasma IgA+':"Plasma.IgA", 'Plasma IgG+':"Plasma.IgG",
    'Mitotic plasma cell': "Cyc.Plasma",
    'Cyc.T': "Cyc.T"
    }

cells = list(long_to_short.values())
cells_sorted = sorted(
    cells,
    key=lambda x: len(x),
    reverse=True
)

for c in cells_sorted:
    print(f"{c}\t{len(c)}")

# ----------------------------
# 读取数据
# ----------------------------
all_dfs = []
folder_path = analysis_dir

for filename in os.listdir(folder_path):
    if filename.startswith("CPDB") and filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True)
df_all.columns = [c.strip() for c in df_all.columns]

del all_dfs;
gc.collect()

# 进行改名：
df_all["cell_left"] = df_all["cell_left"].tolist()
df_all["cell_right"] = df_all["cell_right"].tolist()
df_all = df_all.assign(
    cell_left=df_all["cell_left"].map(long_to_short).fillna(df_all["cell_left"]),
    cell_right=df_all["cell_right"].map(long_to_short).fillna(df_all["cell_right"])
)

# --- 更新 celltype_group ---
df_all["old_cellpair"] = df_all["celltype_group"]
del df_all["celltype_group"]
df_all["celltype_group"] = pad_strings(
    df_all["cell_left"] + "-" + df_all["cell_right"]
)

print(df_all["celltype_group"].unique())
print(df_all["cell_right"].unique())

# 保存
df_all.to_csv(f"{analysis_dir}/260309_Combine_inShort.csv")

# ----------------------------
# 后续读取
# ----------------------------
df_all = pd.read_csv(f"{analysis_dir}/260309_Combine_inShort.csv", index_col=0)
# ----------------------------
# 围绕细胞类群查找
# ----------------------------
# ct_dict = {
#     "Colon": ['Col.GN+', 'Colon'],
#     "Stem_undiff": ['ISC', 'Quie.SC', 'TA cell', 'M_like cell'],
#     'M cell': ['M cell'],
#     'GZMK': ['CD4 GZMK+', 'CD8 GZMK+'],
#     'Macro_Neutro': ['M1 Mph', 'M2 Mph', 'Neutro'],
#     'M_like cell': ['M_like cell'],
# }
#
# for k, v in ct_dict.items():
#     df_full = search_df(df_all, cell_any=v)
#     group_cols = ["interaction_group", "celltype_group"]
#
#     mask_keep = (
#         df_full
#         .groupby(group_cols)["scaled_means"]
#         .transform(lambda x: (x != 0).any())
#     )
#
#     df_filtered = df_full[mask_keep].copy()
#
#     df_filtered.to_csv(f"{analy_fig_dir}/Celltype_Filter_by_{k}.csv")

#############################
# 围绕通路绘图
cell_chat_dict = {  # 全部仅按照 ‘interaction_group’ 进行查询
    'GZMK_PARs_interaction': ['GZMK-F2R', 'GZMK-F2RL1', 'GZMK-F2RL2', 'GZMK-F2RL3'],
    'GZMK_complement_interaction': ['C3a-byGZMK-C3AR1', 'C5a-byGZMK-C5AR1', 'C5a-byGZMK-C5AR2'],
    'Guanylins_interaction': ['GUCA2A-GUCY2C', 'GUCA2B-GUCY2C'],
    'Patrol_Retention_1(CXCL16)': ['CXCL16-CXCR6'],
    'Patrol_Retention_2(CX3CL1)': ['CX3CL1-CX3CR1'],
    'Patrol_Retention_3(CCL28)': ['CCL28-CCR3', 'CCL28-CCR10'],
    'Inflam-Chronic_Retention(CCL15&CCL23)': ['CCL15-CCR1', 'CCL23-CCR1'],
    'Lymph_Retention_1(CXCR4)': ["CXCL12-CXCR4", "CXCL14-CXCR4"],
    'Lymph_Retention_2(CCR4&CCR7)': ["CCL5-CCR4", "CCL19-CCR7"],
    'Inflam-Acute_1(CCL3&CCL4&CCL5)': ['CCL3-CCR1', 'CCL3-CCR5',  # 无 CCR3
                                       'CCL4-CCR5',
                                       'CCL5-CCR1', 'CCL5-CCR5'],
    'Inflam-Acute_2(CCL20-CCR6)': ['CCL20-CCR6'],
    'Inflam-Acute_3(CCL8_CXCR2)': ['CCL8-CCR1', 'CCL8-CCR2',
                                   'CCL13–CCR2', 'CCL14–CCR1',  # 这两个似乎检测不到
                                   'CXCL1-CXCR2', 'CXCL2-CXCR2', 'CXCL3-CXCR2'],
    'Inflam-Acute_4(Other_CXCR2)': ['CXCL8-CXCR2', 'CXCL5-CXCR2', 'CXCL6-CXCR2', 'CXCL7-CXCR2'],  # 中性粒细胞为主
    'Inflam-Acute_5(CXCR3)': ["CXCL9-CXCR3", "CXCL10-CXCR3", "CXCL11-CXCR3"],
    'Follicule(CXCL13_IL21)': ["CXCL13-CXCR5", "IL21-IL21-receptor"],
    'Costimulatory_1(T_cell)': ["CD70–CD27",  #
                                'CD80-CD28',
                                'TNFSF9–TNFRSF9',  # 4-1BBL/4-1BB, CD8/NK
                                'TNFSF4–TNFRSF4',  # OX40L/OX40, CD4
                                "TNFSF15–TNFRSF25",  # TL1A/DR3
                                'TNFSF15-TNFSF15',  # TL1A/DcR3
                                'TNFSF18-TNFRSF18',  # GITRL/GITR, Teff
                                ],
    'Costimulatory_2(B_cell)': ["TNFSF13–TNFRSF13C", 'TNFSF13-TNFRSF17',
                                'TNFSF13–TNFRSF13B', 'TNFSF13B-TNFRSF13B',
                                'FCER2–CR2', 'FCER2–FGFR2'],
    # 'TNFSF_1(apoptotic)':["TNF-TNFRSF1A",'TNF-TNFRSF1B',
    #                       'FASLG-FAS',
    #                       'FASLG–TNFRSF6B', # 诱饵受体 DcR3
    #                       ],
    'TNFSF_2(TRAIL)': ['TNFSF10-TNFRSF10A', 'TNFSF10-TNFRSF10B',  # DR4, DR5
                       'TNFSF10-TNFRSF10C', 'TNFSF10-TNFRSF10D', 'TNFSF10-TNFRSF11B'  # 竞争性/诱饵受体
                       ],
    'TNFSF_3(homeostatic)': ["TNFSF11–TNFRSF11A", 'TNFSF11-TNFRSF11B',  # RANKL/RANK, OPG
                             "TNFSF14-TNFRSF6B", 'TNFSF14-TNFRSF14', 'BTLA-TNFRSF14', 'TNFSF14-LTBR',
                             # LIGHT/DcR3, HVEM, ..
                             "TNFSF12-TNFRSF12A"  # TWEAK/Fn14
                             ],
    'TNFSF_4(lymph_tissue)': ['LTA-TNFRSF1A', 'LTA-TNFRSF1B', "LTB-LTBR", 'CD40LG-CD40'],
    "IFNG": ["IFNG-Type-II-IFNR"],
    'TGFB': ["TGFB1-TGFbeta-receptor1", "TGFB1-TGFbeta-receptor2"],
    'Interleukin_1(IL1_superfamily)': ["IL1A-IL1-receptor-inhibitor", 'IL1B-IL1-receptor-inhibitor',
                                       'IL1RN-IL1-receptor-inhibitor',
                                       'IL18-IL18_receptor', 'IL33-IL33_receptor',
                                       'IL36A-IL36_receptor', 'IL36B-IL36_receptor', 'IL36G-IL36_receptor',
                                       'IL36RA-IL36_receptor'
                                       ],
    'Interleukin_2(inflammatory)': ["IL6-IL6-receptor",
                                    "IL17_AF-IL17_receptor_AC",
                                    "IL17F-IL17_receptor_AC", "IL23-IL23_receptor",
                                    ],
    'Interleukin_3(anti-inflammatory)': ['IL10-IL10_receptor',
                                         'IL27-IL27_receptor',
                                         'IL35-IL35_receptor',
                                         'IL37-IL37_receptor'],
    'Interleukin_4(T_cell_diff)': ["IL2-IL2_receptor_I", 'IL2-IL2_receptor_HA',
                                   'IL4-IL4_receptor', "IL7-IL7-receptor", 'IL9-IL9_receptor',
                                   'IL12-IL12_receptor',
                                   'IL15-IL15_receptor'],
    # 'Interleukin_5(hematopoietic_myeloid)':['IL3-IL3_receptor','IL5-IL5_receptor','IL11-IL11_receptor'],
    # 'Interleukin_6(epi_regulatory)':['IL19-IL20_receptor_Type_I',
    #                                  'IL20-IL20_receptor_Type_I','IL20-IL20_receptor_Type_II',
    #                                  'IL22-IL22_receptor',
    #                                  'IL24-IL20_receptor_Type_I','IL24-IL20_receptor_Type_II'],
    "CSF": ["CSF1-CSF1R",
            "IL34-CSF1R",
            "CSF2-GMCSFR",
            "CSF3-CSF3R"
            ],
    'TNFA': ["TNF-TNFRSF1A", 'TNF-TNFRSF1B'],
    'FAS': ['FASLG-FAS',
            'FASLG–TNFRSF6B',  # 诱饵受体 DcR3
            ],
    # 'ephrin': ['EFNA1-EPHA1', 'EFNA1-EPHA2', 'EFNA1-EPHA3', 'EFNA1-EPHA4', 'EFNA1-EPHA7',
    #            'EFNA2-EPHA1', 'EFNA2-EPHA2', 'EFNA2-EPHA3', 'EFNA2-EPHA4', 'EFNA2-EPHA7',
    #            'EFNA3-EPHA1', 'EFNA3-EPHA2', 'EFNA3-EPHA3', 'EFNA3-EPHA4', 'EFNA3-EPHA7',
    #            'EFNA4-EPHA1', 'EFNA4-EPHA2', 'EFNA4-EPHA3', 'EFNA4-EPHA4', 'EFNA4-EPHA7', ]
}

for k, v in cell_chat_dict.items():
    search_dict = {"interaction_group": v}
    
    df_full = search_df(df_all, search_dict=search_dict)
    print(df_full["cell_left"].unique())
    print(df_full["cell_right"].unique())
    
    # print(long_to_short)
    df_sorted, vline_pairs = vline_generator(df_full, by_ligand=True)
    
    # 自动尺寸
    n_celltypes = df_sorted["celltype_group"].nunique()
    n_interactions = df_sorted["interaction_group"].nunique()
    n_groups = df_sorted["group"].nunique()
    min_fig_width = 6;
    width_per_celltype = 0.30;
    extra_width = 6;
    extra_height = 10;
    facet_height = 0.4
    fig_width = max(min_fig_width,
                    n_celltypes * width_per_celltype + extra_width)
    fig_height = n_groups * n_interactions * facet_height + extra_height
    print(f"Print with width = {fig_width} and height = {fig_height}")
    
    draw_combine_dotplot(df_sorted, save_addr=analy_fig_dir,
                         filename=f"PPI_{k}",  # ←←←← 必须改
                         vline_pairs=vline_pairs,
                         facet_aspect=3,  # 如果点和点之间上下左右不协调，调整这个
                         facet_height=1.1,  # 如果 group = xxx 位置过于拥挤，增加这个
                         fig_width=fig_width, fig_height=fig_height,
                         # 每种疾病只有一对交互（df_sorted.interaction_group）参考值为 height=12, 两个交互 height=14
                         # 宽度最小为 6，410 row 1 个交互 约为 30，1500 row 2 个交互 约为 50
                         bottom=0.2, top=0.95,
                         interaction_order=search_dict["interaction_group"]
                         )

##################################################
# 对以下进行手动微调
top_cell_dict = {
    'TNFSF_2(TRAIL)': [['Col.GN+', 'Quie.SC', 'TA cell', 'Goblet', 'Mast cell', 'ISC', 'Col.Prog.'],  # TRAIL
                       ['Quie.SC', 'TA cell', 'pre_TA cell', 'ISC', 'Col.Prog.', 'M_like cell', 'M2 Mph', 'cDC2',
                        'M cell']],  # TRAILR
    'TNFA': [['M1 Mph', 'M2 Mph', 'MAIT', 'CD4 Th17', 'cDC2', 'GC B cell', 'g9d2T', 'Cyc.T', 'ILC3'],  # TNF
             ['Col.GN+', 'Quie.SC', 'CD4 Treg', 'TA cell', 'M2 Mph', 'CD4 Tfh', 'C.Mono.', 'Goblet', 'CD8 Trm']],
    'GZMK_PARs_interaction': [['CD8 GZMK+', 'CD4 GZMK+', 'MAIT', 'g9d2T', 'gdTnaive', 'CD8 Trm', 'Cyc.T'],
                              ['Col.GN+', 'Quie.SC', 'Col.Prog.', 'Endothelium', 'Cyc.T', 'pDC']],
    'GZMK_complement_interaction': [['CD8 GZMK+', 'CD4 GZMK+', 'MAIT', 'g9d2T', 'gdTnaive', 'CD8 Trm', 'Cyc.T'],
                                    ['M2 Mph', 'C.Mono.', 'M1 Mph', 'cDC2', 'Macrophage', 'Neutro']],
    'IFNG': [['CD8 Trm', 'CD8 GZMK+', 'Cyc.T', 'g9d2T', 'CD4 GZMK+', 'NK CD56+', 'NK CD16+'],
             ['Col.GN+', 'Bmem_k', 'Quie.SC', 'Bmem_l', 'M2 Mph', 'B cell IL6+', 'pre_TA cell', 'ISC']],
    'Inflam-Acute_1(CCL3&CCL4&CCL5)': [["CD8 Trm", "CD8 GZMK+", "gdTrm", "ILC1", "gdTnaive",
                                        "MAIT", "g9d2T", "CD8 Trm GZMA+", "M2 Mph", "CD4 GZMK+"],
                                       ['M2 Mph', 'C.Mono.', 'M1 Mph', 'cDC2', 'Neutro']],
    'Inflam-Acute_2(CCL20-CCR6)': [['Col.GN+', 'Quie.SC', 'Col.Prog', 'Col.BEST4+', 'M_like cell'],
                                   ['Bmem_k', 'Bmem_l', 'B cell IL6+', 'CD4 Th17' 'CD4 Tmem', 'CD4 Treg', 'MAIT',
                                    "ILC3"]],
    'ephrin': [['Col.GN+', 'Quie.SC', 'Col.Prog', 'Col.BEST4+', 'M_like cell', 'Endothelium', 'M cell'],
               ['Quie.SC', 'Fb act.']],
    'FAS': [['gdTrm', 'ILC1', 'CD8 Trm GZMA+', 'NK CD56+', 'Cyc.T', 'g9d2T', 'NK CD16+'],
            ['Quie.SC', 'TA cell', 'CD4 Treg', 'Col.Prog', 'CD4 Tfh', 'CD4 Th17']],
    'TNFSF_4(lymph_tissue)': [['Bmem_k', 'Bmem_l', 'CD4 Tnaive', 'B cell IL6+', 'CD4 Tmem',
                               'CD4 Treg', 'CD8 Tmem', 'CD4 Th17'],
                              ['Col.GN+', 'Quie.SC', 'CD4 Treg', 'TA cell', 'M2 Mph']],
    'TNFSF_3(homeostatic)': [['Fb act.', 'ILC1', 'NK CD56+', 'Cyc.T', 'ILC3', 'pDC'],
                             ['Col.GN+', 'Quie.SC', 'Bmem_k', 'TA cell', 'pre_TA cell', 'Bmem_l', 'Goblet']],
    'TGFB': [['Bmem_k', 'Bmem_l', 'CD4 Tnaive', 'B cell IL6+', 'Plasma IgA+', 'CD4 Tmem', 'M2 Mph'],
             ['Bmem_k', 'Quie.SC', 'CD4 Tnaive', 'TA cell', 'Bmem_l', 'CD8 Tnaive', 'ISC']],
    'Patrol_Retention_1(CXCL16)': [['Col.GN+', 'Quie.SC', 'M2 Mph'],
                                   ['CD4 Treg', 'CD4 Th17', 'CD8 Trm', 'MAIT', 'Cyc.T', 'gdTrm', 'g9d2T',
                                    'CD8 Trm GZMA+', 'ILC1']],
    'Patrol_Retention_3(CCL28)': [['Col.GN+', 'Quie.SC', 'Col.Prog', 'TA cell', 'Goblet', 'Colon'],
                                  ['Plasma IgA+', 'Cyc.Plasma']],
    'Lymph_Retention_1(CXCR4)': [['Fb act.', 'Fb', 'M2 Mph', 'Endothelium'],
                                 ['CD8 GZMK+', "CD4 GZMK+"]],
    'Follicule(CXCL13_IL21)': [['CD4 Tfh'],
                               ['Bmem_k', 'Bmem_l', 'B cell IL6+', 'GC B cell',
                                'CD4 Treg', 'gdTnaive', 'CD8 Trm', 'CD8 GZMK+', 'g9d2T', 'ILC1']],
    'Inflam-Chronic_Retention(CCL15&CCL23)': [['Col.GN+', 'Quie.SC', 'Goblet', 'TA cell'],
                                              ['M2 Mph', 'C.Mono.', 'M1 Mph', 'cDC2', 'Neutro']],
    'Inflam-Acute_5(CXCR3)': [['M1 Mph', 'M2 Mph', 'cDC2'],
                              ['CD8 Trm', 'gdTnaive', 'CD8 GZMK+', 'CD4 Treg', 'gdTrm', 'CD4 GZMK+']]
    
}

for k, v in top_cell_dict.items():
    search_dict = {"interaction_group": cell_chat_dict[k]}
    df_full = search_df(df_all, search_dict=search_dict)
    df_sorted, vline_pairs = vline_generator(df_full, by_ligand=False,
                                             ligand_order=v[0],
                                             receptor_order=v[1])
    
    # 自动尺寸
    n_celltypes = df_sorted["celltype_group"].nunique()
    n_interactions = df_sorted["interaction_group"].nunique()
    n_groups = df_sorted["group"].nunique()
    min_fig_width = 6;
    width_per_celltype = 0.30;
    extra_width = 6;
    extra_height = 10;
    facet_height = 0.4
    fig_width = max(min_fig_width,
                    n_celltypes * width_per_celltype + extra_width)
    fig_height = n_groups * n_interactions * facet_height + extra_height
    print(f"Print with width = {fig_width} and height = {fig_height}")
    
    draw_combine_dotplot(df_sorted, save_addr=analy_fig_dir,
                         filename=f"PPI_{k}",  # ←←←← 必须改
                         vline_pairs=vline_pairs,
                         facet_aspect=3,  # 如果点和点之间上下左右不协调，调整这个
                         facet_height=1.1,  # 如果 group = xxx 位置过于拥挤，增加这个
                         fig_width=fig_width, fig_height=fig_height,
                         # 每种疾病只有一对交互（df_sorted.interaction_group）参考值为 height=12, 两个交互 height=14
                         # 宽度最小为 6，410 row 1 个交互 约为 30，1500 row 2 个交互 约为 50
                         bottom=0.2, top=0.95,
                         interaction_order=search_dict["interaction_group"]
                         )

# 单独进行
print(cell_chat_dict.keys())

key = 'TNFSF_2(TRAIL)'
search_dict = {"interaction_group": cell_chat_dict[key]}

key = 'Neutrophil(CXCR2)'
search_dict = {'interaction_group': ['CXCL1-CXCR2', 'CXCL2-CXCR2', 'CXCL3-CXCR2']}

df_full = search_df(df_all, search_dict=search_dict)
print(df_full["cell_left"].unique())
print(df_full["cell_right"].unique())

df_sorted, vline_pairs = vline_generator(df_full, by_ligand=False,
                                         ligand_order=top_cell_dict[key][0],
                                         receptor_order=top_cell_dict[key][1])

# 自动尺寸
n_celltypes = df_sorted["celltype_group"].nunique()
n_interactions = df_sorted["interaction_group"].nunique()
n_groups = df_sorted["group"].nunique()
min_fig_width = 6;
width_per_celltype = 0.30;
extra_width = 6
extra_height = 10;
facet_height = 0.4
fig_width = max(min_fig_width,
                n_celltypes * width_per_celltype + extra_width)
fig_height = n_groups * n_interactions * facet_height + extra_height
print(f"Print with width = {fig_width} and height = {fig_height}")

draw_combine_dotplot(df_sorted, save_addr=analy_fig_dir,
                     filename=f"PPI_{key}",  # ←←←← 必须改
                     vline_pairs=vline_pairs,
                     facet_aspect=3,  # 如果点和点之间上下左右不协调，调整这个
                     facet_height=1.1,  # 如果 group = xxx 位置过于拥挤，增加这个
                     fig_width=fig_width, fig_height=fig_height,
                     # 每种疾病只有一对交互（df_sorted.interaction_group）参考值为 height=12, 两个交互 height=14
                     # 宽度最小为 6，410 row 1 个交互 约为 30，1500 row 2 个交互 约为 50
                     bottom=0.2, top=0.95,
                     interaction_order=search_dict["interaction_group"]
                     )

####################################
# 绘制基因表达-基因来源泡泡图
####################################################
# 将细胞按照高表达和低表达分为两组
adata_sub = anndata.read_h5ad(
    "/public/home/xiongyuehan/data/IBD_analysis/output/Step07/Step07a_Summary/final_plus/01141549_T Cell.h5ad")

####################################################
# 绘制 chord

#########
mat = build_interaction_matrix(
    df_all,
    # cells=cells,
    # interactions=interactions,
    weight_col="scores"
)
fig, ax = plot_chord_diagram(
    mat,
    min_weight=0.2,
    title="Total signaling",
    # cell_colors=cell_colors
)

plt.savefig(f"{analy_fig_dir}/Chord_All.pdf", bbox_inches="tight")
plt.savefig(f"{analy_fig_dir}/Chord_All.png", dpi=300, bbox_inches="tight")
plt.close()
#########
chord_dict = {
    "Total": None,
    "Myeloid_Attractant": ["CCL3-CCR1", "CCL3-CCR3", "CCL3-CCR5",
                           "CCL4-CCR5",
                           "CCL5-CCR1", "CCL5-CCR3", "CCL5-CCR5",
                           "CCL8-CCR1", "CCL8-CCR2",
                           "CCL15-CCR1", "CCL23-CCR1",
                           "CXCL1-CXCR2", "CXCL2-CXCR2", "CXCL3-CXCR2"],
    "Lymphoid_Attractant": ["CX3CL1-CX3CR1", "CXCL16-CXCR6", 'CCL20-CCR6',
                            'CXCL9-CXCR3', 'CXCL10-CXCR3', 'CXCL11-CXCR3',
                            'CCL5-CCR4', 'CCL18-CCR7'],
}
cell_type_dict = {
    "Myeloid": ['C.Mono.', 'NC.Mono.',
                "Macrophage", "M1 Mph", "M2 Mph", 'Neutro',
                'cDC1', 'cDC2', 'pDC', 'Mast cell', ],
    "Epithelium": ['ISC', 'pre_TA cell', 'TA cell', 'Quie.SC', 'M_like cell',
                   'Col.Prog.', 'Colon', 'Col.GN+', 'Col.BEST4+',
                   'M cell', 'Goblet', 'Paneth cell', 'Endocrine.', 'Tuft cell', ],
    "Stromal Cell": ['Fibroblast', 'Fb act.', 'Endothelium'],
    "B-Plasma Cell": ['Bmem_k', 'Bmem_l', 'B cell IL6+', 'GC B cell', 'Plasma IgA+', 'Plasma IgG+'],
    'T-NK Cell': ['CD4 Tnaive', 'CD4 Tmem', 'CD4 Tfh', 'CD4 Th17', 'CD4 Treg', 'CD4 GZMK+',
                  'CD8 Tnaive', 'CD8 Tmem', 'CD8 GZMK+', 'CD8 Trm', 'CD8 Trm GZMA+', 'CD8+ NKT',
                  'CD8aa IEL',
                  'g9d2T', 'gdTnaive', 'gdTrm',
                  'NK CD16+', 'NK CD56+', 'ILC1', 'ILC3', 'MAIT', ]
}

for k, v in chord_dict.items():
    mat = build_interaction_matrix(
        df_all,
        # cells=['Macrophage', 'M1 Mph','M2 Mph', 'Neutro'],
        interactions=v,
        weight_col="scores"
    )
    mat = prune_cells_by_activity(mat, min_value=10)
    
    fig, ax = plot_chord_diagram(
        mat,
        min_weight=0.2,
        title="Myeloid Chemotaxis signaling",
        group_cells=cell_type_dict,
        group_arc_width=4
        # cell_colors=cell_colors
    )
    plt.savefig(f"{analy_fig_dir}/Chord_{k}.pdf", bbox_inches="tight")
    plt.savefig(f"{analy_fig_dir}/Chord_{k}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    for i in df_all["group"].unique():
        df_sub = df_all[df_all["group"] == i]
        mat = build_interaction_matrix(
            df_sub,
            # cells=['Macrophage', 'M1 Mph','M2 Mph', 'Neutro'],
            interactions=v,
            weight_col="scores"
        )
        mat = prune_cells_by_activity(mat, min_value=10)
        
        fig, ax = plot_chord_diagram(
            mat,
            min_weight=0.2,
            title=f"Group {i}",
            group_cells=cell_type_dict,
            group_arc_width=4
            
            # cell_colors=cell_colors
        )
        
        del mat
        plt.savefig(f"{analy_fig_dir}/Chord_{k}_{i}.pdf", bbox_inches="tight")
        plt.savefig(f"{analy_fig_dir}/Chord_{k}_{i}.png", dpi=300, bbox_inches="tight")
        plt.close()

#########
# 白塞病情况
big_chemotaxis_list = chord_dict["Myeloid_Attractant"] + chord_dict["Lymphoid_Attractant"]
#######
for dis in df_all["group"].unique():
    df_sub = df_all[df_all["group"] == dis]
    df_sub = df_sub[df_sub["interaction_group"].isin(big_chemotaxis_list)]
    nodes, links_df = CCI_sankey_table(df_sub, center_cell=cell_type_dict["Myeloid"], min_score=10.0)
    CCI_sankey_plot_top5(nodes, links_df,
                         save_addr=analy_fig_dir,
                         filename=f"SankeyPlot_Myeloid-centered_chemotaxis_{dis}",
                         title="Myeloid-centered Chemotaxis")

#############################
cell_of_interest = ['ISC', 'Quie.SC', 'TA cell']

df_full = search_df(df_all, search_dict={'cell_right': cell_of_interest})
group_cols = ["interaction_group", "celltype_group"]

mask_keep = (
    df_full
    .groupby(group_cols)["scaled_means"]
    .transform(lambda x: (x != 0).any())
)

df = df_full[mask_keep].copy()

# df = pd.read_csv(f"{analy_fig_dir}/Celltype_Filter_by_Stem_undiff.csv",index_col=0)

# -----------------------------
# 1. 定义分组
# -----------------------------
group_A = ["UC", "Colitis"]  # inflamed
group_B = ["HC", "CD"]  # control-like

# -----------------------------
# 2. 只保留这四类 group
# -----------------------------
df_sub = df[df["group"].isin(group_A + group_B)].copy()

# -----------------------------
# 3. 标记大组
# -----------------------------
df_sub["group_bin"] = np.where(
    df_sub["group"].isin(group_A),
    "Inflamed",
    "Control"
)

# -----------------------------
# 4. 按 interaction 聚合
# -----------------------------
key_cols = [
    "interaction_group",
    "cell_left",
    "cell_right"
]

agg = (
    df_sub
    .groupby(key_cols + ["group_bin"])
    .agg(
        mean_scaled=("scaled_means", "mean"),
        mean_score=("scores", "mean"),
        min_pval=("pvals", "min")
    )
    .reset_index()
)

# -----------------------------
# 5. pivot 成宽表
# -----------------------------
wide = agg.pivot_table(
    index=key_cols,
    columns="group_bin",
    values=["mean_scaled", "mean_score", "min_pval"]
)

# flatten columns
wide.columns = [
    f"{a}_{b}" for a, b in wide.columns
]
wide = wide.reset_index()

# -----------------------------
# 6. 计算差异指标
# -----------------------------
wide["delta_scaled"] = (
        wide["mean_scaled_Inflamed"] -
        wide["mean_scaled_Control"]
)

wide["abs_delta_scaled"] = wide["delta_scaled"].abs()

wide["log2FC_scaled"] = np.log2(
    (wide["mean_scaled_Inflamed"] + 1e-6) /
    (wide["mean_scaled_Control"] + 1e-6)
)

# -----------------------------
# 7. 筛选“差异较大”的 interaction
# （阈值你可以自己调）
# -----------------------------
result = wide[
    (wide["abs_delta_scaled"] > 0.3) &  # 表达差异
    (wide["min_pval_Inflamed"] < 0.05)  # 至少在 inflamed 里显著
    ].sort_values(
    "abs_delta_scaled",
    ascending=False
)

# 查看结果
result.head(20)

result.to_csv(f"{analy_fig_dir}/Colitis_vs._HC&CD(Stem_undiff).csv")