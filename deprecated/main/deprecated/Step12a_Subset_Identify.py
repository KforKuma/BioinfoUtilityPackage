# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 备注：本项目主要将Step11整合、清理之后的代码进行重新拆分，并分类保存，以供进一步检查。
# 并进行笼统地重新分群。
# 运行环境: scvpy10
# 主要输出：Step12_cleansed
# 以及拆分的小群，格式如：/data/HeLab/bio/IBD_analysis/tmp/12_adata_En_final.h5ad, .../12_adata_T_final
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##——————————————————————————————————————————————————————————————————————————
import scanpy as sc
import anndata
import os
##——————————————————————————————————————————————————————————————————————————
# 设置scanpy基本属性
import yaml
with open('/data/HeLab/bio/IBD_analysis/assets/io_config.yaml', 'r') as f:
    config = yaml.safe_load(f) # 读取的格式为一系列字典的列表
##——————————————————————————————————————————————————————————————————————————
os.chdir("/data/HeLab/bio/IBD_analysis/")
from src.ScanpyTools.ScanpyTools import ScanpyPlotWrapper, AnnotationMaker, Geneset, filter_and_save_adata

rank_genes_groups_dotplot = ScanpyPlotWrapper(func = sc.pl.rank_genes_groups_dotplot)
dotplot = ScanpyPlotWrapper(func = sc.pl.dotplot)
umap_plot = ScanpyPlotWrapper(func = sc.pl.umap)
tsne_plot = ScanpyPlotWrapper(func = sc.pl.tsne)
##——————————————————————————————————————————————————————————————————————————
sc.set_figure_params(dpi_save=450, color_map = 'viridis_r',fontsize=6)
sc.settings.verbosity = 1
sc.logging.print_header()
##——————————————————————————————————————————————————————————————————————————
# # 读取原始数据
adata = anndata.read(config["Step11"]["general_output_anndata"])

# remap_disease = {'HRR025262':'Control', 'HRR025263':'Control', 'HRR025264':'Control', 'HRR025265':'Control', 'HRR025266':'Control', 'HRR025267':'Control', 'HRR025268':'Control', 'HRR025269':'Control', 'HRR025278':'Control', 'HRR025279':'Control', 'HRR025280':'Control', 'HRR025281':'Control', 'HRR025282':'Control', 'HRR025283':'Control', 'HRR025284':'Control', 'HRR025285':'Control', 'HRR025294':'UC', 'HRR025295':'UC', 'HRR025296':'UC', 'HRR025297':'UC', 'HRR025298':'UC', 'HRR025299':'UC', 'HRR025300':'UC', 'HRR025301':'UC', 'HRR025310':'UC', 'HRR025311':'UC', 'HRR025312':'UC', 'HRR025313':'UC', 'HRR025314':'UC', 'HRR025315':'UC', 'HRR025316':'UC', 'HRR025317':'UC', 'HRR025326':'Colitis', 'HRR025327':'Colitis', 'HRR025328':'Colitis', 'HRR025329':'Colitis', 'HRR025330':'Colitis', 'HRR025331':'Colitis', 'HRR025332':'Colitis', 'HRR025333':'Colitis', 'HRR025342':'Control', 'HRR025343':'Control', 'HRR025344':'Control', 'HRR025345':'Control', 'HRR025346':'Control', 'HRR025347':'Control', 'HRR025348':'Control', 'HRR025349':'Control', 'HRR025358':'Control', 'HRR025359':'Control', 'HRR025360':'Control', 'HRR025361':'Control', 'HRR025362':'Control', 'HRR025363':'Control', 'HRR025364':'Control', 'HRR025365':'Control', 'HRR025374':'Colitis', 'HRR025375':'Colitis', 'HRR025376':'Colitis', 'HRR025377':'Colitis', 'HRR025378':'Colitis', 'HRR025379':'Colitis', 'HRR025380':'Colitis', 'HRR025381':'Colitis', 'HRR025390':'CD', 'HRR025391':'CD', 'HRR025392':'CD', 'HRR025393':'CD', 'HRR025394':'CD', 'HRR025395':'CD', 'HRR025396':'CD', 'HRR025397':'CD', 'HRR025406':'Colitis', 'HRR025407':'Colitis', 'HRR025408':'Colitis', 'HRR025409':'Colitis', 'HRR025410':'Colitis', 'HRR025411':'Colitis', 'HRR025412':'Colitis', 'HRR025413':'Colitis', 'HRR025422':'CD', 'HRR025423':'CD', 'HRR025424':'CD', 'HRR025425':'CD', 'HRR025426':'CD', 'HRR025427':'CD', 'HRR025428':'CD', 'HRR025429':'CD', 'HRR025438':'Control', 'HRR025439':'Control', 'HRR025440':'Control', 'HRR025441':'Control', 'HRR025442':'Control', 'HRR025443':'Control', 'HRR025444':'Control', 'HRR025445':'Control', 'HRR025454':'Control', 'HRR025455':'Control', 'HRR025456':'Control', 'HRR025457':'Control', 'HRR025458':'Control', 'HRR025459':'Control', 'HRR025460':'Control', 'HRR025461':'Control', 'HRR025470':'CD', 'HRR025471':'CD', 'HRR025472':'CD', 'HRR025473':'CD', 'HRR025474':'CD', 'HRR025475':'CD', 'HRR025476':'CD', 'HRR025477':'CD', 'HRR025486':'Colitis', 'HRR025487':'Colitis', 'HRR025488':'Colitis', 'HRR025489':'Colitis', 'HRR025490':'Colitis', 'HRR025491':'Colitis', 'HRR025492':'Colitis', 'HRR025493':'Colitis', 'HRR025502':'Colitis', 'HRR025503':'Colitis', 'HRR025504':'Colitis', 'HRR025505':'Colitis', 'HRR025506':'Colitis', 'HRR025507':'Colitis', 'HRR025508':'Colitis', 'HRR025509':'Colitis', 'HRR025518':'Colitis', 'HRR025519':'Colitis', 'HRR025520':'Colitis', 'HRR025521':'Colitis', 'HRR025522':'Colitis', 'HRR025523':'Colitis', 'HRR025524':'Colitis', 'HRR025525':'Colitis',
#                  'A3':'CD', 'A2':'CD', 'A1':'Control', 'B3':'CD', 'B2':'CD', 'B1':'Control', 'C3':'CD', 'C2':'CD', 'C1':'Control', 'GSM7041323_Peri_P5':'CD', 'GSM7041324_Peri_P6':'CD', 'GSM7041325_Peri_P7':'CD', 'GSM7041326_Peri_P8':'CD', 'GSM7041327_Peri_P11':'CD', 'GSM7041328_Peri_P13':'CD', 'GSM7041329_Peri_P15':'CD', 'GSM7041330_Peri_P16':'CD', 'GSM7041331_Peri_P17':'CD', 'GSM7041332_Peri_P18':'CD', 'GSM7041333_Peri_P19':'Control', 'GSM7041334_Peri_P20':'CD', 'GSM7041335_Peri_P21':'CD',
#                  'BS1':"BS", 'BS2':"BS", 'CD1':"CD", 'CD2':"CD", 'CD3':"CD", 'CD4':"CD", 'N1':"Control", 'N2':"Control", 'N3':"Control", 'N4':"Control", 'N5':"Control", 'UC1':"UC", 'UC2':"UC", 'UC3':"UC"}
# adata.obs["disease"] = adata.obs["orig.ident"].map(remap_disease)

# 定义细胞类型与对应的文件名
cell_groups = {
    'T_and_ILC_gdT': (["T", "ILC-gdT"], 'Step12_adata_T.h5ad'),
    'BP_group': (["Plasma", "GC-B", "B"], 'Step12_adata_BP.h5ad'),
    'Epi_group': (["Epi", "Stem.Epi", "Abs.Epi", "Sec.Epi", "Goblet", "Ent.Endocrine", "Tuft Cell"], 'Step12_adata_Epi.h5ad'),
    'Macrophage': (["Mph"], 'Step12_adata_M.h5ad'),
    'Fibroblast': (["Fibroblast"], 'Step12_adata_Fb.h5ad'),
    'Endothelium': (["Endothelium"], 'Step12_adata_En.h5ad')
}

# 遍历每个细胞群体进行筛选和保存
os.chdir("/data/HeLab/bio/IBD_analysis/tmp")
for group, (cell_types, file_name) in cell_groups.items():
    filter_and_save_adata(adata, cell_types, file_name)
os.chdir("/data/HeLab/bio/IBD_analysis/")
##——————————————————————————————————————————————————————————————————————————
# import importlib
## 重新加载该模块
# importlib.reload(src.EasyInterface.ScanpyTools)
# 导入自定义函数
from src.ScanpyTools.ScanpyTools import update_doublet, process_adata
##——————————————————————————————————————————————————————————————————————————
# 绘制主要marker点图
my_markers = Geneset(config['Env']['assets']+"Markers.xlsx") # 读取为Geneset对象
## 调用函数
adata_subset = anndata.read("/data/HeLab/bio/IBD_analysis/tmp/Step12_adata_T.h5ad")
process_adata(adata_subset = adata_subset, file_name = "T_Cell",
              my_markers = my_markers, marker_sheet = "thymocyte",output_dir="/data/HeLab/bio/IBD_analysis/output/Step12/T_Cell/",
              do_subcluster=True, do_DEG_enrich=True,
              DEG_enrich_key="leiden_res",resolutions_list = [1.5])
## 鉴定
ann_list = ["Tcm",
            "CD8 Trm","CD8 Tem","CD8 Tem","DN T","CD4 Tem",
            "CD8 Trm-CD16 NK","Tcm","CD4 Tfh","CD8 Trm","CD4 Treg-Th17", # 10
            "CD4 Treg-Th17","CD4 Treg","CD4 Tem","DN T-CD8 Trm","CD8 Tem", # 15
            "ILC3-gdT-CD56NK","Tcm","CD8 Trm proliferative","CD8 Tem","CD8 Tcm", # 20
            "CD8 Tcm","CD8 Trm","Tcm","CD8 Trm-CD56 NK","CD8 stem-like", # 25
            "CD4 Tem"]
make_anno = AnnotationMaker(adata_subset,
                            obs_key = "leiden_res1.5",
                            anno_key = "manual_cellsubtype_annotation")
make_anno.annotate_by_list(ann_list)
make_anno.make_annotate()
## 再次调用函数
process_adata(adata_subset = make_anno.data, file_name = "T_Cell_annotated",
              my_markers = my_markers, marker_sheet = "thymocyte",output_dir="/data/HeLab/bio/IBD_analysis/output/Step12/T_Cell/",
              do_subcluster=False, do_DEG_enrich=True,
              DEG_enrich_key="manual_cellsubtype_annotation")
## 存储
make_anno.data.write_h5ad('/data/HeLab/bio/IBD_analysis/tmp/12_adata_T_final.h5ad')
##——————————————————————————————————————————————————————————————————————————
##——————————————————————————————————————————————————————————————————————————
## 调用函数
adata_subset = anndata.read("/data/HeLab/bio/IBD_analysis/tmp/Step12_adata_Epi.h5ad")
process_adata(adata_subset = adata_subset, file_name = "Epithelial",
              my_markers = my_markers, marker_sheet = "epithelium",output_dir="/data/HeLab/bio/IBD_analysis/output/Step12/Epithelium/",
              do_subcluster=True, do_DEG_enrich=True,
              DEG_enrich_key="leiden_res",resolutions_list = [1.5])
## 鉴定
ann_list = ["Secretive",
            "Tuft cell","Epi. Stem cell","Absorp","Goblet","Absorp", # 5
            "Absorp","Intermediate","Epi. Stem cell","Goblet","Absorp", # 10
            "Intermediate","Absorp","Absorp","Secretive","Goblet", # 15
            "Epi. Stem cell","Goblet","Absorp","Epi. Stem cell","Absorp", # 20
            "Goblet","Secretive","Enteroendocrine","Tuft cell","Goblet", #25
            "Doublet","Doublet","Doublet","Doublet","Epi. Stem cell" # 30
            ]
make_anno = AnnotationMaker(adata_subset,obs_key = "leiden_res1.5",
                            anno_key = "manual_cellsubtype_annotation")
make_anno.annotate_by_list(ann_list)
make_anno.make_annotate()
## 再次调用函数
process_adata(adata_subset = make_anno.data, file_name = "Epithelial_annotated",
              my_markers = my_markers, marker_sheet = "epithelium",output_dir="/data/HeLab/bio/IBD_analysis/output/Step12/Epithelium/",
              do_subcluster=False, do_DEG_enrich=True,
              DEG_enrich_key="manual_cellsubtype_annotation")
adata = update_doublet(adata, adata_subset, obs_key="manual_cellsubtype_annotation", delete=True)
adata_subset = update_doublet(adata_subset, adata_subset, obs_key="manual_cellsubtype_annotation", delete=True)
adata.write("/data/HeLab/bio/IBD_analysis/tmp/Step12_cleansed.h5ad")

## 存储
make_anno.data.write_h5ad('/data/HeLab/bio/IBD_analysis/tmp/12_adata_Epi_final.h5ad')
##——————————————————————————————————————————————————————————————————————————
##——————————————————————————————————————————————————————————————————————————
## 调用函数
adata_subset = anndata.read("/data/HeLab/bio/IBD_analysis/tmp/Step12_adata_M.h5ad")
process_adata(adata_subset = adata_subset, file_name = "Myeloid",
              my_markers = my_markers, marker_sheet = "myeloid",output_dir="/data/HeLab/bio/IBD_analysis/output/Step12/Myeloid/",
              do_subcluster=True, do_DEG_enrich=True,
              DEG_enrich_key="leiden_res",resolutions_list = [1.5])
## 鉴定
ann_list = ["Macrophage",
            "DC","Neutrophil","M1","Neutrophil","Neutrophil", # 5
            "M2-like","M2-like","M1","DC","M2", # 10
            "DC","M2","M1","M1","Neutrophil", # 15
            "DC","DC","DC","DC","M2", # 20
            "Myeloid proliferative","M2-like","Macrophage","Microglia" # 24
            ]
make_anno = AnnotationMaker(adata_subset,obs_key = "leiden_res1.5",
                            anno_key = "manual_cellsubtype_annotation")
make_anno.annotate_by_list(ann_list)
# print(f"Annotation dict: {make_anno.anno_dict}")
# print(f"Annotation keys: {make_anno.anno_key}")
make_anno.make_annotate()
## 再次调用函数
process_adata(adata_subset = make_anno.data, file_name = "Myeloid_annotated",
              my_markers = my_markers, marker_sheet = "myeloid",output_dir="/data/HeLab/bio/IBD_analysis/output/Step12/Myeloid/",
              do_subcluster=False, do_DEG_enrich=True,
              obs_subset=True,
              DEG_enrich_key="manual_cellsubtype_annotation")

## 存储
make_anno.data.write_h5ad('/data/HeLab/bio/IBD_analysis/tmp/12_adata_M_final.h5ad')
##——————————————————————————————————————————————————————————————————————————
##——————————————————————————————————————————————————————————————————————————
## 调用函数
adata_subset = anndata.read("/data/HeLab/bio/IBD_analysis/tmp/Step12_adata_BP.h5ad")
process_adata(adata_subset = adata_subset, file_name = "BP",
              my_markers = my_markers, marker_sheet = "B_cell",output_dir="/data/HeLab/bio/IBD_analysis/output/Step12/BP/",
              do_subcluster=True, do_DEG_enrich=True,
              obs_subset=True,
              DEG_enrich_key="leiden_res",resolutions_list = [1.5])
## 鉴定
ann_list = ["B memory",
            "B memory","B germinal","B memory","B memory","B memory", # 5
            "B memory","B memory","B memory","B memory","B memory", # 10
            "B memory","B memory","B memory","Plasma","B memory", # 15
            "Plasma","Plasma","B memory","B memory","B germinal", # 20
            "B memory","B memory","B memory","B memory","B memory", # 25
            "B memory","B memory","Plasma","B germinal","B memory", # 30
            "B memory"
            ]
make_anno = AnnotationMaker(adata_subset,obs_key = "leiden_res1.5",
                            anno_key = "manual_cellsubtype_annotation")
make_anno.annotate_by_list(ann_list)
make_anno.make_annotate()
## 再次调用函数
process_adata(adata_subset = make_anno.data, file_name = "BP_annotated",
              my_markers = my_markers, marker_sheet = "B_cell",output_dir="/data/HeLab/bio/IBD_analysis/output/Step12/BP/",
              do_subcluster=False, do_DEG_enrich=True,
              obs_subset=True,
              DEG_enrich_key="manual_cellsubtype_annotation")

## 存储
make_anno.data.write_h5ad('/data/HeLab/bio/IBD_analysis/tmp/12_adata_BP_final.h5ad')
##——————————————————————————————————————————————————————————————————————————
##——————————————————————————————————————————————————————————————————————————
## 调用函数
adata_subset = anndata.read("/data/HeLab/bio/IBD_analysis/tmp/Step12_adata_Fb.h5ad")
process_adata(adata_subset = adata_subset, file_name = "Fb",
              my_markers = my_markers, marker_sheet = "fibroblast",output_dir="/data/HeLab/bio/IBD_analysis/output/Step12/Fb/",
              do_subcluster=True, do_DEG_enrich=True,
              # obs_subset=True,
              DEG_enrich_key="leiden_res",resolutions_list = [1.5])
## 鉴定
ann_list = ["Fibroblast",
            "Fibroblast","Fibroblast","Fibroblast","Fibroblast","Fibroblast",
            "Fibroblast PDE4D+","Fibroblast","Fibroblast","Fibroblast","Fibroblast",
            "Fibroblast","Fibroblast","Fibroblast","Fibroblast","Fibroblast",
            "Fibroblast","Fibroblast","Fibroblast","Fibroblast","Neuroendocrine",
            "Fibroblast","Fibroblast"
            ]
make_anno = AnnotationMaker(adata_subset,obs_key = "leiden_res1.5",
                            anno_key = "manual_cellsubtype_annotation")
make_anno.annotate_by_list(ann_list)
make_anno.make_annotate()
## 再次调用函数
process_adata(adata_subset = make_anno.data, file_name = "Fb_annotated",
              my_markers = my_markers, marker_sheet = "fibroblast",output_dir="/data/HeLab/bio/IBD_analysis/output/Step12/Fb/",
              do_subcluster=False, do_DEG_enrich=True,
              obs_subset=True,
              DEG_enrich_key="manual_cellsubtype_annotation")

## 存储
make_anno.data.write_h5ad('/data/HeLab/bio/IBD_analysis/tmp/12_adata_Fb_final.h5ad')
##——————————————————————————————————————————————————————————————————————————
##——————————————————————————————————————————————————————————————————————————
## 调用函数
adata_subset = anndata.read("/data/HeLab/bio/IBD_analysis/tmp/Step12_adata_En.h5ad")
process_adata(adata_subset = adata_subset, file_name = "En",
              my_markers = my_markers, marker_sheet = "endothelium",output_dir="/data/HeLab/bio/IBD_analysis/output/Step12/En/",
              do_subcluster=True, do_DEG_enrich=True,
              # obs_subset=True,
              DEG_enrich_key="leiden_res",resolutions_list = [1.5])
## 鉴定
ann_list = ["Endothelium",
            "Endothelium","Endothelium","Endothelium","Endothelium","Endothelium",
            "Endothelium","Endothelium CLEC3B+","Endothelium","LEC","Endothelium",
            "Endothelium CLEC3B+","Endothelium"
            ]
make_anno = AnnotationMaker(adata_subset,obs_key = "leiden_res1.5",
                            anno_key = "manual_cellsubtype_annotation")
make_anno.annotate_by_list(ann_list)
make_anno.make_annotate()
## 再次调用函数
process_adata(adata_subset = make_anno.data, file_name = "En_annotated",
              my_markers = my_markers, marker_sheet = "endothelium",output_dir="/data/HeLab/bio/IBD_analysis/output/Step12/En/",
              do_subcluster=False, do_DEG_enrich=True,
              obs_subset=True,
              DEG_enrich_key="manual_cellsubtype_annotation")

## 存储
make_anno.data.write_h5ad('/data/HeLab/bio/IBD_analysis/tmp/12_adata_En_final.h5ad')
##——————————————————————————————————————————————————————————————————————————


