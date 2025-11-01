"""
Step09b_Pyscenic_analysis.py
Author: John Hsiung
Update date: 2025-08-29
Description:
    - 进行 pyscenic 结果的分析
Notes:
    - 使用环境：conda activate scvpy10
"""
import pandas as pd
import os, gc, sys
import numpy as np
sys.path.append('/data/HeLab/bio/IBD_analysis/')


from src.EasyInterface.PyscenicTools import get_regulons_auc_from_h5,get_regulons_incid_matrix, regulons_to_gene_lists,get_most_var_regulon,scale_tf_matrix,plot_rss_heatmap, pyscenic_pheatmap,calc_rss,plot_rss_one_set

###################################################
# 批量读取存储和进行初步处理
# 目前进行测试


save_dir = "/data/HeLab/bio/IBD_analysis/output/Step09_PYSCENIC/Pyscenic_0621"
output_dir = "/data/HeLab/bio/IBD_analysis/output/Step09_PYSCENIC/Output_0621"

file_list = os.listdir(save_dir)

for file in file_list[7:]:
    print(file)
    os.makedirs(f"{output_dir}/{file}",exist_ok=True)
    readin_loom = f"{save_dir}/{file}/Output.loom"
    print(f"Reading from loom file {readin_loom}.")
    
    # 读取 AUC 矩阵
    regulon_AUC = get_regulons_auc_from_h5(loom_file = readin_loom)
    # 通过查表把 regulon 恢复为字典，keys 为文件中包含的 regulon；values 为 regulon 所对应的基因
    regulons_incidMat = get_regulons_incid_matrix(loom_file = readin_loom)
    regulons = regulons_to_gene_lists(regulons_incidMat)
    # 读取 meta 信息，用来匹配细胞的身份
    meta = pd.read_csv(f"{save_dir}/{file}/meta_data.csv")
    meta.columns = ['Barcodes', 'orig.ident', 'Cell_Identity','disease'] # 注意：Cell_Idenity列可以为任意
    
    # 筛选感兴趣的转录因子
    tf_oi = get_most_var_regulon(data = regulon_AUC,fit_thr = 0.5,min_mean_for_fit = 0.001,
                                 picture = True,plt_savedir = f"{output_dir}/{file}",plt_name=f"{file}_cv2_summary"
    )
    tf_all = get_most_var_regulon(data = regulon_AUC,fit_thr = 0,min_mean_for_fit = 0)
    
    tf_oi_scaled = scale_tf_matrix(tf_oi)
    tf_all_scaled = scale_tf_matrix(tf_all)
    
    pyscenic_pheatmap(tf_oi_scaled,meta,plt_savedir=f"{output_dir}/{file}",plt_name=f"{file}_tf_oi_heatmap")
    pyscenic_pheatmap(tf_all_scaled,meta,plt_savedir=f"{output_dir}/{file}",plt_name=f"{file}_tf_all_heatmap")
    pyscenic_pheatmap(tf_all_scaled,meta,plt_savedir=f"{output_dir}/{file}",plt_name=f"{file}_by_disease_tf_all_heatmap",
                      meta_key=["disease","Cell_Identity"])
    
    # regulon特异性分数(Regulon Specificity Score, RSS)：基于 Jensen-Shannon Divergence
    selectedResolution = "celltype"
    rss = calc_rss(regulon_AUC, cell_annotation=meta["Cell_Identity"])
    plot_rss_heatmap(rss,
                     plt_savedir=f"{output_dir}/{file}",plt_name=f"{file}_RSS",
                     thr=0.01) # 阈值设置为0.01，用来尽量打印非0的全部
    for set in rss.columns:
        plot_rss_one_set(rss,
                         plt_savedir=f"{output_dir}/{file}",plt_name=f"{file}_{set}_RSS_Rank",set_name=set)



