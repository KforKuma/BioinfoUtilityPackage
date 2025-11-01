# 加载内容
# from main.Step11_urgent_repair import spline_df
dict_for_pseudo.keys()
# dict_for_pseudo.keys().index("T Cell_MAIT")
i=25
name = list(dict_for_pseudo.keys())[i]
ident = dict_for_pseudo[name]
combined_flag = isinstance(ident, list) and len(ident) > 1
proj_name = name.replace(" ", "_")
save_addr = f"/data/HeLab/bio/IBD_analysis/output/Step11_scfates/{proj_name}"


###################################################
# 自定义函数
from src.scFatesTools.scFatesTools import easy_split, easy_hvg, summarize_nodes, fork_check, filter_adata_by_seg
###################################################
# 备选：如果整个palantir principal graph看起来太奇怪，需要重新跑一遍上游
# 但 hvg 和之前步骤应该是没必要了
adata_allgenes = anndata.read_h5ad(f"{save_addr}/01_palantirred.h5ad")
sc.pp.pca(adata_allgenes, use_highly_variable=True)

sc.pl.pca_variance_ratio(adata_allgenes, log=False, save="PCA_ratio.png")

pca_projections = pd.DataFrame(adata_allgenes.obsm["X_pca"], index=adata_allgenes.obs_names)

# 使用自适应各向异性核运行扩散图。
dm_res = palantir.utils.run_diffusion_maps(pca_projections,seed=0,
                                           n_components=50, # 默认值 10
                                           alpha=1, # 默认值 0，不考虑密度（保留稀疏区细节）；最大为 1
                                           knn=100 # 默认值 30; 过大 → 全局模糊、分支不清。
                                           )

# 确定数据的多尺度空间
ms_data = palantir.utils.determine_multiscale_space(dm_res,
                                                    n_eigs=5 # 默认为 None；若结构碎裂，减少 n_eigs（如从 30 → 15）
                                                    )
adata_allgenes.obsm["X_palantir"] = ms_data.values
sc.pp.neighbors(adata_allgenes, n_neighbors=30, use_rep="X_palantir")
adata_allgenes.obsm["X_pca2d"] = adata_allgenes.obsm["X_pca"][:, :2]
sc.tl.draw_graph(adata_allgenes, init_pos='X_pca2d')
# --- 富文本形式记录参数说明 ---
adata_allgenes.uns["palantir_params"] = """
dm_res = palantir.utils.run_diffusion_maps(pca_projections,seed=0,
                                           n_components=50, # 默认值 10
                                           alpha=1, # 默认值 0，不考虑密度（保留稀疏区细节）；最大为 1
                                           knn=100 # 默认值 30; 过大 → 全局模糊、分支不清。
                                           )
ms_data = palantir.utils.determine_multiscale_space(dm_res,
                                                    n_eigs=5 # 默认为 None；若结构碎裂，减少 n_eigs（如从 30 → 15）
                                                    )
"""
adata_allgenes.write_h5ad(f"{save_addr}/01_palantirred.h5ad")


###################################################
# 备选：如果树最终看起来太奇怪，需要重新画
adata_allgenes = anndata.read_h5ad(f"{save_addr}/01_palantirred.h5ad")
scf.tl.tree(
    adata_allgenes,
    method="ppt",
    Nodes=100,  # 默认 200，参考 100 ~ 2,000
    use_rep="palantir",
    device="cpu",
    seed=1,
    ppt_lambda=100,  # 默认 100，参考 1；越大越平滑
    ppt_sigma=0.05,  # 默认 0.025，参考值 0.1；越大，邻域越大
    ppt_nsteps=200  # 默认 200，参考 50；不过收敛之后就会停止
)
# --- 富文本形式记录参数说明 ---
adata_allgenes.uns["tree_params"] = """
scf.tl.tree(
    method="ppt",
    Nodes=50,          # 默认 200，参考 100 ~ 2,000
    use_rep="palantir",
    device="cpu",
    seed=1,
    ppt_lambda=100,    # 默认 100，参考 1；越大越平滑
    ppt_sigma=0.05,      # 默认 0.025，参考值 0.1；越大，邻域越大
    ppt_nsteps=200      # 默认 200，参考 50；不过收敛之后就会停止
)
"""
adata_allgenes.write_h5ad(f"{save_addr}/02_treed.h5ad")
###################################################
# 直接读取树
adata_tree = anndata.read_h5ad(f"{save_addr}/02_treed.h5ad")
Celltype = adata_tree.obs["Celltype"][i]
Naive_marker = naive_resting_dict[Celltype]
# print(adata_tree.uns["tree_params"])
###################################################
# 清理树
scf.tl.cleanup(adata_tree)
R = adata_tree.obsm["X_R"]
small_sum_rows = np.where(R.sum(axis=1) < 1e-10)[0]

print("Any NaN in R?", np.isnan(R).any())
print("Any Inf in R?", np.isinf(R).any())
print("Rows with extremely small sum:", small_sum_rows)
###################################################
# 对节点画图
pl_graph(save_addr=save_addr,
         filename="Nodes_Tip(FA)",tips=True,forks=False,
         adata=adata_tree)

pl_graph(save_addr=save_addr,
         filename="Nodes_Fork(FA)",tips=False,forks=True,
         adata=adata_tree)
###################################################
# 查看高表达基因并构建需要绘制基因表达量的基因集
import gseapy as gp
hvg_oi = easy_hvg(adata_tree)
enr = gp.enrichr(gene_list=hvg_oi,
                 gene_sets=['GO_Biological_Process_2023', 'KEGG_2021_Human'],
                 organism='Human',
                 cutoff=0.05)
# 结果是一个 pandas.DataFrame
print(enr.results.head())
resultsdf = enr.results
resultsdf = resultsdf.sort_values(by='Combined Score',ascending=False)
resultsdf.to_csv(f"{save_addr}/hvg_GO_KEGG_enrichr.csv", index=False)

###################################################
# 打印基本信息
summarize_nodes(adata=adata_tree, save_addr=save_addr)
# adata_tree.uns['graph']['tips']
# 检查结构
fork_check(adata_tree)
###################################################
# 对感兴趣的基因、以及基本信息画图
clean_genes, removed = filter_genes(adata_tree.var_names)
print(f"保留基因数: {len(clean_genes)}，移除基因数: {len(removed)}")

# gene_of_interest = []
# goi = ['IL6','VAV2','VAV3','CLASP1','CLASP2','IGHA1','JCHAIN']
# goi = ["CXCL12", "CCL21","PER3","PTGDS","CACNA2D1","CACNA1C","SLC8A1"]
# goi = ["GLRA2","MT2A","MT1M","HLA-DRA","CD55","DDIT3","ZC3H12A","LGALS9C",'DEFA5', 'DEFA6','CXCL1', 'CXCL10',
#        'CXCL11', 'CXCL12', 'CXCL14', 'CXCL16', 'CXCL2', 'CXCL3', 'CXCL5', 'CXCL8', 'CXCL9'] # absorp_col_all
# goi = ["RGS2","ANGPTL4","PLIN5", "CLCN7","SLC26A6", "SLC26A3", "IFITM1","TRIM15","TRIM31","TRIM11"]
# goi = ["DNAJB1","HSPA2","HSPA1B", "EGR1","SOX9", "UBA7","ISG15","TFF3","TFF1","SOX9",'MUC4']
# goi = ["CA2","AGTR1","JUN","MT1G","MT1X","OR51E2","SCTR"]
# goi = easy_split("SELENBP1;TST;ETHE1;BPNT1;PAPSS2;MT2A;MT1M;MT1F;MT1G;MT1X;MT1H")
# goi = easy_split("UBA7;ISG15;PLK2;SHANK2;MT2A;MT1G;MT1X;MT1H;MT1E")
# goi = easy_split("JUN;MT1F;MT1G;MT1X;MT1H;FOS;MT1E;UBA7;ISG15")
# goi = easy_split("HNF1B;MTSS1;FTH1;FTL;JUN;CITED2;PDE4B;SLC9A1")
# goi = easy_split("EDN1;CCL20;PIK3R3;TNFAIP3;CXCL1;FOS;CXCL3;CXCL2;NFKBIA;SOCS3;BCL3;MAP3K8;RIPK1;JUNB;DNAJB1;HSPA2") # Epi.Stem.Cell
# goi = easy_split("NR4A1;TNFRSF25;VEGFA;THBS1;HSPA1B;HSPA1A;NOXA1;NOXO1") # Tuft cell
# goi = easy_split("PTPRS;SEMA4D;ULK1;FGF13;SERPINE1;PLAT;THBS1;KDM4C;UTY;KDM6A") # fibroblast
# goi = easy_split("CCL5;CCL4;CCL3;IL1B;S100A9;S100A8;CREB3;CXCL10;CXCL12;PLA2G7;LGMN") # macro
# goi = easy_split("CXCL8;IL1B;PDE4D;PDE4B;CAMK2D;PRKCE;STK39;NEDD4L;HSPA1B;HSPA1A") # mono
# goi = easy_split("PTPN22;HSPA1B;HSPA1A;IGHA1;IGHA2;JCHAIN;PRKX;SOX4;PDE3A;RAPGEF2;CD74;TBXAS1;PLA2G4A;PTGS2") # mast
# goi = easy_split("CXCL9;CXCL8;CCL20;CXCL3;CXCL2;CXCL10;CXCL11;CCL5;CCL4;PDE4B;CCL3;CCL19;S100A9;S100A8;IL1B;TNF")
# goi = easy_split("CCL5;CCL4;CCL3;INSIG1;RORA;TGFBR2;EGR1;HIF1A;IGHG3;IGHM;IGHG4;IGHG1;IGHG2") # plasma
# goi = easy_split("STAT4;PLCB1;IL12RB2;NR4A1;NR4A3;CCL5;CCL4;RUNX3;RUNX1;KLF2;SREBF2;NCOA1;NCOA2") # CD4_all
# goi = easy_split("IFNG;CD28;GZMB;HLA-DRA;KLRC1;TNF;HLA-DRB1;KLF2;SREBF2;STK39;CCL3;XCL2;XCL1;HSPA1B;HSPA1A") # cd8_all
goi = ['THEMIS','VAV3','GRAP2','RUNX3','SMAD3','RICTOR','FOXO3','RASGRP3','MAP3K5','ADGRE5','MEF2C',"KLRK1","CCL5"] # DN T
# goi = easy_split("CCL5;CCL4;CCL3;XCL1HSPA6;HSPA1B;HSPA1A;PTGER4;GNG2;PTGER2;CD74;CSF1;MYC;IRF7;HLA-DRB1") # ilc xcl1+
# goi = easy_split("STAT5B;JUN;MAML2;CD3G;FOS;GATA3;CD3D;NFKBIA;HLA-DMA;IFNG;STAT4;PRKCQ;MAML3;HLA-DRB1;IL12RB2") # ILC1
# goi = easy_split("TNFSF11;TNFRSF11A;PTGS2;TNF;CCL5;PLAUR;TNFSF11;PRKCA;CD81;HLA-DRA;XCL1;HLA-DRB1;STAT4;PLCB1;IL12RB2") # ilc3
# goi = easy_split("KLRK1;CCL4;CCL3;XCL1;IFNG;TNF;NFKB1") # MAIT
# goi = easy_split("IFNG;TNF;NFKB1;CCL4;CCL3;XCL1;IL23R;NFKBIZ;NFKBID") # innate all
# goi = easy_split("IFNG;HLA-DRA;HLA-DRB1;PDE4D;PDE4B;PLCG2;CD28;CARD11;RUNX1;LEF1;XCL1;GATA3") # gdT_all
goi = easy_split("IFNG;TNF;NFKB1;TRGC1;JAML;TCF7;CD3G;TRGC2;PTPN22;HSPA1B;HSPA1A") # NK_all
# gene_of_interest.append(goi)

# 定义红–淡灰–蓝渐变
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("red_gray_blue", ["midnightblue", "orange", "red"])

sc_draw_graph(save_addr=save_addr,filename="Gene_Expression(FA)2",
              color=[genes for genes in Naive_marker + goi if genes in adata_tree.var_names],
              cmap=cmap,             # 红蓝渐变，反转顺序
              adata=adata_tree)

if combined_flag:
    sc_draw_graph(save_addr=save_addr, filename="Quality_inform(FA)",
                  adata=adata_tree,
                  color=["disease_type", "Cell_Subtype", "percent.ribo", "percent.mt"])
else:
    sc_draw_graph(save_addr=save_addr, filename="Quality_inform(FA)",
                  adata=adata_tree,
                  color=["disease_type", "percent.ribo", "percent.mt"])

###################################################
# 选择根
scf.tl.root(adata_tree, 146)

# scf.tl.pseudotime(adata_tree, n_jobs=20, n_map=1, seed=42)
# adata_tree.write_h5ad(f"{save_addr}/03_pseudotimed(cleaned_tmp-1).h5ad")

scf.tl.pseudotime(adata_tree, n_jobs=20, n_map=100, seed=42)
adata_tree.write_h5ad(f"{save_addr}/03_pseudotimed(cleaned_tmp-100).h5ad")

#-----------------------------------------------------------
adata_pseudo = adata_tree
# adata_pseudo = anndata.read_h5ad(f"{save_addr}/03_pseudotimed(cleaned_tmp-100).h5ad")

# 非常有趣的是，在经过 scf.tl.cleanup之后，就没有 na 了
# adata_notna = adata_pseudo[adata_pseudo.obs['milestones'].isna()==False]
# adata_na = adata_pseudo[adata_pseudo.obs['milestones'].isna()==True]

pl_trajectory(
    save_addr=save_addr,
    filename="Traj(FA)",
    adata=adata_pseudo,
    # basis="pca", # 默认值即 X_draw_graph_fa
    arrows=True, arrow_offset=3
)

pl_milestones(
    save_addr=save_addr,
    filename="Milestones(FA)",
    adata=adata_pseudo,
    annotate=True
    )
###################################################
# 如果采取 n_map=1 的方案，或者说
# 测试并拟合与树相关的特征
# adata_pseudo.obs.loc[:, ["t", "seg"]]
# value_df = adata_pseudo.obs.seg.value_counts()
# spline_df=5
# adata_pseudo_cutof = adata_pseudo[adata_pseudo.obs.seg.isin(value_df.index[value_df.values>=spline_df])]
# # 235545 → 235494
spline_df=5 # 数据越大数字越大哈
adata_pseudo_cutof = filter_adata_by_seg(adata_pseudo, spline_df=spline_df, seg_col="seg")

scf.tl.test_association(adata_pseudo_cutof, n_map=1,spline_df=spline_df, n_jobs=20)
adata_pseudo_cutof.write_h5ad(f"{save_addr}/04_test_assoc(splined={spline_df}).h5ad")
#
# scf.tl.test_association(adata_pseudo_cutof, n_map=100,spline_df=spline_df, n_jobs=20)
# adata_pseudo_cutof.write_h5ad(f"{save_addr}/04_test_assoc(splined={spline_df}-nmap=100).h5ad")

del adata_pseudo_cutof