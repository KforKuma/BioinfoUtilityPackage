# 先拿巨噬细胞测试
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # DejaVu 是常用开源字体

i=21
name = list(dict_for_pseudo.keys())[i]
ident = dict_for_pseudo[name]
combined_flag = isinstance(ident, list) and len(ident) > 1
proj_name = name.replace(" ", "_")
save_addr = f"/data/HeLab/bio/IBD_analysis/output/Step11_scfates/{proj_name}"
###################################################
# 自定义函数
from src.scFatesTools.scFatesTools import choose_assoc_file

# 使用示例：
# save_addr = "/path/to/dir"
# path, spline_val = choose_assoc_file(save_addr)
# print("选择结果：", path, spline_val)


# 包装需要包装的函数

###################################################
# adata_pseudo = anndata.read_h5ad(f"{save_addr}/03_pseudotimed(cleaned_tmp-100).h5ad")
# adata_pseudo_cutof = filter_adata_by_seg(adata_pseudo, spline_df=spline_df, seg_col="seg")
# scf.tl.test_association(adata_pseudo_cutof, n_map=100,spline_df=spline_df, n_jobs=20)
# adata_pseudo_cutof.write_h5ad(f"{save_addr}/04_test_assoc(splined={spline_df}-nmap=100).h5ad")


###################################################
# path, spline_val = choose_assoc_file(save_addr)
# print("选择结果：", path, spline_val)
# adata_pseudo_cutof = anndata.read_h5ad(path)
adata_pseudo_cutof = anndata.read_h5ad(f"{save_addr}/04_test_assoc(splined=5).h5ad")




# pl_test_association(save_addr=save_addr,
#                     filename="Features_over_Pseudotime",
#                     adata=adata_pseudo_cutof
# )
# 为了节约时间，目前， tl.test_association 参数采用的是 n_map=1 ，因此不能正确地计算 st 和 A
# 在这个前提下，我们快速地重新划分阈值，以做示意
scf.tl.test_association(adata_pseudo_cutof,reapply_filters=True,A_cut=.5, st_cut=0)
pl_test_association(save_addr=save_addr,
                    filename="Features_over_Pseudotime_tmp",
                    adata=adata_pseudo_cutof
)
    # A: amplitude，振幅是 GAM 预测值的最大值减去预测值的最小值。如果 A > A_cut，则具有显著性。
    # st: stability，关联稳定性的截止值（具有显著 (fdr,A) 对的映射分数）；显著性，如果 st > st_cut，则显著性。
adata_pseudo_cutof.var.query("fdr < 1e-4")[["A", "st", "signi"]].head(10)

scf.tl.fit(adata_pseudo_cutof,n_jobs=10)
    # 跨平台加载比较缓慢，测验数据耗时 5+7 分钟
    # 默认情况下，函数 fit 会将整个数据集保存在 adata.raw 下（参数 save_raw 默认设置为 True）

adata_pseudo_cutof.write_h5ad(f"{save_addr}/05_fitted.h5ad")

scf.tl.dendrogram(adata_pseudo_cutof)
for item in ["disease_type","Subset_Identity","seg","milestones"]:
    pl_dendro(
        save_addr=save_addr,
        filename=f"Dendro_{item}",
        adata=adata_pseudo_cutof,
        color=item
    )
###################################################
# 绘制单个特征
# Celltype = adata_pseudo_cutof.obs["Celltype"][i]
# Naive_marker = naive_resting_dict[Celltype]
from src.scFatesTools.scFatesTools import filter_genes
clean_genes, removed = filter_genes(adata_pseudo_cutof.var_names)
print(f"保留基因数: {len(clean_genes)}，移除基因数: {len(removed)}")
print(clean_genes)

# genes_for_trend = ['CCL5', 'CCR7', 'CD19', 'CD37', 'CD44', 'CD52', 'CD59', 'CD83','CXCL16', 'CXCR4', 'CXCR5','IL23R', 'IL32'] # macro
genes_for_trend = ['APOO', 'CD3D', 'CREM', 'FGFBP2', 'FOS', 'GZMH', 'GZMK', 'IFI6', 'IL12RB2', 'ISG15', 'KLRK1', 'NR4A2', 'PTPRCAP', 'SIK3', 'SPON2', 'STAT4', 'SYTL3', 'TRBC2', 'XCL1', 'XCL2']

 # epi stem

for genes in genes_for_trend:
    pl_single_trend(
        save_addr=save_addr,
        filename=f"Singletrend_{genes}",
        adata=adata_pseudo_cutof,
        feature=genes,
        basis="dendro",wspace=0.25,color_exp="k"
    )
###################################################
# 找一下差异基因
from src.ScanpyTools.ScanpyTools import easy_DEG
easy_DEG(bdata = adata_pseudo_cutof,
         save_addr = save_addr,
         filename = "DEG_By_Segs", obs_key="seg")
easy_DEG(bdata = adata_pseudo_cutof,
         save_addr = save_addr,
         filename = "DEG_By_MSs", obs_key="milestones")
###################################################
# 重命名
pl_trajectory(
    save_addr=save_addr,
    filename="Milestones(FA)",
    adata=adata_pseudo_cutof,
    # basis="pca", # 默认值即 X_draw_graph_fa
    color_cells="milestones",
    arrows=True, arrow_offset=3
)
pl_trajectory(
    save_addr=save_addr,
    filename="Subset_Identity(FA)",
    adata=adata_pseudo_cutof,
    # basis="pca", # 默认值即 X_draw_graph_fa
    color_cells="Subset_Identity",
    arrows=True, arrow_offset=3
)
scf.tl.rename_milestones(adata_pseudo_cutof,
                         new={"15": "IFN-act. NK16",
                              "3": "KLRC3+ NK16",
                              "49": "MHC-II+ NK16",
                              "50": "KLRC2+ NK16",
                              "7": "Act. NK56",
                              "74": "Int. NK56",
                              "93": "Resting NK56",
                              "94": "Int. NK16"
                              })
adata_pseudo_cutof.write_h5ad(f"{save_addr}/06_renamed.h5ad")

pl_milestones(save_addr=save_addr,
              filename="Milestones_renamed",
              adata=adata_pseudo_cutof,
              annotate=True)


###################################################
tl_linearity_deviation = ScanpyPlotWrapper(func=scf.tl.linearity_deviation)
pl_linearity_deviation = ScanpyPlotWrapper(func=scf.pl.linearity_deviation)
tl_linearity_deviation(adata=adata_pseudo_cutof,
                       start_milestone="Radial Glia",
                       end_milestone="Neurons",
                       n_jobs=20,plot=True,basis="pca")

###################################################
scf.tl.cluster(adata_pseudo_cutof,n_neighbors=100,metric="correlation")
adata_pseudo_cutof.var.clusters.unique()



for c in adata_pseudo_cutof.var["clusters"].unique():
    pl_trends(
        save_addr=save_addr,
        filename=f"Trends_of_Cluster_{c}",
        adata=adata_pseudo_cutof,
        features=adata_pseudo_cutof.var_names[adata_pseudo_cutof.var.clusters==c],
        basis="draw_graph_fa"
    )


