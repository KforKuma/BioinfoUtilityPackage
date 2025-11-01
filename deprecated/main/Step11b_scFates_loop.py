list_idents = adata.obs["Subset_Identity"].unique().tolist()
# 去除
list_idents = [item for item in list_idents if not item.endswith("mitotic")]
# 合并，写一个字典
dict_for_pseudo = {'B Cell_all': ['B Cell_B.mem.activated', 'B Cell_B.mem.naive', 'B Cell_GC.B'],
                   'Endo_Endothelium': 'Endo_Endothelium',
                   'Epi_Absorp_Col_all': ['Epi_Absorp.Col', 'Epi_Absorp.Col.AQP8+', 'Epi_Absorp.Col.DUOX2+'],
                   'Epi_Col': ['Epi_Col', 'Epi_Prolif.Col'],
                   'Epi_Col.BEST4+': 'Epi_Col.BEST4+',
                   'Epi_EEC': 'Epi_EEC',
                   'Epi_Goblet_all': ['Epi_Goblet', 'Epi_Goblet.TFF1+'],
                   'Epi_Microfold': 'Epi_Microfold',
                   'Epi_Paneth': 'Epi_Paneth',
                   'Epi_Rec': 'Epi_Rec',
                   'Epi_Stem_all': ['Epi_Stem.LGR5', 'Epi_Stem.OLFM4'],
                   'Epi_Tuft': 'Epi_Tuft',
                   'Fibroblast_all': ['Fibroblast_Fb.activated', 'Fibroblast_Fb.resting'],
                   'Myeloid_Monocyte_all': ['Myeloid_C.Monocyte', 'Myeloid_Int.Monocyte', 'Myeloid_N.C.Monocyte'],
                   'Myeloid_Macrophage_all': ['Myeloid_M2.Macrophage', 'Myeloid_Macrophage'],
                   'Myeloid_Mast': 'Myeloid_Mast',
                   'Myeloid_Megakaryocyte': 'Myeloid_Megakaryocyte',
                   'Myeloid_DC_all': ['Myeloid_cDC1', 'Myeloid_cDC2', 'Myeloid_cDC3', 'Myeloid_pDC'],
                   'Plasma': ['Plasma_IgA', 'Plasma_IgG', 'Plasma_IgM'],
                   'T Cell_CD4_all': ['T Cell_CD4.Tfh',
                                      'T Cell_CD4.Th1', 'T Cell_CD4.Th17',
                                      'T Cell_CD4.Tr1', 'T Cell_CD4.Treg',
                                      'T Cell_CD4.early', 'T Cell_CD4.mem', 'T Cell_CD4.naive'],
                   'T Cell_CD8_all': ['T Cell_CD8.NKT', 'T Cell_CD8.Trm', 'T Cell_CD8.Trm.KLRC2+',
                                      'T Cell_CD8.mem', 'T Cell_CD8.mem.GZMK', 'T Cell_CD8.naive', ],
                   'T Cell_DN': 'T Cell_DN',
                   'T Cell_ILC.XCL1+': 'T Cell_ILC.XCL1+',
                   'T Cell_ILC1': 'T Cell_ILC1',
                   'T Cell_ILC3': 'T Cell_ILC3',
                   'T Cell_MAIT': 'T Cell_MAIT',
                   'T Cell_innate_all': ['T Cell_ILC1', 'T Cell_ILC3', 'T Cell_MAIT',
                                         'T Cell_NK.CD16+', 'T Cell_NK.CD56+', 'T Cell_gdT.Trm.GZMA+',
                                         'T Cell_gdT.Trm.XCL1+',
                                         'T Cell_gdT.g9d2', 'T Cell_gdT.naive'],
                   'T Cell_NK_all': ['T Cell_NK.CD16+', 'T Cell_NK.CD56+'],
                   'T Cell_gdT_all': ['T Cell_gdT.Trm.GZMA+', 'T Cell_gdT.Trm.XCL1+',
                                      'T Cell_gdT.g9d2', 'T Cell_gdT.naive']
                   }

naive_resting_dict = {
    'B Cell': ["MS4A1", "CD19", "CD79A", "CD79B", "BANK1"],
    'Endo': ["PECAM1", "VWF", "CLDN5", "ESAM", "CDH5"],
    'Epi': ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1"],
    'Fibroblast': ["COL1A1", "COL1A2", "DCN", "LUM", "FBLN1"],
    'Myeloid': ["CSF1R", "LYZ", "CD14", "MRC1", "C1QA"],
    'Plasma': ["SDC1", "XBP1", "MZB1", "DERL3", "IGHM"],
    'T Cell': ["CD3D", "LEF1", "TCF7", "CCR7", "CD3D"]
}

# # 删除多余的文件夹
# import shutil
# root_dir = "/data/HeLab/bio/IBD_analysis/output/Step11_scfates"   # 你的目标文件夹
# file_to_keep = [item.replace(" ","_") for item in dict_for_pseudo.keys()]
# for name in os.listdir(root_dir):
#     full_path = os.path.join(root_dir, name)
#     if os.path.isdir(full_path) and name not in file_to_keep:
#         print(f"删除文件夹: {full_path}")
#         shutil.rmtree(full_path)


for name, ident in dict_for_pseudo.items():
    start_time = time.time()
    proj_name = name.replace(" ", "_")
    combined_flag = isinstance(ident, list) and len(ident) > 1
    save_addr = f"/data/HeLab/bio/IBD_analysis/output/Step11_scfates/{proj_name}"
    os.makedirs(save_addr, exist_ok=True)
    print(f"\n==============================")
    print(f"▶ 开始处理 {ident}  (保存路径: {save_addr})")
    print(f"==============================")
    try:
        # 1. 预处理
        print(f"[{ident}] Step 1: 复制 raw -> adata_allgenes")
        if not combined_flag:
            adata_allgenes = adata[adata.obs["Subset_Identity"] == ident].raw.to_adata()
        else:
            adata_allgenes = adata[adata.obs["Subset_Identity"].isin(ident)].raw.to_adata()
        Celltype = adata_allgenes.obs["Celltype"][0]
        Naive_maker = naive_resting_dict[Celltype]
        print(f"[{ident}] Celltype: {Celltype}, Naive markers: {Naive_maker}")
        print(f"[{ident}] Step 1: 过滤基因")
        sc.pp.filter_genes(adata_allgenes, min_cells=3)
        adata_allgenes = adata_allgenes[adata_allgenes.obs["phase"].isin(["G1", "S"])]
        print(f"[{ident}] Step 1: 归一化 & log1p")
        sc.pp.normalize_total(adata_allgenes)
        sc.pp.log1p(adata_allgenes, base=10)
        print(f"[{ident}] Step 1: 计算 HVG")
        sc.pp.highly_variable_genes(adata_allgenes, n_top_genes=5000, flavor='cell_ranger')
        adata_allgenes = adata_allgenes[:, adata_allgenes.var['highly_variable']].copy()
        print(f"[{ident}] Step 1: HVG genes count: {adata_allgenes.shape[1]}")
        print(f"[{ident}] Step 1: PCA 降维")
        sc.pp.pca(adata_allgenes, use_highly_variable=True)
        print(f"[{ident}] Step 1: X_pca shape: {adata_allgenes.obsm['X_pca'].shape}")
    except Exception as e:
        print(f"[{ident}] ❌ Step 1 失败，跳过此 ident")
        traceback.print_exc()
        continue
    try:
        # 2. palantir 过程
        print(f"[{ident}] Step 2: 运行 diffusion map")
        pca_projections = pd.DataFrame(adata_allgenes.obsm["X_pca"], index=adata_allgenes.obs_names)
        dm_res = palantir.utils.run_diffusion_maps(pca_projections)
        print(f"[{ident}] Step 2: diffusion map shape: {dm_res.eigenvectors.shape}")
        print(f"[{ident}] Step 2: 多尺度空间")
        ms_data = palantir.utils.determine_multiscale_space(dm_res, n_eigs=4)
        print(f"[{ident}] Step 2: multiscale space shape: {ms_data.shape}")
    except Exception as e:
        print(f"[{ident}] ❌ Step 2 失败，跳过此 ident")
        traceback.print_exc()
        continue
    try:
        # 3. 邻居 & 绘图
        print(f"[{ident}] Step 3: 保存 palantir 空间 & neighbors")
        adata_allgenes.obsm["X_palantir"] = ms_data.values
        print(f"[{ident}] X_palantir shape: {adata_allgenes.obsm['X_palantir'].shape}")
        sc.pp.neighbors(adata_allgenes, n_neighbors=30, use_rep="X_palantir")
        print(f"[{ident}] Step 3: 设置初始二维坐标")
        adata_allgenes.obsm["X_pca2d"] = adata_allgenes.obsm["X_pca"][:, :2]
        print(f"[{ident}] Step 3: 运行 draw_graph")
        sc.tl.draw_graph(adata_allgenes, init_pos='X_pca2d')
        try:
            adata_allgenes.write_h5ad(f"{save_addr}/01_palantirred.h5ad")
            print(f"[{ident}] ✅ 保存 01_palantirred.h5ad 完成")
        except Exception:
            print(f"[{ident}] ⚠️ 保存 01_palantirred.h5ad 失败，继续执行")
    except Exception as e:
        print(f"[{ident}] ❌ Step 3 失败，跳过此 ident")
        traceback.print_exc()
        continue
    try:
        # 4. scfates tree
        print(f"[{ident}] Step 4: 构建 PPT 树")
        scf.tl.tree(
            adata_allgenes,method="ppt",
            Nodes=200,
            use_rep="palantir",device="cpu",
            seed=1,
            ppt_lambda=100,ppt_sigma=0.025,ppt_nsteps=200)
        print(f"[{ident}] Step 4: X_R shape: {adata_allgenes.obsm['X_R'].shape}")
        # nan_rows = np.where(np.isnan(adata_allgenes.obsm['X_R']).any(axis=1))[0]
        # print("Number of rows with NaN:", len(nan_rows))
        # print("Example row indices:", nan_rows[:20])
        # print("Example row values:", adata_allgenes.obsm['X_R'][nan_rows[0], :20])
        
        try:
            adata_allgenes.write_h5ad(f"{save_addr}/02_treed.h5ad")
            print(f"[{ident}] ✅ 保存 02_treed.h5ad 完成")
        except Exception:
            print(f"[{ident}] ⚠️ 保存 02_treed.h5ad 失败，跳过保存")
    except Exception as e:
        print(f"[{ident}] ❌ Step 4 失败，跳过此 ident")
        traceback.print_exc()
        continue
    try:
        # 5. plot some pics
        print(f"[{ident}] Step 5: 绘制基本信息图")
        if combined_flag:
            pca_plot(save_addr=save_addr,filename="Integrated_Inform(PCA)",adata=adata_allgenes,color=["disease_type", "Cell_Subtype", "percent.ribo", "percent.mt"])
        else:
            pca_plot(save_addr=save_addr,filename="Integrated_Inform(PCA)",
                     adata=adata_allgenes,color=["disease_type", "percent.ribo", "percent.mt"])
        pca_plot(save_addr=save_addr,filename="Naive_maker(PCA)",adata=adata_allgenes,color=Naive_maker)
    except Exception as e:
        print(f"[{ident}] ❌ Step 5 失败，跳过此 ident")
        traceback.print_exc()
        continue
    try:
        print(f"[{ident}] Step 5 plus: 绘制增补信息图")
        pl_graph(save_addr=save_addr,filename="Nodes(FA)",adata=adata_allgenes)
        sc_draw_graph(save_addr=save_addr,filename="Combined_info(FA)",
                      color=Naive_maker + ["disease_type", "Cell_Subtype", "percent.ribo", "percent.mt"],adata=adata_allgenes)
    except Exception as e:
        print(f"[{ident}] ❌ Step 5 plus 失败，跳过此 ident")
        traceback.print_exc()
        continue
    try:
        print(f"[{ident}] Step 6: 特殊信息保存")
        node_assign = np.argmax(adata_allgenes.obsm["X_R"], axis=1)
        adata_allgenes.obs["assigned_node"] = node_assign
        print(f"[{ident}] assigned_node shape: {adata_allgenes.obs['assigned_node'].shape}")
        # 离散变量统计
        count_df = adata_allgenes.obs.groupby(["assigned_node", "Subset_Identity"]).size().reset_index(name="count")
        count_df["proportion"] = count_df.groupby("assigned_node")["count"].transform(lambda x: x / x.sum())
        # 连续变量统计
        continuous_vars = ["percent.ribo", "percent.mt", "percent.hb"]
        mean_df = adata_allgenes.obs.groupby("assigned_node")[continuous_vars].mean().reset_index()
        count_df.to_csv(f"{save_addr}/Subset_by_Nodes.csv", index=False)
        mean_df.to_csv(f"{save_addr}/Percentage_by_Nodes.csv", index=False)
        cross_df = adata_allgenes.obs.groupby("Subset_Identity")[continuous_vars].mean().reset_index()
        cross_df.to_csv(f"{save_addr}/Percentage_by_Subset.csv", index=False)
        cross_count_df = adata_allgenes.obs.groupby(["assigned_node", "disease_type"]).size().reset_index(name='count')
        cross_count_df["proportion"] = cross_count_df.groupby("assigned_node")["count"].transform(lambda x: x / x.sum())
        wide_df = cross_count_df.pivot(index='assigned_node', columns='disease_type', values='proportion').reset_index()
        wide_df.to_csv(f"{save_addr}/Disease_by_Nodes.csv", index=False)
    except Exception as e:
        print(f"[{ident}] ❌ Step 6 失败，跳过此 ident")
        traceback.print_exc()
        continue
    end_time = time.time()
    print(f"[{ident}] ✅ 完成，耗时 {end_time - start_time:.1f} 秒")

###################################################
# 接下来我们一个一个地手动进行
keys = list(dict_for_pseudo.keys())
name = keys[13]
print(name)
ident = dict_for_pseudo[name]
if isinstance(ident, list) and len(ident) > 1:
    combined_flag = True
else:
    combined_flag = False

proj_name = name.replace(" ", "_")
save_addr = f"/data/HeLab/bio/IBD_analysis/output/Step11_scfates/{proj_name}"

adata_allgenes = anndata.read_h5ad(f"{save_addr}/02_treed.h5ad")
###################################################
# 加载之后先检查，如果有问题我们得重新跑一遍 tree
adata = adata_allgenes
B = adata.uns["graph"]["B"]

g = igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected")
degrees = g.degree()
high_degree_nodes = [i for i, d in enumerate(degrees) if d >= 3]

for edge in g.es:
    u, v = edge.tuple
    if u in high_degree_nodes and v in high_degree_nodes:
        print((u, v))
###################################################
# 有问题的时候
adata_allgenes = anndata.read_h5ad(f"{save_addr}/01_palantirred.h5ad")
scf.tl.tree(
    adata_allgenes,
    method="ppt",
    Nodes=150,  # 默认 200，参考 100 ~ 2,000
    use_rep="palantir",
    device="cpu",
    seed=1,
    ppt_lambda=300,  # 默认 100，参考 1；越大越平滑
    ppt_sigma=0.1,  # 默认 0.025，参考值 0.1；越大，邻域越大
    ppt_nsteps=200  # 默认 200，参考 50；不过收敛之后就会停止
)

adata_allgenes.write_h5ad(f"{save_addr}/02_treed.h5ad")

