"""
Step11a_scFates_demo.py
Author: John Hsiung
Update date: 2025-09-17
Description:
    - 检验 scFates-cellrank 工作流程
Notes:
    - 使用环境：conda activate scfates
    - export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6.0.34
    - 教程参考：https://scfates.readthedocs.io/en/latest/Basic_Curved_trajectory_analysis.html
"""

###################################################
# 为了避免由于 rpy2 未在 conda 上找到 R 安装而导致的任何崩溃，请运行以下导入命令：

import os, sys
os.environ['R_HOME'] = sys.exec_prefix + "/lib/R"

import time
import traceback

import numpy as np
import pandas as pd
import scipy
import igraph
import sklearn
import scanpy as sc
import scFates as scf
import anndata
import palantir
import matplotlib.pyplot as plt
import seaborn

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # DejaVu 是常用开源字体
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # DejaVu 是常用开源字体

###################################################
# scanpy / scFates 的默认设置
sc.settings.verbosity = 3
sc.settings.logfile = sys.stdout
seaborn.reset_orig()  # fix palantir breaking some plots
sc.set_figure_params()
scf.set_figure_pubready()

###################################################
# 图像保存的自定义行为
sys.path.append("/data/HeLab/bio/IBD_analysis/src")
from ScanpyTools.Scanpy_Plot import ScanpyPlotWrapper

pl_graph = ScanpyPlotWrapper(func=scf.pl.graph)
pca_plot = ScanpyPlotWrapper(func=sc.pl.pca)
pl_trajectory = ScanpyPlotWrapper(func=scf.pl.trajectory)
# trajec_3d_plot = ScanpyPlotWrapper(func=scf.pl.trajectory_3d)
pl_milestones = ScanpyPlotWrapper(func=scf.pl.milestones)
tl_linearity_deviation = ScanpyPlotWrapper(func=scf.tl.linearity_deviation)
pl_linearity_deviation = ScanpyPlotWrapper(func=scf.pl.linearity_deviation)
pl_test_association = ScanpyPlotWrapper(func=scf.pl.test_association)
pl_single_trend = ScanpyPlotWrapper(func=scf.pl.single_trend)
pl_trends = ScanpyPlotWrapper(func=scf.pl.trends)
sc_draw_graph = ScanpyPlotWrapper(func=sc.pl.draw_graph)
pl_dendro = ScanpyPlotWrapper(func=scf.pl.dendrogram)
dotplot = ScanpyPlotWrapper(func=sc.pl.dotplot)
###################################################
# 测试数据
# adata = sc.read('/data/HeLab/bio/IBD_analysis/assets/scvelo-pancreas.h5ad')
# adata.var_names_make_unique()

# 数据加载
adata = anndata.read_h5ad("/data/HeLab/bio/IBD_analysis/output/Step07/Step07_finalversion_5.h5ad")
print(adata.obs["Subset_Identity"].unique().tolist())

proj_name = "T Cell_CD4.Th17".replace(" ", "_") # 作为文件夹名称，尽量避免空格
adata_subset  = adata[adata.obs["Subset_Identity"] == "T Cell_CD4.Th17"]
save_addr = f"/data/HeLab/bio/IBD_analysis/output/Step11_scfates/{proj_name}"
os.makedirs(save_addr, exist_ok=True)

###################################################
# 因为小群的关键基因可能和大群很不一样，为了捕捉这些差异
# 也同时因为 palantir 能够扩散平滑排除掉一些噪音和小的分叉
# 我们选择恢复 raw，不过还是要筛选一下的
# 1. 预处理
adata_allgenes = adata_subset.raw.to_adata()  # 全部基因
# pre-processing，通常忽略
sc.pp.filter_genes(adata_allgenes,min_cells=3)
sc.pp.normalize_total(adata_allgenes)
sc.pp.log1p(adata_allgenes,base=10)
# 取 HVG
sc.pp.highly_variable_genes(adata_allgenes, n_top_genes=1500, flavor='cell_ranger')
# 归一化、PCA
sc.pp.pca(adata_allgenes, use_highly_variable=True)

###################################################
# 2. palantir 过程
pca_projections = pd.DataFrame(adata_allgenes.obsm["X_pca"],
                               index=adata_allgenes.obs_names)
dm_res = palantir.utils.run_diffusion_maps(pca_projections)
ms_data = palantir.utils.determine_multiscale_space(dm_res,n_eigs=4)
###################################################
# 3. 从多尺度扩散空间生成嵌入
adata_allgenes.obsm["X_palantir"]=ms_data.values
sc.pp.neighbors(adata_allgenes,n_neighbors=30,use_rep="X_palantir")
# draw ForceAtlas2 embedding using 2 first PCs as initial positions
adata_allgenes.obsm["X_pca2d"]=adata_allgenes.obsm["X_pca"][:,:2]
sc.tl.draw_graph(adata_allgenes,init_pos='X_pca2d')
adata_allgenes.write_h5ad(f"{save_addr}/01_palantirred.h5ad")
scf.tl.tree(
            adata_allgenes,method="ppt",Nodes=200,
            use_rep="palantir",
            device="cpu",seed=1,
            ppt_lambda=100,ppt_sigma=0.025,ppt_nsteps=200
)
adata_allgenes.write_h5ad(f"{save_addr}/02_treed.h5ad")


###################################################
# 如何抉择具体的 root？

adata_allgenes = anndata.read_h5ad(f"{save_addr}/02_treed.h5ad")


# 方法1：查看“金标准”基因
adata.var_names[adata.var['highly_variable']]

pca_plot(
    save_addr="/data/HeLab/bio/IBD_analysis/output/Step11_scfates",
    filename="PCA_plot",
    adata=adata,color="Fhl2",cmap="RdBu_r"
    )

# 方法2：查看细胞的分簇信息
pca_plot(
    save_addr="/data/HeLab/bio/IBD_analysis/output/Step11_scfates",
    filename="PCA_cluster",
    adata=adata,
    color=["clusters_coarse","clusters"]
    )

# 方法3：使用 ElPiGraph 算法学习曲线 + 展示软分配（但是在）
scf.tl.curve(adata,
             Nodes=30, # 对于 ElPiGraph 方法使用 10 到 100 的范围，对于 PPT 方法使用 100 到 2000 的范围
             use_rep="X_pca", # “X_scvi”
             ndims_rep=3, # 用于推理的维度数，rep 空间有多少维度就可以写多少；如 10
             )
pl_graph(
        save_addr="/data/HeLab/bio/IBD_analysis/output/Step11_scfates",
        filename="PCA_treeplot",
        adata=adata,basis="pca"
    )

# 显示细胞的软分配
    # tl.curve 算法返回的信息包括一个 obsm[‘X_R’] 里储存的 soft assignment，和几个 uns 中的字典/变量赋值信息
    # obsm['X_R'] 会为每个细胞赋予一个 [0, 1] 之间的 float 赋值；它的行和细胞数量对齐，列和 Nodes 数量对齐
    # 其含义为：对每个细胞，ElPiGraph 先找到它到所有 Node 的距离，随后把距离转换成权重。
    # 对于每个 Nodes，我们可以考察赋值高低，实际上能够形成边界有重叠的 'soft' 分配

string_list = [str(i) for i in range(30)]
pca_plot(
    save_addr="/data/HeLab/bio/IBD_analysis/output/Step11_scfates",
    filename="Soft_assignment_All",
    adata=sc.AnnData(adata.obsm["X_R"],obsm=adata.obsm),
    color=string_list, # 对应 node ID 2；反正 Node 有 30 个的话，取值就是 [0, 29]
    cmap="Reds"
    )
###################################################
# 选择 root 并计算 pseudotime
    # 使用 FAM64A 标记，我们可以自信地判断尖端 1 是根。
scf.tl.root(adata,"Fhl2")
scf.tl.pseudotime(adata,n_jobs=8,n_map=100,seed=42)
    # 运行 100 次映射来解释不确定性，obs 中保存的伪时间将是所有计算出的伪时间的平均值
    # 这一步相对比较花费时间
pca_plot(
    save_addr="/data/HeLab/bio/IBD_analysis/output/Step11_scfates",
    filename="PCA_Pseudotime",
    adata=adata,
    color="t"
    )

pl_trajectory(
    save_addr="/data/HeLab/bio/IBD_analysis/output/Step11_scfates",
    filename="PCA_traj",
    adata=adata,
    basis="pca",arrows=True,arrow_offset=3
    )
# 3d 的前提是最起码用了 3 个维度
# trajec_3d_plot(
#     save_addr="/data/HeLab/bio/IBD_analysis/output/Step11_scfates",
#     filename="PCA_traj_3d",
#     adata=adata,basis="pca"
#     )
###################################################
# 分配并绘制 milestones
pca_plot(
    save_addr="/data/HeLab/bio/IBD_analysis/output/Step11_scfates",
    filename="PCA_milestones",
    adata=adata,
    color="milestones"
    )

start = adata.uns['graph']['root'] # 这个 root 对应值实际上就是 obs['milestone'] 里对应着根节点的名字（一个str格式的整数）
end = adata.uns['graph']['tips'][adata.uns['graph']['tips']!=start][0]
# scf.tl.rename_milestones(adata,
#                          new={str(start):"Radial Glia",
#                               str(end): "Neurons"}) # 一个简单的改名方法
scf.tl.rename_milestones(adata,
                         new={'8':"Origin",
                              "0": "Diff_left",
                              "2": "Diff_right"})

pca_plot(
    save_addr="/data/HeLab/bio/IBD_analysis/output/Step11_scfates",
    filename="PCA_milestones(updated)",
    adata=adata,
    color="milestones"
    )
# 下图和上图几乎完全一致，除了 label 直接标注在细胞类群上；以及增加了 show/save 参数
# 不过这样确实看得清楚了点儿；但是似乎无法显示大于 2 群？弃用待定
pl_milestones(
    save_addr="/data/HeLab/bio/IBD_analysis/output/Step11_scfates",
    filename="PCA_milestones_annot",
    adata=adata,
    basis="pca",
    annotate=True
    )

###################################################
# 线性偏差评估
# 实现了 Kameneva et al. 2021 提出的“线性偏离检验”，用来检测一个“过渡/桥接细胞群”是否可能是生物学上真实的状态过渡，
# 还是只是 progenitor 和 progeny 混合（甚至是双细胞 doublet）造成的假象。
# adata.uns["graph"]["milestones"] = {'Diff_right': 2, 'Diff_left':0, 'Origin': 22}
# # 请注意：这一步必须从 root 指向 end，二者之间必须要联通；快速检查是否联通的方法：
# start_node = graph["milestones"].get('Diff_left', None) # 自定义
# end_node   = graph["milestones"].get('Diff_right', None) # 自定义
# print('start_node:', start_node)
# print('end_node:', end_node)
# cells = getpath(img, start_node, graph.get("tips"), end_node, graph, df)
# print('getpath returned:', type(cells))
# print(cells if cells is None else cells.head())

tl_linearity_deviation(
    save_addr="/data/HeLab/bio/IBD_analysis/output/Step11_scfates",
    filename="PCA_linearity_detect",
    adata=adata,
    start_milestone="Origin",
    end_milestone="Diff_right",
    n_jobs=20,plot=True,basis="pca"
)

pl_linearity_deviation(
    save_addr="/data/HeLab/bio/IBD_analysis/output/Step11_scfates",
    filename="PCA_linearity_pl",
    adata=adata,
    start_milestone="Origin",
    end_milestone="Diff_right"
)
# 参照上图，我们可以对感兴趣的梯度基因投射到 PCA 上
pca_plot(
    save_addr="/data/HeLab/bio/IBD_analysis/output/Step11_scfates",
    filename="PCA_GeneOI",
    adata=adata,
    color=["Rps19","Rps4x","Cd24a"]
    )
###################################################
# 伪时间测试中显著变化的特征

scf.tl.test_association(adata,n_jobs=20)
    # 比较耗时，3600细胞大约10分钟多
    # 我们可以改变振幅(amplitude)参数来获得更重要的基因，
    # 这可以在不重新进行所有测试的情况下完成(reapply_filters)
    # 这没有什么特别的含义，只是决定了绘图时多少基因点被绘制为红色的（显著）
# scf.tl.test_association(adata,
#                         reapply_filters=True, # 避免重新计算并重新应用 filter
#                         A_cut=.5 # 振幅 `A` 是 GAM 预测值的最大值减去预测值的最小值。如果 A > A_cut，则具有显著性。默认为 1。
#                         )

pl_test_association(
    save_addr="/data/HeLab/bio/IBD_analysis/output/Step11_scfates",
    filename="Test_association",
    adata=adata
)

###################################################
# 拟合和聚类重要特征
    # 默认情况下，函数 fit 将把整个数据集保存在 adata.raw 下（参数 save_raw）
    # 重聚类替代之前的节点 node，增加可解释性
scf.tl.fit(adata,n_jobs=20)
    # 跨平台加载比较缓慢，测验数据耗时 5+7 分钟
for genes in ['Sbspon', 'Mcm3', 'Fhl2', 'Akr1cl']:
    pl_single_trend(
        save_addr="/data/HeLab/bio/IBD_analysis/output/Step11_scfates",
        filename=f"Singletrend_{genes}",
        adata=adata,feature=genes,basis="pca",color_exp="k"
    )


scf.tl.cluster(adata,n_neighbors=100,metric="correlation")
    # 很快！ as 快 as scanpy
    # 集群特性。使用 CPU 时使用 scanpy 后端，使用 GPU 时使用 cuml。
    # 数据集经过转置，计算主成分分析 (PCA)，并从主成分空间生成最近邻图。使用 leiden 算法进行 community detection。

adata.var.clusters.unique()
for c in adata.var["clusters"].unique():
    pl_trends(save_addr="/data/HeLab/bio/IBD_analysis/output/Step11_scfates",
              filename=f"Clustertrends_{c}",
              adata=adata,features=adata.var_names[adata.var.clusters==c],basis="pca")
    # 绘图函数的参数 highlight_features = 'A' 或 'fdr'，按照此标准筛选需要高亮标记的基因



