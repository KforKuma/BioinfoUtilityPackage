import anndata
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import gc, os
import re
import time
import sys
sys.path.append("/data/HeLab/bio/IBD_analysis/")
sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')  # 使用无GUI的后端

import src.ScanpyTools.Scanpy_Plot as scanpy_plot



def subcluster(adata, n_neighbors=20, n_pcs=50,skip_DR=False, resolutions=None, use_rep="X_scVI"):
    """
    打包的聚类降维综合函数
    :param adata: anndata格式文件
    :param n_neighbors: KNN最近邻的近邻数量
    :param n_pcs: 维度数量
    :param resolutions: 分辨率
    :param use_rep: 默认为harmony方法降维，可以选用.obsm中任何一项
    :return: 返回聚类降维后的anndata文件
    """
    if resolutions is None:
        resolutions = [1.0, 1.5]
        print("No resolutions provided. Using default resolutions: ", resolutions)
    
    if not skip_DR:
        print(f"Calculating neighbors with n_neighbors={n_neighbors}, n_pcs={n_pcs}, use_rep='{use_rep}'")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep)
        print("Neighbors calculated successfully.")
        print("Calculating UMAP...")
        sc.tl.umap(adata)
        print("UMAP calculation completed.")
    else:
        print("Dimensional reduction is skipped.")
    
    for res in resolutions:
        print(f"Running Leiden clustering with resolution={res}...")
        sc.tl.leiden(adata, key_added="leiden_res" + str(res), resolution=res)
        print(f"Leiden clustering completed for resolution={res}.")
        gc.collect()
        print("Garbage collection completed.")
    
    print("Subclustering process finished.")
    return adata


import numpy as np


def obs_key_wise_subsampling(bdata, obs_key, downsample):
    """
    下采样bdata，保证obs_key所限定的列中每一个值（身份）都至少有N个样本（N>1）；或对每个值所对应的样本量以N为缩放系数进行缩放（0<N<1)
    :param obs_key: 分组键
    :param bdata: 需要下采样的anndata文件
    :param downsample: 下采样系数；当N>1时为最大样本数，当0<N<1时为缩放系数
    :return: 返回下采样后的bdata
    """
    counts = bdata.obs[obs_key].value_counts()
    indices = []  # 用列表存储选取的样本索引
    
    if downsample > 1 and downsample % 1 == 0:  # 截断模式
        print("Start cutoff.")
        for group, count in counts.items():
            use_size = min(count, downsample)  # 限制样本数不超过scale
            selected_indices = np.random.choice(
                bdata.obs_names[bdata.obs[obs_key] == group],
                size=use_size,
                replace=False
            )
            indices.extend(selected_indices)  # 使用extend避免嵌套列表
            print(len(indices))
    elif 0 < downsample < 1:  # 缩放模式
        print("Start zoom.")
        for group, count in counts.items():
            use_size = max(1, round(count * downsample))  # 确保至少选择1个样本
            selected_indices = np.random.choice(
                bdata.obs_names[bdata.obs[obs_key] == group],
                size=use_size,
                replace=False
            )
            indices.extend(selected_indices)
    else:
        raise ValueError("Please recheck parameter `downsample`. It must be >1 (integer) or 0<N<1 (float).")
    
    return bdata[indices].copy()


def easy_DEG(bdata, save_addr, filename, obs_key="Subset_Identity",
             save_plot=True, plot_gene_num=5, downsample=False,
             method='wilcoxon',
             use_raw=True):
    """
    快速进行差异基因富集（DEG）
    """
    import os
    # import scanpy as sc
    # import pandas as pd
    import matplotlib.pyplot as plt
    if use_raw and bdata.raw is None:
        print("Warning: use_raw=True, but .raw not found in AnnData. Will fallback to .X.")
    
    hvg_key = "hvg_" + obs_key
    save_addr = save_addr if save_addr.endswith("/") else save_addr + "/"
    os.makedirs(save_addr, exist_ok=True)
    
    if downsample:
        print("Start downsampling...")
        bdata = obs_key_wise_subsampling(bdata, obs_key, downsample)
    
    print(f"Starting HVG ranking for '{obs_key}'...")
    sc.tl.rank_genes_groups(bdata, groupby=obs_key, use_raw=use_raw, method=method,
                            key_added=hvg_key)
    
    if save_plot:
        with plt.rc_context():
            sc.pl.rank_genes_groups_dotplot(bdata, groupby=obs_key, key=hvg_key, standard_scale="var",
                                            n_genes=plot_gene_num, dendrogram=False, use_raw=use_raw, show=False)
            plt.savefig(f"{save_addr}{filename}_HVG_dotplot_by_{obs_key}.pdf", bbox_inches="tight", dpi=300)
            plt.savefig(f"{save_addr}{filename}_HVG_dotplot_by_{obs_key}.png", bbox_inches="tight", dpi=300)
    groups = bdata.uns[hvg_key]['names'].dtype.names
    # 合并所有 group 的结果
    df_all = pd.concat([
        sc.get.rank_genes_groups_df(bdata, group=grp, key=hvg_key).assign(cluster=grp)
        for grp in groups
    ])
    
    # 第一种排序方式：按 logfoldchanges 降序，再按 names 升序
    df_sorted_logfc = df_all.sort_values(by=['names','logfoldchanges'], ascending=[False, True])
    
    # 第二种排序方式：按 pvals_adj 升序，再按 cluster 升序
    df_sorted_pval = df_all.sort_values(by=['cluster','pvals_adj'], ascending=[True, True])
    
    # 保存到 Excel 两个 sheet 中
    try:
        with pd.ExcelWriter(f"{save_addr}{filename}_HVG_{method}_{obs_key}.xlsx", engine='xlsxwriter') as writer:
            df_sorted_logfc.to_excel(writer, sheet_name='Sorted_by_logFC', index=False)
            df_sorted_pval.to_excel(writer, sheet_name='Sorted_by_pval', index=False)
            print("Excel file saved successfully.")
    except Exception as e:
        print(f"Error saving Excel file: {e}")
    
    print("Successfully saved.")
    return bdata


def score_gene_analysis(marker_dict, adata_subset,
                        downsample=False, plot=None, obs_key=None, save_addr=None):
    """
    对一个基因进行打分的函数，运行速度较慢
    :param downsample:
    :param marker_dict: 字典格式，形如{'GeneSetName1':['Gene1','Gene2',...]}
    :param adata_subset: anndata文件
    :param plot：是否作图
    :param obs_key：如果作图则必须提供，以该列为分组、marker_dict为基因集进行打分绘制dotplot
    :param save_addr：作图时使用的图片保存地址，缺省值为在工作目录下创建名为fig的子目录
    :return: 返回一个简化版的anndata，仅包含基因集打分作为基因表达矩阵，允许直接对接dotplot
    """
    if (plot or downsample) and (obs_key is None):
        raise ValueError("Value of obs_key must be passed.")
    if downsample:
        print("Start downsampling...")
        adata_subset = obs_key_wise_subsampling(adata_subset, obs_key, downsample)
    for key in marker_dict:
        sc.tl.score_genes(adata=adata_subset, gene_list=marker_dict[key], score_name=str(key), use_raw=False)
    var_df = pd.DataFrame(data=list(marker_dict.keys()), index=list(marker_dict.keys()), columns=['features'])
    obs_remained = ([x for x in ['orig.ident', 'disease', "Subset", 'Subset_Identity', 'Celltype_Identity'] if
                     x in adata_subset.obs_keys()] +
                    [x for x in adata_subset.obs.columns.tolist() if x.startswith('leiden')] +
                    [x for x in adata_subset.obs.columns.tolist() if x.startswith('exploration')])
    adata_score = anndata.AnnData(X=adata_subset.obs[list(marker_dict.keys())],
                                  obs=adata_subset.obs[obs_remained],
                                  var=var_df)
    if plot:
        with plt.rc_context():
            sc.pl.dotplot(adata_subset, groupby=obs_key, standard_scale="var",
                          var_names=list(marker_dict.keys()), show=False)
            plt.savefig(save_addr + "HVG_wilcoxin(ElasticNet)_by_" + obs_key + ".pdf", bbox_inches="tight")
            plt.savefig(save_addr + "HVG_wilcoxin(ElasticNet)_by_" + obs_key + ".png", bbox_inches="tight")
    return adata_score


def filter_and_save_adata(adata, cell_types_list, output_file, from_obs = "manual_cellsubtype_annotation"):
    '''
    我他妈的为什么要写这么个函数，还用了这么多次？？？

    :param adata:
    :param cell_types_list:
    :param output_file:
    :param from_obs:
    :return:
    '''
    filtered_adata = adata[adata.obs[from_obs].isin(cell_types_list)]
    filtered_adata.write_h5ad(output_file)
    del filtered_adata
    gc.collect()


def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

class Timer:
    def __init__(self, name="Code"):
        self.name = name
    def __enter__(self):
        self.start_time = time.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.time() - self.start_time
        print(f"{self.name} running costs: {self.elapsed_time:.4f} seconds")



    

def analysis_DEG(adata_subset, file_name, groupby_key, output_dir,downsample,use_raw,skip_QC=False):
    from src.EasyInterface.QualityControl import Basic_QC_Plot
    print(f"--> Starting differential expression analysis for group '{groupby_key}'...")
    easy_DEG(adata_subset, save_addr=output_dir, filename=file_name, obs_key=groupby_key,
             save_plot=True, plot_gene_num=5, downsample=downsample,use_raw=use_raw)
    # 基础QC图
    Basic_QC_Plot(
        adata_subset,
        prefixx=f"{file_name}_{groupby_key}",
        out_dir=output_dir
    )

def write_scenic_input(adata_subset,save_addr,use_col, file_name):
    import loompy as lp
    import numpy as np
    path = f"{save_addr}/{file_name}"
    isExist = os.path.exists(path)
    print(path)
    print(isExist)
    if isExist == False:
        os.makedirs(path)
        print("The new directory is created!")
    row_attrs = {"Gene": np.array(adata_subset.var_names),}
    col_attrs = {"CellID": np.array(adata_subset.obs_names),
                 "nGene": np.array(np.sum(adata_subset.X.transpose() > 0, axis=0)).flatten(),
                 "nUMI": np.array(np.sum(adata_subset.X.transpose(), axis=0)).flatten(),}
    print("Writing into loom.")
    lp.create(f"{path}/matrix.loom",
              adata_subset.X.transpose(),
              row_attrs,
              col_attrs)
    print("Writing meta data.")
    Adata2Csv = adata_subset.obs[["orig.ident", use_col, 'disease']]
    Adata2Csv.to_csv(f"{path}/meta_data.csv")
    print("Finished.")
    
def update_doublet(adata_old,adata_subset_new,obs_key="manual_cellsubtype_annotation",delete=True):
    # 获取在 adata_subset 中被标记为 "doublet" 的细胞名称
    doublet_cells = adata_subset_new.obs_names[adata_subset_new.obs[obs_key] == "Doublet" | "doublet"]
    print(f"Cells marked as 'doublet': {len(doublet_cells)}")
    if delete:
        print(f"Whole original anndata object has total cell number of: {adata_old.n_obs}")
        # 从 adata 中删除这些细胞
        adata_old = adata_old[~adata_old.obs_names.isin(doublet_cells)]
        # 验证删除的结果
        print(f"Cells marked as 'doublet' have been removed. Remaining cells in adata: {adata_old.n_obs}")
        return adata_old
    else:
        return doublet_cells

def process_adata(
    adata_subset,
    file_name,
    my_markers,
    marker_sheet,
    output_dir,
    do_subcluster=True,
    do_DEG_enrich=True,
    downsample=False,
    DEG_enrich_key="leiden_res",
    resolutions_list=[],
    use_rep="X_scVI",
    use_raw=True,
    **kwargs
):

    """
    主流程：处理子集 adata，对其进行子聚类、DEG富集、绘图。
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directory {output_dir} created or already exists.")
    
    # ==== 1. 可选：降维聚类 ====
    if do_subcluster:
        print("==> Starting subclustering...")
        adata_subset = subcluster(
            adata_subset,
            n_neighbors=20,
            n_pcs=min(adata_subset.obsm[use_rep].shape[1], 50),
            resolutions=resolutions_list,
            use_rep=use_rep
        )
        print("==> Subclustering completed.")
    
    # ==== 2.1 使用 leiden_res 作为分组方式；如果省略第一步则依赖原有adata.obs中的列，需要确保`resolutions_list`能对应实际存在的列 ====
    if DEG_enrich_key == "leiden_res":
        if not resolutions_list:
            raise ValueError("resolutions_list cannot be empty when using 'leiden_res' as DEG enrichment key.")
        if not all(isinstance(res, (int, float)) for res in resolutions_list):
            raise TypeError("All elements in resolutions_list must be integers or floats.")
        
        if do_DEG_enrich:
            print("==> Processing DEG enrichment for different resolutions...")
        
        # 2.1.1 分辨率比较图，和基础 QC 图
        from src.EasyInterface.QualityControl import Basic_QC_Plot
        process_resolution_umaps(adata_subset, output_dir, resolutions_list,use_raw=use_raw,**kwargs)
        Basic_QC_Plot(
            adata_subset,
            prefixx=f"{file_name}",
            out_dir=output_dir
        )
        
        # 2.1.2 每个分辨率进行绘图 + DEG
        for res in resolutions_list:
            groupby_key = f"leiden_res{res}"
            print(f"--> Drawing dotplot for group '{groupby_key}'...")
            scanpy_plot.geneset_dotplot(
                adata=adata_subset,
                markers=my_markers,
                marker_sheet=marker_sheet,
                output_dir=output_dir,
                filename_prefix=f"{file_name}_Geneset({marker_sheet})",
                groupby_key=groupby_key,
                use_raw=use_raw,
                **kwargs
            )
            print(f"--> Dotplot done for '{groupby_key}'.")
            
            if do_DEG_enrich:
                print(f"--> Running DEG enrichment for '{groupby_key}'...")
                easy_DEG(adata_subset, save_addr=output_dir, filename=file_name, obs_key=groupby_key,
                         save_plot=True, plot_gene_num=5, downsample=downsample, use_raw=use_raw)
    
    
    # ==== 2.2 其他 obs 中的分组变量 ====
    else:
        print(f"==> Creating UMAP plot for key '{DEG_enrich_key}'...")
        from src.ScanpyTools.Scanpy_Plot import ScanpyPlotWrapper
        umap_plot = ScanpyPlotWrapper(func=sc.pl.umap)
        umap_plot(
            save_addr=output_dir,
            filename=DEG_enrich_key,
            adata=adata_subset,
            color=[DEG_enrich_key],
            legend_loc="right margin",
            use_raw=use_raw,
            **kwargs
        )
        print(f"==> UMAP plot saved.")
        
        print(f"==> Drawing gene marker dotplot for key '{DEG_enrich_key}'...")
        from src.ScanpyTools.Scanpy_Plot import geneset_dotplot
        geneset_dotplot(
            adata=adata_subset,
            markers=my_markers,
            marker_sheet=marker_sheet,
            output_dir=output_dir,
            filename_prefix=f"{file_name}_Geneset({marker_sheet}_{DEG_enrich_key})",
            groupby_key=DEG_enrich_key,
            use_raw=use_raw,
            **kwargs
        )
        
        if do_DEG_enrich:
            print(f"--> Starting DEG enrichment analysis for '{DEG_enrich_key}'...")
            analysis_DEG(adata_subset=adata_subset,
                         file_name=file_name,
                         groupby_key=DEG_enrich_key,
                         output_dir=output_dir,downsample=10000,use_raw=use_raw)
                
    print("==> Process completed.")

def process_resolution_umaps(adata, output_dir, resolutions,use_raw=True,**kwargs):
    """
    生成 UMAP 图像，用于不同 Leiden 分辨率对比。
    """
    from src.ScanpyTools.Scanpy_Plot import ScanpyPlotWrapper
    umap_plot = ScanpyPlotWrapper(sc.pl.umap)
    color_keys = [f"leiden_res{res}" for res in resolutions]
    umap_plot(
        save_addr=output_dir,
        filename="Comparison_Leiden_Res",
        adata=adata,
        color=color_keys,
        legend_loc="on data",
        use_raw=use_raw,
        **kwargs
    )
    print("==> UMAP plot for resolution comparisons saved.")




