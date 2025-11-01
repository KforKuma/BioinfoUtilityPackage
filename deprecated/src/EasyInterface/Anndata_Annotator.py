import pandas as pd
import anndata
import os,gc
# 迁移完成，打包成新的 class ObsEditor


def make_new_ident(anno_list, anno_obs_key, adata_obs_key_list, adata):
    """
    将列表形式的聚类结果的身份判定，修订（放入）anndata文件中
    :param anno_list: 按照[identity_for_cluster1, identity_for_cluster2, ...,]格式写好的细胞亚群身份定义
    :param anno_obs_key: 上述列表所参照的obs_key，一般为诸如‘leiden0_5’形式的数字聚类信息列
    :param adata_obs_key_list: 列表，包含所需要更改的.obs项目
    :param adata: 需要修订的anndata文件
    :return: 返回修订后的anndata文件
    """
    cl_annotation = dict()
    for i in range(0, len(anno_list)):
        cl_annotation[str(i)] = anno_list[i]
    print(cl_annotation)
    for i in adata_obs_key_list:
        del adata.obs[i]
        adata.obs[i] = adata.obs[anno_obs_key].map(cl_annotation)
    return adata


def copy_all_ident(adata_children_obs_key, adata_parent_obs_key, adata_children, adata_parent):
    """
    按照adata_children的某一obs_key定义好的细胞信息，修正adata_parent的细胞信息
    :param adata_children_obs_key: 所依据的adata_children的obs_key
    :param adata_parent_obs_key: adata_parent需要修订的obs_key
    :param adata_children: 一般为某一细分亚群
    :param adata_parent: 一般为某一大类群，或原始anndata文件
    :return: 无，但打印修订所涉及的细胞身份及其在大群中的数量
    """
    adata_parent.obs[adata_parent_obs_key] = adata_parent.obs[adata_parent_obs_key].tolist()
    # adata_parent.obs[adata_parent_obs_key] = pd.Series(adata_parent.obs[adata_parent_obs_key].tolist(), ) # deprecated
    for i in adata_children.obs[adata_children_obs_key].unique().tolist():
        print(i)
        index = adata_parent.obs_names[(adata_children.obs[adata_parent_obs_key] == i)]
        adata_parent.obs.loc[index, adata_parent_obs_key] = i
        print(len(adata_parent[adata_parent.obs[adata_parent_obs_key] == i]))


def change_one_ident_fast(adata, key, old, new):
    """
    更快速地替换分类列中的值，仅当必要时添加类别，且不执行慢的remove操作
    """
    if pd.api.types.is_categorical_dtype(adata.obs[key]):
        if new not in adata.obs[key].cat.categories:
            adata.obs[key] = adata.obs[key].cat.add_categories([new])
        # 布尔索引一次完成
        mask = adata.obs[key] == old
        adata.obs.loc[mask, key] = new
        print(f"Replaced {mask.sum()} cells from '{old}' to '{new}'.")
        # 取消 remove_categories —— 慢且通常没必要
    else:
        mask = adata.obs[key] == old
        adata.obs.loc[mask, key] = new
        print(f"Replaced {mask.sum()} cells from '{old}' to '{new}'.")


def generate_subclusters_by_identity(
        adata: anndata.AnnData,
        identity_key: str = "Subset_Identity",
        cell_idents_list: list = None,
        resolutions: list = [0.5, 1.0],
        output_dir: str = ".",
        use_rep: str = "X_scVI",
        subcluster_func=None,
        n_neighbors: int = 20,
        filename_prefix: str = "Step06_Subset"
):
    """
    对指定的细胞群体进行子聚类分析并保存为独立文件。

    Parameters:
        adata: AnnData
            原始 AnnData 数据对象。
        identity_key: str
            用于选择子集的 obs 列名，默认 "Subset_Identity"。
        identities: list
            需要处理的细胞身份列表，默认使用该列中的所有唯一值。
        resolutions: list
            聚类分辨率列表，例如 [0.5, 1.0]。
        output_dir: str
            子集 h5ad 文件的保存目录。
        use_rep: str
            用于聚类的表示空间（例如 "X_scVI"）。
        subcluster_func: callable
            聚类函数，例如 subcluster(adata_subset, ...)，必须传入。
        n_neighbors: int
            聚类时使用的邻居数。
        filename_prefix: str
            输出文件名前缀。
    """
    assert subcluster_func is not None, "请传入 subcluster 函数作为参数 subcluster_func"
    os.makedirs(output_dir, exist_ok=True)
    if cell_idents_list is None:
        cell_idents_list = adata.obs[identity_key].unique()
    
    for ident in cell_idents_list:
        print(f"\n🔍 Now processing subset: {ident}")
        adata_subset = adata[adata.obs[identity_key] == ident].copy()
        
        # 删除 leiden_res 相关列（obs）
        leiden_cols = [col for col in adata_subset.obs.columns if 'leiden_res' in col]
        if leiden_cols:
            adata_subset.obs.drop(columns=leiden_cols, inplace=True)
        
        # 删除 leiden_res 相关项（uns）
        leiden_keys = [key for key in adata_subset.uns.keys() if 'leiden_res' in key]
        for key in leiden_keys:
            del adata_subset.uns[key]
        
        # 子聚类
        adata_subset = subcluster_func(
            adata_subset,
            n_neighbors=n_neighbors,
            n_pcs=min(adata.obsm[use_rep].shape[1], 50),
            resolutions=resolutions,
            use_rep=use_rep
        )
        
        # 保存
        filename = os.path.join(output_dir, f"{filename_prefix}_{ident}.h5ad")
        adata_subset.write(filename)
        print(f"💾 Saved to {filename}")
        
        # 清理内存
        del adata_subset
        gc.collect()


def run_deg_on_subsets(
        cell_idents_list: list = None,
        resolutions: list = [0.5, 1.0],
        base_input_path: str = ".",
        base_output_path: str = ".",
        deg_method: str = "wilcoxon",
        save_plot: bool = True,
        plot_gene_num: int = 5,
        use_raw: bool = True,
        obs_subset: bool = False,
        save_prefix: str = "Step06_Subset",
        output_suffix: str = "_DEG.h5ad",
        easy_deg_func=None,
):
    """
    对 AnnData 中不同子集执行 DEG 分析（基于 leiden 聚类）。

    Parameters:
        resolutions: list
            需要运行 DEG 的 leiden 分辨率列表。
        base_input_path: str
            子集 h5ad 文件的基础目录。
        base_output_path: str
            输出 DEG 文件的基础目录。
        deg_method: str
            使用的 DEG 方法，例如 "wilcoxon"。
        save_plot: bool
            是否保存 DEG 的 marker gene 图。
        plot_gene_num: int
            每个 cluster 显示的 top marker gene 数量。
        use_raw: bool
            是否使用原始表达值进行 DEG。
        save_prefix: str
            输入输出文件名的前缀部分。
        output_suffix: str
            输出 DEG 后缀，例如 "_DEG.h5ad"。
        easy_deg_func: callable
            你自己的 easy_DEG 函数，必须作为参数传入。
    """
    
    assert easy_deg_func is not None, "请传入 easy_DEG 函数作为参数 easy_deg_func"
    
    for cell_ident in cell_idents_list:
        print(f"\n=== Now processing subset: {cell_ident} ===")
        
        input_file = os.path.join(base_input_path, f"{save_prefix}_{cell_ident}.h5ad")
        print(f"Loading file: {input_file}")
        adata_subset = anndata.read_h5ad(input_file)
        
        for res in resolutions:
            group_key = f"leiden_res{res}"
            print(f"Running easy_DEG for resolution {res} with group key '{group_key}'...")
            os.makedirs(base_output_path,exist_ok=True)
            adata_subset = easy_deg_func(
                adata_subset,
                save_addr=base_output_path,
                filename=f"Secondary_Cluster_{cell_ident}(For clean up)",
                obs_key=group_key,
                save_plot=save_plot,
                plot_gene_num=plot_gene_num,
                downsample=obs_subset,
                method=deg_method,
                use_raw=use_raw
            )
            print(f"Finished DEG at resolution {res}.")
        
        output_file = os.path.join(base_output_path, f"{save_prefix}_{cell_ident}{output_suffix}")
        adata_subset.write_h5ad(output_file)
        print(f"Saved DEG results to: {output_file}")


def apply_assignment_annotations(
        assignment_file: str,
        adata_main: anndata.AnnData,
        h5ad_dir: str,
        obs_key_col: str = "Obs_key_select",
        subset_file_col: str = "Subset_File",
        subset_no_col: str = "Subset_No",
        identity_col: str = "Identity",
        output_key: str = "Subset_Identity",
        fillna_from_col: str = None
):
    """
    根据 assignment 表格更新主 AnnData 对象中的细胞亚群注释。

    参数:
    - assignment_file: assignment Excel 文件路径
    - adata_main: 主 AnnData 对象（将被更新）
    - h5ad_dir: 子集 h5ad 文件所在目录
    - obs_key_col, subset_file_col, subset_no_col, identity_col: assignment 表格中对应的列名
    - output_key: 主 AnnData 中需要写入的列名
    - fillna_from_col: 用于填充 output_key 中空值的备用列
    """
    
    excel_data = pd.ExcelFile(assignment_file)
    assignment_sheet = excel_data.parse(excel_data.sheet_names[0])
    
    for subset_filename in set(assignment_sheet[subset_file_col]):
        print(f"\nNow reading {subset_filename} subset.")
        input_path = f"{h5ad_dir}/{subset_filename}"
        adata_subset = anndata.read(input_path)
        
        # 提取 obs_key
        obs_key_series = assignment_sheet.loc[
            assignment_sheet[subset_file_col] == subset_filename, obs_key_col
        ].dropna().drop_duplicates()
        obs_key = obs_key_series.iat[0] if not obs_key_series.empty else None
        print(f"Obs key for {subset_filename}: {obs_key}")
        
        # identity 映射字典
        subset_data = assignment_sheet[assignment_sheet[subset_file_col] == subset_filename]
        result_dict = subset_data.set_index(subset_no_col)[identity_col].to_dict()
        updated_dict = {str(k): v for k, v in result_dict.items()}
        print(f"Created identity dictionary for {subset_filename} with {len(updated_dict)} entries")
        
        adata_subset.obs["tmp"] = adata_subset.obs[obs_key].map(updated_dict)
        unique_identities = adata_subset.obs["tmp"].dropna().unique()
        
        # 如果 output_key 不存在，则初始化为空列
        if output_key not in adata_main.obs.columns:
            adata_main.obs[output_key] = pd.Series(index=adata_main.obs_names, dtype="str")
        
        # 处理 Categorical 类型的列，扩展类别
        if pd.api.types.is_categorical_dtype(adata_main.obs[output_key]):
            existing_categories = set(adata_main.obs[output_key].cat.categories)
            new_categories = set(unique_identities) - existing_categories
            if new_categories:
                adata_main.obs[output_key] = adata_main.obs[output_key].cat.add_categories(list(new_categories))
        
        for cell_identity in unique_identities:
            print(f"  Processing identity: {cell_identity}")
            index = adata_subset.obs_names[adata_subset.obs["tmp"] == cell_identity]
            adata_main.obs.loc[index, output_key] = cell_identity
            updated_cells = (adata_main.obs[output_key] == cell_identity).sum()
            print(f"  -> Updated {updated_cells} cells with identity '{cell_identity}'")
    
    # 用其他列补全缺失值
    if fillna_from_col and fillna_from_col in adata_main.obs.columns:
        n_missing = adata_main.obs[output_key].isna().sum()
        adata_main.obs[output_key] = adata_main.obs[output_key].fillna(adata_main.obs[fillna_from_col])
        print(f"Filled {n_missing} missing '{output_key}' values using '{fillna_from_col}'")
    
    print("\nAll assignments applied.")
