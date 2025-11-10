import anndata
import pandas as pd
import os, gc




class IdentifyFocus():
    def __init__(self, focus_file, adata):
        self.adata = adata

        excelFile = pd.ExcelFile(focus_file)
        focus_sheet = excelFile.parse(excelFile.sheet_names[0])
        # 去除方括号并拆分基因集
        focus_sheet['Subsets'] = (
            focus_sheet['Subsets']
            .str.strip('[]')  # 去除外部方括号
            .str.split(',')  # 拆分为列表
        )
        # 去除每个元素中的多余引号和空格
        focus_sheet['Subsets'] = focus_sheet['Subsets'].apply(
            lambda gene_list: [gene.strip().strip("'").strip('"').strip() for gene in gene_list]
        )
        self.focus = focus_sheet

    def filter_and_save_subsets(self,
                                h5ad_prefix,
                                output_dir,  # 取消预设值以避免储存在意外的地方
                                obs_key="Subset_Identity"):
        for index, row in self.focus.iterrows():
            name = row['Name']
            subsets = row['Subsets']

            if subsets:
                index_list = self.adata.obs[obs_key].isin(subsets)
                adata_subset = self.adata[index_list]
                print(f"Name: {name}, Subsets: {subsets}")
                print(adata_subset.obs[obs_key].value_counts())

                output_path = os.path.join(output_dir, f"{h5ad_prefix}_{name}.h5ad")
                try:
                    adata_subset.write(output_path)
                    print(f"Data for {name} written to {output_path}")
                except Exception as e:
                    print(f"Error saving {name} to {output_path}: {e}")
            else:
                print(f"Subsets for {name} is empty, skipping.")

    def process_filtered_files(self,
                               Geneset_class,
                               storage_dir,
                               h5ad_prefix,
                               process_adata_func,
                               resolutions_list=[1.0, 1.5],
                               use_rep="X_scVI",
                               use_raw=True,
                               **kwargs):
        for index, row in self.focus.iterrows():
            name = row['Name']
            resubset = row['Resubset']

            input_path = os.path.join(storage_dir, f"{h5ad_prefix}_{name}.h5ad")
            if not os.path.exists(input_path):
                print(f"File {input_path} does not exist. Giving up reading {name} from h5ad.")
                continue

            adata_subset = anndata.read(input_path)
            output_dir = f"{storage_dir}/{name}/"
            os.makedirs(output_dir, exist_ok=True)

            if resubset:
                process_adata_func(  #
                    adata_subset=adata_subset,
                    file_name=name,
                    my_markers=Geneset_class,
                    marker_sheet=row["Marker_class"],
                    output_dir=output_dir,
                    do_subcluster=True,
                    do_DEG_enrich=True,
                    DEG_enrich_key="leiden_res",
                    resolutions_list=resolutions_list,
                    use_rep=use_rep,
                    use_raw=use_raw,
                    **kwargs
                )
            else:
                process_adata_func(  #
                    adata_subset=adata_subset,
                    file_name=name,
                    my_markers=Geneset_class,
                    marker_sheet=row["Marker_class"],
                    output_dir=output_dir,
                    do_subcluster=False,
                    do_DEG_enrich=True,
                    DEG_enrich_key="Subset_Identity",
                    resolutions_list=[],
                    use_rep=use_rep,
                    use_raw=use_raw,
                    **kwargs
                )

            adata_subset.write(input_path)
            del adata_subset
            gc.collect()
