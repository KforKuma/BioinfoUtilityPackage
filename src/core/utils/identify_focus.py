import anndata
import pandas as pd
import os, gc
from src.core.ext_anndata_ops import process_adata
from src.core.utils import geneset_editor


def make_a_focus(adata,filename,
                 cat_key="Celltype", type_key="Subset_Identity", resubset=False):
    '''
    自动生成一个 IdentityFocus() 对象可读取的 focus_file
    如果细胞大类标签与 Geneset() 类的 sheet_name 标签一致，这个 focus_file 可以直接用于 IdentityFocus() 的分析
    等同于对细胞大类(categories)进行拆分，然后以细胞小类(types)为基准计算 DEG、按 signature 绘图。
    （当 Resubset 为 True 时）或对细胞大类拆分后，无视细胞小类进行重新的降维聚类，并计算 DEG、按 signature 绘图。

    :param adata:
    :param filename:
    :param cat_key:
    :param type_key:
    :return:
    '''
    df_grouped = adata.obs.groupby(cat_key, as_index=False).agg(list)
    df_grouped = df_grouped[[cat_key,type_key]]
    df_grouped.rename(columns={cat_key: "Name", type_key: 'Subsets'})
    df_grouped["Resubset"] = resubset
    df_grouped["Marker_class"] = df_grouped["Name"]
    df_grouped.to_csv(filename,index=False)




class IdentifyFocus():
    def __init__(self, focus_file, adata):
        '''
        根据 focus_file 中的亚群信息，对 adata 进行亚群拆分，并进行后续分析
        focus_file 的格式包含以下四列：
        Name : str
        Subsets : list of str
        Resubset : bool
        Marker_class : str


        :param focus_file:
        :param adata:
        '''
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

    @staticmethod
    def _log(msg):
        print(f"[IdentifyFocus Message] {msg}")

    def filter_and_save_subsets(self,
                                h5ad_prefix, # 建议使用时间控制版本
                                save_addr,  # 取消预设值以避免储存在意外的地方
                                obs_key="Subset_Identity"):
        for index, row in self.focus.iterrows():
            name = row['Name']
            subsets = row['Subsets']

            if subsets:
                index_list = self.adata.obs[obs_key].isin(subsets)
                adata_subset = self.adata[index_list]
                self._log(f"Name: {name}, Subsets: {subsets}")
                self._log(adata_subset.obs[obs_key].value_counts())

                output_path = os.path.join(save_addr, f"{h5ad_prefix}_{name}.h5ad")
                try:
                    adata_subset.write(output_path)
                    self._log(f"Data for {name} written to {output_path}")
                except Exception as e:
                    self._log(f"Error saving {name} to {output_path}: {e}")
            else:
                self._log(f"Subsets for {name} is empty, skipping.")

    def process_filtered_files(self,
                               Geneset_class,
                               save_addr,
                               h5ad_prefix,
                               **kwargs):
        for index, row in self.focus.iterrows():
            name = row['Name']
            resubset = row['Resubset']

            self._log(f"Processing cat {name}, containing types {resubset}.")

            input_path = os.path.join(save_addr, f"{h5ad_prefix}_{name}.h5ad")
            if not os.path.exists(input_path):
                self._log(f"File {input_path} does not exist. Giving up reading {name} from h5ad.")
                continue

            adata_subset = anndata.read_h5ad(input_path)

            output_dir =os.path.join(save_addr,name)
            os.makedirs(output_dir, exist_ok=True)


            default_pars = {"resolutions_list":None,
                            "use_rep":"X_scVI",
                            "use_raw":True,
                            "do_DEG_enrich":True
                            }

            default_pars.update(**kwargs)

            if resubset:
                default_pars.update({"do_subcluster":True})
            else:
                default_pars.update({"do_subcluster":False})

            process_adata(  #
                adata_subset=adata_subset,
                file_name=name,
                my_markers=Geneset_class,
                marker_sheet=row["Marker_class"],
                output_dir=output_dir,
                **default_pars
            )


            adata_subset.write_h5ad(input_path)
            self._log(f"Finished cat {name}, h5ad saved inplace.")
            del adata_subset
            gc.collect()
