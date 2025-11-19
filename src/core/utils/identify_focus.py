import anndata
import pandas as pd
import numpy as np

import os, gc
from src.core.ext_anndata_ops import process_adata
from src.core.utils import geneset_editor

import logging
from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


@logged
def make_a_focus(adata, filename,
                 cat_key="Celltype", type_key="Subset_Identity", resubset=False):
    if not filename.endswith("csv"):
        raise ValueError("Please recheck filename, must end with .csv")
    
    df = adata.obs.astype(object).copy()
    df = df[[cat_key, type_key]]
    
    # 防止 Categorical 报错，转换为纯 Python list
    df[cat_key] = df[cat_key].tolist()
    df[type_key] = df[type_key].tolist()
    
    # group by 并合并 Subsets，同时去除重复并保持顺序
    df_grouped = (
        df.groupby(cat_key, as_index=False)
        .agg(lambda x: list(dict.fromkeys(v for v in x if pd.notna(v))))
    )
    
    # rename
    df_grouped.rename(columns={cat_key: "Name", type_key: 'Subsets'}, inplace=True)
    
    # 假设 df['genes'] 是列表
    df_grouped['Subsets'] = df_grouped['Subsets'].apply(lambda x: ",".join(map(str, x)) if isinstance(x, list) else x)
    
    # add extra columns
    df_grouped["Resubset"] = resubset
    df_grouped["Marker_class"] = df_grouped["Name"]
    
    # save
    df_grouped.to_csv(filename, index=False)
    
    return df_grouped


class IdentifyFocus():
    def __init__(self, focus_file, adata):
        '''
        根据 focus_file 中的亚群信息，对 adata 进行亚群拆分，并进行后续分析
        focus_file 的格式包含以下四列：
        Name : str
        Subsets : list of str
        Resubset : bool
        Marker_class : str
        
        TODO: 给每个大类都加个print()工具用来打印信息和提示
        
        :param focus_file:
        :param adata:
        '''
        self.adata = adata
        self.logger = logging.getLogger(self.__class__.__name__)
        if focus_file.endswith(".csv"):
            focus_sheet = pd.read_csv(focus_file)
        elif focus_file.endswith(".xlsx") or focus_file.endswith(".xls"):
            excelFile = pd.ExcelFile(focus_file)
            focus_sheet = excelFile.parse(excelFile.sheet_names[0])
        else:
            raise ValueError("focus_file must be csv, xlsx, xls format.")
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
    
    @logged
    def filter_and_save_subsets(self,
                                h5ad_prefix,  # 建议使用时间控制版本
                                save_addr,  # 取消预设值以避免储存在意外的地方
                                obs_key="Subset_Identity"):
        self.obs_key = obs_key
        
        os.makedirs(save_addr,exist_ok=True)
        
        for index, row in self.focus.iterrows():
            name = row['Name']
            subsets = row['Subsets']
            
            if subsets:
                index_list = self.adata.obs[obs_key].isin(subsets)
                adata_subset = self.adata[index_list]
                self.logger.info(f"Name: {name}, Subsets: {subsets}")
                self.logger.info(adata_subset.obs[obs_key].value_counts())
                
                output_path = os.path.join(save_addr, f"{h5ad_prefix}_{name}.h5ad")
                try:
                    adata_subset.write(output_path)
                    self.logger.info(f"Data for {name} written to {output_path}")
                except Exception as e:
                    self.logger.info(f"Error saving {name} to {output_path}: {e}")
            else:
                self.logger.info(f"Subsets for {name} is empty, skipping.")
    
    @logged
    def process_filtered_files(self,
                               Geneset_class,
                               save_addr,
                               h5ad_prefix,
                               **kwargs):
        for index, row in self.focus.iterrows():
            name = row['Name']
            val = row['Resubset']
            
            if isinstance(val, str):
                val_lower = val.lower()
                if val_lower in ("true", "t", "1"):
                    resubset = True
                elif val_lower in ("false", "f", "0"):
                    resubset = False
                else:
                    # 非法值处理，按需求可以抛出异常或设置默认值
                    resubset = False
            elif isinstance(val, (int, float)):
                # 数字类型，0 -> False, 非0 -> True
                resubset = bool(val)
            else:
                # 已经是 bool 或其他类型
                resubset = bool(val)
            
            self.logger.info(f"Processing cat {name}, containing types {resubset}.")
            
            input_path = os.path.join(save_addr, f"{h5ad_prefix}_{name}.h5ad")
            if not os.path.exists(input_path):
                self.logger.info(f"File {input_path} does not exist. Giving up reading {name} from h5ad.")
                continue
            
            adata_subset = anndata.read_h5ad(input_path)
            
            output_dir = os.path.join(save_addr, name)
            os.makedirs(output_dir, exist_ok=True)
            
            default_pars = {"resolutions_list": None,
                            "use_rep": "X_scVI",
                            "use_raw": True,
                            "do_DEG_enrich": True,
                            "DEG_enrich_key":self.obs_key,
                            "do_subcluster": False
                            }
            # 因为 DEG_enrich_key 不为 leiden_res 的时候对 resolution_list 是无视的
            # 所以可以先进性对已有 obs_key 的差异基因的计算
            default_pars.update(**kwargs)
            
            process_adata(  #
                adata_subset=adata_subset,
                filename_prefix=name,
                my_markers=Geneset_class,
                marker_sheet=row["Marker_class"],
                save_addr=output_dir,
                **default_pars
            )
            
            if resubset:
                default_pars.update({"DEG_enrich_key":"leiden_res",
                                    "do_subcluster": True})
                process_adata(  #
                    adata_subset=adata_subset,
                    filename_prefix=name,
                    my_markers=Geneset_class,
                    marker_sheet=row["Marker_class"],
                    save_addr=output_dir,
                    **default_pars
                )
            
            adata_subset.write_h5ad(input_path)
            self.logger.info(f"Finished cat {name}, h5ad saved inplace.")
            del adata_subset
            gc.collect()
