import pandas as pd
import numpy as np

import anndata

import os,gc,re



def _kdk_data_prepare(adata,meta, unit_key="orig.ident",type_key="Subset_Identity"):
    count_dataframe = (
        adata.obs[[unit_key, type_key]]
        .groupby([unit_key, type_key])
        .size()
        .reset_index(name='count')
    )
    merge_df = pd.merge(count_dataframe, meta, how='inner', on=unit_key)
    count_group_df = merge_df


    count_group_df["log_count"] = np.log1p(count_group_df["count"])
    count_group_df["percent"] = count_group_df["count"] / count_group_df.groupby(unit_key)["count"].transform("sum")
    count_group_df["logit_percent"] = np.log(count_group_df["percent"] + 1e-5 / (1 - count_group_df["percent"] + 1e-5))
    count_group_df["total_count"] = count_group_df.groupby(unit_key)["count"].transform("sum")

    return count_group_df




def _kdk_make_meta(adata, group_key="orig.ident"):
    '''
    生成一个用来进行下游分析 meta 文件，包含必要控制的变量。

    :param adata:
    :return:
    '''
    # 选出字符串列（object 或 string）
    string_cols = [c for c in adata.obs.columns if type(adata.obs[c][0]) == str ]

    # 确保 group_key 也在结果里
    if group_key not in string_cols:
        string_cols.append(group_key)

    def unique_or_none(x):
        vals = x.dropna().unique()
        if len(vals) == 1:
            return vals[0]
        else:
            return None  # 多值或空值用 None
    # 聚合
    df_grouped = adata.obs[string_cols].groupby(group_key).agg(unique_or_none).reset_index()

    # 去除全 None 列
    cols_remain = [c for c in df_grouped.columns if df_grouped[c].unique() is None]
    df_grouped = df_grouped.drop(columns=cols_remain)

    return df_grouped


def make_a_meta(adata, meta_file, group_key="orig.ident"):
    '''
    生成一个可读的 meta 文件，也可以手动在上面修改

    :param adata:
    :param meta_file:
    :param group_key:
    :return:
    '''
    meta = _kdk_make_meta(adata, group_key)
    meta.to_csv(meta_file)


def kdk_prepare(adata, meta_file=None, group_key="orig.ident", type_key="Subset_Identity"):
    '''

    :param adata:
    :param meta_file: 包含样本制作信息的表格，兼容 csv 和 xlsx，默认 header=True index=False
    :param group_key:
    :param type_key:
    :return:
    '''
    # 读取 meta 信息
    if meta_file is None:
        meta = _kdk_make_meta(adata, group_key)
    else:
        meta_file = meta_file.strip()
        if meta_file.lower().endswith("csv"):
            meta = pd.read_csv(meta_file)
        elif meta_file.lower().endswith("xlsx"):
            meta = pd.read_excel(meta_file)
        else:
            raise ValueError("[kdk_prepare] Meta file must ends with 'csv' or 'xlsx'.")

    # 准备 KW 分析所需矩阵
    count_df = _kdk_data_prepare(adata, meta, unit_key="orig.ident", type_key="Subset_Identity")


