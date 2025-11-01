import os, sys
os.environ['R_HOME'] = sys.exec_prefix + "/lib/R"

# import time
# import traceback

import numpy as np
import pandas as pd
import scipy
import igraph
import sklearn
import scanpy as sc
import scFates as scf
import anndata
# import palantir
import matplotlib.pyplot as plt
# import seaborn

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # DejaVu 是常用开源字体

def easy_hvg(adata, min_cells_fraction=0.05, n_top_genes=300):
    '''
    先去除线粒体、核糖体、Hemoglobin基因；
    再过滤掉在过少细胞中表达的基因；
    最后计算高变基因HVG并返回基因列表
    :param adata: AnnData对象（通常为log1p后的表达矩阵）
    :param min_cells_fraction: 至少多少比例的细胞表达才保留
    :param n_top_genes: HVG数量
    :return: hvg_list (list of str)
    '''
    import re
    # 1. 正则匹配过滤掉不需要的基因
    pattern = re.compile(r"^(MT-|RP[SL]\d+\w?\d*|HB[AB])", re.IGNORECASE)
    genes_to_remove = adata.var_names[adata.var_names.str.match(pattern)]
    mask = ~adata.var_names.isin(genes_to_remove)
    adata_filtered = adata[:, mask].copy()
    
    # 2. 过滤掉表达过少的基因
    min_cells = int(adata_filtered.n_obs * min_cells_fraction)
    print(min_cells)
    sc.pp.filter_genes(adata_filtered, min_cells=min_cells)
    
    # 3. 计算高变基因
    sc.pp.highly_variable_genes(
        adata_filtered,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
    )
    
    # 4. 返回HVG列表
    hvg_list = adata_filtered.var_names[adata_filtered.var['highly_variable']].tolist()
    return hvg_list


def summarize_nodes(adata, save_addr,
                    embedding_key="X_R",
                    discrete_vars="Subset_Identity",
                    continuous_vars=("percent.ribo", "percent.mt", "percent.hb"),
                    disease_var="disease_type"):
    """
    统计 assigned_node 分布、连续变量均值及疾病类型比例，并保存结果。

    Parameters
    ----------
    adata : AnnData
        包含 obsm[embedding_key] 和 obs 的 AnnData 对象
    save_addr : str
        输出文件保存路径
    embedding_key : str, default "X_R"
        obsm 中节点赋值用的矩阵
    discrete_vars : tuple of str
        分组计算 count/proportion 的分类变量
    continuous_vars : tuple of str
        计算均值的连续变量
    disease_var : str, default "disease_type"
        计算疾病分布比例的变量
    """
    os.makedirs(save_addr, exist_ok=True)
    
    # 0. 看一下 tip 和 fork 节点
    tip_nodes = adata.uns['graph']['tips'].tolist()
    fork_nodes = adata.uns['graph']['forks'].tolist()
    
    # 1. 计算 assigned_node
    node_assign = np.argmax(adata.obsm[embedding_key], axis=1)
    adata.obs["assigned_node"] = node_assign
    
    def node_type(n):
        if n in tip_nodes:
            return "tip"
        elif n in fork_nodes:
            return "fork"
        else:
            return "other"
    
    # 2. 离散变量统计
    def identity_simplify(count_df_annot, simplify_col="Subset_Identity", thr=0.66):
        result = []
        for node, subdf in count_df_annot.groupby("assigned_node"):
            # 寻找是否存在占比>阈值的亚群
            dominant = subdf[subdf["proportion"] > thr]
            if not dominant.empty:
                result.append(dominant.iloc[0])
            else:
                # 构造“mixed”行
                mixed_row = subdf.iloc[0].copy()
                mixed_row[simplify_col] = "mixed"
                mixed_row["proportion"] = 0
                result.append(mixed_row)
        
        result_df = pd.DataFrame(result).reset_index(drop=True)
        return result_df
    
    count_df = adata.obs.groupby(["assigned_node", discrete_vars], observed=True).size().reset_index(name="count")
    count_df["proportion"] = count_df.groupby("assigned_node")["count"].transform(lambda x: x / x.sum())
    # 可读性处理
    count_df_annot = count_df[count_df["assigned_node"].isin(tip_nodes + fork_nodes)]
    count_df_annot["Nodes"] = count_df_annot["assigned_node"].apply(node_type)
    count_df_annot = identity_simplify(count_df_annot, simplify_col="Subset_Identity")
    # count_df.to_csv(f"{save_addr}/Subset_by_Nodes.csv", index=False)
    
    # 3. 连续变量统计
    mean_df = adata.obs.groupby("assigned_node")[list(continuous_vars)].mean().reset_index()
    mean_df = mean_df[mean_df["assigned_node"].isin(tip_nodes + fork_nodes)]
    
    # mean_df.to_csv(f"{save_addr}/Percentage_by_Nodes.csv", index=False)
    
    # 4. 跨 Subset_Identity 的均值统计，进行身份估计
    def assign_subset_by_distance(mean_df, cross_df, features=None):
        if features is None:
            features = ["percent.ribo", "percent.mt", "percent.hb"]
        
        results = []
        for _, node_row in mean_df.iterrows():
            node_vec = node_row[features].values.astype(float)
            
            distances = []
            for _, sub_row in cross_df.iterrows():
                sub_vec = sub_row[features].values.astype(float)
                dist = np.linalg.norm(node_vec - sub_vec)
                distances.append((sub_row["Subset_Identity"], dist))
            
            # 取距离最小的 subset
            best_subset, min_dist = min(distances, key=lambda x: x[1])
            node_row = node_row.copy()
            node_row["closest_subset"] = best_subset
            node_row["distance"] = min_dist
            results.append(node_row)
        
        return pd.DataFrame(results)
    
    cross_df = adata.obs.groupby(discrete_vars)[list(continuous_vars)].mean().reset_index()
    annotated_nodes = assign_subset_by_distance(mean_df, cross_df)
    # cross_df.to_csv(f"{save_addr}/Percentage_by_Subset.csv", index=False)
    
    # 5. 疾病类型分布
    cross_count_df = adata.obs.groupby(["assigned_node", disease_var]).size().reset_index(name='count')
    cross_count_df["proportion"] = cross_count_df.groupby("assigned_node")["count"].transform(lambda x: x / x.sum())
    cross_count_df = identity_simplify(cross_count_df, simplify_col="disease_type", thr=0.5)
    cross_count_df = cross_count_df[cross_count_df["assigned_node"].isin(tip_nodes + fork_nodes)]
    
    dfs = [count_df_annot, annotated_nodes, cross_count_df]
    from functools import reduce
    result_table = reduce(lambda left, right: pd.merge(left, right, on="assigned_node", how="outer"), dfs)
    # result_table = result_table.loc[:, ['assigned_node','Subset_Identity','dat']]
    result_table.to_csv(f"{save_addr}/Combined_Nodes_Info.csv", index=False)


def fork_check(adata):
    B = adata.uns["graph"]["B"]
    
    g = igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected")
    degrees = g.degree()
    high_degree_nodes = [i for i, d in enumerate(degrees) if d >= 3]
    
    for edge in g.es:
        u, v = edge.tuple
        if u in high_degree_nodes and v in high_degree_nodes:
            print((u, v))


def filter_adata_by_seg(adata, spline_df=5, seg_col="seg"):
    """
    根据每个 seg 的细胞数过滤 AnnData 对象，并打印筛选前后细胞数量变化。

    Parameters
    ----------
    adata : AnnData
        输入的 AnnData 对象，要求 .obs 里有 seg 列。
    min_cells_per_seg : int
        每个 seg 至少保留多少细胞，否则丢弃。
    seg_col : str
        .obs 中存放 seg 信息的列名，默认 "seg"。

    Returns
    -------
    AnnData
        过滤后的 AnnData。
    """
    import pandas as pd
    
    # 打印筛选前信息
    total_before = adata.n_obs
    value_df = adata.obs[seg_col].value_counts()
    print(f"Before filtering: total cells = {total_before}")
    print(f"Segment counts:\n{value_df}")
    
    # 进行筛选
    keep_segs = value_df.index[value_df.values >= spline_df]
    adata_filtered = adata[adata.obs[seg_col].isin(keep_segs)].copy()
    
    # 打印筛选后信息
    total_after = adata_filtered.n_obs
    print(f"\nAfter filtering: total cells = {total_after} (removed {total_before - total_after})")
    value_df_after = adata_filtered.obs[seg_col].value_counts()
    print(f"Remaining segment counts:\n{value_df_after}")
    
    return adata_filtered


def easy_split(gene_str):
    """
    将形如 "SELENBP1;TST;ETHE1;BPNT1;PAPSS2" 的字符串
    拆分为 ["SELENBP1", "TST", "ETHE1", "BPNT1", "PAPSS2"]
    自动处理空格、逗号、分号及多余分隔符。
    """
    if not isinstance(gene_str, str):
        return []
    import re
    # 按分号或逗号分割，strip 去除空白，并过滤空字符串
    genes = [g.strip() for g in re.split(r'[;,]', gene_str) if g.strip()]
    return genes


def choose_assoc_file(save_addr):
    """
    在 save_addr 目录中查找所有形如:
      04_test_assoc(splined=XXX).h5ad
    的文件，提取括号内的 spline 值。
    如果只有一个匹配文件 -> 自动返回它的完整路径和 spline 值。
    如果多个 -> 列表显示并由用户输入编号选择。
    返回 (full_path, spline_value) 或 (None, None)（未找到时）。
    """
    # 更宽松的正则：捕获括号里任意非右括号内容（支持数字、浮点、字符串等）
    import os
    import re
    import datetime
    
    pattern = re.compile(r"^04_test_assoc\(splined=([^)]*)\)\.h5ad$")
    
    try:
        entries = os.listdir(save_addr)
    except FileNotFoundError:
        print(f"目录不存在: {save_addr}")
        return None, None
    
    matches = []
    for fname in entries:
        m = pattern.match(fname)
        if m:
            spline_val = m.group(1)
            full = os.path.join(save_addr, fname)
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                mtime = None
            matches.append({
                "fname": fname,
                "full": full,
                "spline": spline_val,
                "mtime": mtime
            })
    
    if not matches:
        print("❌ 未找到任何匹配文件：04_test_assoc(splined=...).h5ad")
        return None, None
    
    # 如果只有一个，自动选择
    if len(matches) == 1:
        sel = matches[0]
        print(f"✅ 仅发现一个匹配文件，自动选择：{sel['fname']}")
        return sel["full"], sel["spline"]
    
    # 多个时，按修改时间降序显示（最近的在前）
    matches.sort(key=lambda x: x["mtime"] or 0, reverse=True)
    
    print("⚠️ 发现多个匹配文件，请选择：")
    for i, m in enumerate(matches):
        mtime_str = datetime.datetime.fromtimestamp(m["mtime"]).strftime("%Y-%m-%d %H:%M:%S") if m["mtime"] else "未知时间"
        print(f"[{i}] {m['fname']}    (spline='{m['spline']}', mtime={mtime_str})")
    
    # 交互选择
    while True:
        try:
            idx = input("请输入要选择的编号（回车选择[0]=最近修改的）：").strip()
            if idx == "":
                chosen = matches[0]
                break
            idx_int = int(idx)
            if 0 <= idx_int < len(matches):
                chosen = matches[idx_int]
                break
            else:
                print("编号超出范围，请重新输入。")
        except ValueError:
            print("输入无效，请输入数字编号或直接回车。")
    
    print(f"已选择：{chosen['fname']}")
    return chosen["full"], chosen["spline"]


import re
import pandas as pd


def filter_genes(gene_names):
    """
    过滤掉不希望保留的基因，如线粒体、核糖体蛋白、免疫球蛋白、orf类、克隆编号等。

    参数：
    ----------
    gene_names : list 或 pandas.Index
        基因名列表（例如 adata.var_names）

    返回：
    ----------
    clean_genes : list
        过滤后的基因名列表（按字母排序）
    removed_genes : list
        被移除的基因名列表
    """
    if isinstance(gene_names, (pd.Index,)):
        gene_names = gene_names.to_list()
    
    pattern = re.compile(
        r"^(?:"
        r"MT-"  # 线粒体基因
        r"|RP[SL]\d+\w?\d*"  # 核糖体蛋白
        r"|HB[AB]"  # 血红蛋白
        r"|IGH[AGM]\d*"  # 免疫球蛋白重链
        r"|IG[KL]C\d*"  # 免疫球蛋白轻链
        r"|C?\d*orf\d*"  # 含orf的基因
        r"|AC\d+"  # AC编号
        r"|AL\d+"  # AL编号
        r"|LINC\d+"  # 长链非编码RNA
        r"|SNOR[AD]\d+"  # 小核RNA
        r"|MIR\d+"  # microRNA
        r"|LOC\d+"  # LOC基因
        r")",
        re.IGNORECASE,
    )
    
    removed_genes = [x for x in gene_names if re.match(pattern, x)]
    clean_genes = sorted([x for x in gene_names if x not in removed_genes])
    
    return clean_genes, removed_genes



