# ===== Third-party =====
import os.path

import numpy as np
import pandas as pd

import scanpy as sc
import gseapy as gp

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import scanpy as sc
import anndata as ad

from src.core.plot.utils import matplotlib_savefig

def balance_cell_subsets(adata, subset_col='Subset_Identity', cond_col='cond_group', groups=['Inflammatory', 'Control'],
                         random_state=42):
    """
    对每个细胞亚群，平衡其在两个实验组别中的数量（取较小值）。
    """
    # 1. 提取元数据
    obs = adata.obs[[subset_col, cond_col]].copy()
    
    # 2. 计算每个亚群在每个组别中的频数
    # 得到一个表：Index是Subset_Identity, Columns是cond_group
    counts = obs.groupby([subset_col, cond_col]).size().unstack(fill_value=0)
    
    # 3. 确定每个亚群应该保留的数量（取两组中的最小值）
    # 如果某组完全没有该细胞，则该亚群保留 0
    min_counts_per_subset = counts[groups].min(axis=1)
    
    sampled_indices = []
    
    # 4. 遍历每个亚群进行下采样
    for subset, n_target in min_counts_per_subset.items():
        if n_target <= 0:
            continue
        
        for group in groups:
            # 找到当前亚群且属于当前组别的所有细胞索引
            current_idx = obs[(obs[subset_col] == subset) & (obs[cond_col] == group)].index
            
            # 随机抽样
            sampled_idx = pd.Series(current_idx).sample(n=int(n_target), random_state=random_state)
            sampled_indices.extend(sampled_idx.tolist())
    
    # 5. 根据索引切片并返回新的 AnnData
    adata_balanced = adata[sampled_indices].copy()
    
    print(f"平衡完成。原始细胞数: {adata.n_obs}, 平衡后细胞数: {adata_balanced.n_obs}")
    return adata_balanced


def analyze_DEG(adata,save_addr,filename,
                subset_col='Subset_Identity',
                group1='Absorptive colonocyte Guanylins+',
                group2='Absorptive colonocyte'):
    """
    计算 group1 vs group2 的 DEG 并绘制火山图
    """
    # 1. 提取这两个亚群
    subset_adata = adata[adata.obs[subset_col].isin([group1, group2])].copy()
    
    # 2. 差异表达分析 (使用 t-test 或 wilcoxon)
    # reference='Absorptive colonocyte' 表示计算 Guanylins+ 相对于它的变化
    sc.tl.rank_genes_groups(subset_adata,
                            groupby=subset_col,
                            reference=group2,
                            groups=[group1],
                            method='wilcoxon')
    
    # 3. 提取结果到 DataFrame
    result = sc.get.rank_genes_groups_df(subset_adata, group=group1)
    result.to_csv(f"{save_addr}/{filename}.csv", index=False)


from adjustText import adjust_text


def plot_volcano(result, save_addr, filename,
                 cluster_genes_dict=None,
                 lfc_limit=10,
                 p_thresh=0.05,
                 lfc_thresh=1.0):
    """
    使用 fig, ax 对象绘制高性能火山图，并自动优化标签布局。
    """
    # 0. 确定基因名所在的列 (自动兼容 'names' 或 'gene')
    gene_col = 'names' if 'names' in result.columns else 'gene'
    
    # 1. 数据预处理
    df = result.copy()
    df['sig'] = 'Normal'
    df.loc[(df['pvals_adj'] < p_thresh) & (df['logfoldchanges'] > lfc_thresh), 'sig'] = 'Up'
    df.loc[(df['pvals_adj'] < p_thresh) & (df['logfoldchanges'] < -lfc_thresh), 'sig'] = 'Down'
    
    df['plot_lfc'] = df['logfoldchanges'].clip(-lfc_limit, lfc_limit)
    df['nlog10'] = -np.log10(df['pvals_adj'] + 1e-300)
    
    # 2. 初始化画布
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {'Up': '#e41a1c', 'Down': '#377eb8', 'Normal': '#d9d9d9'}
    
    # 3. 绘制散点
    sns.scatterplot(data=df, x='plot_lfc', y='nlog10', hue='sig',
                    palette=colors, s=25, alpha=0.7, edgecolor=None, ax=ax, rasterized=True)
    
    # 4. 绘制阈值线
    ax.axvline(x=lfc_thresh, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(x=-lfc_thresh, color='gray', linestyle='--', linewidth=0.8)
    ax.axhline(y=-np.log10(p_thresh), color='gray', linestyle='--', linewidth=0.8)
    
    # 5. 标注指定基因并优化布局
    texts = []
    if cluster_genes_dict is not None:
        for cluster, genes in cluster_genes_dict.items():
            # 筛选匹配的行
            genes_to_plot = df[(df['cluster'] == cluster) & (df[gene_col].isin(genes))]
            
            # 1) 打印未匹配到的基因
            found_genes = genes_to_plot[gene_col].tolist()
            missing_genes = set(genes) - set(found_genes)
            if missing_genes:
                print(f"⚠️ [Warning] 在 Cluster '{cluster}' 中未找到基因: {missing_genes}")
            if not genes_to_plot.empty:
                print(f"✅ [Info] 正在为 Cluster '{cluster}' 标注 {len(found_genes)} 个基因")
            
            # 2) 创建文本对象
            for _, row in genes_to_plot.iterrows():
                texts.append(ax.text(row['plot_lfc'], row['nlog10'], row[gene_col],
                                     fontsize=9, fontweight='medium'))
        
        # 3) 核心优化：自动排版防止重叠
        if texts:
            adjust_text(texts, ax=ax,
                        only_move={'points': 'y', 'text': 'xy'},  # 允许在xy方向移动文字以避开点
                        arrowprops=dict(arrowstyle='->', color='black', lw=0.5),  # 添加指引线
                        expand_points=(1.5, 1.5))  # 增加点周围的排斥力
    
    # 6. 细节修饰
    clusters = df['cluster'].unique()
    title = f"{clusters[0]} vs Others" if len(clusters) > 0 else "Volcano Plot"
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel('$\log_{2}(\text{Fold Change})$ (Clipped)', fontsize=12)
    ax.set_ylabel('$-\log_{10}(\text{Adjusted P-value})$', fontsize=12)
    ax.set_xlim(-lfc_limit * 1.1, lfc_limit * 1.1)
    
    # 图例处理
    ax.legend(title='Expression', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    
    sns.despine()  # 去除上方和右侧边框
    plt.tight_layout()
    # 保存 PDF 和 PNG
    abs_path = os.path.join(save_addr, filename)
    matplotlib_savefig(fig,abs_path)
    
    print(f"Volcano plot saved as PNG and PDF. LFC values clipped at ±{lfc_limit}.")


def trim_redundant_terms(res_df, overlap_thresh=0.6):
    """
    根据基因重叠度自动剔除冗余的 GO Term
    """
    # 确保按显著性排序
    res_df = res_df.sort_values('Adjusted P-value').reset_index(drop=True)
    keep_indices = []
    
    # 提取基因集列表 (gseapy 结果中基因通常以 ';' 分隔)
    gene_sets = [set(genes.split(';')) for genes in res_df['Genes']]
    
    for i in range(len(gene_sets)):
        is_redundant = False
        for j in keep_indices:
            # 计算 Jaccard 相似度: 交集 / 并集
            intersection = len(gene_sets[i].intersection(gene_sets[j]))
            union = len(gene_sets[i].union(gene_sets[j]))
            jaccard = intersection / union if union > 0 else 0
            
            if jaccard > overlap_thresh:
                is_redundant = True
                break
        
        if not is_redundant:
            keep_indices.append(i)
    
    return res_df.iloc[keep_indices]


def run_go_enrichment(result,save_addr, filename, dataset_paths, go_types=['BP'], overlap_thresh=0.6,
                      p_thr=0.001, logFC_thr=4, target_sig='Up', topN=10,
                      organism='Human',
                      font_family='DejaVu Sans', term_of_interest=None):
    """
    读取DEG文件，精简Term名称，并绘制美观的GO条形图。

    新增功能：
        - term_of_interest: list 或 dict
            * list: 对所有 go_type 通用
            * dict: { 'BP': [...], 'MF': [...], 'CC': [...] }
        - 优先从 term_of_interest 中选择
        - 若不足 topN，则按 Adjusted P-value 顺延补齐
        - 若超过 topN，则只取前 topN
    """
    matplotlib.rcParams['font.family'] = font_family  # 设置字体
    
    # 1. 筛选基因
    df = result.copy()
    if target_sig == 'Up':
        gene_list = df[(df['pvals_adj'] < p_thr) & (df['logfoldchanges'] > logFC_thr)]['names'].tolist()
    else:
        gene_list = df[(df['pvals_adj'] < p_thr) & (df['logfoldchanges'] < -logFC_thr)]['names'].tolist()
    
    if len(gene_list) < 5:
        print(f"Warning: Too few {target_sig} genes for enrichment.")
        return
    
    # 2. 针对每种 GO 类型运行富集
    for go_type in go_types:
        if go_type not in dataset_paths:
            print(f"Warning: {go_type} not found in dataset_paths, skipping.")
            continue
        
        dataset_path = dataset_paths[go_type]
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=dataset_path,
            organism=organism,
            outdir=None
        )
        res = enr.results
        if res.empty:
            print(f"No significant GO terms found for {go_type}.")
            continue
        
        # 3. 文本清洗
        res['Term'] = res['Term'].str.replace(f'^GO{go_type}_', '', regex=True)
        res['Term'] = res['Term'].str.split(' \(GO:').str[0]
        res['Term'] = res['Term'].apply(lambda x: x[:47] + '...' if len(x) > 60 else x)
        res['Term'] = res['Term'].str.lower().str.replace('_', ' ')
        
        # 4. 去冗余
        res_trimmed = trim_redundant_terms(res, overlap_thresh=overlap_thresh)
        
        # 5. 按显著性排序（后面顺延要用）
        res_trimmed = res_trimmed.sort_values('Adjusted P-value').reset_index(drop=True)
        
        # ===== 新增核心逻辑：term_of_interest =====
        if term_of_interest is not None:
            # 允许 list 或 dict
            if isinstance(term_of_interest, dict):
                toi = term_of_interest.get(go_type, [])
            else:
                toi = term_of_interest
            
            toi = [t.lower().replace('_', ' ') for t in toi]
            
            # 5a. 先选 term_of_interest 中存在的
            selected = res_trimmed[res_trimmed['Term'].isin(toi)]
            
            # 5b. 如果不足 topN，按显著性顺延补齐
            if len(selected) < topN:
                remaining = res_trimmed[~res_trimmed.index.isin(selected.index)]
                selected = pd.concat(
                    [selected, remaining.head(topN - len(selected))],
                    axis=0
                )
            
            # 5c. 如果超过 topN，截断
            top_terms = selected.head(topN).copy()
        else:
            # 原始行为：直接取 TopN
            top_terms = res_trimmed.head(topN).copy()
        
        # 6. 计算 -log10(p)
        top_terms['nlog10'] = -np.log10(top_terms['Adjusted P-value'] + 1e-10)
        
        # 7. 保存 CSV（完整去冗余结果）
        csv_file = f"{save_addr}/{filename}_{target_sig}_{go_type}.csv"
        res_trimmed.to_csv(csv_file, index=False)
        print(f"GO term results saved to {csv_file}")
        
        # 8. 绘图
        plt.figure(figsize=(10, 6))
        sns.set_style("white")
        sns.barplot(data=top_terms, x='nlog10', y='Term', palette='magma')
        plt.title(f'Top {topN} {go_type} terms ({target_sig}regulated)', fontsize=14, pad=15)
        plt.xlabel('-log10(Adjusted P-value)', fontsize=12)
        plt.ylabel('')
        sns.despine()
        plt.tight_layout()
        
        # 9. 保存 PNG + PDF
        plt.savefig(f"{save_addr}/{filename}_{target_sig}_{go_type}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_addr}/{filename}_{target_sig}_{go_type}.pdf", bbox_inches='tight')
        plt.close()
        
        print(f"Refined GO ({go_type}) barplot saved for {target_sig} genes.")
