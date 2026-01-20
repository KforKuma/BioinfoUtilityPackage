# ===== Third-party =====
import numpy as np
import pandas as pd

import scanpy as sc
import gseapy as gp

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


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


def plot_volcano(result,save_addr,filename,
                 cluster_genes_dict=None,  # 新增：字典 {cluster_name: [gene1, gene2, ...]}
                 lfc_limit=10,
                 p_thresh=0.05,
                 lfc_thresh=1.0):
    """
    绘制火山图，并可标注指定基因。

    参数:
        result: pd.DataFrame, 包含列 ['cluster', 'logfoldchanges', 'pvals_adj', 'gene']
        cluster_genes_dict: dict, {cluster_name: [gene1, gene2, ...]}，标注这些基因
        lfc_limit: log fold change 最大截断值
        p_thresh: 调整后的 p 值阈值
        lfc_thresh: log fold change 阈值
    """
    
    # 1. 分类 Up/Down/Normal
    result['sig'] = 'Normal'
    result.loc[(result['pvals_adj'] < p_thresh) & (result['logfoldchanges'] > lfc_thresh), 'sig'] = 'Up'
    result.loc[(result['pvals_adj'] < p_thresh) & (result['logfoldchanges'] < -lfc_thresh), 'sig'] = 'Down'
    
    # 2. LFC 截断
    result['plot_lfc'] = result['logfoldchanges'].clip(-lfc_limit, lfc_limit)
    
    # 3. -log10(p)
    result['nlog10'] = -np.log10(result['pvals_adj'] + 1e-300)
    
    # 4. 绘图
    plt.figure(figsize=(7, 6))
    colors = {'Up': '#e41a1c', 'Down': '#377eb8', 'Normal': '#d9d9d9'}
    
    sns.scatterplot(data=result, x='plot_lfc', y='nlog10', hue='sig',
                    palette=colors, s=20, alpha=0.6, edgecolor=None)
    
    # 阈值线
    plt.axvline(x=lfc_thresh, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.axvline(x=-lfc_thresh, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.axhline(y=-np.log10(p_thresh), color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # 5. 标注指定基因
    if cluster_genes_dict is not None:
        for cluster, genes in cluster_genes_dict.items():
            genes_to_plot = result[(result['cluster'] == cluster) & (result['names'].isin(genes))]
            for _, row in genes_to_plot.iterrows():
                ha = 'right' if row['plot_lfc'] < 0 else 'left'
                plt.text(x=row['plot_lfc'], y=row['nlog10'], s=row['names'],
                         fontsize=8, ha=ha, va='bottom', color='black')
    
    # 从 result["cluster"] 自动取组名作为标题
    clusters = result['cluster'].unique()
    if len(clusters) == 2:
        title = f'{clusters[0]} vs {clusters[1]}'
    else:
        title = 'Volcano Plot'
    
    plt.title(title, fontsize=12)
    plt.xlabel('log2(Fold Change) [Clipped]', fontsize=11)
    plt.ylabel('-log10(Adjusted P-value)', fontsize=11)
    plt.xlim(-lfc_limit * 1.05, lfc_limit * 1.05)
    plt.legend(title='Expression', loc='upper right', frameon=False)
    plt.tight_layout()
    
    # 保存 PDF 和 PNG
    plt.savefig(f"{save_addr}/{filename}.png", dpi=300)
    plt.savefig(f"{save_addr}/{filename}.pdf")
    plt.close()
    
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
