for b in range(n_bootstrap):
    # 2. 对每个 donor 做随机采样（可选：保持类别比例）
    sampled_preds = []
    for y in y_pred_list:
        idx = np.random.choice(len(y), size=len(y), replace=True)
        sampled_preds.append(y[idx])
    
    # 3. 构建 donor x donor 相似性矩阵（可用类别占比或 co-clustering）
    co_matrix = compute_donor_similarity(sampled_preds, n_classes=n_classes)
    dist_matrix = 1 - co_matrix
    
    # 4. 构建 linkage 矩阵
    Z = linkage(dist_matrix, method=method)
    
    # 5. 遍历 linkage，记录每个内部节点包含的 donor 集合
    # 注意：这里只是伪逻辑，实际需要解析 Z 矩阵
    for node in internal_nodes(Z):
        donors_in_node = get_donors(node, donor_labels,Z)
        branch_counter[frozenset(donors_in_node)] += 1