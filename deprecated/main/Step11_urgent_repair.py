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
