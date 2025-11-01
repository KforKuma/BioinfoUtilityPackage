library(dplyr)
library(tibble)
library(Seurat)


ReadIn10X <- function(InputDir, suffix = "outs") {
  # 获取目录下的所有子文件夹（假设每个子文件夹是一个样本名）
  samples <- list.dirs(InputDir, recursive = FALSE, full.names = FALSE)
  if (length(samples) == 0) {
    stop("未在 ", InputDir, " 下找到任何样本子目录。")
  }

  sce_list <- vector("list", length(samples))
  names(sce_list) <- samples

  for (i in seq_along(samples)) {
    sample_name <- samples[i]
    read_dir <- file.path(InputDir, sample_name, suffix)

    if (!dir.exists(read_dir)) {
      warning("路径不存在，跳过：", read_dir)
      next
    }

    message("正在读取：", read_dir)
    sc_data <- Read10X(data.dir = read_dir)

    # 创建 Seurat 对象并赋值
    seurat_obj <- CreateSeuratObject(counts = sc_data, project = sample_name)

    # 拆分样本名为 orig.ident 与 disease
    parts <- strsplit(sample_name, "_")[[1]]
    if (length(parts) >= 2) {
      seurat_obj@meta.data$orig.ident <- parts[1]
      seurat_obj@meta.data$disease    <- parts[2]
    } else {
      seurat_obj@meta.data$orig.ident <- sample_name
      seurat_obj@meta.data$disease    <- NA
      warning("样本名 '", sample_name, "' 未包含下划线，无法拆分 disease 信息。")
    }

    # 存入列表
    sce_list[[i]] <- seurat_obj
  }

  # 去除空白条目（若有跳过的样本）
  sce_list <- Filter(Negate(is.null), sce_list)

  return(sce_list)
}

RunSoupX <- function(input_dir,
                     sample_names,
                     pollution_list = NULL,
                     suffix = "",
                     mode = c("auto", "manual"),
                     max_iter = 10,
                     max_genes = 500
) {
  tic("Total RunSoupX")
  message("[Start] RunSoupX on ", length(sample_names), " samples")
  mode <- match.arg(mode)

  # 限制污染基因列表长度，提高速度
  if (!is.null(pollution_list) && length(pollution_list) > max_genes) {
    pollution_list <- pollution_list[seq_len(max_genes)]
    message("[Info] Limited pollution_list to first ", length(pollution_list), " genes for speed")
  }

  n <- length(sample_names)
  sce_list <- vector("list", length = n)
  names(sce_list) <- sample_names

  for (i in seq_along(sample_names)) {
    sample <- sample_names[i]
    message(sprintf("\n[Start Sample %d/%d] %s", i, n, sample))
    # tic(sprintf("Sample %s total", sample))

    # 构造路径并检查
    read_dir <- file.path(input_dir, sample, suffix)
    message("[Step] Read directory: ", read_dir)
    if (!dir.exists(read_dir)) stop("Directory not found: ", read_dir)

    # 读取原始与过滤矩阵
    tic("Read10X")
    raw_matrix <- Read10X(file.path(read_dir, "raw_feature_bc_matrix"))
    filt_matrix <- Read10X(file.path(read_dir, "filtered_feature_bc_matrix"))
    toc(log = TRUE)

    # 初始化 SoupChannel
    message("[Step] Initialize SoupChannel")
    tic("SoupChannel")
    soup.channel <- SoupChannel(raw_matrix, filt_matrix)
    toc(log = TRUE)

    # 创建 Seurat 对象并执行标准预处理
    message("[Step] Create SeuratObject and preprocess")
    tic("CreateSeuratObject")
    seurat_obj <- CreateSeuratObject(counts = filt_matrix, project = sample)
    toc(log = TRUE)

    tic("NormalizeData")
    seurat_obj <- NormalizeData(seurat_obj)
    toc(log = TRUE)

    tic("FindVariableFeatures")
    seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2000)
    toc(log = TRUE)

    tic("ScaleData")
    seurat_obj <- ScaleData(seurat_obj)
    toc(log = TRUE)

    # 降维与聚类
    tic("RunPCA")
    seurat_obj <- RunPCA(seurat_obj, npcs = 50)
    toc(log = TRUE)

    tic("RunUMAP")
    seurat_obj <- RunUMAP(seurat_obj, dims = 1:20, verbose = FALSE)
    toc(log = TRUE)

    tic("FindNeighbors")
    seurat_obj <- FindNeighbors(seurat_obj, dims = 1:20)
    toc(log = TRUE)

    tic("FindClusters")
    seurat_obj <- FindClusters(seurat_obj, resolution = 1.2)
    toc(log = TRUE)

    # 将聚类和降维结果传回 SoupChannel
    message("[Step] Transfer clustering and DR to SoupChannel")
    meta <- seurat_obj@meta.data
    umap <- Embeddings(seurat_obj, "umap")
    soup.channel <- setClusters(soup.channel, setNames(meta$seurat_clusters, rownames(meta)))
    soup.channel <- setDR(soup.channel, umap)

    # 污染估算
    if (mode == "auto") {
      message("[Step] Auto estimating contamination")
      tic("autoEstCont")
      soup.channel <- autoEstCont(soup.channel, doPlot = FALSE)
      toc(log = TRUE)
    } else {
      message("[Step] Manual contamination estimation")
      tic("estimateNonExpressingCells")
      ute <- estimateNonExpressingCells(
        soup.channel,
        nonExpressedGeneList = pollution_list
      )
      toc(log = TRUE)

      message("[Step] calculateContaminationFraction (max_iter = ", max_iter, ")")
      tic("calculateContaminationFraction")
      soup.channel <- calculateContaminationFraction(
        soup.channel,
        nonExpressedGeneList = pollution_list,
        useToEst = ute,
        forceAccept = TRUE,
        maxIter = max_iter
      )
      toc(log = TRUE)
    }

    # 扣减污染并创建最终对象
    message("[Step] adjustCounts and finalize SeuratObject")
    tic("adjustCounts")
    adjusted <- adjustCounts(soup.channel)
    toc(log = TRUE)

    tic("Final SeuratObject")
    final_obj <- CreateSeuratObject(counts = adjusted, project = sample)
    toc(log = TRUE)

    sce_list[[i]] <- final_obj
    # toc(sprintf("Sample %s total", sample), log = TRUE, quiet = FALSE)
  }

  message("[End] RunSoupX completed")
  toc(log = TRUE)
  return(sce_list)
}

updateSeuratMeta <- function(scList, meta_clean, col.by="orig.ident") {
  #' 批量更新 Seurat 对象列表的 meta.data
  #'
  #' @param scList 一个包含 Seurat 对象的列表
  #' @param meta_clean 已经清理并以 orig.ident 唯一化的元数据 data.frame
  #'                   （需包含 orig.ident 列，行无特殊要求）
  #' @return 返回一个新的 Seurat 对象列表，meta.data 已被更新
  # 校验输入
  if (!is.list(scList) || length(scList) == 0) {
    stop("scList must be a non-empty list of Seurat objects.")
  }
  if (!col.by %in% colnames(meta_clean)) {
    stop(paste0("meta_clean must contain ", col.by, " column."))
  }

  # 遍历列表，更新每个对象
  for (i in seq_along(scList)) {
    obj <- scList[[i]]

    # 检查是否为 Seurat 对象
    if (!inherits(obj, "Seurat")) {
      warning(sprintf("Element %d is not a Seurat object—skipping.", i))
      next
    }

    # 原始 meta.data
    md <- obj@meta.data %>%
      rownames_to_column(var = "cell_id")

    # 左连接 clean 后的 meta
    md2 <- md %>%
      left_join(meta_clean, by = col.by)

    # 恢复行名并赋值回 Seurat 对象
    new_meta <- md2 %>%
      column_to_rownames(var = "cell_id")
    obj@meta.data <- new_meta

    # 存回列表
    scList[[i]] <- obj
  }

  return(scList)
}

Score_organelle <- function(sce_list) {
  n <- length(sce_list)
  message("Start computing organelle scores for ", n, " objects.")

  for (i in seq_along(sce_list)) {
    message("  [", i, "/", n, "] Processing object: ", names(sce_list)[i] %||% i)

    # 线粒体基因百分比
    sce_list[[i]][["percent.mt"]] <- PercentageFeatureSet(
      object = sce_list[[i]],
      pattern = "^MT-"
    )
    message("    percent.mt computed.")

    # 核糖体蛋白百分比
    sce_list[[i]][["percent.ribo"]] <- PercentageFeatureSet(
      object = sce_list[[i]],
      pattern = "^RP[SL][[:digit:]]|^RPLP[[:digit:]]|^RPSA"
    )
    message("    percent.ribo computed.")

    # （可选）血红蛋白基因百分比
    sce_list[[i]][["percent.hb"]] <- PercentageFeatureSet(
      object = sce_list[[i]],
      pattern = "^HB[^(P)]"
    )
    message("    percent.hb computed.")
  }

  message("Finished computing organelle scores.")
  return(sce_list)
}

Compute_QC_metrics <- function(sce_list, method = "Rfast") {
  # 检查 method 参数
  valid_methods <- c("apply", "rowOrderStats", "Rfast")
  if (!method %in% valid_methods) {
    stop("Invalid method. Please choose one of: ", paste(valid_methods, collapse = ", "))
  }

  n <- length(sce_list)
  message("Start computing QC metrics for ", n, " objects (method = '", method, "').")

  for (i in seq_along(sce_list)) {
    obj_name <- names(sce_list)[i] %||% i
    message("  [", i, "/", n, "] Processing '", obj_name, "' ...")

    # 1. log1p_total_counts
    sce_list[[i]]@meta.data$log1p_total_counts <- log1p(
      sce_list[[i]]@meta.data$nCount_RNA
    )
    message("    ✓ log1p_total_counts added.")

    # 2. log1p_n_genes_by_counts
    sce_list[[i]]@meta.data$log1p_n_genes_by_counts <- log1p(
      sce_list[[i]]@meta.data$nFeature_RNA
    )
    message("    ✓ log1p_n_genes_by_counts added.")

    # 3. pct_counts_in_top_20_genes
    message("    → Computing pct_counts_in_top_20_genes with method '", method, "' ...")
    counts_mat <- GetAssayData(
      object = sce_list[[i]],
      assay  = "RNA",
      slot   = "counts"
    )

    if (method == "apply") {
      message("      * Using base apply + sort")
      top20_counts <- apply(counts_mat, 2, function(col) {
        sum(sort(col, decreasing = TRUE)[1:20])
      })
      message("      * apply method done.")

    } else if (method == "rowOrderStats") {
      message("      * Using sparseMatrixStats::colOrderStats")
      if (!requireNamespace("sparseMatrixStats", quietly = TRUE)) {
        stop("Please install the Bioconductor package 'sparseMatrixStats' first.")
      }
      kth_vals <- sparseMatrixStats::colOrderStats(
        counts_mat,
        which = nrow(counts_mat) - 19
      )
      top20_counts <- vapply(
        seq_len(ncol(counts_mat)),
        function(j) {
          col   <- counts_mat[, j]
          pivot <- kth_vals[j]
          sum(col[col >= pivot])
        },
        numeric(1)
      )
      message("      * rowOrderStats method done.")

    } else if (method == "Rfast") {
      message("      * Using Rfast::colSort on dense matrix")
      dense_mat  <- as.matrix(counts_mat)
      sorted_mat <- Rfast::colSort(dense_mat, descending = TRUE)
      top20_mat  <- sorted_mat[1:20, , drop = FALSE]
      top20_counts <- colSums(top20_mat)
      message("      * Rfast method done.")
    }

    # 保存结果
    sce_list[[i]]@meta.data$pct_counts_in_top_20_genes <- 100 *
      top20_counts / sce_list[[i]]@meta.data$nCount_RNA
    message("    ✓ pct_counts_in_top_20_genes added.")
  }

  message("Finished computing QC metrics for all objects.")
  return(sce_list)
}

Is_Outlier <- function(sce_list, metric, nmads = 3) {
  # sce_list: 一个 Seurat 对象列表
  # metric   : meta.data 中的列名（字符串）
  # nmads    : 用于定义离群阈值的 MAD 倍数，默认 3

  n <- length(sce_list)
  message("Start outlier detection on ", n, " objects.")

  for (i in seq_along(sce_list)) {
    obj_name <- names(sce_list)[i] %||% as.character(i)
    message(sprintf(" [%d/%d] Processing object '%s' …", i, n, obj_name))

    # 1. 提取指标向量
    M <- sce_list[[i]]@meta.data[[metric]]  # 使用 [[ ]] 取向量 :contentReference[oaicite:4]{index=4}

    # 2. 计算中位数和 MAD（constant=1 以匹配 Python 的 median_abs_deviation）
    med  <- median(M, na.rm = TRUE)                 # medians via median() :contentReference[oaicite:5]{index=5}
    madv <- mad(M, center = med, constant = 1, na.rm = TRUE)  # mad constant param :contentReference[oaicite:6]{index=6}

    # 3. 判断离群值
    outlier <- (M < med - nmads * madv) | (M > med + nmads * madv)

    # 4. 写回 metadata
    new_col <- paste0(metric, "_outlier")
    sce_list[[i]]@meta.data[[new_col]] <- outlier

    message(sprintf("   -> Added '%s' (%d TRUEs)",
                    new_col, sum(outlier, na.rm = TRUE)))
  }

  message("Finished outlier detection for all objects.")
  return(sce_list)  # 返回更新后的列表
}

Remove_Outlier <- function(sce_list, nmads = 5, mt_nmads = 3, mt_thresh = 8) {
  #' Remove_Outlier: 对每个 Seurat 对象计算多种离群指标，并剔除任一离群细胞
  #' @param sce_list   Seurat 对象列表
  #' @param nmads      普通指标使用的 MAD 倍数（默认 5）
  #' @param mt_nmads   percent.mt 使用的 MAD 倍数（默认 3）
  #' @param mt_thresh  percent.mt 的硬阈值（默认 8%）
  #' @return           去掉任一离群细胞后的 Seurat 对象列表

  n <- length(sce_list)
  message("Starting Remove_Outlier on ", n, " objects.")

  # 1. 计算离群：log1p_total_counts、log1p_n_genes_by_counts、pct_counts_in_top_20_genes
  sce_list <- Is_Outlier(sce_list, "log1p_total_counts",      nmads)
  sce_list <- Is_Outlier(sce_list, "log1p_n_genes_by_counts", nmads)
  sce_list <- Is_Outlier(sce_list, "pct_counts_in_top_20_genes", nmads)
  # 额外针对 percent.mt
  sce_list <- Is_Outlier(sce_list, "percent.mt", mt_nmads)

  for (i in seq_along(sce_list)) {
    obj_name <- names(sce_list)[i] %||% as.character(i)
    message(sprintf(" [%d/%d] Processing '%s' …", i, n, obj_name))

    # 2. 合并所有离群标记：四列任一为 TRUE 则 any_outlier 为 TRUE
    outlier_cols <- c(
            "log1p_n_genes_by_counts_outlier",
            "pct_counts_in_top_20_genes_outlier",
            "log1p_total_counts_outlier",
            "percent.mt_outlier"
    )
    sce_list[[i]]@meta.data$any_outlier <- rowSums(sce_list[[i]]@meta.data[, outlier_cols], na.rm = TRUE) > 0
    message(sprintf("    any_outlier: %d cells flagged", sum(sce_list[[i]]@meta.data$any_outlier, na.rm = TRUE)))

    # 3. （可选）基于硬阈值再标记 percent.mt 离群
    sce_list[[i]]@meta.data$mt_hard_outlier <- sce_list[[i]]@meta.data$percent.mt > mt_thresh
    message(sprintf("    mt_hard_outlier (>%.1f%%): %d cells flagged",
                    mt_thresh, sum(sce_list[[i]]@meta.data$mt_hard_outlier, na.rm = TRUE)))

    # 4. 子集：去除任一离群或硬阈值线粒体离群
    message("Available meta columns: ",
           paste(colnames(sce_list[[i]]@meta.data), collapse = ", "))
    keep.cells <- names(which(! sce_list[[i]]$any_outlier|sce_list[[i]]$mt_hard_outlier))

    sce_list[[i]] <- subset(
      sce_list[[i]],
      cells = keep.cells
    )
    message(sprintf("    After removal: %d cells remain", ncol(sce_list[[i]])))
  }

  message("Finished Remove_Outlier.")
  return(sce_list)
}

RunDimReduc <- function(sce,dims=20,resolution=1.0){
  sce <- NormalizeData(sce)
  sce <- FindVariableFeatures(sce, selection.method = "vst", nfeatures = 2000)
  sce <- ScaleData(sce)
  sce <- RunPCA(sce)
  sce <- FindNeighbors(object = sce, dims = 1:dims)
  sce <- FindClusters(object = sce, resolution = resolution)
  sce <- RunUMAP(sce, dims = 1:dims)
  return(sce)
}

RunDoubletFinder <- function(
  sce_list,
  pN       = 0.25,
  dims     = 10,
  expected_doublet_rate = 0.075,
  speed_up = TRUE,
  homotypic_adjust     = TRUE
) {
  #' RunDoubletFinder: 对一系列 Seurat 对象批量执行 DoubletFinder
  #'
  #' @param sce_list              一个包含 Seurat 对象的列表
  #' @param pN                    DoubletFinder 中的 pN 参数 (默认 0.25)
  #' @param dims                  用于构建 pANN 的 PCA 维度数量 (默认 20)
  #' @param expected_doublet_rate 预期双倍体比例 (默认 0.075)，我们接受这个默认值因为 10x genomics 官方给出的 v2 chemistry 在恢复细胞量约为一万时的二倍体率约为 7.6%
  #' @param homotypic_adjust      是否根据同源簇比例调整 nExp (默认 TRUE)
  #'
  #' @return 返回已添加双倍体标记并过滤后的 Seurat 对象列表

  n <- length(sce_list)
  message("Starting RunDoubletFinder on ", n, " objects.")

  for (i in seq_along(sce_list)) {
    obj <- sce_list[[i]]
    # 清除之前可能做过DF留下的列
    obj@meta.data <- obj@meta.data[,-grep("^DF.classifications_", colnames(obj@meta.data))]


    name_i <- names(sce_list)[i] %||% paste0("Sample", i)
    message(sprintf("---- [%d/%d] %s ----", i, n, name_i))

    # 1. 类型检查
    # if (!inherits(obj, "Seurat")) {
    #   warning("Element ", i, " is not a Seurat object; skipping.")
    #   next
    # }
    # if (is.null(obj@meta.data$seurat_clusters)) {
    #   warning("No 'seurat_clusters' in meta.data of ", name_i, "; skipping.")
    #   next
    # }

    # 2 pK 参数搜寻 (no ground-truth)
    message("  * Running paramSweep without ground-truth …")
    if(speed_up){
      if(!exists("hvgs")){ # 对于一个列表来说只有第一次需要计算一下
        FindVariableFeatures(obj,selection.method = "vst", nfeatures = 3000)
        hvgs <- VariableFeatures(obj)
      }
      obj_sub <- subset(x = obj, downsample = 500)
      obj_sub <- subset(x = obj_sub, features = hvgs)
    }

    sweep.res.list <- DoubletFinder::paramSweep(obj_sub, PCs = 1:dims, sct = FALSE, num.cores = 4) # 非常非常耗时
    sweep.stats    <- DoubletFinder::summarizeSweep(sweep.res.list, GT = FALSE)
    bcmvn          <- DoubletFinder::find.pK(sweep.stats) # 均值方差归一化双峰系数 (BCmvn)；不知为何print一个NULL
    idx    <- which.max(bcmvn$BCmetric)
    opt_pK <- as.numeric(as.character(bcmvn$pK[idx]))
    message("    -> Optimal pK = ", opt_pK)
    rm(obj_sub)

    # 3. 计算期望双倍体数
    nCells <- ncol(obj)
    nExp   <- round(expected_doublet_rate * nCells)
    if (homotypic_adjust) {
      hom.prop <- modelHomotypic(obj@meta.data$seurat_clusters)
      nExp.adj <- round(nExp * (1 - hom.prop))
      message("    -> Homotypic-adjusted nExp = ", nExp.adj)
    } else {
      nExp.adj = nExp
      message("    -> Using raw nExp = ", nExp.adj)
    }

    # 4. 首次调用 doubletFinder 生成 pANN
    message("  * Running doubletFinder (initial pANN) …")
    obj <- DoubletFinder::doubletFinder(
      obj,
      PCs     = 1:dims,
      pN       = pN,
      pK       = opt_pK,
      nExp     = nExp,
      reuse.pANN = NULL,
      sct = FALSE
    )
    # 找到新产生的 pANN 列名
    pann_col1 <- tail(grep("^pANN_", colnames(obj@meta.data)), 1)
    pann_name1 <- colnames(obj@meta.data)[pann_col1]
    message("    -> Produced pANN column: ", pann_name1)

    # 5. 再次调用以生成二次分类并标记
    message("  * Running doubletFinder (using existing pANN) …")
    obj <- DoubletFinder::doubletFinder(
      obj,
      PCs      = 1:dims,
      pN        = pN,
      pK        = opt_pK,
      nExp      = nExp.adj,
      reuse.pANN = pann_name1,
      sct = FALSE
    )
    pann_col2  <- tail(grep("^DF.classifications_", colnames(obj@meta.data)), 1)
    df_name2   <- colnames(obj@meta.data)[pann_col2]
    message("    -> Produced classification column: ", df_name2)

    # 6. 合并分类并标记高/低置信度双倍体
    #    新列: DF_hi_lo = Doublet_lo / Doublet_hi / Singlet
    obj@meta.data$DF_hi_lo <- with(
      obj@meta.data,
      ifelse(
        get(df_name2) == "Doublet" & get(df_name2) != get(df_name2),
        "Doublet_lo",   # 逻辑保留示意，可按需求调整
        as.character(get(df_name2))
      )
    )
    message("    -> Created DF_hi_lo column.")

    # 7. 过滤双倍体（如果需要）
    # obj_filtered <- subset(obj, subset = DF.classifications == "Singlet")
    # message("    -> Filtered to singlets: ", ncol(obj_filtered), " cells.")
    # obj <- obj_filtered

    sce_list[[i]]@meta.data["DF_hi_lo"] <- obj@meta.data$DF_hi_lo
    message("---- Finished ", name_i, " ----\n")
  }

  message("RunDoubletFinder completed.")
  return(sce_list)
}


Sce_Cleanse <- function(sce_list, remove_doublet = TRUE, keep_cols = NULL, combine = TRUE) {
  message("▶ 开始运行 Sce_Cleanse()")

  # 默认要保留的 meta.data 列
  if (is.null(keep_cols)) {
    keep_cols <- c(
      "orig.ident", "disease", "orig.project", "Patient",
      "tissue-type", "presorted",
      "percent.mt", "percent.ribo", "percent.hb"
    )
    message("  • 使用默认的 keep_cols: ", paste(keep_cols, collapse = ", "))
  } else {
    message("  • 用户指定的 keep_cols: ", paste(keep_cols, collapse = ", "))
  }
  orig_names <- names(sce_list)
  if(is.null(orig_names)){
    names(sce_list) <- paste0("Sample", seq_along(sce_list))
  }
  # 1. 可选地剔除双倍体
  if (remove_doublet) {
    message("  ▶ 正在剔除双倍体 (DF_hi_lo != 'Singlet') …")
    sce_list <- lapply(names(sce_list), function(nm) {
      obj <- sce_list[[nm]]
      n_before <- ncol(obj)
      obj2 <- subset(obj, subset = DF_hi_lo == "Singlet")
      n_after <- ncol(obj2)
      message(sprintf("    - 样本 %s: 细胞数 %d → %d", nm, n_before, n_after))
      obj2
    })
    names(sce_list) <- orig_names  # 保留名字
    message("  ✔ 双倍体剔除完成")
  } else {
    message("  • skip remove_doublet 步骤")
  }

  # 2. 保留指定的 meta.data 列
  message("  ▶ 开始保留 meta.data 指定列 …")
  sce_list <- lapply(names(sce_list), function(nm) {
    obj <- sce_list[[nm]]
    md <- obj@meta.data
    missing_cols <- setdiff(keep_cols, colnames(md))
    if (length(missing_cols) > 0) {
      warning("样本 ", nm, " 缺失列: ", paste(missing_cols, collapse = ", "))
      message("    ! 警告：样本 ", nm, " 缺失列 ", paste(missing_cols, collapse = ", "))
    }
    keep_present <- intersect(keep_cols, colnames(md))
    obj@meta.data <- md[, keep_present, drop = FALSE]
    message("    - 样本 ", nm, " 保留列: ", paste(keep_present, collapse = ", "))
    obj
  })
  names(sce_list) <- orig_names
  message("  ✔ meta.data 列筛选完成")

  # 3. 合并或返回列表
  if (combine) {
    message("  ▶ 开始合并所有样本 …")
    sce_combined <- merge(
      x             = sce_list[[1]],
      y             = sce_list[-1],
      project       = "CombinedProject",
      add.cell.ids  = names(sce_list)
    )
    message("  ✔ 合并完成，共计样本数: ", length(sce_list), "，细胞数: ", ncol(sce_combined))
    message("▶ Sce_Cleanse() 运行结束，返回合并后的 SingleCellExperiment 对象")
    return(sce_combined)
  } else {
    message("▶ Sce_Cleanse() 运行结束，返回处理后的列表（不合并）")
    return(sce_list)
  }
}

check_seurat_assay_slots_simple <- function(seu, assay = NULL) {
  stopifnot(inherits(seu, "Seurat"))

  # 1. 确定 Assay
  if (is.null(assay)) assay <- DefaultAssay(seu)
  if (! assay %in% Assays(seu)) {
    stop(sprintf("Assay '%s' 不存在，可用 assays: %s",
                 assay, paste(Assays(seu), collapse = ", ")))
  }

  # 2. 获取 slot 列表
  slot_names <- slotNames(seu@assays[[assay]])
  message("检测到 Assay '", assay, "' 下的 slots: ",
          paste(slot_names, collapse = ", "))

  # 3. 逐 slot 检查
  results <- lapply(slot_names, function(sl) {
    mat <- tryCatch(
      GetAssayData(seu, assay = assay, slot = sl),
      error = function(e) NULL
    )
    if (is.null(mat)) {
      return(data.frame(slot = sl, available = FALSE, is_raw_counts = NA))
    }
    # 仅提取非零部分：对稀疏矩阵读 x；对普通矩阵退回到 as.numeric（少数情况）
    if (inherits(mat, "sparseMatrix")) {
      vals <- mat@x
    } else {
      vals <- as.numeric(mat)
    }
    # 判断所有条目是否非负整数（隐式零也视作整数）
    ok <- length(vals) > 0 && all(vals >= 0 & vals == floor(vals))
    data.frame(slot = sl, available = TRUE, is_raw_counts = ok)
  })

  # 4. 合并并打印结果
  df <- do.call(rbind, results)
  print(df, row.names = FALSE)

  # 5. 提示可能的 raw count 层
  candidates <- df$slot[which(df$is_raw_counts)]
  if (length(candidates) > 0) {
    message("✅ 可能存有原始整数 counts 的 slot: ",
            paste(candidates, collapse = ", "))
  } else {
    message("⚠️ 未发现明显的原始整数 counts slot，请手动确认。")
  }

  invisible(df)
}
