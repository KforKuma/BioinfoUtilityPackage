args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: run_da_method.R <propeller|dcats|sccomp> <work_dir>")
}

# CmdStan/sccomp emit subprocess text that is parsed by R.  Force an installed
# UTF-8 English locale so Windows does not mix UTF-8 data labels with CP936
# toolchain messages.
suppressWarnings(Sys.setlocale("LC_ALL", "English_United States.utf8"))

method <- args[[1L]]
work_dir <- normalizePath(args[[2L]], winslash = "/", mustWork = TRUE)

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("there is no package called 'jsonlite'")
}

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L || (length(x) == 1L && is.na(x))) y else x
}

read_utf8_csv <- function(name) {
  read.csv(
    file.path(work_dir, name),
    stringsAsFactors = FALSE,
    check.names = FALSE,
    fileEncoding = "UTF-8"
  )
}

abundance <- read_utf8_csv("abundance.csv")
sample_manifest <- read_utf8_csv("sample_manifest.csv")
cell_type_manifest <- read_utf8_csv("cell_type_manifest.csv")
spec <- jsonlite::fromJSON(file.path(work_dir, "run_spec.json"), simplifyVector = TRUE)
contrast <- spec$contrast
options <- spec$options

factor_name <- as.character(contrast$factor)
group_1 <- as.character(contrast$group_1)
group_2 <- as.character(contrast$group_2)
sample_order <- as.character(sample_manifest$sample_id)
cell_order <- as.character(cell_type_manifest$cell_type)

if (!factor_name %in% colnames(sample_manifest)) {
  stop("Requested factor is missing from sample_manifest")
}
if (!setequal(unique(as.character(sample_manifest[[factor_name]])), c(group_1, group_2))) {
  stop("The R backend expects the already-filtered pairwise sample manifest")
}

covariates <- options$covariates %||% character()
covariates <- as.character(covariates)
missing_covariates <- setdiff(covariates, colnames(sample_manifest))
if (length(missing_covariates)) {
  stop(paste("Missing requested covariates:", paste(missing_covariates, collapse = ", ")))
}

count_table <- xtabs(count ~ sample_id + cell_type, data = abundance)
count_mat <- as.matrix(count_table[sample_order, cell_order, drop = FALSE])
storage.mode(count_mat) <- "numeric"

observed_effect <- function() {
  props <- count_mat / rowSums(count_mat)
  group_values <- as.character(sample_manifest[[factor_name]])
  colMeans(props[group_values == group_1, , drop = FALSE]) -
    colMeans(props[group_values == group_2, , drop = FALSE])
}

find_column <- function(x, candidates) {
  normalized <- gsub("[^a-z0-9]+", "", tolower(colnames(x)))
  for (candidate in candidates) {
    hit <- which(normalized == gsub("[^a-z0-9]+", "", tolower(candidate)))
    if (length(hit)) return(colnames(x)[hit[[1L]]])
  }
  NULL
}

run_propeller <- function() {
  if (!requireNamespace("speckle", quietly = TRUE)) {
    stop("there is no package called 'speckle'")
  }
  transform <- as.character(options$transform %||% "logit")
  prop_list <- speckle::convertDataToList(
    t(count_mat), data.type = "counts", transform = transform
  )
  design_data <- data.frame(
    tested_group = factor(
      as.character(sample_manifest[[factor_name]]),
      levels = c(group_1, group_2)
    ),
    check.names = FALSE
  )
  for (covariate in covariates) design_data[[covariate]] <- sample_manifest[[covariate]]
  rhs <- c("0 + tested_group", sprintf("`%s`", covariates))
  design <- model.matrix(as.formula(paste("~", paste(rhs, collapse = " + "))), data = design_data)
  group_columns <- grep("^tested_group", colnames(design))
  if (length(group_columns) != 2L) stop("Could not identify two Propeller group design columns")
  contrast_vector <- rep(0, ncol(design))
  contrast_vector[group_columns[[1L]]] <- 1
  contrast_vector[group_columns[[2L]]] <- -1
  result <- speckle::propeller.ttest(
    prop.list = prop_list,
    design = design,
    contrasts = contrast_vector,
    robust = isTRUE(options$robust),
    trend = isTRUE(options$trend),
    sort = FALSE
  )
  result <- as.data.frame(result, check.names = FALSE)
  p_column <- find_column(result, c("P.Value", "PValue", "pvalue"))
  fdr_column <- find_column(result, c("FDR", "adj.P.Val", "adjusted.pvalue"))
  statistic_column <- find_column(result, c("t", "statistic"))
  if (is.null(p_column) || is.null(fdr_column)) {
    stop(paste("Unrecognized Propeller result columns:", paste(colnames(result), collapse = ", ")))
  }
  transformed <- prop_list$TransformedProps[cell_order, sample_order, drop = FALSE]
  group_values <- as.character(sample_manifest[[factor_name]])
  estimate <- rowMeans(transformed[, group_values == group_1, drop = FALSE]) -
    rowMeans(transformed[, group_values == group_2, drop = FALSE])
  data.frame(
    cell_type = cell_order,
    estimate = as.numeric(estimate[cell_order]),
    pvalue_raw = as.numeric(result[cell_order, p_column]),
    pvalue_adjusted = as.numeric(result[cell_order, fdr_column]),
    statistic = if (is.null(statistic_column)) NA_real_ else as.numeric(result[cell_order, statistic_column]),
    transformation = transform,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

run_dcats <- function() {
  if (!requireNamespace("DCATS", quietly = TRUE)) {
    stop("there is no package called 'DCATS'")
  }
  design <- sample_manifest[, covariates, drop = FALSE]
  design[[factor_name]] <- factor(
    as.character(sample_manifest[[factor_name]]), levels = c(group_2, group_1)
  )
  base_model <- as.character(options$base_model %||% if (length(covariates)) "FULL" else "NULL")
  pseudo_count <- options$pseudo_count %||% NULL
  fix_phi <- options$fix_phi %||% NULL
  reference <- options$reference_cell_types %||% NULL
  native <- DCATS::dcats_GLM(
    count_mat = count_mat,
    design_mat = design,
    similarity_mat = NULL,
    pseudo_count = pseudo_count,
    base_model = base_model,
    fix_phi = fix_phi,
    reference = reference
  )
  required_native <- c("ceoffs", "coeffs_err", "LR_vals", "LRT_pvals", "fdr")
  if (!all(required_native %in% names(native))) {
    stop(paste("Unrecognized DCATS result members:", paste(names(native), collapse = ", ")))
  }
  if (!factor_name %in% colnames(native$LRT_pvals)) {
    stop("DCATS result does not contain the requested tested-factor column")
  }
  p_values <- as.numeric(native$LRT_pvals[cell_order, factor_name])
  data.frame(
    cell_type = cell_order,
    estimate = as.numeric(native$ceoffs[cell_order, factor_name]),
    standard_error = as.numeric(native$coeffs_err[cell_order, factor_name]),
    statistic = as.numeric(native$LR_vals[cell_order, factor_name]),
    pvalue_raw = p_values,
    pvalue_adjusted = p.adjust(p_values, method = "BH"),
    native_fdr = as.numeric(native$fdr[cell_order, factor_name]),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

run_sccomp <- function() {
  if (!requireNamespace("sccomp", quietly = TRUE)) {
    stop("there is no package called 'sccomp'")
  }
  cmdstan_path <- Sys.getenv("CMDSTAN", unset = "")
  if (nzchar(cmdstan_path)) {
    if (!requireNamespace("cmdstanr", quietly = TRUE)) {
      stop("CMDSTAN is configured but there is no package called 'cmdstanr'")
    }
    # cmdstanr 0.9.0 does not expose the later `quiet` argument.
    cmdstanr::set_cmdstan_path(cmdstan_path)
  }
  if (!requireNamespace("instantiate", quietly = TRUE) || !instantiate::stan_cmdstan_exists()) {
    stop("CmdStan is unavailable for sccomp; configure a valid CMDSTAN path")
  }
  model_data <- merge(
    abundance[, c("sample_id", "cell_type", "count")],
    sample_manifest,
    by = "sample_id",
    all.x = TRUE,
    sort = FALSE
  )
  model_data$tested_group <- factor(
    as.character(model_data[[factor_name]]), levels = c(group_2, group_1)
  )
  composition_terms <- c("0 + tested_group", sprintf("`%s`", covariates))
  variability_terms <- if (isTRUE(options$include_variability)) "0 + tested_group" else "1"
  formula_composition <- as.formula(paste("~", paste(composition_terms, collapse = " + ")))
  formula_variability <- as.formula(paste("~", variability_terms))
  group_design <- model.matrix(~ 0 + tested_group, data = unique(model_data[, c("sample_id", "tested_group")]))
  group_coefficients <- colnames(group_design)
  if (length(group_coefficients) != 2L) stop("Could not identify two sccomp group coefficients")
  contrast_expression <- paste0("`", group_coefficients[[2L]], "` - `", group_coefficients[[1L]], "`")
  model_cache_dir <- as.character(options$model_cache_dir %||% file.path(work_dir, "sccomp_model_cache"))
  dir.create(model_cache_dir, recursive = TRUE, showWarnings = FALSE)
  fit <- sccomp::sccomp_estimate(
    model_data,
    formula_composition = formula_composition,
    formula_variability = formula_variability,
    sample = "sample_id",
    cell_group = "cell_type",
    abundance = "count",
    cores = as.integer(options$cores %||% 1L),
    cache_stan_model = model_cache_dir,
    output_directory = file.path(work_dir, "sccomp_draws_files"),
    inference_method = as.character(options$inference_method %||% "hmc"),
    max_sampling_iterations = as.integer(options$max_sampling_iterations %||% 1000L),
    adapt_delta = as.numeric(options$adapt_delta %||% 0.95)
  )
  stan_fit <- attr(fit, "fit")
  model_rhat_max <- NA_real_
  model_ess_bulk_min <- NA_real_
  model_ess_tail_min <- NA_real_
  model_divergences <- NA_real_
  if (!is.null(stan_fit)) {
    stan_summary <- as.data.frame(stan_fit$summary())
    finite_rhat <- stan_summary$rhat[is.finite(stan_summary$rhat)]
    finite_ess_bulk <- stan_summary$ess_bulk[is.finite(stan_summary$ess_bulk)]
    finite_ess_tail <- stan_summary$ess_tail[is.finite(stan_summary$ess_tail)]
    if (length(finite_rhat)) model_rhat_max <- max(finite_rhat)
    if (length(finite_ess_bulk)) model_ess_bulk_min <- min(finite_ess_bulk)
    if (length(finite_ess_tail)) model_ess_tail_min <- min(finite_ess_tail)
    sampler_diagnostics <- stan_fit$sampler_diagnostics(format = "matrix")
    if ("divergent__" %in% dimnames(sampler_diagnostics)[[2L]]) {
      model_divergences <- sum(sampler_diagnostics[, "divergent__"])
    }
  }
  result <- sccomp::sccomp_test(
    fit,
    contrasts = contrast_expression,
    percent_false_positive = as.numeric(options$percent_false_positive %||% 5),
    test_composition_above_logit_fold_change = as.numeric(options$effect_threshold %||% 0.1),
    pass_fit = FALSE
  )
  result <- as.data.frame(result, check.names = FALSE)
  result$model_rhat_max <- model_rhat_max
  result$model_ess_bulk_min <- model_ess_bulk_min
  result$model_ess_tail_min <- model_ess_tail_min
  result$model_divergences <- model_divergences
  if ("cell_group" %in% colnames(result)) {
    result$cell_type <- as.character(result$cell_group)
  } else if ("cell_type" %in% colnames(result)) {
    result$cell_type <- as.character(result$cell_type)
  } else {
    stop(paste("sccomp output lacks a cell-type column; columns:", paste(colnames(result), collapse = ", ")))
  }
  if (!"c_n_eff" %in% colnames(result) && all(c("c_ess_bulk", "c_ess_tail") %in% colnames(result))) {
    result$c_n_eff <- pmin(result$c_ess_bulk, result$c_ess_tail)
  }
  if (!"c_R_k_hat" %in% colnames(result) && "c_rhat" %in% colnames(result)) {
    result$c_R_k_hat <- result$c_rhat
  }
  if (!"v_n_eff" %in% colnames(result) && all(c("v_ess_bulk", "v_ess_tail") %in% colnames(result))) {
    result$v_n_eff <- pmin(result$v_ess_bulk, result$v_ess_tail)
  }
  if (!"v_R_k_hat" %in% colnames(result) && "v_rhat" %in% colnames(result)) {
    result$v_R_k_hat <- result$v_rhat
  }
  keep <- c(
    "cell_type", "parameter", "factor",
    "c_lower", "c_effect", "c_upper", "c_pH0", "c_FDR", "c_rhat", "c_ess_bulk", "c_ess_tail", "c_n_eff", "c_R_k_hat",
    "v_lower", "v_effect", "v_upper", "v_pH0", "v_FDR", "v_rhat", "v_ess_bulk", "v_ess_tail", "v_n_eff", "v_R_k_hat",
    "model_rhat_max", "model_ess_bulk_min", "model_ess_tail_min", "model_divergences"
  )
  keep <- intersect(keep, colnames(result))
  result[, keep, drop = FALSE]
}

started_at <- format(Sys.time(), tz = "UTC", usetz = TRUE)
output <- switch(
  method,
  propeller = run_propeller(),
  dcats = run_dcats(),
  sccomp = run_sccomp(),
  stop(paste("Unsupported R backend method:", method))
)

write.csv(
  output,
  file.path(work_dir, "native_output.csv"),
  row.names = FALSE,
  na = "",
  fileEncoding = "UTF-8"
)
package_name <- switch(method, propeller = "speckle", dcats = "DCATS", sccomp = "sccomp")
converged <- TRUE
diagnostic_rule <- "native_execution_completed"
min_effective_sample_size <- NA_real_
max_r_k_hat_deviation <- NA_real_
if (method == "sccomp") {
  required_diagnostics <- c("model_rhat_max", "model_ess_bulk_min", "model_ess_tail_min", "model_divergences")
  model_diagnostics_available <- all(required_diagnostics %in% colnames(output))
  if (model_diagnostics_available) {
    model_values <- suppressWarnings(as.numeric(output[1L, required_diagnostics]))
    names(model_values) <- required_diagnostics
    min_effective_sample_size <- min(model_values[c("model_ess_bulk_min", "model_ess_tail_min")])
    max_r_k_hat_deviation <- abs(model_values[["model_rhat_max"]] - 1)
    converged <- all(is.finite(model_values)) && model_values[["model_rhat_max"]] <= 1.05 &&
      min_effective_sample_size >= 50 && model_values[["model_divergences"]] == 0
    diagnostic_rule <- "model_rhat_max<=1.05; model bulk/tail ESS>=50; divergences=0"
  } else {
    converged <- FALSE
    diagnostic_rule <- "required sccomp model diagnostics unavailable"
  }
}
diagnostics <- list(
  method = method,
  package = package_name,
  package_version = as.character(utils::packageVersion(package_name)),
  R_version = R.version.string,
  started_at = started_at,
  finished_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
  n_samples = nrow(count_mat),
  n_cell_types = ncol(count_mat),
  factor = factor_name,
  group_1 = group_1,
  group_2 = group_2,
    covariates = covariates,
    cmdstan_path = if (method == "sccomp") Sys.getenv("CMDSTAN", unset = NA_character_) else NA_character_,
    converged = converged,
    diagnostic_rule = diagnostic_rule,
    min_effective_sample_size = min_effective_sample_size,
    max_r_k_hat_deviation = max_r_k_hat_deviation,
    session_info = paste(capture.output(sessionInfo()), collapse = "\n")
  )
jsonlite::write_json(
  diagnostics,
  file.path(work_dir, "diagnostics.json"),
  auto_unbox = TRUE,
  pretty = TRUE,
  na = "null"
)
