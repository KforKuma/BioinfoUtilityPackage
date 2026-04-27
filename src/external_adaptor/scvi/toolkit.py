"""scVI 数据预处理与模型训练辅助函数。"""

import logging
import os
import time
from typing import Optional, Sequence

import numpy as np
import scanpy as sc
import scvi

from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


def _print_stage(message: str) -> None:
    """统一输出阶段信息。"""
    print(f"[process_adata] {message}")


@logged
def process_adata(
    adata,
    prefix,
    save_addr,
    batch_key="orig.ident",
    continuous_covariate_keys=("percent.mt", "percent.ribo"),
    max_epochs=360,
    batch_size=128,
    train_size=0.6,
    validation_size=0.1,
    early_stopping=True,
    early_stopping_monitor="elbo_validation",
    early_stopping_patience=20,
    check_val_every_n_epoch=5,
    target_sum=1e4,
    span=0.3,
    n_top_genes=1000,
):
    """执行标准的 scVI 预处理、训练和结果导出流程。

    Args:
        adata: 输入 AnnData 对象。
        prefix: 本次任务的名称前缀，用于保存模型与结果文件。
        save_addr: 输出目录。
        batch_key: `adata.obs` 中的批次列名。
        continuous_covariate_keys: 连续协变量列名列表。
        max_epochs: 最大训练轮数。
        batch_size: 训练批大小。
        train_size: 训练集比例。
        validation_size: 验证集比例。
        early_stopping: 是否启用早停。
        early_stopping_monitor: 早停监控指标。
        early_stopping_patience: 早停耐心轮数。
        check_val_every_n_epoch: 验证频率。
        target_sum: `normalize_total` 目标总 counts。
        span: `seurat_v3` HVG 选择的 `span` 参数。
        n_top_genes: 高变基因数量。

    Returns:
        处理并写回 scVI 结果后的 AnnData 对象。

    Example:
        adata_scvi = process_adata(
            adata=adata,
            prefix="SampleA",
            save_addr=save_addr,
            batch_key="orig.ident",
            n_top_genes=2000,
            max_epochs=200,
        )
    """
    if not isinstance(prefix, str) or prefix.strip() == "":
        raise ValueError("Argument `prefix` must be a non-empty string.")
    if not isinstance(save_addr, str) or save_addr.strip() == "":
        raise ValueError("Argument `save_addr` must be a non-empty string.")
    if batch_key not in adata.obs.columns:
        raise KeyError(
            f"Column `{batch_key}` was not found in `adata.obs`. "
            f"Available columns are: {list(adata.obs.columns)}."
        )

    missing_covariates = [key for key in continuous_covariate_keys if key not in adata.obs.columns]
    covariates = list(continuous_covariate_keys)
    if missing_covariates:
        logger.info(
            f"[process_adata] Warning! Continuous covariates {missing_covariates} were not found in `adata.obs` "
            "and will be ignored."
        )
        covariates = [key for key in covariates if key not in missing_covariates]

    save_addr = save_addr.strip()
    os.makedirs(save_addr, exist_ok=True)
    start_time = time.time()
    prefix = prefix.strip()

    _print_stage(f"Starting scVI workflow for prefix: '{prefix}'.")
    _print_stage(f"Input size: {adata.n_obs} cells x {adata.n_vars} genes.")

    if hasattr(adata.X, "data"):
        nz = adata.X.data
        all_integer = np.allclose(nz, np.round(nz))
        if all_integer:
            logger.info("[process_adata] The non-zero matrix entries looked like integer counts.")
            adata.X.data = adata.X.data.astype("int32")
        else:
            logger.info(
                "[process_adata] Warning! The non-zero matrix entries were not all integer-like. "
                "The original values will be kept."
            )
    else:
        logger.info(
            "[process_adata] Warning! `adata.X` did not expose sparse `.data`; integer count checking was skipped."
        )

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.raw = adata
    _print_stage("Normalization and log1p finished.")

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        subset=True,
        layer="counts",
        flavor="seurat_v3",
        batch_key=batch_key,
        span=span,
    )
    _print_stage(f"HVG selection finished with {adata.n_vars} genes.")

    scvi.model.SCVI.setup_anndata(
        adata,
        layer="counts",
        batch_key=batch_key,
        continuous_covariate_keys=covariates if covariates else None,
    )
    _print_stage("SCVI anndata setup finished.")

    train_start = time.time()
    model = scvi.model.SCVI(adata)
    _print_stage("SCVI model initialized.")
    model.train(
        max_epochs=max_epochs,
        early_stopping=early_stopping,
        batch_size=batch_size,
        train_size=train_size,
        validation_size=validation_size,
        early_stopping_monitor=early_stopping_monitor,
        early_stopping_patience=early_stopping_patience,
        check_val_every_n_epoch=check_val_every_n_epoch,
    )
    train_elapsed = time.time() - train_start
    _print_stage(f"Training finished in {train_elapsed / 3600:.2f} hours.")

    try:
        last_elbo = model.history[early_stopping_monitor][-1]
        _print_stage(f"Final monitored metric `{early_stopping_monitor}`: {float(last_elbo):.4f}")
    except Exception:
        logger.info(
            f"[process_adata] Warning! The final value of `{early_stopping_monitor}` could not be retrieved from model history."
        )

    model_save_path = os.path.join(save_addr, prefix)
    model.save(dir_path=model_save_path, overwrite=True)
    _print_stage(f"Model was saved to: '{model_save_path}'.")

    latent = model.get_latent_representation()
    adata.obsm["X_scVI"] = latent
    _print_stage(f"Latent representation shape: {latent.shape}.")

    normalized = model.get_normalized_expression(
        library_size=target_sum,
        n_samples=1,
        transform_batch=None,
    )
    adata.layers["scvi_normalized"] = normalized
    _print_stage("Normalized expression matrix was computed.")

    out_path = os.path.join(save_addr, f"Step04_{prefix}.h5ad")
    adata.write_h5ad(out_path)
    total_time = time.time() - start_time
    _print_stage(f"Corrected AnnData was saved to: '{out_path}'.")
    _print_stage(f"Total elapsed time: {total_time / 60:.1f} minutes.")
    return adata
