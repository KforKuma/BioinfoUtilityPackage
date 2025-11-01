def process_adata(adata, prefix, save_addr,
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
                  n_top_genes=1000):
    """
    Process a single .h5ad file through normalization, HVG selection, and SCVI model training.

    Parameters
    ----------
    save_addr : str
        Directory path where output files and model will be saved.
    max_epochs : int
        Maximum number of training epochs for SCVI.
    batch_size : int
        Batch size for training.
    train_size : float
        Proportion of data for training.
    validation_size : float
        Proportion of data for validation.
    early_stopping : bool
        Whether to apply early stopping.
    early_stopping_monitor : str
        Metric to monitor for early stopping.
    early_stopping_patience : int
        Patience epochs for early stopping.
    check_val_every_n_epoch : int
        Frequency of validation checks.
    target_sum : float
        Target sum for normalization.
    n_top_genes : int
        Number of highly variable genes to select.

    Returns
    -------
    anndata.AnnData
        The processed AnnData object with SCVI results.
    """
    start_time = time.time()
    print(f"\n=== Processing {prefix} ===")
    
    # 1. Check data
    print(f"Read {prefix}: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # 2. Preserve counts and normalize + log1p
    nz = adata.X.data
    all_integer = np.all(nz % 1 == 0)
    if all_integer:
        print("所有非零元素均为整数")
        adata.X.data = adata.X.data.astype("int32")
    else:
        print("所有非零元素并非均为整数")
        rows, cols = adata.X.nonzero()
        values = adata.X.data
        for i in range(min(100, len(values))):
            r, c = rows[i], cols[i]
            v = values[i]
            print(f"cell {adata.obs_names[r]} (row {r}), gene {adata.var_names[c]} (col {c}): {v}")
    
    adata.layers["counts"] = adata.X.copy()
    # normalization and log1p
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.raw = adata
    print("  • Normalization & log1p: done")
    
    # 3. Highly variable genes
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        subset=True,
        layer="counts",
        flavor="seurat_v3",
        batch_key="orig.ident",
        span=span,
    )
    print(f"  • Selected HVGs: {adata.n_vars} genes")
    
    # 4. Setup AnnData for scvi-tools
    scvi.model.SCVI.setup_anndata(
        adata,
        layer="counts",
        batch_key="orig.ident",
        continuous_covariate_keys=["percent.mt", "percent.ribo"],
    )
    print("  • SCVI anndata setup: done")
    
    # 5. Initialize and train SCVI
    t0 = time.time()
    model = scvi.model.SCVI(adata)
    print("  • SCVI model initialized")
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
    elapsed = time.time() - t0
    print(f"  • Training completed in {elapsed / 3600:.2f} h")
    try:
        last_elbo = model.history[early_stopping_monitor][-1]
        print(f"  • Final validation ELBO: {last_elbo:.2f}")
    except Exception:
        pass
    
    # 6. Save model
    model_save_path = os.path.join(save_addr, prefix)
    model.save(dir_path=model_save_path)
    print(f"  • Model saved at '{model_save_path}'")
    
    # 7. Extract latent representation
    latent = model.get_latent_representation()
    adata.obsm["X_scVI"] = latent
    print(f"  • Latent repr shape: {latent.shape}")
    
    # 8. Get batch-corrected expressions
    normalized = model.get_normalized_expression(
        library_size=target_sum,
        n_samples=1,
        transform_batch=None,
    )
    adata.layers["scvi_normalized"] = normalized
    print("  • Normalized expression matrix computed")
    
    # 9. Write output
    out_path = os.path.join(save_addr, f"Step04_{prefix}.h5ad")
    adata.write_h5ad(out_path)
    total_time = time.time() - start_time
    print(f"Saved corrected AnnData to {out_path} ({total_time / 60:.1f} min elapsed)")
    
    return adata
