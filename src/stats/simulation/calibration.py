from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.stats.evaluation.evaluator_dm import estimate_DM_parameters


def get_refined_noise_estimates(df_real: pd.DataFrame) -> dict[str, float]:
    """Estimate donor and sample CLR noise for simulation calibration."""
    required = {"donor_id", "sample_id", "disease", "tissue", "cell_type", "count"}
    if missing := required - set(df_real.columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    wide = df_real.pivot_table(
        index=["donor_id", "sample_id", "disease", "tissue"],
        columns="cell_type",
        values="count",
        fill_value=0,
    )
    if wide.empty:
        raise ValueError("Input data contains no count rows after pivoting.")
    values = wide.values + 1.0
    clr = pd.DataFrame(
        np.log(values) - np.log(values).mean(axis=1, keepdims=True),
        index=wide.index,
        columns=wide.columns,
    ).reset_index()
    donor_sd: list[float] = []
    sample_sd: list[float] = []
    samples_per_donor = clr.groupby("donor_id").size().mean()
    for cell_type in wide.columns:
        try:
            model = smf.mixedlm(
                f"Q('{cell_type}') ~ disease + tissue", clr, groups=clr["donor_id"]
            ).fit(reml=True)
            donor_sd.append(float(np.sqrt(max(model.cov_re.iloc[0, 0], 0))))
            sample_sd.append(float(np.sqrt(max(model.scale, 0))))
        except Exception:
            design = sm.add_constant(
                pd.get_dummies(clr[["disease", "tissue"]], drop_first=True, dtype=float)
            )
            residual = sm.OLS(clr[cell_type], design).fit().resid
            work = clr.assign(_residual=residual)
            within = work.groupby("donor_id")["_residual"].std().median()
            between = work.groupby("donor_id")["_residual"].mean().var()
            donor_sd.append(float(np.sqrt(max(between - within ** 2 / samples_per_donor, 0))))
            sample_sd.append(float(within))
    return {
        "donor_noise_sd": float(np.nanmedian(donor_sd)),
        "sample_noise_sd": float(np.nanmedian(sample_sd)),
    }


def get_all_simulation_params(
    df_real: pd.DataFrame,
    collected_results: dict[str, Any],
    ref_disease: str = "HC",
    ref_tissue: str = "nif",
) -> dict[str, dict[str, Any]]:
    """Build the existing DM/LN/resample parameter bundles outside evaluation."""
    required = {"donor_id", "sample_id", "disease", "tissue", "cell_type", "count"}
    if missing := required - set(df_real.columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "disease_levels" not in collected_results:
        raise ValueError("Missing required key: disease_levels")
    noise = get_refined_noise_estimates(df_real)
    reference = df_real.loc[
        df_real["disease"].eq(ref_disease) & df_real["tissue"].eq(ref_tissue)
    ]
    proportions = reference.groupby("cell_type")["count"].sum() + 1
    if proportions.empty:
        raise ValueError("No reference samples found for simulation calibration.")
    baseline_mu_scale = float((np.log(proportions) - np.log(proportions).mean()).std())
    base = estimate_DM_parameters(collected_results)
    sample_sums = df_real.groupby("sample_id")["count"].sum()
    n_donors = df_real["donor_id"].nunique()
    common = {
        "n_donors": n_donors,
        "n_samples_per_donor": int(np.ceil(df_real["sample_id"].nunique() / n_donors)),
        "n_celltypes": df_real["cell_type"].nunique(),
        "disease_effect_size": base["disease_effect_size"],
        "tissue_effect_size": base["tissue_effect_size"],
        "interaction_effect_size": base.get("interaction_effect_size", 0.0),
        "inflamed_cell_frac": base["inflamed_cell_frac"],
        "disease_levels": collected_results["disease_levels"],
    }
    ln = {
        **common, **noise, "baseline_mu_scale": baseline_mu_scale,
        "total_count_mean": float(sample_sums.mean()),
        "total_count_sd": min(float(sample_sums.std()), 500),
    }
    dm = {
        **base, **common, **noise, "sampling_bias_strength": 0.0,
        "total_count_mean": float(sample_sums.mean()),
        "total_count_sd": min(float(sample_sums.std()), 500),
    }
    resample = {**common, **noise}
    return {"ln_params": ln, "dm_params": dm, "resample_params": resample}
