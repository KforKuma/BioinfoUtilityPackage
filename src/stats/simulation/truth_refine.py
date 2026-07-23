from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import softmax


def attach_population_effects(
    df_true_effect: pd.DataFrame,
    *,
    baseline_logits: np.ndarray,
    cell_types: Sequence[str],
    disease_effects: Mapping[str, np.ndarray],
    tissue_effect: np.ndarray,
    interaction_effects: Mapping[str, np.ndarray],
    population_null_tolerance: float = 1e-12,
    population_reference_cell_type: str | None = None,
) -> pd.DataFrame:
    """Attach an explicit population estimand and preserve marginal closure effects.

    With a reference cell type, the primary population estimand is the exact change
    in ``log(p_cell / p_reference)`` induced by the model's additive log effect. The
    marginal expected proportion difference is always retained separately.
    """
    truth = df_true_effect.copy()
    baseline = np.asarray(baseline_logits, dtype=float)
    if baseline.ndim == 1:
        baseline = baseline[None, :]
    if baseline.ndim != 2 or baseline.shape[1] != len(cell_types):
        raise ValueError("`baseline_logits` must have one column per cell type.")
    if not np.isfinite(baseline).all():
        raise ValueError("`baseline_logits` must contain only finite values.")
    if not np.isfinite(population_null_tolerance) or population_null_tolerance < 0:
        raise ValueError("`population_null_tolerance` must be finite and non-negative.")

    cell_index = {str(cell_type): index for index, cell_type in enumerate(cell_types)}
    reference_index = (
        cell_index.get(str(population_reference_cell_type))
        if population_reference_cell_type is not None else None
    )
    if population_reference_cell_type is not None and reference_index is None:
        raise ValueError("`population_reference_cell_type` is absent from cell_types.")
    baseline_probability = softmax(baseline, axis=1).mean(axis=0)
    population_effects: list[float] = []
    marginal_effects: list[float] = []
    population_available: list[bool] = []

    for _, row in truth.iterrows():
        factor = str(row["contrast_factor"])
        group = str(row["contrast_group"])
        cell_type = str(row["cell_type"])
        effect_vector: np.ndarray | None
        if factor == "disease":
            effect_vector = disease_effects.get(group)
        elif factor == "tissue":
            effect_vector = tissue_effect
        elif factor in {"interaction", "addition"}:
            disease_group = group.split(" x ", 1)[0]
            disease_effect = disease_effects.get(disease_group)
            interaction_effect = interaction_effects.get(disease_group)
            effect_vector = (
                None
                if disease_effect is None or interaction_effect is None
                else disease_effect + tissue_effect + interaction_effect
            )
        else:
            effect_vector = None

        index = cell_index.get(cell_type)
        if effect_vector is None or index is None:
            population_effects.append(np.nan)
            marginal_effects.append(np.nan)
            population_available.append(False)
            continue
        effect_vector = np.asarray(effect_vector, dtype=float)
        if effect_vector.shape != (len(cell_types),) or not np.isfinite(effect_vector).all():
            raise ValueError("Every population effect vector must be finite and match cell_types.")
        expected_probability = softmax(baseline + effect_vector, axis=1).mean(axis=0)
        marginal_effect = float(expected_probability[index] - baseline_probability[index])
        marginal_effects.append(marginal_effect)
        population_effects.append(
            float(effect_vector[index] - effect_vector[reference_index])
            if reference_index is not None else marginal_effect
        )
        population_available.append(True)

    truth["Population_Effect"] = population_effects
    truth["Population_Effect_Estimand"] = (
        "reference_relative_population_log_ratio_change"
        if reference_index is not None else
        "noise_free_marginal_expected_proportion_difference"
    )
    truth["Population_Effect_Scale"] = (
        "natural_log_ratio" if reference_index is not None else "proportion_difference"
    )
    truth["Population_Reference_Cell_Type"] = (
        str(population_reference_cell_type) if population_reference_cell_type is not None else pd.NA
    )
    population_nonzero = [
        abs(value) > population_null_tolerance if available else pd.NA
        for value, available in zip(population_effects, population_available, strict=True)
    ]
    truth["Is_Population_Nonzero"] = pd.Series(
        population_nonzero,
        dtype="boolean",
    )
    truth["Marginal_Population_Effect"] = marginal_effects
    truth["Marginal_Population_Effect_Estimand"] = (
        "noise_free_marginal_expected_proportion_difference"
    )
    truth["Marginal_Population_Effect_Scale"] = "proportion_difference"
    marginal_nonzero = [
        abs(value) > population_null_tolerance if available else pd.NA
        for value, available in zip(marginal_effects, population_available, strict=True)
    ]
    truth["Is_Marginal_Population_Nonzero"] = pd.Series(marginal_nonzero, dtype="boolean")
    injected_support = truth["True_Significant"].astype("boolean")
    closure_induced = pd.Series(
        [
            bool(marginal) and not bool(injected)
            if pd.notna(marginal) and pd.notna(injected) else pd.NA
            for marginal, injected in zip(marginal_nonzero, injected_support, strict=True)
        ],
        dtype="boolean",
    )
    truth["Is_Marginal_Population_Closure_Induced"] = closure_induced
    truth["Is_Population_Closure_Induced"] = closure_induced
    truth["Population_Null_Tolerance"] = population_null_tolerance
    truth["Population_Effect_Status"] = np.where(population_available, "available", "unavailable")
    return truth


def refine_ground_truth_by_observation(
    df_long: pd.DataFrame,
    df_true_effect: pd.DataFrame,
    lfc_threshold: float = 0.2,
    *,
    injected_effect_scale: str = "model_native_log_effect",
) -> pd.DataFrame:
    """Add observed detectability without redefining injected/population truth.

    Observed labels are secondary sensitivity metadata and missing contrasts
    remain missing rather than being replaced by zero.
    """
    required_long_cols = {"cell_type", "disease", "tissue", "prop"}
    missing_long_cols = required_long_cols - set(df_long.columns)
    if missing_long_cols:
        raise ValueError(f"Missing required columns in `df_long`: {sorted(missing_long_cols)}")
    required_truth_cols = {
        "cell_type", "contrast_group", "contrast_ref", "contrast_factor",
        "True_Effect", "True_Significant",
    }
    missing_truth_cols = required_truth_cols - set(df_true_effect.columns)
    if missing_truth_cols:
        raise ValueError(f"Missing required columns in `df_true_effect`: {sorted(missing_truth_cols)}")
    if lfc_threshold < 0:
        raise ValueError("`lfc_threshold` must be non-negative.")

    observed = df_long.copy()
    truth = df_true_effect.copy()
    truth["Injected_Effect"] = pd.to_numeric(truth["True_Effect"], errors="coerce")
    truth["Injected_Effect_Scale"] = injected_effect_scale
    truth["Is_Injected_Nonzero"] = (truth["Injected_Effect"].abs() > 0).astype("boolean")
    if "Population_Effect" not in truth.columns:
        truth["Population_Effect"] = np.nan
        truth["Population_Effect_Scale"] = pd.NA
        truth["Is_Population_Nonzero"] = pd.Series(pd.NA, index=truth.index, dtype="boolean")
        truth["Population_Effect_Status"] = "unavailable"

    observed["status_key"] = (
        observed["disease"].astype(str) + " x " + observed["tissue"].astype(str)
    )
    medians_inter = observed.groupby(["cell_type", "status_key"])["prop"].median().unstack("status_key")
    medians_disease = observed.groupby(["cell_type", "disease"])["prop"].median().unstack("disease")
    medians_tissue = observed.groupby(["cell_type", "tissue"])["prop"].median().unstack("tissue")

    observed_effects: list[float] = []
    detectable_values: list[Any] = []
    observed_status: list[str] = []
    for _, row in truth.iterrows():
        cell_type = row["cell_type"]
        group = str(row["contrast_group"])
        reference = str(row["contrast_ref"])
        factor = row["contrast_factor"]
        try:
            if factor == "disease":
                group_value = medians_disease.loc[cell_type, group]
                reference_value = medians_disease.loc[cell_type, reference]
            elif factor == "tissue":
                group_value = medians_tissue.loc[cell_type, group]
                reference_value = medians_tissue.loc[cell_type, reference]
            else:
                group_value = medians_inter.loc[cell_type, group]
                reference_value = medians_inter.loc[cell_type, reference]
            lfc = float(np.log2((group_value + 1e-6) / (reference_value + 1e-6)))
        except KeyError:
            observed_effects.append(np.nan)
            detectable_values.append(pd.NA)
            observed_status.append("unavailable")
            continue

        injected = bool(row["Is_Injected_Nonzero"])
        if not np.isfinite(group_value) or not np.isfinite(reference_value) or not np.isfinite(lfc):
            observed_effects.append(np.nan)
            detectable_values.append(pd.NA)
            observed_status.append("unavailable")
            continue
        direction_consistent = row["Injected_Effect"] * lfc >= 0
        detectable = injected and abs(lfc) >= lfc_threshold and direction_consistent
        observed_effects.append(lfc)
        detectable_values.append(bool(detectable))
        observed_status.append("available")

    truth["Observed_Effect"] = observed_effects
    truth["Observed_Effect_Scale"] = "log2_fold_change"
    truth["Is_Observed_Detectable"] = pd.Series(detectable_values, index=truth.index, dtype="boolean")
    truth["Observed_Effect_Status"] = observed_status

    # Deprecated compatibility aliases. Benchmark code does not read these.
    truth["Observed_LFC"] = truth["Observed_Effect"]
    truth["Is_Detectable_True"] = truth["Is_Observed_Detectable"]
    return truth
