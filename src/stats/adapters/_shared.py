from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pandas as pd

from src.stats.schemas import CanonicalDAInput, check_estimand_compatibility
from src.stats.validation import parse_boolean_value


class NativeAdapterError(RuntimeError):
    """Expected native-backend failure with a stable canonical reason code."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class NativeInput:
    abundance: pd.DataFrame
    sample_manifest: pd.DataFrame
    cell_type_manifest: pd.DataFrame
    contrast: dict[str, Any]
    options: dict[str, Any]


NativeExecutor = Callable[[NativeInput, pd.Series], pd.DataFrame]


def prepare_pairwise_input(
    canonical_input: CanonicalDAInput,
    contrast: pd.Series,
    *,
    options: dict[str, Any] | None = None,
) -> NativeInput:
    if str(contrast["contrast_type"]) != "pairwise":
        raise NativeAdapterError("unsupported_contrast", "This adapter currently supports pairwise contrasts only.")
    factor = contrast.get("factor")
    if pd.isna(factor) or str(factor) not in canonical_input.sample_manifest.columns:
        raise NativeAdapterError("factor_unavailable", "The contrast factor is absent from sample_manifest.")
    factor = str(factor)
    group_1, group_2 = str(contrast["group_1"]), str(contrast["group_2"])

    samples = canonical_input.sample_manifest.loc[
        canonical_input.sample_manifest["inclusion_status"].eq("included")
    ].copy()
    samples = samples.loc[samples[factor].astype(str).isin([group_1, group_2])].copy()
    levels = set(samples[factor].astype(str))
    if levels != {group_1, group_2}:
        raise NativeAdapterError("insufficient_groups", "Both requested contrast groups must be present.")
    counts_per_group = samples.groupby(samples[factor].astype(str), observed=True)["sample_id"].nunique()
    if (counts_per_group.reindex([group_1, group_2], fill_value=0) < 2).any():
        raise NativeAdapterError("insufficient_replicates", "Each contrast group requires at least two samples.")

    cell_types = canonical_input.cell_type_manifest.loc[
        canonical_input.cell_type_manifest["inclusion_status"].eq("included")
    ].copy()
    abundance = canonical_input.abundance_long.loc[
        canonical_input.abundance_long["sample_id"].astype(str).isin(samples["sample_id"].astype(str))
        & canonical_input.abundance_long["cell_type"].astype(str).isin(cell_types["cell_type"].astype(str))
    ].copy()
    abundance["sample_id"] = abundance["sample_id"].astype(str)
    abundance["cell_type"] = abundance["cell_type"].astype(str)
    samples["sample_id"] = samples["sample_id"].astype(str)
    cell_types["cell_type"] = cell_types["cell_type"].astype(str)

    spec = {key: (None if pd.isna(value) else value) for key, value in contrast.to_dict().items()}
    spec["factor"] = factor
    return NativeInput(abundance, samples, cell_types, spec, dict(options or {}))


def stable_ids(
    analysis_id: str,
    method_id: str,
    contrast_id: str,
    cell_type: str,
    effect_component: str,
) -> tuple[str, str]:
    result_id = str(uuid5(
        NAMESPACE_URL,
        f"{analysis_id}:{method_id}:{contrast_id}:{cell_type}:{effect_component}",
    ))
    return result_id, str(uuid5(NAMESPACE_URL, f"evidence:{result_id}"))


def effect_direction(estimate: float, *, null: float = 0.0) -> str:
    if not np.isfinite(estimate):
        return "not_applicable"
    return "group_1_higher" if estimate > null else (
        "group_2_higher" if estimate < null else "no_effect"
    )


def public_row(
    *,
    method_id: str,
    method_version: str,
    analysis_id: str,
    diagnostic_id: str,
    contrast: pd.Series,
    cell_type: str,
    effect_component: str,
    estimate: float,
    effect_estimand: str,
    effect_scale: str,
    direction_basis: str,
    decision_rule_id: str,
    reference_cell_type: Any = pd.NA,
    effect_estimate_source: str = "method_native",
    result_interpretation: Any = pd.NA,
    effect_null: float = 0.0,
    reference_strategy: Any = "not_applicable",
    reference_selection_reason: Any = pd.NA,
    reference_is_fixed: bool = False,
    is_benchmark_eligible: bool = True,
    benchmark_estimand: str = "proportion_difference",
    derived_from_native_effect: bool = False,
) -> tuple[dict[str, Any], str, str]:
    result_id, evidence_id = stable_ids(
        analysis_id, method_id, str(contrast["contrast_id"]), str(cell_type), effect_component
    )
    estimate_value = float(estimate) if pd.notna(estimate) else float("nan")
    row = {
        "result_id": result_id,
        "evidence_id": evidence_id,
        "method": method_id,
        "method_version": method_version,
        "analysis_id": analysis_id,
        "cell_type": cell_type,
        "contrast_id": contrast["contrast_id"],
        "contrast_definition": contrast["contrast_definition"],
        "contrast_type": contrast["contrast_type"],
        "result_scope": "cell_type_specific",
        "group_1": contrast["group_1"],
        "group_2": contrast["group_2"],
        "reference_group": contrast.get("reference_group", contrast["group_2"]),
        "reference_cell_type": reference_cell_type,
        "effect_component": effect_component,
        "estimate": estimate_value,
        "effect_estimand": effect_estimand,
        "effect_scale": effect_scale,
        "effect_null": effect_null,
        "effect_direction": effect_direction(estimate_value),
        "direction_basis": direction_basis,
        "reference_strategy": reference_strategy,
        "reference_selection_reason": reference_selection_reason,
        "reference_is_fixed": reference_is_fixed,
        "is_benchmark_eligible": is_benchmark_eligible,
        "estimand_compatibility": check_estimand_compatibility(
            method_id,
            effect_estimand,
            effect_scale,
            reference_cell_type,
            benchmark_estimand,
        ),
        "derived_from_native_effect": derived_from_native_effect,
        "effect_estimate_source": effect_estimate_source,
        "result_interpretation": result_interpretation,
        "primary_decision": pd.NA,
        "decision_metric": pd.NA,
        "decision_value": pd.NA,
        "decision_operator": pd.NA,
        "decision_threshold": pd.NA,
        "decision_rule_id": decision_rule_id,
        "decision_rule_description": pd.NA,
        "is_available": True,
        "is_valid": True,
        "contrast_status": "success",
        "failure_reason": pd.NA,
        "diagnostic_id": diagnostic_id,
    }
    return row, result_id, evidence_id


def frequentist_evidence(
    *,
    evidence_id: str,
    result_id: str,
    pvalue_raw: Any,
    pvalue_adjusted: Any,
    native_rule_id: str,
    adjustment_family: str,
    alpha: float,
    test_name: str,
    statistic: Any = pd.NA,
    statistic_type: Any = pd.NA,
    standard_error: Any = pd.NA,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = pd.to_numeric(pd.Series([pvalue_raw]), errors="coerce").iloc[0]
    adjusted = pd.to_numeric(pd.Series([pvalue_adjusted]), errors="coerce").iloc[0]
    native = bool(np.isfinite(adjusted) and adjusted < alpha)
    row = {
        "evidence_id": evidence_id,
        "result_id": result_id,
        "evidence_paradigm": "frequentist",
        "native_decision": native,
        "native_decision_metric": "pvalue_adjusted",
        "native_decision_value": adjusted,
        "native_decision_rule_id": native_rule_id,
        "standard_error": standard_error,
        "statistic": statistic,
        "statistic_type": statistic_type,
        "pvalue_raw": raw,
        "pvalue": raw,
        "pvalue_adjusted": adjusted,
        "pvalue_type": "adjusted",
        "test_name": test_name,
        "adjustment_method": "BH",
        "adjustment_family": adjustment_family,
        "nominal_alpha": alpha,
    }
    row.update(extra or {})
    return row


def bayesian_evidence(
    *,
    evidence_id: str,
    result_id: str,
    native_decision: Any,
    native_metric: str,
    native_value: Any,
    native_rule_id: str,
    posterior_probability: Any = pd.NA,
    posterior_probability_type: Any = pd.NA,
    posterior_inclusion_probability: Any = pd.NA,
    native_discovery_metric_name: Any = pd.NA,
    native_discovery_metric_value: Any = pd.NA,
    credible_interval_lower: Any = pd.NA,
    credible_interval_upper: Any = pd.NA,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "evidence_id": evidence_id,
        "result_id": result_id,
        "evidence_paradigm": "bayesian",
        "native_decision": parse_boolean_value(native_decision),
        "native_decision_metric": native_metric,
        "native_decision_value": native_value,
        "native_decision_rule_id": native_rule_id,
        "posterior_probability": posterior_probability,
        "posterior_probability_type": posterior_probability_type,
        "posterior_inclusion_probability": posterior_inclusion_probability,
        "native_discovery_metric_name": native_discovery_metric_name,
        "native_discovery_metric_value": native_discovery_metric_value,
        "credible_interval_lower": credible_interval_lower,
        "credible_interval_upper": credible_interval_upper,
        "pvalue_raw": pd.NA,
        "pvalue": pd.NA,
        "pvalue_adjusted": pd.NA,
        "pvalue_type": "not_applicable",
        "adjustment_method": "not_applicable",
    }
    row.update(extra or {})
    return row


def require_columns(frame: pd.DataFrame, columns: set[str], method: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise NativeAdapterError(
            "invalid_native_result",
            f"{method} native output is missing columns: {sorted(missing)}",
        )
