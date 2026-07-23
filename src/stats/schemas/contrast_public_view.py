from __future__ import annotations

import numpy as np
import pandas as pd

from src.stats.schemas.decision_rules import DecisionRule
from src.stats.schemas.estimands import ESTIMAND_COMPATIBILITY_LEVELS
from src.stats.validation import parse_boolean_series


REQUIRED_PUBLIC_COLUMNS = {
    "result_id", "evidence_id", "method", "method_version", "run_id", "analysis_id", "cell_type",
    "contrast_id", "contrast_definition", "contrast_type", "result_scope", "group_1", "group_2",
    "reference_group", "reference_cell_type", "effect_component", "estimate", "effect_estimand",
    "effect_scale", "effect_null", "effect_direction", "direction_basis", "primary_decision", "decision_metric",
    "decision_value", "decision_operator", "decision_threshold", "decision_rule_id",
    "decision_rule_description", "is_available", "is_valid", "contrast_status", "failure_reason",
    "diagnostic_id", "reference_strategy", "reference_selection_reason", "reference_is_fixed",
    "is_benchmark_eligible", "estimand_compatibility", "derived_from_native_effect",
}

_FORBIDDEN_PUBLIC_COLUMNS = {
    "evidence_paradigm", "native_decision", "native_decision_metric", "native_decision_value",
    "native_decision_rule_id", "standard_error", "statistic", "statistic_type", "pvalue",
    "pvalue_raw", "pvalue_adjusted", "pvalue_type", "confidence_interval_lower",
    "confidence_interval_upper", "credible_interval_lower", "credible_interval_upper",
    "posterior_probability", "posterior_probability_type", "posterior_inclusion_probability",
    "posterior_probability_direction", "native_discovery_metric_name",
    "native_discovery_metric_value",
    "confidence_score", "calibrated_confidence", "calibrated_score", "benchmark_positive",
}

_STATUS_MAP = {
    "success": (True, True, True),
    "invalid": (True, False, False),
    "unavailable": (False, False, False),
    "failed": (False, False, False),
    "not_tested": (False, False, False),
    "reference": (False, False, False),
}

_EFFECT_DIRECTIONS = {
    "group_1_higher", "group_2_higher", "no_effect", "undetermined", "not_applicable"
}


def validate_contrast_public_view(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_PUBLIC_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Public contrast view is missing required columns: {sorted(missing)}")
    forbidden = _FORBIDDEN_PUBLIC_COLUMNS & set(frame.columns)
    if forbidden:
        raise ValueError(f"Paradigm-specific evidence is forbidden in the public view: {sorted(forbidden)}")
    if frame["result_id"].isna().any() or frame["result_id"].duplicated().any():
        raise ValueError("`result_id` must be non-missing and unique.")
    if frame[["run_id", "analysis_id"]].isna().any().any():
        raise ValueError("`run_id` and `analysis_id` must both be non-missing.")
    if frame["run_id"].astype(str).eq(frame["analysis_id"].astype(str)).any():
        raise ValueError("`run_id` and `analysis_id` are distinct identifiers and cannot be aliases.")

    result = frame.copy()
    for column in (
        "primary_decision", "is_available", "is_valid", "reference_is_fixed",
        "is_benchmark_eligible", "derived_from_native_effect",
    ):
        result[column] = parse_boolean_series(result[column])
    nonnullable_booleans = (
        "is_available", "is_valid", "reference_is_fixed",
        "is_benchmark_eligible", "derived_from_native_effect",
    )
    if result[list(nonnullable_booleans)].isna().any().any():
        raise ValueError(f"{nonnullable_booleans} must be non-missing booleans.")

    for index, row in result.iterrows():
        status = row["contrast_status"]
        if status not in _STATUS_MAP:
            raise ValueError(f"Unknown contrast status at row {index}: {status!r}")
        expected_available, expected_valid, expects_decision = _STATUS_MAP[status]
        if bool(row["is_available"]) != expected_available or bool(row["is_valid"]) != expected_valid:
            raise ValueError(
                f"Status/availability/validity mismatch at row {index}: "
                f"status={status!r}, is_available={row['is_available']!r}, "
                f"is_valid={row['is_valid']!r}."
            )
        if expects_decision != pd.notna(row["primary_decision"]):
            raise ValueError(f"Status/primary_decision mismatch at row {index}.")
        if status == "success" and pd.isna(row["evidence_id"]):
            raise ValueError(f"Successful result lacks evidence_id at row {index}.")
        if status in {"invalid", "unavailable", "failed"} and pd.isna(row["failure_reason"]):
            raise ValueError(f"Non-success result lacks failure_reason at row {index}.")

        estimate = pd.to_numeric(pd.Series([row["estimate"]]), errors="coerce").iloc[0]
        effect_null = pd.to_numeric(pd.Series([row["effect_null"]]), errors="coerce").iloc[0]
        if row["effect_direction"] not in _EFFECT_DIRECTIONS:
            raise ValueError(f"Unknown effect_direction at row {index}: {row['effect_direction']!r}")
        if pd.isna(estimate):
            direction_only_result = (
                row["estimand_compatibility"] == "decision_only"
                and row["effect_direction"] in {
                    "group_1_higher", "group_2_higher", "no_effect", "undetermined",
                }
                and pd.notna(row["direction_basis"])
            )
            if row["effect_direction"] != "not_applicable" and not direction_only_result:
                raise ValueError(
                    "A missing estimate may carry direction only for a decision-only "
                    f"consensus result with explicit direction_basis (row {index})."
                )
        else:
            required_effect = (
                "effect_estimand", "effect_scale", "effect_null", "effect_direction", "direction_basis"
            )
            if any(pd.isna(row[column]) or row[column] == "" for column in required_effect):
                raise ValueError(f"Incomplete effect semantics at row {index}.")
            if row["effect_direction"] == "not_applicable":
                raise ValueError(f"Finite estimate cannot have not_applicable direction at row {index}.")
            expected_direction = "group_1_higher" if estimate > effect_null else (
                "group_2_higher" if estimate < effect_null else "no_effect"
            )
            if row["effect_direction"] != expected_direction:
                raise ValueError(f"Effect direction conflicts with effect_null at row {index}.")

        if row["estimand_compatibility"] not in ESTIMAND_COMPATIBILITY_LEVELS:
            raise ValueError(f"Unknown estimand_compatibility at row {index}.")
        if bool(row["derived_from_native_effect"]) and row.get("effect_estimate_source") != "derived":
            raise ValueError("Derived effects must be stored separately with effect_estimate_source='derived'.")

        if pd.notna(row["reference_cell_type"]):
            forbidden_estimands = {"absolute_abundance_difference", "proportion_difference"}
            if row["effect_estimand"] in forbidden_estimands:
                raise ValueError(
                    "Reference-cell-relative compositional effects cannot be absolute abundance/proportion differences."
                )
            if pd.notna(estimate) and "reference_cell_type=" not in str(row["direction_basis"]):
                raise ValueError("Relative compositional direction_basis must name the reference cell type.")
            if pd.isna(row["reference_strategy"]) or pd.isna(row["reference_selection_reason"]):
                raise ValueError("Reference-cell-relative effects require strategy and selection reason.")

        if status == "reference":
            if pd.notna(estimate) or pd.notna(row["primary_decision"]):
                raise ValueError("Reference rows require estimate=NA and primary_decision=NA.")
            if pd.isna(row["reference_cell_type"]) or not bool(row["reference_is_fixed"]):
                raise ValueError("Reference rows must preserve the fixed reference cell type.")
        if bool(row["is_benchmark_eligible"]):
            if status != "success" or row["effect_component"] != "composition":
                raise ValueError("Only successful composition rows can be benchmark eligible.")
            if (
                pd.notna(row["reference_cell_type"])
                and str(row["cell_type"]) == str(row["reference_cell_type"])
            ):
                raise ValueError("The reference cell-type row cannot be benchmark eligible.")

        if expects_decision:
            provenance = (
                "decision_metric", "decision_value", "decision_operator",
                "decision_threshold", "decision_rule_id",
            )
            if any(pd.isna(row[column]) or row[column] == "" for column in provenance):
                raise ValueError(f"Incomplete primary decision provenance at row {index}.")
            rule = DecisionRule(
                rule_id=str(row["decision_rule_id"]),
                metric=str(row["decision_metric"]),
                operator=str(row["decision_operator"]),
                threshold=row["decision_threshold"],
                description=str(row["decision_rule_description"]),
            )
            if rule.evaluate(row["decision_value"]) != bool(row["primary_decision"]):
                raise ValueError(f"Primary decision cannot be recomputed at row {index}.")
        else:
            provenance = (
                "decision_metric", "decision_value", "decision_operator",
                "decision_threshold", "decision_rule_id",
            )
            if any(pd.notna(row[column]) for column in provenance):
                raise ValueError(f"Unavailable/invalid result contains decision provenance at row {index}.")
    return result
