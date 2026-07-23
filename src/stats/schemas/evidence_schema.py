from __future__ import annotations

import numpy as np
import pandas as pd

from src.stats.validation import parse_boolean_series


REQUIRED_EVIDENCE_COLUMNS = {
    "evidence_id",
    "result_id",
    "evidence_paradigm",
    "native_decision",
    "native_decision_metric",
    "native_decision_value",
    "native_decision_rule_id",
}

_COMMON_SCORE_COLUMNS = {"confidence_score", "calibrated_confidence", "calibrated_score"}
_PROBABILITY_COLUMNS = {
    "pvalue_raw",
    "pvalue_adjusted",
    "posterior_probability",
    "posterior_inclusion_probability",
    "posterior_probability_direction",
}


def validate_evidence_layer(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_EVIDENCE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Evidence layer is missing required columns: {sorted(missing)}")
    forbidden = _COMMON_SCORE_COLUMNS & set(frame.columns)
    if forbidden:
        raise ValueError(f"Common calibrated evidence scores are forbidden: {sorted(forbidden)}")
    if frame["evidence_id"].isna().any() or frame["evidence_id"].duplicated().any():
        raise ValueError("`evidence_id` must be non-missing and unique.")
    if frame["result_id"].isna().any() or frame["result_id"].duplicated().any():
        raise ValueError("Evidence layer must have one row per unique `result_id`.")

    result = frame.copy()
    result["native_decision"] = parse_boolean_series(result["native_decision"])
    allowed_paradigms = {"frequentist", "bayesian", "other_native"}
    invalid = set(result["evidence_paradigm"].dropna()) - allowed_paradigms
    if invalid:
        raise ValueError(f"Unknown evidence paradigms: {sorted(invalid)}")

    for column in _PROBABILITY_COLUMNS & set(result.columns):
        numeric = pd.to_numeric(result[column], errors="coerce")
        supplied = result[column].notna()
        if (supplied & (numeric.isna() | ~numeric.between(0.0, 1.0))).any():
            raise ValueError(f"{column!r} must be in [0, 1] or missing.")

    bayesian = result["evidence_paradigm"].eq("bayesian")
    frequentist = result["evidence_paradigm"].eq("frequentist")
    for column in ("pvalue_raw", "pvalue", "pvalue_adjusted"):
        if column in result.columns and result.loc[bayesian, column].notna().any():
            raise ValueError(f"Bayesian evidence must not populate {column!r}.")
    for column in (
        "posterior_probability", "posterior_inclusion_probability",
        "posterior_probability_direction", "credible_interval_lower",
        "credible_interval_upper",
    ):
        if column in result.columns and result.loc[frequentist, column].notna().any():
            raise ValueError(f"Frequentist evidence must not populate {column!r}.")
    if "pvalue_type" in result.columns:
        invalid_type = bayesian & result["pvalue_type"].fillna("not_applicable").ne("not_applicable")
        if invalid_type.any():
            raise ValueError("Bayesian evidence requires pvalue_type='not_applicable'.")
    if {"pvalue", "pvalue_raw"}.issubset(result.columns):
        both = result["pvalue"].notna() & result["pvalue_raw"].notna()
        if not np.allclose(
            pd.to_numeric(result.loc[both, "pvalue"], errors="coerce"),
            pd.to_numeric(result.loc[both, "pvalue_raw"], errors="coerce"),
            equal_nan=True,
        ):
            raise ValueError("`pvalue` compatibility alias must equal `pvalue_raw` elementwise.")

    native_decision = result["native_decision"].notna()
    native_provenance = (
        "native_decision_metric", "native_decision_value", "native_decision_rule_id"
    )
    incomplete_native = native_decision & result[list(native_provenance)].isna().any(axis=1)
    if incomplete_native.any():
        raise ValueError("A populated native_decision requires native decision provenance.")
    return result
