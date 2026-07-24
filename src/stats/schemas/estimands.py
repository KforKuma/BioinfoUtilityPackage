from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


ESTIMAND_COMPATIBILITY_LEVELS = {
    "compatible",
    "direction_only",
    "decision_only",
    "incompatible",
    "unavailable",
}


@dataclass(frozen=True)
class DerivedEffect:
    value: float
    source_scale: str
    target_scale: str
    status: str
    reason: str | None
    derived_from_native_effect: bool


def derive_effect(native_value: Any, source_scale: str, target_scale: str) -> DerivedEffect:
    """Perform only explicit reversible scale conversions; otherwise return unavailable."""
    value = pd.to_numeric(pd.Series([native_value]), errors="coerce").iloc[0]
    if not np.isfinite(value):
        return DerivedEffect(np.nan, source_scale, target_scale, "unavailable", "native_effect_unavailable", True)
    if (source_scale, target_scale) in {("log_ratio", "ratio"), ("log_odds", "odds_ratio")}:
        return DerivedEffect(float(np.exp(value)), source_scale, target_scale, "available", None, True)
    if (source_scale, target_scale) in {("ratio", "log_ratio"), ("odds_ratio", "log_odds")}:
        if value <= 0:
            return DerivedEffect(np.nan, source_scale, target_scale, "unavailable", "nonpositive_ratio", True)
        return DerivedEffect(float(np.log(value)), source_scale, target_scale, "available", None, True)
    return DerivedEffect(
        np.nan,
        source_scale,
        target_scale,
        "unavailable",
        "unsupported_or_information_incomplete_conversion",
        True,
    )


def check_estimand_compatibility(
    method: str,
    effect_estimand: Any,
    effect_scale: Any,
    reference_cell_type: Any,
    benchmark_estimand: str,
) -> str:
    """Classify native effects without numerically converting across estimands."""
    if pd.isna(effect_estimand) or pd.isna(effect_scale) or not benchmark_estimand:
        return "unavailable"
    estimand = str(effect_estimand)
    scale = str(effect_scale)
    benchmark = str(benchmark_estimand)
    if "unknown" in estimand or "unknown" in scale:
        return "unavailable"

    if estimand == benchmark:
        return "compatible"
    if "variability" in estimand and "composition" in benchmark:
        return "incompatible"
    if pd.notna(reference_cell_type) and str(reference_cell_type):
        if "relative_compositional" not in estimand and "pairwise_celltype_log_ratio" not in estimand:
            return "incompatible"
        return "direction_only"
    if method.lower() in {
        "propeller", "dcats", "sccomp", "sccoda", "naive_welch_proportion",
        "clr_lmm",
    }:
        if scale in {
            "logit", "arcsine_sqrt", "log_odds", "logit_unconstrained", "log_ratio",
            "proportion_difference", "clr_log_ratio",
        }:
            return "direction_only"
    return "incompatible"
